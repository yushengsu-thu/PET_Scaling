# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import copy
import os
import nltk


import functools
import logging
from opendelta.utils.delta_hub import create_hub_repo_name
from opendelta import BitFitModel, AdapterModel, LoraModel, PrefixModel, AutoDeltaModel, SoftPromptModel

import random
import torch
import numpy as np
import sys
import subprocess
import json
import dataclasses
from typing import Optional, List, Tuple
from pathlib import Path
from datasets import load_dataset, load_metric, concatenate_datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from transformers.generation.configuration_utils import GenerationConfig

from examples_seq2seq.data_processors import AutoTask, TaskDataCollatorForSeq2Seq, AutoPostProcessor
from examples_seq2seq.seq2seq_trainer import Seq2SeqTrainer
from examples_seq2seq.ed2lm_seq2seq_trainer import ED2LMSeq2SeqTrainer

from examples_seq2seq.trainers.trainer_utils import save_training_config, EvalPrediction
from dataclasses import dataclass, field

from transformers.models.t5.modeling_t5 import T5Config, T5ForConditionalGeneration
from examples_seq2seq.trainers.model_args import ModelArguments
from examples_seq2seq.trainers.trainer_args import TrainingArguments, DataTrainingArguments
from examples_seq2seq.metrics.metrics import squad_em, squad_NAF1, squad_f1, normalize_answer

from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
from transformers.file_utils import PaddingStrategy
from typing import Optional, Union, List, Dict, Any

from termcolor import colored

logger = logging.getLogger(__name__)


def run_command(command):
    output = subprocess.getoutput(command)
    return output


class MyCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        # self.delta_args = kwargs.pop("delta_args")
        # self.trainer_args = kwargs.pop("trainer_args")
        # self.model_args = kwargs.pop("model_args")
        super(MyCallback, self).__init__(*args, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        control.should_log = True
        return control


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (:obj:`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        tmp_raw_query_list = []
        raw_query_ids = []
        # dealing with pad in .generate(allowed_prefix_token)

        features_copy = copy.deepcopy(features)

        for ins in features_copy:

            if "extra_fields" in ins:
                ins.pop("extra_fields")
            if "task" in ins:
                ins.pop("task")

            if "options" in ins:
                ins.pop("options")

            if "icl_test_source" in ins:
                ins.pop("icl_test_source")

            if "ed2lm_prefix_allowed_text" in ins:
                raw_query_text = ins['ed2lm_prefix_allowed_text']


                tmp_raw_query_list.append(raw_query_text)

                raw_query_id = self.tokenizer.encode(raw_query_text, add_special_tokens=False)
                raw_query_ids.append(raw_query_id)
                ins.pop('ed2lm_prefix_allowed_text')

        # pop in and push back again to avoid error in :`meth`: pad
        try:
            batch = self.tokenizer.pad(
                features_copy,
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,

            )
        except:
            print("error")

        # for index, ins in enumerate(features):
        #     ins['ed2lm_prefix_allowed_tokens'] = tmp_raw_query_list[index]

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        if len(raw_query_ids) != 0:
            batch['ed2lm_prefix_allowed_tokens'] = raw_query_ids

        return batch


@dataclass
class MyTrainingArgument:
    """
    Arguments pertaining to my custom training configuration.
    """

    freeze_encoder: bool = field(
        default=False, metadata={"help": "Used in ed2lm transfer setting"}
    )
    delta_type: str = field(
        default=None, metadata={"help": "Used in ed2lm transfer adapter setting"}
    )

    delta_checkpoint: str = field(
        default=None, metadata={"help": "Used in ed2lm transfer adapter setting"}
    )

    saved_mask_path: str = field(
        default=None, metadata={"help": "calculated mask based on l2_norm"}
    )

    dropout_rate: float = field(
        default=None, metadata={"help": "I think we should specify the dropout rate to 0 when doing mask_tuning"}
    )


class RemainArgHfArgumentParser(HfArgumentParser):
    def parse_json_file(self, json_file: str, return_remaining_args=True):
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        import argparse
        import json
        from pathlib import Path
        import dataclasses

        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: data.pop(k) for k in list(data.keys()) if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)

        remain_args = argparse.ArgumentParser()
        remain_args.__dict__.update(data)
        if return_remaining_args:
            return (*outputs, remain_args)
        else:
            return (*outputs,)

    def parse_args_into_dataclasses(
        self, args=None, return_remaining_strings=False, look_for_args_file=True, args_filename=None
    ):
        """
        Parse command-line args into instances of the specified dataclass types.

        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

        Args:
            args:
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            look_for_args_file:
                If true, will look for a ".args" file with the same base name as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename:
                If not None, will uses this file instead of the ".args" file specified in the previous argument.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """
        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}")

            # my custom hack here, because i need to handle the argument which is a list

            if len(self.dataclass_types) != len(outputs):
                # shift the last arguments to my_delta dataclass
                for remain in outputs[len(self.dataclass_types):]:
                    for key, value in remain.__dict__.items():
                        # self.dataclass_types[-1].__dict__[key] = value #wrong
                        setattr(self.dataclass_types[-1], key, value)



            return (*outputs[:len(self.dataclass_types)],)


def get_attr(args)->Dict:
    # vars() lost some field
    new_dict = {}
    for key in dir(args):
        if not key.startswith("__"):
            new_dict[key] = getattr(args, key)

    return new_dict


def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = RemainArgHfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MyTrainingArgument))

    parser.add_argument("--soft_token_num", type=int, default=100)
    parser.add_argument("--token_init", default=True, action="store_true")
    parser.add_argument("--init_range", type=float, default=0.1)
    parser.add_argument("--unfrozen_modules", type=str, nargs="*", default=["deltas"])
    parser.add_argument("--other_expand_ids", type=json.loads, help="can directly pass '{\"value1\":\"key1\"}' ")

    parser.add_argument("--lora_r", type=int, help="used in lora setting", default=8)
    parser.add_argument("--lora_alpha", type=int, help="used in lora setting", default=16)
    parser.add_argument("--lora_dropout", type=float, help="used in lora setting", default=0.0)

    parser.add_argument("--use_delta", type=int, help="used in delta setting", default=0)
    parser.add_argument("--apet_scale", type=str, help="used in apet setting, it means the param's num can", default=None)
    parser.add_argument("--apet_seed", type=int, help="used in apet setting", default=None)


    parser.add_argument("--label_mapping_type", type=str, default=None, help=" select in number or original ")

    print(sys.argv)
    # exit()

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, delta_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, delta_args = parser.parse_args_into_dataclasses(return_remaining_strings=False)

    # Check parameters before run
    # assert 'google' in model_args.model_name_or_path
    # assert 'v1_1' in model_args.model_name_or_path
    # assert training_args.learning_rate == 3e-4 or training_args.learning_rate == 3e-2
    # assert (training_args.per_device_train_batch_size == 32) or (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps == 32)
    # assert training_args.split_validation_test == False

    ## Uncomment to resume from latest checkpoint
    # training_args.overwrite_output_dir = False


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            '''
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            '''
            pass 
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if delta_args.use_delta == 1:
        config.use_delta = delta_args.use_delta
        config.delta_type = delta_args.delta_type
        # lora
        config.lora_r = delta_args.lora_r
        config.lora_alpha = delta_args.lora_alpha
        config.lora_dropout = delta_args.lora_dropout
        # soft_prompt
        config.soft_token_num = delta_args.soft_token_num
        config.init_range = delta_args.init_range

        if delta_args.delta_type == "soft_prompt":
            data_args.max_target_length += config.soft_token_num
            data_args.val_max_target_length += config.soft_token_num
            data_args.test_max_target_length += config.soft_token_num

    try:
        generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path)
    except:
        generation_config = GenerationConfig.from_pretrained("google/t5-v1_1-small")

    generation_config.max_length = data_args.max_target_length

    if delta_args.dropout_rate is not None:
        config.dropout_rate = delta_args.dropout_rate
        logger.info("we set config.dropout_rate to {}".format(config.dropout_rate))
    # config.dropout_rate = 0.0
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        low_cpu_mem_usage=True if "xxl" in model_args.model_name_or_path and training_args.deepspeed is None else False
    )
    model.resize_token_embeddings(len(tokenizer))



        # if delta_tuning freeze other parameters
    if delta_args.use_delta == 1:

        if delta_args.delta_type == "apet":

            assert hasattr(delta_args, "apet_scale")
            assert hasattr(delta_args, "apet_seed")

            from apet_scale import scale2num

            # t5-specific
            # for name, module in model.named_modules():
            #     if isinstance(module, torch.nn.Linear) and module.bias is None:
            #         module.bias = torch.nn.Parameter(torch.zeros(module.out_features))

            # remember that i change bias=True in modeling_t5
            allowed_num = scale2num[model_args.model_name_or_path][delta_args.apet_scale]

            inner_param = []
            delta_param = []

            all_param = []
            picked_param = []

            # make sure that if we pick lora_A, then lora_B should be picked as well

            for name, param in model.named_parameters():
                param.requires_grad_(False)
                if hasattr(param, "ds_shape"):

                    if "lora" in name or "adapter" in name:
                        delta_param.append([name, param])
                    else:
                        inner_param.append([name, param])

                    print(f"name: {name} \t shape: {param.ds_shape}, numel: {param.ds_numel}")

            # group together
            index = 0
            while index < len(delta_param):
                if "lora" in delta_param[index][0]:
                    all_param.append(delta_param[index: index + 2])
                    index += 2
                elif "adapter" in delta_param[index][0]:
                    all_param.append(delta_param[index: index + 4])
                    index += 4
                else:
                    raise ValueError("find no lora or adapter")

            # all_param += inner_param

            random.seed(allowed_num * delta_args.apet_seed)
            random.shuffle(all_param)
            random.shuffle(inner_param)

            inner_allowed = delta_allowed = allowed_num // 2

            while len(all_param) > 0 and delta_allowed > 0:
                # if all_param[0][1].ds_numel <= allowed_num:

                if all_param[0][0][1].ds_numel <= delta_allowed:
                    picked_param.extend(all_param[0])
                    for ins in all_param[0]:
                        delta_allowed -= ins[1].ds_numel
                all_param.pop(0)

            while len(inner_param) > 0 and inner_allowed > 0:

                if inner_param[0][1].ds_numel <= inner_allowed:
                    picked_param.append(inner_param[0])
                    inner_allowed -= inner_param[0][1].ds_numel

                inner_param.pop(0)

            print(f"we pick this param to be updated {picked_param}")
            for name, param in picked_param:
                param.requires_grad_(True)


        else:
            for name, param in model.named_parameters():
                param.requires_grad_(False)
                if delta_args.delta_type in name or \
                        ("bias" in name if delta_args.delta_type == "bitfit" else False):
                    param.requires_grad_(True)
                print(f"name:{name}, param:{param}")
            print(f"using original pet methods : {delta_args.delta_type}")

    data_args.dataset_name = [data_args.task_name]
    data_args.eval_dataset_name = [data_args.eval_dataset_name]
    data_args.test_dataset_name = [data_args.test_dataset_name]
    data_args.dataset_config_name = [data_args.dataset_config_name]
    data_args.eval_dataset_config_name = [data_args.eval_dataset_config_name]
    data_args.test_dataset_config_name = [data_args.test_dataset_config_name]
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(data_args.test_dataset_config_name)

    # Temporarily set max_target_length for training.
    #max_target_length = data_args.max_target_length

    assert data_args.pad_to_max_length, "if not pad to max length will raise error when collating"
    assert training_args.remove_unused_columns is False, "avoid removed our prefix allowed tokens used for ed2lm"
    padding = "max_length" if data_args.pad_to_max_length else True

    def preprocess_function(examples, max_target_length):

        model_inputs = tokenizer([s for s in examples['source']], max_length=data_args.max_source_length,
                                 padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer([t for t in examples['target']], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        # model_inputs["extra_fields"] = examples['extra_fields']
        return model_inputs

    # column_names = ['source', 'target', 'extra_fields']
    column_names = ['source', 'target']
    performance_metrics = {}
    if training_args.do_train:
        train_datasets = [AutoTask.get(dataset_name,
                                       dataset_config_name,
                                       seed=training_args.data_seed).get(
            split="train",
            split_validation_test=training_args.split_validation_test,
            add_prefix=False,
            n_obs=data_args.max_train_samples,
            label_mapping_type=delta_args.label_mapping_type)
            for dataset_name, dataset_config_name\
            in zip(data_args.dataset_name, data_args.dataset_config_name)]
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(\
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)\
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)]
        for i, train_dataset in enumerate(train_datasets):
            train_datasets[i] = train_datasets[i].map(
                functools.partial(preprocess_function, max_target_length=max_target_lengths[i]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names, # if train_dataset != "superglue-record" else column_names+["answers"],
                load_from_cache_file=False,
            )
        train_dataset = concatenate_datasets(train_datasets)


    if training_args.do_eval:
        eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
            seed=training_args.data_seed).get(
            split="validation", 
            split_validation_test=training_args.split_validation_test,
            add_prefix=False,
            n_obs=data_args.max_val_samples,
            label_mapping_type=delta_args.label_mapping_type)
            for eval_dataset, eval_dataset_config in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)}
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length( \
            tokenizer=tokenizer, default_max_length=data_args.max_target_length) \
            for dataset_name, dataset_config_name in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)]
        for k, name in enumerate(eval_datasets):
            eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(preprocess_function, max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names, # if name != "superglue-record" else column_names+["answers"],
                    load_from_cache_file=False,
            )

    # TODO important! deal with generation target length
    # for debug
    if "max_target_lengths" in locals() and training_args.generation_max_length != max_target_lengths[0]:
        print("reset generation_max_length from {} to {}".format(training_args.generation_max_length,
                                                                 max_target_lengths[0]))
        training_args.generation_max_length = max_target_lengths[0]
    else:
        training_args.generation_max_length = data_args.max_target_length
    print("##### cur generation_max_length : {} ############".format(training_args.generation_max_length))


    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id


    # if data_args.pad_to_max_length:
    #     data_collator = default_data_collator
    # else:
    #     data_collator = TaskDataCollatorForSeq2Seq(
    #         tokenizer,
    #         label_pad_token_id=label_pad_token_id,
    #         pad_to_multiple_of=8 if training_args.fp16 else None,
    #     )

    # unlike opendelta's implement, we do our custom collator, and do not need to pass data_info in trainer
    # use with extra_fields in post_process func
    # we don't need extra_fields in input, different from opendelta's implement, we will use extra_field directly by passing
    # original eval dataset
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # Metric, we assume we have only one training task.
    eval_metrics = [AutoTask.get(dataset_name, dataset_config_name).metric\
        for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]

    if training_args.predict_with_generate is False:
        # we need logits pos here
        classification_logits_pos = AutoTask.get(data_args.eval_dataset_name[0],
                                                 data_args.eval_dataset_config_name[0]).classification_logits_pos
    else:
        classification_logits_pos = None

    # Extracts the extra information needed to evaluate on each dataset.
    # These information are only used in the compute_metrics.
    # We will assume that the test/eval dataloader does not change the order of 
    # the data.
    # TODO we do not need data_info in this implementation
    # data_info = {"eval": eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'] if training_args.do_eval else None,
    #              "test": test_datasets[data_args.test_dataset_name[0]]['extra_fields'] if training_args.do_test else None,
    #              "train": train_dataset['extra_fields'] if training_args.do_train and hasattr(train_dataset,"extra_fields") else None}
    # assert len(data_info['eval'][0]) != 0

    def compute_metrics(eval_preds):
        preds, labels, data_info = eval_preds
        # post_processor = AutoPostProcessor.get(data_args.dataset_name[0], tokenizer,
        #                                        data_args.ignore_pad_token_for_loss)
        # decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)

        result = {}
        for metric in eval_metrics:
            # change here for squad, since we will have multiple choice in eval_ans
            result.update(metric(preds, data_info))

        if 'LOCAL_RANK' in os.environ and os.environ['LOCAL_RANK'] == '0':
            # print('rank here:{}'.format(os.environ['LOCAL_RANK']))
            # print('save path here:{}'.format(os.path.join(TEMP_OUT_PATH,'latest_raw_eval_results.txt')))

            with open(os.path.join(training_args.output_dir, 'wrong_text.txt'), 'w', encoding='utf-8') as f:
                for pre, grn in zip(preds, data_info):
                    if pre not in grn:
                        f.write("prenormalized predict: {}\n".format(pre))
                        f.write("prenormalized ground: {}\n".format(str(grn)))

            with open(os.path.join(training_args.output_dir, 'latest_raw_eval_results.txt'), 'w', encoding='utf-8') as f:
                f.write("latest_metrics : {}".format(str(result)))
                for pre, grn in zip(preds, data_info):
                    f.write("prenormalized predictions:{}\ntargets:{}\n".format(pre, str(grn)))

        else:
            with open(os.path.join(training_args.output_dir, 'latest_raw_eval_results.txt'), 'w', encoding='utf-8') as f:
                f.write("latest_metrics : {}".format(str(result)))
                for pre, grn in zip(preds, data_info):
                    f.write("prenormalized predictions:{}\ntargets:{}\n".format(pre, str(grn)))
            with open(os.path.join(training_args.output_dir, 'wrong_text.txt'), 'w', encoding='utf-8') as f:
                for pre, grn in zip(preds, data_info):
                    if pre not in grn:
                        f.write("prenormalized predict: {}\n".format(pre))
                        f.write("prenormalized ground: {}\n".format(str(grn)))


        return result


    def post_processing_function(
        examples, features, outputs, stage="eval"
    ):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # lazy hack
        if "ed2lm" in data_args.eval_dataset_name[0].lower():
            formatted_predictions = []
            for decode_pred, removed_prefix in zip(decoded_preds, features["ed2lm_prefix_allowed_text"]):
                formatted_predictions.append(decode_pred.split(removed_prefix)[-1].strip())
        else:
            for idx, _ in enumerate(decoded_preds):
                if isinstance(decoded_preds[idx], str):
                    decoded_preds[idx] = decoded_preds[idx].strip()

                    # TODO ugly hack  avoid generate other words
                    decoded_preds[idx] = decoded_preds[idx].split("\n")[0].strip()
                    decoded_preds[idx] = decoded_preds[idx].split("</s>")[0].strip()

            formatted_predictions = decoded_preds

        return EvalPrediction(predictions=formatted_predictions, label_ids=outputs.label_ids , data_info=features["extra_fields"])

    # Initialize our Trainer
    trainer = ED2LMSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=list(eval_datasets.values())[0] if training_args.do_eval else None,
        data_info=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        classification_logits_pos=classification_logits_pos,
        generation_config=generation_config
    )
    # evaluation_metrics = TASK_TO_METRICS[data_args.dataset_name[0]],

    trainer.add_callback(MyCallback())

    # Saves training config. 
    if trainer.is_world_process_zero():

        os.makedirs(training_args.output_dir, exist_ok=True)
        if sys.argv[1].endswith('json'):
            save_training_config(sys.argv[1], training_args.output_dir)
        else:

            with open(os.path.join(training_args.output_dir, 'training_config.txt'), 'w', encoding='utf-8') as w:
                w.write(str(sys.argv))


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})
        
        trainer.save_model()  # Saves the tokenizer too for easy upload
        train_metrics = train_result.metrics
        if hasattr(train_dataset, "__len__"):
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        else:
            train_metrics["train_samples"] = "unknown"
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        print(performance_metrics)
        trainer.save_metrics("performance", performance_metrics)
    
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        for task, eval_dataset in eval_datasets.items():
            metrics = trainer.evaluate(eval_dataset=eval_dataset,
               max_length=data_args.val_max_target_length, num_beams=data_args.num_beams,
            )
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        results['evaluate'] = metrics

    # Test
    if training_args.do_test:
        logger.info("*** Test ***")
        for task, test_dataset in test_datasets.items():
            metrics = trainer.evaluate(eval_dataset=test_dataset,
              max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
              metric_key_prefix="test"
            )
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
        results['test'] = metrics


    return results




if __name__ == "__main__":
    result = main()
    print(result)

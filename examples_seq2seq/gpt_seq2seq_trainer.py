# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
A subclass of `Trainer` specific to Question-Answering tasks
"""

import collections
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled

from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)


from typing import Dict, List, Optional

from torch.utils.data import Dataset

from transformers import Seq2SeqTrainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
from inspect import signature

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


def prepare_no_position_ids_inputs_for_generation(input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)
    #TODO do not count position ids, let the model count by its self, because this function used in generation is different from that in training
    # if attention_mask is not None and position_ids is None:
    #     # create position_ids on the fly for batch generation
    #     position_ids = attention_mask.long().cumsum(-1) - 1
    #     position_ids.masked_fill_(attention_mask == 0, 1)
    #     if past:
    #         position_ids = position_ids[:, -1].unsqueeze(-1)
    # else:
    #     position_ids = None

    position_ids = None

    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


class GPTSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, data_info=None, evaluation_metrics=None,
                 classification_logits_pos=None, extract_neuron_or_hidden=False, generation_config,**kwargs):
        #TODO ugly hack , could be wrong, we still need position ids on the fly
        # kwargs["model"].prepare_inputs_for_generation = prepare_no_position_ids_inputs_for_generation

        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.evaluation_metrics = evaluation_metrics
        self.data_info = data_info
        self.classification_logits_pos = classification_logits_pos
        self.extract_neuron_or_hidden = extract_neuron_or_hidden
        self.generation_config = generation_config
        # self.eval_dataset will be init in super().super() aka. BaseTrainer
        # self.compute_metrics will be init in super().super() aka. BaseTrainer

    # def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)


        used_kwargs = list(signature(self.model.forward).parameters.keys())
        generation_inputs = {k: v for k, v in inputs.items() if k in used_kwargs}

        if "input_ids" not in generation_inputs and "input_ids" in inputs:
            generation_inputs["input_ids"] = inputs["input_ids"]

        if 'ed2lm_prefix_allowed_tokens' in inputs:
            self.to_lm = True
            self.cur_constrained_id_list : List[List[int]] = inputs['ed2lm_prefix_allowed_tokens']
        else:
            self.to_lm = False
            self.cur_constrained_id_list = None

        generated_tokens = self.model.generate(
            **generation_inputs,
            generation_config=self.generation_config,
            prefix_allowed_tokens_fn=self.prefix_allowed_fn if self.to_lm else None
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < self.generation_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.generation_config.max_length)

        if not self.extract_neuron_or_hidden:
            # do not want to calculate loss againe when self.extract_neuron_or_hidden because we have register_hook or try to remove the hook
            with torch.no_grad():
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if has_labels:
                    if self.label_smoother is not None:
                        loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    else:
                        loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
                else:
                    loss = None
        else:
            loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < self.generation_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, self.generation_config.max_length)
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def prefix_allowed_fn(self, batch_id, sent):

        assert  self.cur_constrained_id_list

        cur_allowed_prefix = self.cur_constrained_id_list[batch_id]

        if len(sent) <= len(cur_allowed_prefix):
            return cur_allowed_prefix[len(sent) - 1]
        else:
            return list(range(len(self.tokenizer)))


import csv

import random
from collections import OrderedDict
import collections
import abc
import functools
from typing import Callable, List, Mapping
from examples_seq2seq.trainers.trainer_utils import pad_punctuation
from examples_seq2seq.metrics import metrics
from .utils import round_stsb_target
import datasets
import logging
import numpy as np
import torch
import re
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

logger = logging.getLogger(__name__)

import pandas as pd
import pyarrow as pa
from datasets import Dataset

import os


csv.field_size_limit(100000000)


class AbstractTask(abc.ABC):
    name = NotImplemented
    config = NotImplemented
    prefix = NotImplemented
    preprocessor: Callable = NotImplemented
    metric = NotImplemented
    metric_names = NotImplemented
    icl_dataset_seed = NotImplemented
    icl_k_examples = NotImplemented
    dataset = NotImplemented
    split_map = None
    labels_list = None
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
                                         "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
                                         "superglue-boolq"]
    large_data_without_all_splits = ["qqp", "qnli", "superglue-record", "sst2"]

    index = 0

    def __init__(self, config, seed=42, source_data=None, icl_dataset_seed=None, icl_k_examples=None,
                 reset_pos_id_for_instance=False, three_newlines=True, used_task=None):
        self.config = config
        self.seed = seed
        self.source_data = source_data
        self.reset_pos_id_for_instance = reset_pos_id_for_instance
        self.icl_dataset_seed = icl_dataset_seed
        self.icl_k_examples = icl_k_examples
        self.three_newlines = three_newlines
        self.used_task = used_task

    def get_max_target_length(self, tokenizer, default_max_length):
        if self.labels_list is not None:
            return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       add_prefix: bool = False,
                       prefix: str = None,
                       extra_fields={},
                       options=None,
                       ed2lm_prefix_allowed_text=None
                       ):
        src_prefix = self.name if prefix is None else prefix
        sources = [src_prefix] + sources if add_prefix else sources
        if ed2lm_prefix_allowed_text is not None:
            ins = {'source': ' '.join(sources),
                   'target': ' '.join(targets),
                   'task': self.name,
                   'extra_fields': extra_fields,
                   'ed2lm_prefix_allowed_text': ed2lm_prefix_allowed_text
                   }
        else:

            icl_test_source = None

            if self.reset_pos_id_for_instance:
                if self.three_newlines:
                    if "\n\n\n" in ' '.join(sources):
                        icl_test_source = ' '.join(sources).split("\n\n\n")[-1]
                    else:
                        if "icl" in self.name:
                            raise ValueError("format-dependent line in icl setting, check if it raise the error")
                else:
                    if "\n" in ' '.join(sources):
                        icl_test_source = ' '.join(sources).split("\n")[-2] + "\n"
                    else:
                        if "icl" in self.name:
                            raise ValueError("format-dependent line in icl setting, check if it raise the error")

            ins = {'source': ' '.join(sources),
                   'target': ' '.join(targets),
                   'task': self.name,
                   'icl_test_source': icl_test_source,
                   'extra_fields': extra_fields,
                   'options': options}

        if self.index < 3:
            print("{}th examples : {}".format(self.index, ins))
            self.index += 1

        return ins
        # return {'source': ' '.join(sources),
        #         'target': ' '.join(targets),
        #         'task': self.name,
        #         }

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
            indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        print(f"we select following indices' data : {indices}")
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, self.config, split=split, script_version="master")

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def map_dataset(self, dataset, add_prefix):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           remove_columns=dataset.column_names, load_from_cache_file=False)

    def get(self, split, add_prefix=True, n_obs=None, split_validation_test=False, source_data=None,
            label_mapping_type=None):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        # if split_validation_test and self.name in self.small_datasets_without_all_splits \
        #         and split != "train":
        #     mapped_split = self.split_to_data_split["validation"]
        #     dataset = self.load_dataset(split=mapped_split)
        #     indices = self.get_split_indices(split, dataset, validation_size=len(dataset)//2)
        #     dataset = self.subsample(dataset, n_obs, indices)
        # # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # # validation and the rest as training set, keeping the original validation
        # # set as the test set.
        # elif split_validation_test and self.name in self.large_data_without_all_splits \
        #         and split != "test":
        #     dataset = self.load_dataset(split="train")
        #     indices = self.get_split_indices(split, dataset, validation_size=1000)
        #     dataset = self.subsample(dataset, n_obs, indices)
        # else:
        self.source_data = source_data
        mapped_split = self.split_to_data_split[split]
        self.n_obs = n_obs
        self.label_mapping_type = label_mapping_type
        self.dataset = self.load_dataset(split=mapped_split)
        # shuffles the data and samples it.
        if n_obs is not None:
            # hack
            self.dataset = self.subsample(self.dataset, n_obs)
        return self.map_dataset(self.dataset, add_prefix)





class SAMSUM_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "samsum_gpt_icl"
    metric = [metrics.rouge_l]
    metric_names = ["rouge_l"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'samsum/samsum_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'samsum/samsum_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{cur_icl_sample['outputs']}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [example['outputs']]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is SUMMARY, no options")


class SAMSUM_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "samsum_gpt"
    metric = [metrics.rouge_l]
    metric_names = ["rouge_l"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'samsum/samsum_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {example['outputs']}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [example['outputs']]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is SUMMARY, no options")

class SAMSUM_t5(AbstractTask):
    root_path = "./crossfit_data/"
    name = "samsum_t5"
    metric = [metrics.rouge_l]
    metric_names = ["rouge_l"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'samsum/samsum_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        src_texts = [f"Input: {example['inputs']} Output: "]

        tgt_texts = [example['outputs'] + "</s>"]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is SUMMARY, no options")


class MULTI_NEWS_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "multi_news_gpt"
    metric = [metrics.rouge_l]
    metric_names = ["rouge_l"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'multi_news/multi_news_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {example['outputs']}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [example['outputs']]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is SUMMARY, no options")

class MULTI_NEWS_t5(AbstractTask):
    root_path = "./crossfit_data/"
    name = "multi_news_t5"
    metric = [metrics.rouge_l]
    metric_names = ["rouge_l"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'multi_news/multi_news_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        src_texts = [f"Input: {example['inputs']} Output: "]

        tgt_texts = [example['outputs'] + "</s>"]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is SUMMARY, no options")


class xsum_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "xsum_gpt_icl"
    metric = [metrics.rouge_l]
    metric_names = ["rouge_l"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'xsum/xsum_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'xsum/xsum_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{cur_icl_sample['outputs']}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [example['outputs']]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is SUMMARY, no options")


class xsum_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "xsum_gpt"
    metric = [metrics.rouge_l]
    metric_names = ["rouge_l"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'xsum/xsum_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {example['outputs']}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [example['outputs']]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is SUMMARY, no options")


class AQUA_RAT_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "aqua_rat_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'aqua_rat/aqua_rat_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'aqua_rat/aqua_rat_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{cur_icl_sample['outputs']}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [example['outputs']]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is multiple choice qa, no options")


class AQUA_RAT_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "aqua_rat_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'aqua_rat/aqua_rat_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {example['outputs']}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [example['outputs']]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is multiple choice qa, no options")


class MRPC_t5(AbstractTask):
    root_path = "./crossfit_data/"
    name = "mrpc_t5"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "validation"}

    def load_dataset(self, split):

        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "equivalent": "1",
                "not_equivalent": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "equivalent": "equivalent",
                "not_equivalent": "not_equivalent",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-mrpc/glue-mrpc_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        src_texts = [f"Input: {example['inputs']} Output: "]

        tgt_texts = [self.label_mapping[example['outputs']] + "</s>"]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class MRPC_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "mrpc_gpt"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "validation"}

    def load_dataset(self, split):

        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "equivalent": "1",
                "not_equivalent": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "equivalent": "equivalent",
                "not_equivalent": "not_equivalent",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-mrpc/glue-mrpc_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class MRPC_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "mrpc_gpt_icl"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "validation"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                0: "0",
                1: "1"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                0: "not_equivalent",
                1: "equivalent"
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")
        return datasets.load_from_disk(os.path.join(self.root_path, 'glue/mrpc'))[split]
        # return datasets.load_dataset('glue', 'qqp',
        #                              split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["validation", "test"]:
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]

                src_text += f"sentence1:{cur_icl_sample['sentence1'].strip()} sentence2:{cur_icl_sample['sentence2'].strip()}\n{self.label_mapping[cur_icl_sample['label']]}\n"
            src_text += f"sentence1:{example['sentence1'].strip()} sentence2:{example['sentence2'].strip()}\n"
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['label']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields=self.label_mapping[example['label']])

class IMDB_t5(AbstractTask):
    root_path = "./crossfit_data/"
    name = "imdb_t5"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "validation"}

    split = None

    def load_dataset(self, split):

        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "positive": "1",
                "negative": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "positive": "positive",
                "negative": "negative",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'imdb/imdb_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        src_texts = [f"Input: {example['inputs']} Output: "]

        tgt_texts = [self.label_mapping[example['outputs']] + "</s>"]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class IMDB_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "imdb_gpt"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "validation"}

    split = None

    def load_dataset(self, split):

        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "positive": "1",
                "negative": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "positive": "positive",
                "negative": "negative",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'imdb/imdb_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))

class IMDB_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "imdb_gpt_icl"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    # icl has difficulty generating 0,1(not always true); and it cannot generate right ans if the last tokens is whitespace seemingly
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "validation"}

    split = None
    icl_indices = None
    n_obs = None

    def load_dataset(self, split):
        self.split = split
        if self.label_mapping_type == "number":
            self.label_mapping = {
                "negative": "0",
                "positive": "1"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "negative": "negative",
                "positive": "positive"
            }

        random.seed(self.icl_dataset_seed)
        # TODO reformulate following line, now we assure self.n_obs is not None, However it is not always correct
        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")
        # Mine, official datasets' sst2, note that test set do not have labels

        data_files = {split: os.path.join(self.root_path, f'IMDB/{split}.csv')}
        return datasets.load_dataset('csv', data_files=data_files, delimiter='\t', column_names=['sentence', 'label'])[
            split]
        # return datasets.load_dataset('glue', 'sst2',
        #                              split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["validation", "test"]:
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]

                src_text += f"sentence:{cur_icl_sample['sentence'].strip()}\n{self.label_mapping[cur_icl_sample['label']]}\n"
            src_text += f"sentence:{example['sentence'].strip()}\n"
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['label']]]
        else:
            raise ValueError("icl do not need train set")

        # TODO check extra_field afterwards
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields=self.label_mapping[example['label']])





class SUPERGLUE_CB_2label_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "superglue_cb_2label_gpt_icl"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    # dev got only 28 samples so we take train set

    split_to_data_split = {"train": "train",
                           "validation": "train",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "entailment": "1",
                "neutral": "0",
                "contradiction": "0"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "entailment": "entailment",
                "neutral": "contradiction",
                "contradiction": "contradiction"
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'superglue-cb/superglue-cb_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]

                src_text += f"{cur_icl_sample['inputs']}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class SNLI_gpt_2label(AbstractTask):
    root_path = "./crossfit_data/"
    name = "snli_gpt_2label"
    # labels_list = ["0", "1", "2"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_matched",
                           "test": "validation_matched"}

    def load_dataset(self, split):

        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                0: "1",
                1: "0",
                2: "0"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                0: "entailment",
                1: "contradiction",
                2: "contradiction"
            }

        return datasets.load_from_disk(os.path.join(self.root_path, 'glue/mnli'))[split]
        # return datasets.load_dataset('glue', 'qnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [
                f"premise:{example['premise'].strip()} hypothesis:{example['hypothesis'].strip()}\n{self.label_mapping[example['label']]}</s>"]
        elif self.split == "validation_matched":
            src_texts = [f"premise:{example['premise'].strip()} hypothesis:{example['hypothesis'].strip()}\n"]

        tgt_texts = [self.label_mapping[example['label']]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields=self.label_mapping[example['label']])


class SNLI_gpt_icl_2label(AbstractTask):
    root_path = "./crossfit_data/"
    name = "snli_gpt_icl_2label"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_matched",
                           "test": "validation_matched"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                0: "1",
                1: "0",
                2: "0"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                0: "entailment",
                1: "contradiction",
                2: "contradiction"
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")
        return datasets.load_from_disk(os.path.join(self.root_path, 'glue/mnli'))[split]
        # return datasets.load_dataset('glue', 'qqp',
        #                              split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["validation_matched", "test"]:
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]

                src_text += f"premise:{cur_icl_sample['premise'].strip()} hypothesis:{cur_icl_sample['hypothesis'].strip()}\n{self.label_mapping[cur_icl_sample['label']]}{newlines}"
            src_text += f"premise:{example['premise'].strip()} hypothesis:{example['hypothesis'].strip()}\n"
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['label']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields=self.label_mapping[example['label']],
                                   options=list(self.label_mapping.values()))


class dbpedia_14_2label_gpt_icl(AbstractTask):
    # original 14, pick company, athlete here
    root_path = "./crossfit_data/"
    name = "dbpedia_14_2label_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Athlete": "1",
                "Company": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Athlete": "Athlete",
                "Company": "Company",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'dbpedia_14/dbpedia_14_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["Athlete", "Company"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'dbpedia_14/dbpedia_14_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["Athlete", "Company"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class dbpedia_14_2label_gpt(AbstractTask):
    # original 14, pick company, athlete here
    root_path = "./crossfit_data/"
    name = "dbpedia_14_2label_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Athlete": "1",
                "Company": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Athlete": "Athlete",
                "Company": "Company",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'dbpedia_14/dbpedia_14_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["Athlete", "Company"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class dbpedia_14_14label_gpt_icl(AbstractTask):
    # original 14, pick company, athlete here
    root_path = "./crossfit_data/"
    name = "dbpedia_14_14label_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Athlete": "1",
                "Company": "0",
                "Village": "2",
                "Film": "3",
                "OfficeHolder": "4",
                "Artist": "5",
                "Album": "6",
                "Building": "7",
                "Plant": "8",
                "Animal": "9",
                "EducationalInstitution": "10",
                "WrittenWork": "11",
                "MeanOfTransportation": "12",
                "NaturalPlace": "13",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Athlete": "Athlete",
                "Company": "Company",
                "Village": "Village",
                "Film": "Film",
                "OfficeHolder": "OfficeHolder",
                "Artist": "Artist",
                "Album": "Album",
                "Building": "Building",
                "Plant": "Plant",
                "Animal": "Animal",
                "EducationalInstitution": "EducationalInstitution",
                "WrittenWork": "WrittenWork",
                "MeanOfTransportation": "MeanOfTransportation",
                "NaturalPlace": "NaturalPlace",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'dbpedia_14/dbpedia_14_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in list(self.label_mapping.keys()):
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'dbpedia_14/dbpedia_14_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in list(self.label_mapping.keys()):
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class dbpedia_14_14label_gpt(AbstractTask):
    # original 14, pick company, athlete here
    root_path = "./crossfit_data/"
    name = "dbpedia_14_14label_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Athlete": "1",
                "Company": "0",
                "Village": "2",
                "Film": "3",
                "OfficeHolder": "4",
                "Artist": "5",
                "Album": "6",
                "Building": "7",
                "Plant": "8",
                "Animal": "9",
                "EducationalInstitution": "10",
                "WrittenWork": "11",
                "MeanOfTransportation": "12",
                "NaturalPlace": "13",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Athlete": "Athlete",
                "Company": "Company",
                "Village": "Village",
                "Film": "Film",
                "OfficeHolder": "OfficeHolder",
                "Artist": "Artist",
                "Album": "Album",
                "Building": "Building",
                "Plant": "Plant",
                "Animal": "Animal",
                "EducationalInstitution": "EducationalInstitution",
                "WrittenWork": "WrittenWork",
                "MeanOfTransportation": "MeanOfTransportation",
                "NaturalPlace": "NaturalPlace",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'dbpedia_14/dbpedia_14_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in list(self.label_mapping.keys()):
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class ag_news_2label_gpt_icl(AbstractTask):
    # original 4, pick business, sports here
    root_path = "./crossfit_data/"
    name = "ag_news_2label_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Sports": "1",
                "Business": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Sports": "Sports",
                "Business": "Business",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'ag_news/ag_news_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["Sports", "Business"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'ag_news/ag_news_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["Sports", "Business"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]

                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class ag_news_2label_gpt(AbstractTask):
    # original 4, pick business, sports here
    root_path = "./crossfit_data/"
    name = "ag_news_2label_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Sports": "1",
                "Business": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Sports": "Sports",
                "Business": "Business",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'ag_news/ag_news_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["Sports", "Business"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class ag_news_4label_gpt_icl(AbstractTask):
    # original 4, pick business, sports here
    root_path = "./crossfit_data/"
    name = "ag_news_4label_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Sports": "1",
                "Business": "0",
                "Sci/Tech": "2",
                "World": "3"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Sports": "Sports",
                "Business": "Business",
                "Sci/Tech": "Sci/Tech",
                "World": "World"
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'ag_news/ag_news_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["Sports", "Business", "Sci/Tech", "World"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'ag_news/ag_news_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["Sports", "Business", "Sci/Tech", "World"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]

                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class ag_news_4label_gpt(AbstractTask):
    # original 4, pick business, sports here
    root_path = "./crossfit_data/"
    name = "ag_news_4label_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Sports": "1",
                "Business": "0",
                "Sci/Tech": "2",
                "World": "3"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Sports": "Sports",
                "Business": "Business",
                "Sci/Tech": "Sci/Tech",
                "World": "World"
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'ag_news/ag_news_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["Sports", "Business", "Sci/Tech", "World"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class health_fact_2label_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "health_fact_2label_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "true": "1",
                "false": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "true": "true",
                "false": "false",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'health_fact/health_fact_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["true", "false"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'health_fact/health_fact_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["true", "false"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class health_fact_2label_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "health_fact_2label_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "true": "1",
                "false": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "true": "true",
                "false": "false",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'health_fact/health_fact_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["true", "false"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class health_fact_4label_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "health_fact_4label_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "true": "1",
                "false": "0",
                "mixture": "2",
                "unproven": "3"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "true": "true",
                "false": "false",
                "mixture": "mixture",
                "unproven": "unproven"
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'health_fact/health_fact_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["true", "false", "mixture", "unproven"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'health_fact/health_fact_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["true", "false", "mixture", "unproven"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class health_fact_4label_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "health_fact_4label_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "true": "1",
                "false": "0",
                "mixture": "2",
                "unproven": "3"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "true": "true",
                "false": "false",
                "mixture": "mixture",
                "unproven": "unproven"
            }
        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'health_fact/health_fact_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if line[1] not in ["true", "false", "mixture", "unproven"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class climate_fever_2label_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "climate_fever_2label_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Supports": "1",
                "Refutes": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Supports": "Supports",
                "Refutes": "Refutes",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'climate_fever/climate_fever_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) < 2 or line[1] not in ["Supports", "Refutes"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'climate_fever/climate_fever_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) < 2 or line[1] not in ["Supports", "Refutes"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class climate_fever_2label_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "climate_fever_2label_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Supports": "1",
                "Refutes": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Supports": "Supports",
                "Refutes": "Refutes",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'climate_fever/climate_fever_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) < 2 or line[1] not in ["Supports", "Refutes"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class climate_fever_4label_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "climate_fever_2label_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Supports": "1",
                "Refutes": "0",
                "Disputed": "2",
                "Not enough info": "3"

            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Supports": "Supports",
                "Refutes": "Refutes",
                "Disputed": "Disputed",
                "Not enough info": "Not enough info"
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'climate_fever/climate_fever_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) < 2 or line[1] not in ["Supports", "Refutes", "Disputed", "Not enough info"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'climate_fever/climate_fever_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) < 2 or line[1] not in ["Supports", "Refutes", "Disputed", "Not enough info"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class climate_fever_4label_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "climate_fever_2label_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Supports": "1",
                "Refutes": "0",
                "Disputed": "2",
                "Not enough info": "3"

            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Supports": "Supports",
                "Refutes": "Refutes",
                "Disputed": "Disputed",
                "Not enough info": "Not enough info"
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'climate_fever/climate_fever_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) < 2 or line[1] not in ["Supports", "Refutes", "Disputed", "Not enough info"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class kilt_fever_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "kilt_fever_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "SUPPORTS": "1",
                "REFUTES": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "SUPPORTS": "SUPPORTS",
                "REFUTES": "REFUTES",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'kilt_fever/kilt_fever_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["SUPPORTS", "REFUTES"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'kilt_fever/kilt_fever_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["SUPPORTS", "REFUTES"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class kilt_fever_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "kilt_fever_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "SUPPORTS": "1",
                "REFUTES": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "SUPPORTS": "SUPPORTS",
                "REFUTES": "REFUTES",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'kilt_fever/kilt_fever_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["SUPPORTS", "REFUTES"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class MNLI_2label_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "MNLI_2label_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "entailment": "1",
                "contradiction": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "entailment": "entailment",
                "contradiction": "contradiction",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'glue-mnli/glue-mnli_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["entailment", "contradiction"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-mnli/glue-mnli_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["entailment", "contradiction"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class MNLI_2label_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "MNLI_2label_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "entailment": "1",
                "contradiction": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "entailment": "entailment",
                "contradiction": "contradiction",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-mnli/glue-mnli_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["entailment", "contradiction"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class MNLI_3label_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "MNLI_3label_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "entailment": "1",
                "contradiction": "0",
                "neutral": "2"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "entailment": "entailment",
                "contradiction": "contradiction",
                "neutral": "neutral"
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'glue-mnli/glue-mnli_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["entailment", "contradiction", "neutral"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-mnli/glue-mnli_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["entailment", "contradiction", "neutral"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class MNLI_3label_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "MNLI_3label_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "entailment": "1",
                "contradiction": "0",
                "neutral": "2"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "entailment": "entailment",
                "contradiction": "contradiction",
                "neutral": "neutral"
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-mnli/glue-mnli_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["entailment", "contradiction", "neutral"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class MNLI_3label_t5(AbstractTask):
    root_path = "./crossfit_data/"
    name = "MNLI_3label_t5"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "entailment": "1",
                "contradiction": "0",
                "neutral": "2"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "entailment": "entailment",
                "contradiction": "contradiction",
                "neutral": "neutral"
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-mnli/glue-mnli_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["entailment", "contradiction", "neutral"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        src_texts = [f"Input: {example['inputs']} Output: "]

        tgt_texts = [self.label_mapping[example['outputs']] + "</s>"]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class QNLI_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "QNLI_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "entailment": "1",
                "not_entailment": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "entailment": "entailment",
                "not_entailment": "not_entailment",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'glue-qnli/glue-qnli_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["entailment", "not_entailment"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-qnli/glue-qnli_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["entailment", "not_entailment"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class QNLI_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "QNLI_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "entailment": "1",
                "not_entailment": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "entailment": "entailment",
                "not_entailment": "not_entailment",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-qnli/glue-qnli_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["entailment", "not_entailment"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))

class QNLI_t5(AbstractTask):
    root_path = "./crossfit_data/"
    name = "QNLI_t5"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "entailment": "1",
                "not_entailment": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "entailment": "entailment",
                "not_entailment": "not_entailment",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-qnli/glue-qnli_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["entailment", "not_entailment"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        src_texts = [f"Input: {example['inputs']} Output: "]

        tgt_texts = [self.label_mapping[example['outputs']] + "</s>"]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class QQP_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "QQP_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "duplicate": "1",
                "not_duplicate": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "duplicate": "duplicate",
                "not_duplicate": "not_duplicate",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'glue-qqp/glue-qqp_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["duplicate", "not_duplicate"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-qqp/glue-qqp_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["duplicate", "not_duplicate"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class QQP_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "QQP_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "duplicate": "1",
                "not_duplicate": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "duplicate": "duplicate",
                "not_duplicate": "not_duplicate",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-qqp/glue-qqp_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["duplicate", "not_duplicate"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class QQP_t5(AbstractTask):
    root_path = "./crossfit_data/"
    name = "QQP_t5"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "duplicate": "1",
                "not_duplicate": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "duplicate": "duplicate",
                "not_duplicate": "not_duplicate",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-qqp/glue-qqp_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["duplicate", "not_duplicate"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        src_texts = [f"Input: {example['inputs']} Output: "]

        tgt_texts = [self.label_mapping[example['outputs']] + "</s>"]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class MEDICAL_QUESTIONS_PAIRS_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "MEDICAL_QUESTIONS_PAIRS_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Similar": "1",
                "Dissimilar": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Similar": "Similar",
                "Dissimilar": "Dissimilar",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'medical_questions_pairs/medical_questions_pairs_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["Similar", "Dissimilar"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(
                os.path.join(self.root_path, f'medical_questions_pairs/medical_questions_pairs_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["Similar", "Dissimilar"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class MEDICAL_QUESTIONS_PAIRS_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "MEDICAL_QUESTIONS_PAIRS_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "Similar": "1",
                "Dissimilar": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "Similar": "Similar",
                "Dissimilar": "Dissimilar",
            }

        input_ = []
        output_ = []
        with open(
                os.path.join(self.root_path, f'medical_questions_pairs/medical_questions_pairs_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["Similar", "Dissimilar"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class SST2_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "SST2_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "positive": "1",
                "negative": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "positive": "positive",
                "negative": "negative",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'glue-sst2/glue-sst2_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["positive", "negative"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-sst2/glue-sst2_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["positive", "negative"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class SST2_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "SST2_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "positive": "1",
                "negative": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "positive": "positive",
                "negative": "negative",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-sst2/glue-sst2_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["positive", "negative"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class SST2_t5(AbstractTask):
    root_path = "./crossfit_data/"
    name = "SST2_t5"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "positive": "1",
                "negative": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "positive": "positive",
                "negative": "negative",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-sst2/glue-sst2_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["positive", "negative"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        src_texts = [f"Input: {example['inputs']} Output: "]

        tgt_texts = [self.label_mapping[example['outputs']] + "</s>"]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class ROTTEN_TOMATOES_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "ROTTEN_TOMATOES_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "positive": "1",
                "negative": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "positive": "positive",
                "negative": "negative",
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'rotten_tomatoes/rotten_tomatoes_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["positive", "negative"]:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'rotten_tomatoes/rotten_tomatoes_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2 or line[1] not in ["positive", "negative"]:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{self.label_mapping[cur_icl_sample['outputs']]}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['outputs']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class ROTTEN_TOMATOES_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "ROTTEN_TOMATOES_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "positive": "1",
                "negative": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "positive": "positive",
                "negative": "negative",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'rotten_tomatoes/rotten_tomatoes_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["positive", "negative"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class ROTTEN_TOMATOES_t5(AbstractTask):
    root_path = "./crossfit_data/"
    name = "ROTTEN_TOMATOES_t5"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "positive": "1",
                "negative": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "positive": "positive",
                "negative": "negative",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'rotten_tomatoes/rotten_tomatoes_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2 or line[1] not in ["positive", "negative"]:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        src_texts = [f"Input: {example['inputs']} Output: "]

        tgt_texts = [self.label_mapping[example['outputs']] + "</s>"]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class SCIQ_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "sciq_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'sciq/sciq_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                # the question on the last of the sentence
                line[0] = line[0].split(" [SEP] ")[1] + " [SEP] " + line[0].split(" [SEP] ")[0]
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'sciq/sciq_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                # the question on the last of the sentence
                line[0] = line[0].split(" [SEP] ")[1] + " [SEP] " + line[0].split(" [SEP] ")[0]
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{cur_icl_sample['outputs']}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [example['outputs']]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is multiple choice qa, no options")


class SCIQ_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "sciq_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'sciq/sciq_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue
                # the question on the last of the sentence
                line[0] = line[0].split(" [SEP] ")[1] + " [SEP] " + line[0].split(" [SEP] ")[0]
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {example['outputs']}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [example['outputs']]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is multiple choice qa, no options")


class SOCIAL_I_QA_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "social_i_qa_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'social_i_qa/social_i_qa_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                # the question on the last of the sentence
                line[0] = line[0].split(" [SEP] ")[1] + " [SEP] " + line[0].split(" [SEP] ")[0]
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'social_i_qa/social_i_qa_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                # the question on the last of the sentence
                line[0] = line[0].split(" [SEP] ")[1] + " [SEP] " + line[0].split(" [SEP] ")[0]
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{cur_icl_sample['outputs']}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [example['outputs']]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is multiple choice qa, no options")


class SOCIAL_I_QA_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "social_i_qa_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'social_i_qa/social_i_qa_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue
                # the question on the last of the sentence
                line[0] = line[0].split(" [SEP] ")[1] + " [SEP] " + line[0].split(" [SEP] ")[0]
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {example['outputs']}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [example['outputs']]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is multiple choice qa, no options")


class NO_TEXT_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "no_text_gpt_icl"

    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'{self.used_task}/{self.used_task}_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                # the question on the last of the sentence

                if "[SEP]" in line[0]:
                    line[0] = line[0].split(" [SEP] ")[1] + " [SEP] " + line[0].split(" [SEP] ")[0]

                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        self.icl_indices = random.sample(range(len(self.train_dataset)), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        input_ = [""]
        output_ = [""]
        # with open(os.path.join(self.root_path, f'social_i_qa/social_i_qa_{self.split}.tsv')) as f:
        #     reader = csv.reader(f, delimiter="\t")
        #     for line in reader:
        #         if len(line) != 2:
        #             continue
        #         # the question on the last of the sentence
        #         line[0] = line[0].split(" [SEP] ")[1] + " [SEP] " + line[0].split(" [SEP] ")[0]
        #         input_.append(line[0])
        #         output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{cur_icl_sample['outputs']}{newlines}"
            # src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [""]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields="",
                                   options="this is no_text, no ans")


class NO_TEXT_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "no_text_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        input_ = [""]
        output_ = [""]
        # with open(os.path.join(self.root_path, f'social_i_qa/social_i_qa_{self.split}.tsv')) as f:
        #     reader = csv.reader(f, delimiter="\t")
        #     for line in reader:
        #
        #         if len(line) != 2:
        #             continue
        #         # the question on the last of the sentence
        #         line[0] = line[0].split(" [SEP] ")[1] + " [SEP] " + line[0].split(" [SEP] ")[0]
        #         input_.append(line[0])
        #         output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        newlines = "\n\n\n" if self.three_newlines else "\n"

        if self.split == "train":
            src_texts = [f"{newlines}"]
        elif self.split == "dev":
            src_texts = [f"{newlines}"]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [""]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields="",
                                   options="this is no_text, no ans")


class SWAG_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "swag_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'swag/swag_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'swag/swag_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{cur_icl_sample['outputs']}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [example['outputs']]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is multiple choice qa, no options")


class SWAG_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "swag_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'swag/swag_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {example['outputs']}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [example['outputs']]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is multiple choice qa, no options")


class WINO_GRANDE_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "wino_grande_gpt_icl"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")

        train_input_ = []
        train_output_ = []
        with open(os.path.join(self.root_path, f'wino_grande/wino_grande_train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                train_input_.append(line[0])
                train_output_.append(line[1])

        train_data = {"inputs": train_input_, "outputs": train_output_}

        train_df = pd.DataFrame(train_data)
        self.train_dataset = Dataset(pa.Table.from_pandas(train_df))

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'wino_grande/wino_grande_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if len(line) != 2:
                    continue
                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["dev", "test", "train"]:
            newlines = "\n\n\n" if self.three_newlines else "\n"
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]
                # TODO we do ugly truncation here!!!
                src_text += f"{cur_icl_sample['inputs'][:256]}\n{cur_icl_sample['outputs']}{newlines}"
            src_text += f"Input: {example['inputs']} Output: "
            src_texts.append(src_text)

            tgt_texts = [example['outputs']]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is multiple choice qa, no options")


class WINO_GRANDE_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "wino_grande_gpt"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "test"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'wino_grande/wino_grande_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {example['outputs']}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [example['outputs']]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['outputs'],
                                   options="this is multiple choice qa, no options")


class RTE_t5(AbstractTask):
    root_path = "./crossfit_data/"
    name = "rte_t5"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "validation"}

    split = None

    def load_dataset(self, split):

        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "entailment": "1",
                "not_entailment": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "entailment": "entailment",
                "not_entailment": "not_entailment",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-rte/glue-rte_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        src_texts = [f"Input: {example['inputs']} Output: "]

        tgt_texts = [self.label_mapping[example['outputs']] + "</s>"]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))


class RTE_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "rte_gpt"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "validation"}

    split = None

    def load_dataset(self, split):

        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                "entailment": "1",
                "not_entailment": "0",
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                "entailment": "entailment",
                "not_entailment": "not_entailment",
            }

        input_ = []
        output_ = []
        with open(os.path.join(self.root_path, f'glue-rte/glue-rte_{self.split}.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:

                if len(line) != 2:
                    continue

                input_.append(line[0])
                output_.append(line[1])

        data = {"inputs": input_, "outputs": output_}

        df = pd.DataFrame(data)
        dataset = Dataset(pa.Table.from_pandas(df))

        return dataset

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['inputs']} Output: {self.label_mapping[example['outputs']]}</s>"]
        elif self.split == "dev":
            src_texts = [f"Input: {example['inputs']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [self.label_mapping[example['outputs']]]

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=self.label_mapping[example['outputs']],
                                   options=list(self.label_mapping.values()))

class RTE_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "rte_gpt_icl"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "validation"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                0: "0",
                1: "1"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                0: "not_entailment",
                1: "entailment"
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")
        return datasets.load_from_disk(os.path.join(self.root_path, 'glue/rte'))[split]
        # return datasets.load_dataset('glue', 'qqp',
        #                              split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["validation", "test"]:
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]

                src_text += f"sentence1:{cur_icl_sample['sentence1'].strip()} sentence2:{cur_icl_sample['sentence2'].strip()}\n{self.label_mapping[cur_icl_sample['label']]}{newlines}"
            src_text += f"sentence1:{example['sentence1'].strip()} sentence2:{example['sentence2'].strip()}\n"
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['label']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields=self.label_mapping[example['label']],
                                   options=list(self.label_mapping.values()))



class NQOPEN_gpt(AbstractTask):

    name = "nqopen_gpt"
    # labels_list = ["0", "1"]
    metric = [metrics.squad]
    metric_names = ["f1"]

    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    split = None

    def load_dataset(self, split):

        self.split = split

        return datasets.load_dataset("nq_open")[self.split]

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [f"Input: {example['question']} Output: {example['answer'][0]}</s>"]
        elif self.split == "validation":
            src_texts = [f"Input: {example['question']} Output: "]
        else:
            raise ValueError("dev or validation?")

        tgt_texts = [example['answer'][0]]

        # important! extra_fields should be list of list string's
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['answer'],
                                   options="this is multiple choice qa, no options")

class NQOPEN_t5(AbstractTask):

    name = "nqopen_t5"
    # labels_list = ["0", "1"]
    metric = [metrics.squad]
    metric_names = ["f1"]

    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    split = None

    def load_dataset(self, split):

        self.split = split

        return datasets.load_dataset("nq_open")[self.split]

    def preprocessor(self, example, add_prefix=True):

        src_texts = [f"Input: {example['question']} Output: "]

        tgt_texts = [example['answer'][0] + "</s>"]

        # important! extra_fields should be list of list string's
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   extra_fields=example['answer'],
                                   options="this is multiple choice qa, no options")


class WNLI_gpt(AbstractTask):
    root_path = "./crossfit_data/"
    name = "wnli_gpt"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "validation"}

    def load_dataset(self, split):

        self.split = split

        if self.label_mapping_type == "number":
            self.label_mapping = {
                0: "0",
                1: "1"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                0: "not_entailment",
                1: "entailment"
            }

        return datasets.load_from_disk(os.path.join(self.root_path, 'glue/wnli'))[split]
        # return datasets.load_dataset('glue', 'qnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):

        if self.split == "train":
            src_texts = [
                f"sentence1:{example['sentence1'].strip()} sentence2:{example['sentence2'].strip()}\n{self.label_mapping[example['label']]}</s>"]
        elif self.split == "validation":
            src_texts = [f"sentence1:{example['sentence1'].strip()} sentence2:{example['sentence2'].strip()}\n"]

        tgt_texts = [self.label_mapping[example['label']]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields=self.label_mapping[example['label']])


class WNLI_gpt_icl(AbstractTask):
    root_path = "./crossfit_data/"
    name = "wnli_gpt_icl"
    # labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "dev",
                           "test": "validation"}

    split = None
    icl_indices = None

    def load_dataset(self, split):
        self.split = split
        random.seed(self.icl_dataset_seed)

        if self.label_mapping_type == "number":
            self.label_mapping = {
                0: "0",
                1: "1"
            }
        elif self.label_mapping_type == "original":
            self.label_mapping = {
                0: "not_entailment",
                1: "entailment"
            }

        self.icl_indices = random.sample(range(self.n_obs), self.icl_k_examples)
        print(f"we pick indice : {self.icl_indices} samples for icl learning")
        return datasets.load_from_disk(os.path.join(self.root_path, 'glue/wnli'))[split]
        # return datasets.load_dataset('glue', 'qqp',
        #                              split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):

        # Mine
        if self.split in ["validation", "test"]:
            src_texts = []
            src_text = ""
            for i in range(self.icl_k_examples):
                cur_icl_sample = self.train_dataset[self.icl_indices[i]]

                src_text += f"sentence1:{cur_icl_sample['sentence1'].strip()} sentence2:{cur_icl_sample['sentence2'].strip()}\n{self.label_mapping[cur_icl_sample['label']]}{newlines}"
            src_text += f"sentence1:{example['sentence1'].strip()} sentence2:{example['sentence2'].strip()}\n"
            src_texts.append(src_text)

            tgt_texts = [self.label_mapping[example['label']]]
        else:
            raise ValueError("icl do not need train set")

        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields=self.label_mapping[example['label']])




TASK_MAPPING = OrderedDict(
    [
        ('superglue-cb_2label_gpt_icl', SUPERGLUE_CB_2label_gpt_icl),
        ('dbpedia_14_2label_gpt_icl', dbpedia_14_2label_gpt_icl),
        ('dbpedia_14_2label_gpt', dbpedia_14_2label_gpt),
        ('dbpedia_14_14label_gpt_icl', dbpedia_14_14label_gpt_icl),
        ('dbpedia_14_14label_gpt', dbpedia_14_14label_gpt),
        ('ag_news_2label_gpt_icl', ag_news_2label_gpt_icl),
        ('ag_news_2label_gpt', ag_news_2label_gpt),
        ('ag_news_4label_gpt_icl', ag_news_4label_gpt_icl),
        ('ag_news_4label_gpt', ag_news_4label_gpt),
        ('health_fact_2label_gpt_icl', health_fact_2label_gpt_icl),
        ('health_fact_2label_gpt', health_fact_2label_gpt),
        ('health_fact_4label_gpt_icl', health_fact_4label_gpt_icl),
        ('health_fact_4label_gpt', health_fact_4label_gpt),
        ('climate_fever_2label_gpt_icl', climate_fever_2label_gpt_icl),
        ('climate_fever_2label_gpt', climate_fever_2label_gpt),
        ('climate_fever_4label_gpt_icl', climate_fever_4label_gpt_icl),
        ('climate_fever_4label_gpt', climate_fever_4label_gpt),
        ('kilt_fever_gpt_icl', kilt_fever_gpt_icl),
        ('kilt_fever_gpt', kilt_fever_gpt),
        ('mnli_2label_gpt', MNLI_2label_gpt),
        ('mnli_2label_gpt_icl', MNLI_2label_gpt_icl),
        ('mnli_3label_gpt', MNLI_3label_gpt),
        ('mnli_3label_gpt_icl', MNLI_3label_gpt_icl),
        ('qnli_gpt', QNLI_gpt),
        ('qnli_gpt_icl', QNLI_gpt_icl),
        ('qqp_gpt', QQP_gpt),
        ('qqp_gpt_icl', QQP_gpt_icl),
        ('mrpc_gpt', MRPC_gpt),
        ('mrpc_gpt_icl', MRPC_gpt_icl),
        ('imdb_gpt', IMDB_gpt),
        ('imdb_gpt_icl', IMDB_gpt_icl),
        ('rte_gpt', RTE_gpt),
        ('rte_gpt_icl', RTE_gpt_icl),
        ('medical_questions_pairs_gpt', MEDICAL_QUESTIONS_PAIRS_gpt),
        ('medical_questions_pairs_gpt_icl', MEDICAL_QUESTIONS_PAIRS_gpt_icl),
        ('sst2_gpt', SST2_gpt),
        ('sst2_gpt_icl', SST2_gpt_icl),
        ('rottentomatoes_gpt', ROTTEN_TOMATOES_gpt),
        ('rottentomatoes_gpt_icl', ROTTEN_TOMATOES_gpt_icl),
        ('sciq_gpt', SCIQ_gpt),
        ('sciq_gpt_icl', SCIQ_gpt_icl),
        ('social_i_qa_gpt', SOCIAL_I_QA_gpt),
        ('social_i_qa_gpt_icl', SOCIAL_I_QA_gpt_icl),
        ('swag_gpt', SWAG_gpt),
        ('swag_gpt_icl', SWAG_gpt_icl),
        ('wino_grande_gpt', WINO_GRANDE_gpt),
        ('wino_grande_gpt_icl', WINO_GRANDE_gpt_icl),
        ('samsum_gpt_icl', SAMSUM_gpt_icl),
        ('xsum_gpt', xsum_gpt),
        ('xsum_gpt_icl', xsum_gpt_icl),
        ('aqua_rat_gpt', AQUA_RAT_gpt),
        ('aqua_rat_gpt_icl', AQUA_RAT_gpt_icl),
        ('no_text_gpt_icl', NO_TEXT_gpt_icl),
        ('no_text_gpt', NO_TEXT_gpt),
        ('nqopen_gpt', NQOPEN_gpt),
        ('multi_news_gpt', MULTI_NEWS_gpt),
        ('samsum_gpt', SAMSUM_gpt),
        ('sst2_t5', SST2_t5),
        ('rottentomatoes_t5', ROTTEN_TOMATOES_t5),
        ('imdb_t5', IMDB_t5),
        ('mnli_t5', MNLI_3label_t5),
        ('qnli_t5', QNLI_t5),
        ('rte_t5', RTE_t5),
        ('nqopen_t5', NQOPEN_t5),
        ('samsum_t5', SAMSUM_t5),
        ('multi_news_t5', MULTI_NEWS_t5),
        ('qqp_t5', QQP_t5),
        ('mrpc_t5', MRPC_t5),

    ]
)


class AutoTask:
    @classmethod
    def get(self, task, config, seed=42, icl_dataset_seed=None, icl_k_examples=None, reset_pos_id_for_instance=None,
            three_newlines=True, used_task=None):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, seed, icl_dataset_seed=icl_dataset_seed, icl_k_examples=icl_k_examples,
                                      reset_pos_id_for_instance=reset_pos_id_for_instance,
                                      three_newlines=three_newlines, used_task=used_task)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(task, TASK_MAPPING,
                                                    ", ".join(c for c in TASK_MAPPING.keys())
                                                    )
        )


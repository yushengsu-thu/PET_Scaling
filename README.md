# Exploring the Impact of Model Scaling on Parameter-Efficient Tuning

## How to run

#### Prepare the envirionment

```
pip install -r requirements.txt
```

#### Prepare your dataset

Users should put their dataset in `./crossfit_data/` and provide their custom preprocess function in `./examples_seq2seq/data_processors/tasks.py`.
The following is an example:
```python
def preprocessor(self, example, add_prefix=True):

    src_texts = [f"Input: {example['inputs']} Output: "]

    tgt_texts = [self.label_mapping[example['outputs']] + "</s>"]

    return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                               extra_fields=self.label_mapping[example['outputs']],
                               options=list(self.label_mapping.values()))
```


#### Run PET setting
1. Choose a backbone, a pet method and a dataset you want to run. Take `bloom-7b1`, `LORA` and `SST2` for example. 
2. If you want to run T5 as a backbone, use `train_seq2seq.py`.
3. If you want to run Bert as a backbone, use `train_text_classification.py`.
```
deepspeed train_decoder_only.py \
      --deepspeed configs/ds_config_zero3.json \
      --bf16 \
      --do_train \
      --do_eval \
      --model_name_or_path "bigscience/bloom-7b1" \
      --tokenizer_name "bigscience/bloom-7b1" \
      --output_dir "../outputs/test" \
      --label_mapping_type "original" \
      --load_best_model_at_end true \
      --pad_to_max_length true \
      --remove_unused_columns false \
      --overwrite_output_dir true \
      --dropout_rate 0.1 \
      --use_delta 1 \
      --delta_type "lora" \
      --learning_rate "0.0001" \
      --metric_for_best_model "accuracy" \
      --evaluation_strategy "steps" \
      --max_steps "20000"\
      --eval_steps "1000" \
      --save_steps "1000" \
      --save_total_limit 2 \
      --predict_with_generate true \
      --eval_accumulation_steps 20 \
      --max_source_length "128" \
      --max_target_length "138" \
      --task_name "SST2_gpt" \
      --test_dataset_name "SST2_gpt" \
      --eval_dataset_name "SST2_gpt" \
      --gradient_accumulation_steps 1 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 32 \
```

#### Run APET setting

1. Choose a backbone, a pet method and a dataset. Take `bloom-7b1`, parameter amount as `LoRA` and `SST2` for example. 

```
deepspeed train_decoder_only.py \
        --deepspeed configs/ds_config_zero3.json \
        --bf16 \
        --do_train \
        --do_eval \
        --model_name_or_path "bigscience/bloom-7b1" \
        --tokenizer_name "bigscience/bloom-7b1" \
        --output_dir "../outputs/test" \
        --label_mapping_type "original" \
        --load_best_model_at_end true \
        --pad_to_max_length true \
        --remove_unused_columns false \
        --overwrite_output_dir true \
        --dropout_rate 0.0 \
        --use_delta 1 \
        --delta_type "apet" \
        --apet_scale "lora" \
        --apet_seed 2 \
        --learning_rate "0.0001" \
        --metric_for_best_model "accuracy" \
        --evaluation_strategy "steps" \
        --max_steps "20000"\
        --eval_steps "1000" \
        --save_steps "1000" \
        --save_total_limit 2 \
        --predict_with_generate true \
        --eval_accumulation_steps 20 \
        --max_source_length "128" \
        --max_target_length "138" \
        --task_name "SST2_gpt" \
        --test_dataset_name "SST2_gpt" \
        --eval_dataset_name "SST2_gpt" \
        --gradient_accumulation_steps 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
```


#### Run APET ratio setting

1. Choose a backbone, specific parameter amount and a dataset. Take `bloom-7b1`, parameter amount of `4096` and `SST2` for example.

```
deepspeed train_decoder_only.py \
        --deepspeed configs/ds_config_zero3.json \
        --bf16 \
        --do_train \
        --do_eval \
        --model_name_or_path "bigscience/bloom-7b1" \
        --tokenizer_name "bigscience/bloom-7b1" \
        --output_dir "../outputs/test" \
        --label_mapping_type "original" \
        --load_best_model_at_end true \
        --pad_to_max_length true \
        --remove_unused_columns false \
        --overwrite_output_dir true \
        --dropout_rate 0.0 \
        --use_delta 1 \
        --lr_scheduler_type "constant" \
        --delta_type "apet" \
        --apet_scale "4096" \
        --apet_seed "1" \
        --learning_rate "0.0001" \
        --metric_for_best_model "accuracy" \
        --evaluation_strategy "steps" \
        --max_steps "20000"\
        --eval_steps "1000" \
        --save_steps "1000" \
        --save_total_limit 2 \
        --predict_with_generate true \
        --eval_accumulation_steps 20 \
        --max_source_length "128" \
        --max_target_length "138" \
        --task_name "SST2_gpt" \
        --test_dataset_name "SST2_gpt" \
        --eval_dataset_name "SST2_gpt" \
        --gradient_accumulation_steps 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
```

#### Citations
Please cite our paper if it is helpful to your work!
```
@inproceedings{su-etal-2023-exploring,
    title = "Exploring the Impact of Model Scaling on Parameter-Efficient Tuning",
    author = "Su, Yusheng  and
      Chan, Chi-Min  and
      Cheng, Jiali  and
      Qin, Yujia  and
      Lin, Yankai  and
      Hu, Shengding  and
      Yang, Zonghan  and
      Ding, Ning  and
      Sun, Xingzhi  and
      Xie, Guotong  and
      Liu, Zhiyuan  and
      Sun, Maosong",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.931",
    pages = "15062--15078"
}
```

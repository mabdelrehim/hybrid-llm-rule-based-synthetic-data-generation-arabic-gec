#!/bin/bash

env WANDB_ENTITY=mabdelrehim
CHECKPOINTS=../checkpoints
output_dir=../checkpoints/ablation/finetuning/base_model_arat5_ft_data_qalb15_train_combined_real+synth_temp_20_llm_rulebased_hybrid_lr_5e-6

#CUDA_VISIBLE_DEVICES=0 python -m gec.train \
#  --train_file ../data/synthetic/qalb15/train_combined_real+synth_llm_rulebased_hybrid.jsonl \
#  --validation_file ../data/real/clean/qalb-14+qalb-15/jsonl/dev.jsonl \
#  --model_name_or_path UBC-NLP/AraT5v2-base-1024 \
#  --output_dir $output_dir \
#  --do_train --do_eval \
#  --run_name finetune_on_llm_synthetic_temp_20_and_real_qalb15_arat5 \
#  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 \
#  --gradient_accumulation_steps 1 \
#  --learning_rate 5e-6 \
#  --num_train_epochs 10 \
#  --lr_scheduler_type "cosine" \
#  --warmup_ratio 0.1 \
#  --seed 42 \
#  --logging_strategy "steps" \
#  --logging_steps 8 \
#  --save_strategy "steps" \
#  --save_steps 500 \
#  --eval_strategy "steps" \
#  --eval_delay 0 \
#  --eval_steps 500 \
#  --predict_with_generate \
#  --bf16 \
#  --dataloader_num_workers 6 \
#  --preprocessing_num_workers 6 \
#  --ignore_pad_token_for_loss \
#  --include_inputs_for_metrics \
#  --report_to wandb --eval_on_start --save_safetensors false

test_data=../data/real/clean/qalb-14+qalb-15/jsonl/test-l1.jsonl
output_root=../outputs/base_model_arabart_ft_data_qalb15_train_combined_real+synth_temp_10_no_change_UC_llm_rulebased_hybrid_lr_5e-5
output_path=$output_root/qalb15-test-l1-output.txt
model_path=$output_dir

mkdir -p $output_root

python -m gec.infere \
  --test_data $test_data \
  --output_file $output_path \
  --model $model_path \
  --tokenizer $model_path \
  --batch_size 16 \
  --device cuda

test_data=../data/real/clean/qalb-14+qalb-15/jsonl/test-l2.jsonl
output_path=$output_root/qalb15-test-l2-output.txt

python -m gec.infere \
  --test_data $test_data \
  --output_file $output_path \
  --model $model_path \
  --tokenizer $model_path \
  --batch_size 16 \
  --device cuda

## EVALUATE L1 QALB_15
echo "calculating scores for QALB_15 L1 ..."
python -m gec.evaluate.m2scorer \
    --system_output $output_root/qalb15-test-l1-output.txt \
    --m2_file ../data/real/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.m2

## EVALUATE L2 QALB_15
echo "calculating scores for QALB_15 L2 ..."
python -m gec.evaluate.m2scorer \
    --system_output $output_root/qalb15-test-l2-output.txt \
    --m2_file ../data/real/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.m2 
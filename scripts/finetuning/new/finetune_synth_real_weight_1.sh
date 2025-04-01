#!/bin/bash

env WANDB_ENTITY=mabdelrehim
CHECKPOINTS=../checkpoints
DATA=train_combined-qalb_15_real_w1+qalb_15_synth_t20_w1+arabil_sample_50k_t25_w1
output_dir=../checkpoints/ablation/finetuning/jan_4_2025/base_model_arabart_ft_data_${DATA}

for lr in 5e-4 5e-5 5e-6; do
  CUDA_VISIBLE_DEVICES=0 python -m gec.train \
    --train_file ../data/synthetic/qalb15/${DATA}.jsonl \
    --validation_file ../data/real/clean/qalb-14+qalb-15/jsonl/dev.jsonl \
    --model_name_or_path moussaKam/AraBART \
    --output_dir ${output_dir}_lr${lr} \
    --do_train --do_eval \
    --run_name finetune_on_llm_synthetic_jan_4_2025_w1 \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate $lr \
    --max_steps 100000 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --seed 42 \
    --logging_strategy "steps" \
    --logging_steps 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --eval_strategy "steps" \
    --eval_delay 0 \
    --eval_steps 1000 \
    --predict_with_generate \
    --fp16 \
    --dataloader_num_workers 6 \
    --preprocessing_num_workers 6 \
    --ignore_pad_token_for_loss \
    --include_inputs_for_metrics \
    --report_to wandb --eval_on_start
done


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
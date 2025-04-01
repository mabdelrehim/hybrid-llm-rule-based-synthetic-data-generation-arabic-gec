#!/bin/bash

env WANDB_ENTITY=mabdelrehim
CHECKPOINTS=../checkpoints
DATA=train
output_dir=../checkpoints/paper/finetuning_arabart/qalb15_${DATA}


CUDA_VISIBLE_DEVICES=0 python -m gec.train \
  --train_file ../data/real/clean/qalb-14+qalb-15/jsonl/${DATA}.jsonl \
  --validation_file ../data/real/clean/qalb-14+qalb-15/jsonl/dev.jsonl \
  --model_name_or_path moussaKam/AraBART \
  --output_dir ${output_dir}_lr${lr} \
  --do_train --do_eval \
  --run_name PAPER_finetune_arabart \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.1 \
  --seed 42 \
  --logging_strategy "steps" \
  --logging_steps 10 \
  --save_strategy "steps" \
  --save_steps 500 \
  --eval_strategy "steps" \
  --eval_steps 500 \
  --eval_on_start \
  --predict_with_generate \
  --bf16 \
  --dataloader_num_workers 6 \
  --preprocessing_num_workers 6 \
  --ignore_pad_token_for_loss \
  --include_inputs_for_metrics \
  --report_to wandb --eval_on_start --max_source_length 1024 --max_target_length 1024

## best at 3k ups


test_data=../data/real/clean/qalb-14+qalb-15/jsonl/test-l1.jsonl
output_root=../outputs/paper/finetuning_arabart/qalb15_${DATA}
output_path=$output_root/qalb15-test-l1-output.txt
model_path=${output_dir}_lr/checkpoint-3000

mkdir -p $output_root

CUDA_VISIBLE_DEVICES=0 python -m gec.infere \
  --test_data $test_data \
  --output_file $output_path \
  --model $model_path \
  --tokenizer $model_path \
  --batch_size 16 \
  --device cuda


test_data=../data/real/clean/qalb-14+qalb-15/jsonl/test-l2.jsonl
output_path=$output_root/qalb15-test-l2-output.txt

CUDA_VISIBLE_DEVICES=0 python -m gec.infere \
  --test_data $test_data \
  --output_file $output_path \
  --model $model_path \
  --tokenizer $model_path \
  --batch_size 16 \
  --device cuda
test_data=../data/real/clean/qalb-14+qalb-15+ZAEBUC/jsonl/test.jsonl
output_path=$output_root/ZAEBUC-test-output.txt

CUDA_VISIBLE_DEVICES=0 python -m gec.infere \
  --test_data $test_data \
  --output_file $output_path \
  --model $model_path \
  --tokenizer $model_path \
  --batch_size 16 \
  --device cuda

test_data=../data/real/clean/qalb-14+qalb-15/jsonl/test-l1.jsonl
output_path=$output_root/qalb15-test-l1-output-no-punct.txt

CUDA_VISIBLE_DEVICES=0 python -m gec.infere \
  --test_data $test_data \
  --output_file $output_path \
  --model $model_path \
  --tokenizer $model_path \
  --batch_size 16 \
  --device cuda --no-punct


test_data=../data/real/clean/qalb-14+qalb-15/jsonl/test-l2.jsonl
output_path=$output_root/qalb15-test-l2-output-no-punct.txt

CUDA_VISIBLE_DEVICES=0 python -m gec.infere \
  --test_data $test_data \
  --output_file $output_path \
  --model $model_path \
  --tokenizer $model_path \
  --batch_size 16 \
  --device cuda --no-punct
test_data=../data/real/clean/qalb-14+qalb-15+ZAEBUC/jsonl/test.jsonl
output_path=$output_root/ZAEBUC-test-output-no-punct.txt

CUDA_VISIBLE_DEVICES=0 python -m gec.infere \
  --test_data $test_data \
  --output_file $output_path \
  --model $model_path \
  --tokenizer $model_path \
  --batch_size 16 \
  --device cuda --no-punc

# EVALUATE L1 QALB_1
echo "calculating scores for QALB_15 L1 Beam ..."
python -m gec.evaluate.m2scorer \
    --system_output $output_root/qalb15-test-l1-output.txt \
    --m2_file ../data/real/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.m2

## EVALUATE L2 QALB_15
echo "calculating scores for QALB_15 L2 ..."
python -m gec.evaluate.m2scorer \
    --system_output $output_root/qalb15-test-l2-output.txt \
    --m2_file ../data/real/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.m2
# EVALUATE ZAUBUC
echo "calculating scores for ZAUBUC ..."
python -m gec.evaluate.m2scorer \
    --system_output $output_root/ZAEBUC-test-output.txt \
    --m2_file ../data/real/ZAEBUC-v1.0/data/ar/test/test.pnx.tok.m2

echo "calculating scores for QALB_15 L1 No Punct ..."
python -m gec.evaluate.m2scorer \
    --system_output $output_root/qalb15-test-l1-output-no-punct.txt \
    --m2_file ../data/real/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test-no-punct.m2

## EVALUATE L2 QALB_15
echo "calculating scores for QALB_15 L2 No Punct ..."
python -m gec.evaluate.m2scorer \
    --system_output $output_root/qalb15-test-l2-output-no-punct.txt \
    --m2_file ../data/real/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test-no-punct.m2
# EVALUATE ZAUBUC
echo "calculating scores for ZAUBUC No Punct ..."
python -m gec.evaluate.m2scorer \
    --system_output $output_root/ZAEBUC-test-output-no-punct.txt \
    --m2_file ../data/real/ZAEBUC-v1.0/data/ar/test/test.nopnx.tok.m2
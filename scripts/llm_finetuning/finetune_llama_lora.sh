#!/bin/bash

env WANDB_ENTITY=mabdelrehim


CHECKPOINTS=../checkpoints

for run in qalb-14+qalb-15,qalb-14_qalb-15; do
    
    dataset=$(echo $run | cut -f1 -d,)
    run_name=$(echo $run | cut -f2 -d,)
    env  WANDB_PROJECT=arabic-gec-finetune-llama-3.1-8b-lora-$run_name
    
    CUDA_VISIBLE_DEVICES=0 python -m gec.llms.train \
        --train_file ../data/real/clean/${dataset}/jsonl/llm_chat_formatted/Meta-Llama/v1/train.jsonl \
        --validation_file ../data/real/clean/${dataset}/jsonl/llm_chat_formatted/Meta-Llama/v1/dev.jsonl \
        --output_dir ../checkpoints/llms/lora_finetuning/base_model_meta-llama-3.1-8b_ft_data_${dataset}_no_morph_lr_5e-6_run_2_cat \
        --model_name_or_path="meta-llama/Llama-3.1-8B-Instruct" \
        --report_to="wandb" \
        --learning_rate 6e-6 \
        --lr_scheduler_type constant_with_warmup \
        --warmup_ratio 0.05 \
        --bf16 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --logging_steps 1 \
        --num_train_epochs 3 \
        --max_steps -1 \
        --save_steps 512 \
        --save_total_limit 5 \
        --gradient_checkpointing \
        --use_peft \
        --lora_r 64 \
        --lora_alpha 32 \

        

done
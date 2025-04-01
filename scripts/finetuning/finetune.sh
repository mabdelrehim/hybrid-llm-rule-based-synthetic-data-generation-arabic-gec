#!/bin/bash

env WANDB_ENTITY=mabdelrehim


CHECKPOINTS=../checkpoints


for run in qalb-14,qalb-14 qalb-14+qalb-15,qalb-14_qalb-15 qalb-14+qalb-15+ZAEBUC,qalb-14_qalb-15_ZAEBUC; do
    
    dataset=$(echo $run | cut -f1 -d,)
    run_name=$(echo $run | cut -f2 -d,)
    env  WANDB_PROJECT=arabic-gec-finetune-$run_name-ablation-runs
    
    CUDA_VISIBLE_DEVICES=0 python -m gec.train \
        --train_file ../data/real/clean/${dataset}/jsonl/train.jsonl \
        --validation_file ../data/real/clean/${dataset}/jsonl/dev.jsonl \
        --model_name_or_path moussaKam/AraBART \
        --output_dir ../checkpoints/ablation/finetuning/base_model_arabart_ft_data_${dataset}_no_morph_lr_3e-4 \
        --do_train --do_eval \
        --run_name ABLATION_ft_no_morph_${run_name} \
        --per_device_train_batch_size 8 --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --num_train_epochs 10 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.1 \
        --seed 42 \
        --logging_strategy "steps" \
        --logging_steps 8 \
        --save_strategy "steps" \
        --save_steps 128 \
        --eval_strategy "steps" \
        --eval_delay 0 \
        --eval_steps 128 \
        --predict_with_generate \
        --fp16 \
        --dataloader_num_workers 6 \
        --preprocessing_num_workers 6 \
        --ignore_pad_token_for_loss \
        --include_inputs_for_metrics \
        --report_to wandb --eval_on_start

done
#!/bin/bash

env WANDB_ENTITY=mabdelrehim
env WANDB_PROJECT=arabic-gec-test-runs

TRAIN=/mnt/azureml/cr/j/16cf83efd04d4632947702b6967272ab/exe/wd/mount/argecblob/data/synthetic/arabic_billion_words/train.jsonl
DEV=/mnt/azureml/cr/j/16cf83efd04d4632947702b6967272ab/exe/wd/mount/argecblob/data/real/clean/morph_disambiguated/qalb-14+qalb-15/jsonl/dev.jsonl
CHECKPOINTS=/mnt/azureml/cr/j/16cf83efd04d4632947702b6967272ab/exe/wd/mount/argecblob/checkpoints

cd arabic-gec
python -m gec.train \
    --train_file $TRAIN \
    --validation_file $DEV \
    --model_name_or_path moussaKam/AraBART \
    --output_dir $CHECKPOINTS/pretrain_1.3M_synthetic_gec \
    --do_train \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --seed 42 \
    --logging_strategy "steps" \
    --logging_steps 8 \
    --save_strategy "steps" \
    --save_steps 512 \
    --evaluation_strategy "steps" \
    --eval_steps 512 \
    --predict_with_generate \
    --save_total_limit 10 \
    --fp16 \
    --dataloader_num_workers 6 \
    --preprocessing_num_workers 6 \
    --ignore_pad_token_for_loss \
    --report_to wandb
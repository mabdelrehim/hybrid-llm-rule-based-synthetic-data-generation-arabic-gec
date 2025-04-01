#!/bin/bash

CHECKPOINTS=../checkpoints/ablation/finetuning
DATASETS=../data/real/clean/morph_disambiguated
OUTPUTS=../outputs
RUN=base_model_pretrain_1.3M_train_dist_synthetic_gec

for run in qalb-14,qalb-14 qalb-14+qalb-15,qalb-14_qalb-15 qalb-14+qalb-15+ZAEBUC,qalb-14_qalb-15_ZAEBUC ; do
    dataset=$(echo $run | cut -f1 -d,)
    run_name=$(echo $run | cut -f2 -d,)
    model_path=${CHECKPOINTS}/${RUN}_ft_data_${dataset}_morph_lr_3e-4
    if [[ $dataset == "qalb-14+qalb-15" ]]; then
        $echo "here"
        for subsplit in l1 l2; do
            test_data=${DATASETS}/${dataset}/jsonl/test-${subsplit}.jsonl
            output_path=${OUTPUTS}/ablation/${RUN}_ft_data_${dataset}_morph_lr_3e-4/${dataset}-test-${subsplit}-output.txt

            mkdir -p ${OUTPUTS}/ablation/${RUN}_ft_data_${dataset}_morph_lr_3e-4

            python -m gec.infere \
                --test_data $test_data \
                --output_file $output_path \
                --model $model_path \
                --tokenizer $model_path \
                --batch_size 16 \
                --device cuda
        done
    else
        test_data=${DATASETS}/${dataset}/jsonl/test.jsonl
        output_path=${OUTPUTS}/ablation/${RUN}_ft_data_${dataset}_morph_lr_3e-4/${dataset}-test-output.txt

        mkdir -p ${OUTPUTS}/ablation/${RUN}_ft_data_${dataset}_morph_lr_3e-4

        python -m gec.infere \
            --test_data $test_data \
            --output_file $output_path \
            --model $model_path \
            --tokenizer $model_path \
            --batch_size 16 \
            --device cuda
    fi
done
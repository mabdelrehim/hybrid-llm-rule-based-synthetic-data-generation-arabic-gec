#!/bin/bash

export AZURE_OPENAI_API_KEY="AZURE OPENAI KEY"

DATA_ROOT=../data/synthetic/arabic_billion_words_50k_samples/shards
OUTPUT_ROOT=../data/synthetic/arabic_billion_words_50k_samples/shards_output_llm_rule_hybrid_temp_10_excl_UC
CUDA_VISIBLE_DEVICES=3 python -m gec.corruption \
    --input $DATA_ROOT/part_03.txt \
    --output $OUTPUT_ROOT/part_03.jsonl \
    --batch-size 1 \
    --errors-prior ../data/real/clean/qalb-14+qalb-15+ZAEBUC/annotations/qalb-14+qalb-15+ZAEBUC_error-distribution_temp_10_excl_UC.json \
    --llm gpt-4o-mini \
    --error-definitions error_types.json
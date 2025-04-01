#!/bin/bash

root_dir=../data/real/clean/qalb-14

for set in train test dev; do
    echo "Converting $set to jsonl..."
    python scripts/convert_to_jsonl.py \
        --input_source $root_dir/text/$set.sent \
        --input_correct $root_dir/text/$set.cor \
        --output $root_dir/jsonl/$set.jsonl
done
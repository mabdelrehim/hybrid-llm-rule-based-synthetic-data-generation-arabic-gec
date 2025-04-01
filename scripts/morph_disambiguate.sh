#!/bin/bash

MOUNT_DIR=../mount
DATA_DIR=$MOUNT_DIR/argecblob/data/real/clean

for prep in $DATA_DIR $DATA_DIR/dediac $DATA_DIR/dediac/nopnx; do
    for data in qalb-14 qalb-14+qalb-15 qalb-14+qalb-15+ZAEBUC; do
        for split in train dev test; do
            if [[ $data == "qalb-14+qalb-15" && $split == "dev" ]]; then

                for sub_split in $split $split-l2; do
                    echo "disambiguating $prep/$data/text/$sub_split ...	"
                    python -m gec.morph_disambiguate \
                        --input_file $prep/$data/text/$sub_split.sent \
                        --output_file $prep/morph_disambiguated/$data/text/$sub_split.sent
                    cat $prep/$data/text/$sub_split.cor > $prep/morph_disambiguated/$data/text/$sub_split.cor
                done

            elif [[ $data == "qalb-14+qalb-15" && $split == "test" ]]; then
                for sub_split in $split-l1 $split-l2; do
                    echo "disambiguating $prep/$data/text/$sub_split ...	"
                    python -m gec.morph_disambiguate \
                        --input_file $prep/$data/text/$sub_split.sent \
                        --output_file $prep/morph_disambiguated/$data/text/$sub_split.sent
                    cat $prep/$data/text/$sub_split.cor > $prep/morph_disambiguated/$data/text/$sub_split.cor
                done
            else
                echo "disambiguating $prep/$data/text/$split ...	"
                python -m gec.morph_disambiguate \
                    --input_file $prep/$data/text/$split.sent \
                    --output_file $prep/morph_disambiguated/$data/text/$split.sent
                cat $prep/$data/text/$split.cor > $prep/morph_disambiguated/$data/text/$split.cor
            fi
        done
    done
done
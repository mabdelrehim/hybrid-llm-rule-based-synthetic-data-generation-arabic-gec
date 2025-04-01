#!/bin/bash

#eval "$(conda shell.bash hook)"
#source /anaconda/etc/profile.d/conda.sh

# generate error_type annotations for train data
DATA_DIR=/home/azureuser/cloudfiles/code/Users/mo.abdelre/argecblob/data/real

# QALB14
src=$DATA_DIR/clean/qalb-14+qalb-15+ZAEBUC/text/train.cor
system_output=$DATA_DIR/clean/qalb-14+qalb-15+ZAEBUC/text/train.sent
error_analysis_dir=$DATA_DIR/clean/qalb-14+qalb-15+ZAEBUC/annotations

#rm -rf $error_analysis_dir
#mkdir -p $error_analysis_dir

alignment_output=${error_analysis_dir}/qalb-14+qalb-15+ZAEBUC_train.alignment.txt

#printf "Generating alignments for ${src}..\n"

#conda activate arabic-gec
#cd alignment/
#python aligner.py \
#    --src ${src} \
#    --tgt ${system_output} \
#    --output ${alignment_output}
#
#
#conda deactivate

#conda activate arabic-gec-areta
cd areta/

areta_tags_output=${error_analysis_dir}/qalb-14+qalb-15+ZAEBUC_train.areta.txt
enriched_areta_tags_output=${error_analysis_dir}/qalb-14+qalb-15+ZAEBUC_train.areta+.txt

printf "Generating areta tags for ${alignment_output}..\n"

python annotate_err_type_ar.py \
    --alignment $alignment_output \
    --output_path $areta_tags_output \
    --enriched_output_path $enriched_areta_tags_output

rm fout2.basic


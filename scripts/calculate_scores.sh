#!/bin/bash

which python

SCRIPT_DIR=../data/real/QALB-0.9.1-Dec03-2021-SharedTasks/m2Scripts
cd $SCRIPT_DIR

OUTPUTS_DIR=../../../../outputs/ablation
MODEL_OUTS_PREFIX=base_model_pretrain_1.3M_train_dist_synthetic_gec

## EVALUATE ZAEBUC
echo "calculating scores for ZAEBUC ..."
python m2scorer.py \
    ${OUTPUTS_DIR}/${MODEL_OUTS_PREFIX}_ft_data_qalb-14+qalb-15+ZAEBUC_no_morph_lr_3e-4/qalb-14+qalb-15+ZAEBUC-test-output.txt \
    ../../ZAEBUC-v1.0/data/ar/test/test.pnx.tok.m2 > ${OUTPUTS_DIR}/${MODEL_OUTS_PREFIX}_ft_data_qalb-14+qalb-15+ZAEBUC_no_morph_lr_3e-4/scores.txt

## EVALUATE L2 QALB_15
echo "calculating scores for QALB_15 L2 ..."
python m2scorer.py \
    ${OUTPUTS_DIR}/${MODEL_OUTS_PREFIX}_ft_data_qalb-14+qalb-15_no_morph_lr_3e-4/qalb-14+qalb-15-test-l2-output.txt \
    ../data/2015/test/QALB-2015-L2-Test.m2 > ${OUTPUTS_DIR}/${MODEL_OUTS_PREFIX}_ft_data_qalb-14+qalb-15_no_morph_lr_3e-4/scores-l2.txt

## EVALUATE L1 QALB_15
echo "calculating scores for QALB_15 L1 ..."
python m2scorer.py \
    ${OUTPUTS_DIR}/${MODEL_OUTS_PREFIX}_ft_data_qalb-14+qalb-15_no_morph_lr_3e-4/qalb-14+qalb-15-test-l1-output.txt \
    ../data/2015/test/QALB-2015-L1-Test.m2 > ${OUTPUTS_DIR}/${MODEL_OUTS_PREFIX}_ft_data_qalb-14+qalb-15_no_morph_lr_3e-4/scores-l1.txt

## EVALUATE QALB_14
echo "calculating scores for QALB_14 ..."
python m2scorer.py \
    ${OUTPUTS_DIR}/${MODEL_OUTS_PREFIX}_ft_data_qalb-14_no_morph_lr_3e-4/qalb-14-test-output.txt \
    ../data/2014/test/QALB-2014-L1-Test.m2 > ${OUTPUTS_DIR}/${MODEL_OUTS_PREFIX}_ft_data_qalb-14_no_morph_lr_3e-4/scores.txt

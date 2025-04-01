#!/bin/bash

ROOT_DIR=../mount

QALB14_DIR=$ROOT_DIR/data/real/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014
QALB15_DIR=$ROOT_DIR/data/real/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015
ZAEBUC_DIR=$ROOT_DIR/data/real/ZAEBUC-v1.0/data/ar

# collect clean/no_ids files
cat $QALB14_DIR/train/QALB-2014-L1-Train.sent.no_ids.clean > data/real/clean/qalb-14/text/train.sent
cat $QALB14_DIR/train/QALB-2014-L1-Train.cor.no_ids > data/real/clean/qalb-14/text/train.cor

cat $QALB14_DIR/dev/QALB-2014-L1-Dev.sent.no_ids.clean > data/real/clean/qalb-14/text/dev.sent
cat $QALB14_DIR/dev/QALB-2014-L1-Dev.cor.no_ids > data/real/clean/qalb-14/text/dev.cor

cat $QALB14_DIR/test/QALB-2014-L1-Test.sent.no_ids.clean > data/real/clean/qalb-14/text/test.sent
cat $QALB14_DIR/test/QALB-2014-L1-Test.cor.no_ids > data/real/clean/qalb-14/text/test.cor



cat $QALB15_DIR/train/QALB-2015-L2-Train.sent.no_ids > data/real/clean/qalb-14+qalb-15/text/train.sent
cat $QALB14_DIR/train/QALB-2014-L1-Train.sent.no_ids.clean >> data/real/clean/qalb-14+qalb-15/text/train.sent
cat $QALB15_DIR/train/QALB-2015-L2-Train.cor.no_ids > data/real/clean/qalb-14+qalb-15/text/train.cor
cat $QALB14_DIR/train/QALB-2014-L1-Train.cor.no_ids >> data/real/clean/qalb-14+qalb-15/text/train.cor

cat $QALB15_DIR/dev/QALB-2015-L2-Dev.sent.no_ids > data/real/clean/qalb-14+qalb-15/text/dev-l2.sent
cat $QALB15_DIR/dev/QALB-2015-L2-Dev.cor.no_ids > data/real/clean/qalb-14+qalb-15/text/dev-l2.cor
cat $QALB15_DIR/dev/QALB-2015-L2-Dev.sent.no_ids > data/real/clean/qalb-14+qalb-15/text/dev.sent
cat $QALB14_DIR/dev/QALB-2014-L1-Dev.sent.no_ids.clean >> data/real/clean/qalb-14+qalb-15/text/dev.sent
cat $QALB15_DIR/dev/QALB-2015-L2-Dev.cor.no_ids > data/real/clean/qalb-14+qalb-15/text/dev.cor
cat $QALB14_DIR/dev/QALB-2014-L1-Dev.cor.no_ids >> data/real/clean/qalb-14+qalb-15/text/dev.cor

cat $QALB15_DIR/test/QALB-2015-L2-Test.sent.no_ids > data/real/clean/qalb-14+qalb-15/text/test-l2.sent
cat $QALB15_DIR/test/QALB-2015-L2-Test.cor.no_ids > data/real/clean/qalb-14+qalb-15/text/test-l2.cor
cat $QALB15_DIR/test/QALB-2015-L1-Test.sent.no_ids > data/real/clean/qalb-14+qalb-15/text/test-l1.sent
cat $QALB15_DIR/test/QALB-2015-L1-Test.cor.no_ids > data/real/clean/qalb-14+qalb-15/text/test-l1.cor



cat $ZAEBUC_DIR/train/train.sent.raw.pnx.tok > data/real/clean/qalb-14+qalb-15+ZAEBUC/text/train.sent
cat $QALB15_DIR/train/QALB-2015-L2-Train.sent.no_ids >> data/real/clean/qalb-14+qalb-15+ZAEBUC/text/train.sent
cat $QALB14_DIR/train/QALB-2014-L1-Train.sent.no_ids.clean >> data/real/clean/qalb-14+qalb-15+ZAEBUC/text/train.sent
cat $ZAEBUC_DIR/train/train.sent.cor.pnx.tok > data/real/clean/qalb-14+qalb-15+ZAEBUC/text/train.cor
cat $QALB15_DIR/train/QALB-2015-L2-Train.cor.no_ids >> data/real/clean/qalb-14+qalb-15+ZAEBUC/text/train.cor
cat $QALB14_DIR/train/QALB-2014-L1-Train.cor.no_ids >> data/real/clean/qalb-14+qalb-15+ZAEBUC/text/train.cor

cat $ZAEBUC_DIR/dev/dev.sent.raw.pnx.tok > data/real/clean/qalb-14+qalb-15+ZAEBUC/text/dev.sent
cat $ZAEBUC_DIR/dev/dev.sent.cor.pnx.tok > data/real/clean/qalb-14+qalb-15+ZAEBUC/text/dev.cor

cat $ZAEBUC_DIR/test/test.sent.raw.pnx.tok > data/real/clean/qalb-14+qalb-15+ZAEBUC/text/test.sent
cat $ZAEBUC_DIR/test/test.sent.cor.pnx.tok > data/real/clean/qalb-14+qalb-15+ZAEBUC/text/test.cor



# collect dediac files
cat $QALB14_DIR/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac > data/real/clean/dediac/qalb-14/text/train.sent
cat $QALB14_DIR/train/QALB-2014-L1-Train.cor.no_ids.dediac > data/real/clean/dediac/qalb-14/text/train.cor

cat $QALB14_DIR/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac > data/real/clean/dediac/qalb-14/text/dev.sent
cat $QALB14_DIR/dev/QALB-2014-L1-Dev.cor.no_ids.dediac > data/real/clean/dediac/qalb-14/text/dev.cor

cat $QALB14_DIR/test/QALB-2014-L1-Test.sent.no_ids.clean.dediac > data/real/clean/dediac/qalb-14/text/test.sent
cat $QALB14_DIR/test/QALB-2014-L1-Test.cor.no_ids.dediac > data/real/clean/dediac/qalb-14/text/test.cor



cat $QALB15_DIR/train/QALB-2015-L2-Train.sent.no_ids.dediac > data/real/clean/dediac/qalb-14+qalb-15/text/train.sent
cat $QALB14_DIR/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac >> data/real/clean/dediac/qalb-14+qalb-15/text/train.sent
cat $QALB15_DIR/train/QALB-2015-L2-Train.cor.no_ids.dediac > data/real/clean/dediac/qalb-14+qalb-15/text/train.cor
cat $QALB14_DIR/train/QALB-2014-L1-Train.cor.no_ids.dediac >> data/real/clean/dediac/qalb-14+qalb-15/text/train.cor

cat $QALB15_DIR/dev/QALB-2015-L2-Dev.sent.no_ids.dediac > data/real/clean/dediac/qalb-14+qalb-15/text/dev-l2.sent
cat $QALB15_DIR/dev/QALB-2015-L2-Dev.cor.no_ids.dediac > data/real/clean/dediac/qalb-14+qalb-15/text/dev-l2.cor
cat $QALB15_DIR/dev/QALB-2015-L2-Dev.sent.no_ids.dediac > data/real/clean/dediac/qalb-14+qalb-15/text/dev.sent
cat $QALB14_DIR/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac >> data/real/clean/dediac/qalb-14+qalb-15/text/dev.sent
cat $QALB15_DIR/dev/QALB-2015-L2-Dev.cor.no_ids.dediac > data/real/clean/dediac/qalb-14+qalb-15/text/dev.cor
cat $QALB14_DIR/dev/QALB-2014-L1-Dev.cor.no_ids.dediac >> data/real/clean/dediac/qalb-14+qalb-15/text/dev.cor

cat $QALB15_DIR/test/QALB-2015-L2-Test.sent.no_ids.dediac > data/real/clean/dediac/qalb-14+qalb-15/text/test-l2.sent
cat $QALB15_DIR/test/QALB-2015-L2-Test.cor.no_ids.dediac > data/real/clean/dediac/qalb-14+qalb-15/text/test-l2.cor
cat $QALB15_DIR/test/QALB-2015-L1-Test.sent.no_ids.dediac > data/real/clean/dediac/qalb-14+qalb-15/text/test-l1.sent
cat $QALB15_DIR/test/QALB-2015-L1-Test.cor.no_ids.dediac > data/real/clean/dediac/qalb-14+qalb-15/text/test-l1.cor



cat $ZAEBUC_DIR/train/train.sent.raw.pnx.tok.dediac > data/real/clean/dediac/qalb-14+qalb-15+ZAEBUC/text/train.sent
cat $QALB15_DIR/train/QALB-2015-L2-Train.sent.no_ids.dediac >> data/real/clean/dediac/qalb-14+qalb-15+ZAEBUC/text/train.sent
cat $QALB14_DIR/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac >> data/real/clean/dediac/qalb-14+qalb-15+ZAEBUC/text/train.sent
cat $ZAEBUC_DIR/train/train.sent.cor.pnx.tok.dediac > data/real/clean/dediac/qalb-14+qalb-15+ZAEBUC/text/train.cor
cat $QALB15_DIR/train/QALB-2015-L2-Train.cor.no_ids.dediac >> data/real/clean/dediac/qalb-14+qalb-15+ZAEBUC/text/train.cor
cat $QALB14_DIR/train/QALB-2014-L1-Train.cor.no_ids.dediac >> data/real/clean/dediac/qalb-14+qalb-15+ZAEBUC/text/train.cor

cat $ZAEBUC_DIR/dev/dev.sent.raw.pnx.tok.dediac > data/real/clean/dediac/qalb-14+qalb-15+ZAEBUC/text/dev.sent
cat $ZAEBUC_DIR/dev/dev.sent.cor.pnx.tok.dediac > data/real/clean/dediac/qalb-14+qalb-15+ZAEBUC/text/dev.cor

cat $ZAEBUC_DIR/test/test.sent.raw.pnx.tok.dediac > data/real/clean/dediac/qalb-14+qalb-15+ZAEBUC/text/test.sent
cat $ZAEBUC_DIR/test/test.sent.cor.pnx.tok.dediac > data/real/clean/dediac/qalb-14+qalb-15+ZAEBUC/text/test.cor



# collect dediac no punctiuation files
cat $QALB14_DIR/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14/text/train.sent
cat $QALB14_DIR/train/QALB-2014-L1-Train.cor.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14/text/train.cor

cat $QALB14_DIR/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14/text/dev.sent
cat $QALB14_DIR/dev/QALB-2014-L1-Dev.cor.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14/text/dev.cor

cat $QALB14_DIR/test/QALB-2014-L1-Test.sent.no_ids.clean.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14/text/test.sent
cat $QALB14_DIR/test/QALB-2014-L1-Test.cor.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14/text/test.cor



cat $QALB15_DIR/train/QALB-2015-L2-Train.sent.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/train.sent
cat $QALB14_DIR/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac.nopnx >> data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/train.sent
cat $QALB15_DIR/train/QALB-2015-L2-Train.cor.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/train.cor
cat $QALB14_DIR/train/QALB-2014-L1-Train.cor.no_ids.dediac.nopnx >> data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/train.cor

cat $QALB15_DIR/dev/QALB-2015-L2-Dev.sent.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/dev-l2.sent
cat $QALB15_DIR/dev/QALB-2015-L2-Dev.cor.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/dev-l2.cor
cat $QALB15_DIR/dev/QALB-2015-L2-Dev.sent.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/dev.sent
cat $QALB14_DIR/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac.nopnx >> data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/dev.sent
cat $QALB15_DIR/dev/QALB-2015-L2-Dev.cor.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/dev.cor
cat $QALB14_DIR/dev/QALB-2014-L1-Dev.cor.no_ids.dediac.nopnx >> data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/dev.cor

cat $QALB15_DIR/test/QALB-2015-L2-Test.sent.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/test-l2.sent
cat $QALB15_DIR/test/QALB-2015-L2-Test.cor.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/test-l2.cor
cat $QALB15_DIR/test/QALB-2015-L1-Test.sent.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/test-l1.sent
cat $QALB15_DIR/test/QALB-2015-L1-Test.cor.no_ids.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15/text/test-l1.cor



cat $ZAEBUC_DIR/train/train.sent.raw.pnx.tok.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15+ZAEBUC/text/train.sent
cat $QALB15_DIR/train/QALB-2015-L2-Train.sent.no_ids.dediac.nopnx >> data/real/clean/dediac/nopnx/qalb-14+qalb-15+ZAEBUC/text/train.sent
cat $QALB14_DIR/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac.nopnx >> data/real/clean/dediac/nopnx/qalb-14+qalb-15+ZAEBUC/text/train.sent
cat $ZAEBUC_DIR/train/train.sent.cor.pnx.tok.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15+ZAEBUC/text/train.cor
cat $QALB15_DIR/train/QALB-2015-L2-Train.cor.no_ids.dediac.nopnx >> data/real/clean/dediac/nopnx/qalb-14+qalb-15+ZAEBUC/text/train.cor
cat $QALB14_DIR/train/QALB-2014-L1-Train.cor.no_ids.dediac.nopnx >> data/real/clean/dediac/nopnx/qalb-14+qalb-15+ZAEBUC/text/train.cor

cat $ZAEBUC_DIR/dev/dev.sent.raw.pnx.tok.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15+ZAEBUC/text/dev.sent
cat $ZAEBUC_DIR/dev/dev.sent.cor.pnx.tok.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15+ZAEBUC/text/dev.cor

cat $ZAEBUC_DIR/test/test.sent.raw.pnx.tok.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15+ZAEBUC/text/test.sent
cat $ZAEBUC_DIR/test/test.sent.cor.pnx.tok.dediac.nopnx > data/real/clean/dediac/nopnx/qalb-14+qalb-15+ZAEBUC/text/test.cor

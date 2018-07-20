# #!/bin/bash

set -e
set -x

ROOT_DIR=/Users/mac/PycharmProjects/riminder/


# path to scripts directories
M2_SCRIPTS=$ROOT_DIR/preprocess/nucle
LANG8_SCRIPTS=$ROOT_DIR/preprocess/lang8
CLEAN_SCRIPTS=$ROOT_DIR/preprocess/

chmod u+x $LANG8_SCRIPTS/convert_to_parallel.py
chmod u+x $M2_SCRIPTS/convert_m2_to_parallel.py
chmod u+x $CLEAN_SCRIPTS/clean_data.py

mkdir -p $ROOT_DIR/data/nucle
mkdir -p $ROOT_DIR/data/lang-8

# create parallel data
$M2_SCRIPTS/convert_m2_to_parallel.py   $ROOT_DIR/data/conll13st-preprocessed.m2 \
                                     $ROOT_DIR/data/nucle/nucle-train.tok.src \
                                     $ROOT_DIR/data/nucle/nucle-train.tok.trg

$LANG8_SCRIPTS/convert_to_parallel.py   $ROOT_DIR/data/entries.train \
                                     $ROOT_DIR/data/lang-8/lang-8.tok.src \
                                     $ROOT_DIR/data/lang-8/lang-8.tok.trg

# concatenated training data.
mkdir -p $ROOT_DIR/data/concat-train
cat $ROOT_DIR/data/nucle/nucle-train.tok.src $ROOT_DIR/data/lang-8/lang-8.tok.src > $ROOT_DIR/data/concat-train/concat-train.tok.src
cat $ROOT_DIR/data/nucle/nucle-train.tok.trg $ROOT_DIR/data/lang-8/lang-8.tok.trg > $ROOT_DIR/data/concat-train/concat-train.tok.trg


mkdir -p $ROOT_DIR/data/clean-train
$CLEAN_SCRIPTS/clean_data.py    $ROOT_DIR/data/concat-train/concat-train.tok.src \
                            $ROOT_DIR/data/concat-train/concat-train.tok.trg \
                            $ROOT_DIR/data/clean-train/clean-train.tok.src \
                            $ROOT_DIR/data/clean-train/clean-train.tok.trg
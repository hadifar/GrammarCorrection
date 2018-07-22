# #!/bin/bash

set -e
set -x

source ../paths.sh

ROOT_DIR=$BASE_DIR


# path to scripts directories
M2_SCRIPTS=$ROOT_DIR/preprocess/nucle
LANG8_SCRIPTS=$ROOT_DIR/preprocess/lang8
CLEAN_SCRIPTS=$ROOT_DIR/preprocess
TYPO_SCRIPTS=$ROOT_DIR/preprocess

chmod u+x $LANG8_SCRIPTS/convert_to_parallel.py
chmod u+x $M2_SCRIPTS/convert_m2_to_parallel.py
chmod u+x $CLEAN_SCRIPTS/clean_data.py
chmod u+x $CLEAN_SCRIPTS/typo_generator.py

mkdir -p $ROOT_DIR/data/nucle
mkdir -p $ROOT_DIR/data/lang-8

# create parallel data
$M2_SCRIPTS/convert_m2_to_parallel.py   --input_path=$ROOT_DIR/data/conll13st-preprocessed.m2 \
                                     --output_src=$ROOT_DIR/data/nucle/nucle-train.tok.src \
                                     --output_trg=$ROOT_DIR/data/nucle/nucle-train.tok.trg

$LANG8_SCRIPTS/convert_to_parallel.py   --input_path=$ROOT_DIR/data/entries.train \
                                     --output_src=$ROOT_DIR/data/lang-8/lang-8.tok.src \
                                     --output_trg=$ROOT_DIR/data/lang-8/lang-8.tok.trg

# concatenated training data.
mkdir -p $ROOT_DIR/data/concat-train
cat $ROOT_DIR/data/nucle/nucle-train.tok.src $ROOT_DIR/data/lang-8/lang-8.tok.src > $ROOT_DIR/data/concat-train/concat-train.tok.src
cat $ROOT_DIR/data/nucle/nucle-train.tok.trg $ROOT_DIR/data/lang-8/lang-8.tok.trg > $ROOT_DIR/data/concat-train/concat-train.tok.trg

# clean data
mkdir -p $ROOT_DIR/data/clean-train
$CLEAN_SCRIPTS/clean_data.py    --input_src=$ROOT_DIR/data/concat-train/concat-train.tok.src \
                            --input_trg=$ROOT_DIR/data/concat-train/concat-train.tok.trg \
                            --output_src=$ROOT_DIR/data/clean-train/clean-train.tok.src \
                            --input_trg=$ROOT_DIR/data/clean-train/clean-train.tok.trg

# generate typo
mkdir -p $ROOT_DIR/data/final-train
$TYPO_SCRIPTS/typo_generator.py    --input_src=$ROOT_DIR/data/clean-train/clean-train.tok.src \
                            --input_trg=$ROOT_DIR/data/clean-train/clean-train.tok.trg \
                            --output_src=$ROOT_DIR/data/final-train/final-train.tok.src \
                            --output_trg=$ROOT_DIR/data/final-train/final-train.tok.trg
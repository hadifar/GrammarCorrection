# #!/bin/bash

set -e
set -x

source ../paths.sh

ROOT_DIR=$BASE_DIR

chmod u+x $ROOT_DIR/train/prepare_data.py
chmod u+x $ROOT_DIR/train/train.py
chmod u+x $ROOT_DIR/train/predict.py

$ROOT_DIR/train/prepare_data.py --text_A="./data/final-train/final-train.tok.trg" \
                                --text_B="./data/final-train/final-train.tok.src" \
                                --out_file="./data/trg_src_prepped.h5"


mkdir -p $ROOT_DIR/train/weights

$ROOT_DIR/train/train.py --dataset="./data/src_trg_prepared.h5" \
                        --weights_path="./weights/KerasAttentionNMT.h5"



$ROOT_DIR/train/predict.py --dataset="./data/src_trg_prepared.h5" \
                            --weights_path="./weights/KerasAttentionNMT.h5"
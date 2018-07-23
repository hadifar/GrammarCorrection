# #!/bin/bash

set -e
set -x

source ../paths.sh

ROOT_DIR=$BASE_DIR

chmod u+x $ROOT_DIR/models/prepare_data.py
chmod u+x $ROOT_DIR/models/train.py
chmod u+x $ROOT_DIR/models/predict.py

$ROOT_DIR/models/prepare_data.py --text_A=$ROOT_DIR/data/final-train/final-train.tok.src \
                                --text_B=$ROOT_DIR/data/final-train/final-train.tok.trg \
                                --out_file=$ROOT_DIR/data/trg_src_prepped.h5


mkdir -p $ROOT_DIR/train/weights
$ROOT_DIR/models/train.py --dataset=$ROOT_DIR/data/trg_src_prepped.h5 \
                        --weights_path=$ROOT_DIR/train/weights/KerasAttentionNMT.h5


$ROOT_DIR/models/predict.py --dataset=$ROOT_DIR/data/trg_src_prepped.h5 \
                            --weights_path=$ROOT_DIR/weights/KerasAttentionNMT.h5
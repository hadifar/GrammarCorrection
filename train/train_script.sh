

set -e
set -x

ROOT_DIR=/Users/mac/PycharmProjects/riminder/


# prepare data for train
python2 train/prepare_data.py --text_A="./data/final-train/final-train.tok.trg" --text_B="./data/final-train/final-train.tok.src" --out_file="./data/trg_src_prepped.h5"


# train model
mkdir weights
python train.py --dataset="./data/trg_src_prepped.h5" --weights_path="./weights/KerasAttentionNMT_1.h5"


# inference
python predict.py --dataset="./data/trg_src_prepped.h5" --weights_path="./weights/KerasAttentionNMT_1.h5"
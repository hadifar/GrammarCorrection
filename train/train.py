#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random

import h5py
import keras

import config
import model

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="./data/src_trg_prepared.h5",
                    help='Path to HDF5 file')

parser.add_argument('--weights_path', type=str, default="./weights/KerasAttentionNMT.h5",
                    help='Path to Weights checkpoint')

args = parser.parse_args()

hf = h5py.File(args.dataset, 'r')

inp_x = hf['target_sent_mat'][:, : config.MAX_SEQ_LEN]
inp_cond_x = hf['source_sent_mat'][:, : config.MAX_SEQ_LEN]
out_y = hf['source_sent_mat'][:, 1: config.MAX_SEQ_LEN + 1]

tr_data = range(inp_x.shape[0])
random.shuffle(tr_data)


def load_data(batchSize=config.BATCH_SIZE):
    while True:
        for i in range(0, len(tr_data) - batchSize, batchSize):
            inds = tr_data[i: i + batchSize]
            yield [inp_x[inds], inp_cond_x[inds]], \
                  keras.utils.to_categorical(out_y[inds], num_classes=config.MAX_VOCAB_SIZE)


tr_gen = load_data(batchSize=config.BATCH_SIZE)

m = model.getModel(enc_seq_length=config.MAX_SEQ_LEN,
                   enc_vocab_size=config.MAX_VOCAB_SIZE,
                   dec_seq_length=config.MAX_SEQ_LEN,
                   dec_vocab_size=config.MAX_VOCAB_SIZE)

for ep in range(config.EPOCH_NUM):
    print "Epoch", ep
    m.fit_generator(tr_gen, steps_per_epoch=config.STEPS_PER_EPOCH)
    m.save_weights(args.weights_path + "." + str(ep))
    m.save_weights(args.weights_path)

print "Training is finished"

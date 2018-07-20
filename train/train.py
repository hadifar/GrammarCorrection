#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random

import h5py
import keras

import model

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="./data/src_trg_prepared.h5",
                    help='Path to HDF5 file')

parser.add_argument('--weights_path', type=str, default="./weights/KerasAttentionNMT_1.h5",
                    help='Path to Weights checkpoint')

args = parser.parse_args()

hf = h5py.File(args.dataset, 'r')

enc_seq_length = 35
enc_vocab_size = 40005
dec_seq_length = 35
dec_vocab_size = 40005

inp_x = hf['target_sent_mat'][:, : enc_seq_length]
inp_cond_x = hf['source_sent_mat'][:, : dec_seq_length]
out_y = hf['source_sent_mat'][:, 1: dec_seq_length + 1]

tr_data = range(inp_x.shape[0])
random.shuffle(tr_data)


def load_data(batchSize=64):
    while True:
        for i in range(0, len(tr_data) - batchSize, batchSize):
            inds = tr_data[i: i + batchSize]
            yield [inp_x[inds], inp_cond_x[inds]], keras.utils.to_categorical(out_y[inds], num_classes=dec_vocab_size)


tr_gen = load_data(batchSize=64)

m = model.getModel()

for ep in range(1):
    print "Epoch", ep
    m.fit_generator(tr_gen, steps_per_epoch=1, epochs=1)
    m.save_weights(args.weights_path + "." + str(ep))
    m.save_weights(args.weights_path)

print "Training is finished"

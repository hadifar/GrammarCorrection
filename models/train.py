#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random

import keras
import numpy as np

import config
import seq2seq_attention

parser = argparse.ArgumentParser()

parser.add_argument('--cache_dir', required=True, default="../data/",
                    help='Path to cache files')

parser.add_argument('--weights_path', required=True, default="../models/weights/KerasAttentionNMT.h5",
                    help='Path to Weights checkpoint')

args = parser.parse_args()

inp_x = np.load(open(args.cache_dir + config.CACHE_SOURCE, 'rb'))
inp_cond_x = np.load(open(args.cache_dir + config.CACHE_TARGET, 'rb'))
out_y = inp_cond_x[:, 1: config.MAX_SEQ_LEN + 1]  # encoder is one token ahead
out_y = np.pad(out_y, (1, 0), 'constant', constant_values=0)  # add pad for missing index

nb_samples = inp_x.shape[0]
tr_data = range(nb_samples)
random.shuffle(tr_data)
step_per_epoch = nb_samples / config.BATCH_SIZE


def load_data(batchSize=config.BATCH_SIZE):
    while True:
        for i in range(0, len(tr_data) - batchSize, batchSize):
            inds = tr_data[i: i + batchSize]
            yield [inp_x[inds], inp_cond_x[inds]], keras.utils.to_categorical(out_y[inds],
                                                                              num_classes=config.MAX_VOCAB_SIZE)


tr_gen = load_data(batchSize=config.BATCH_SIZE)

# word_index = dict(target_vocab['word2idx'], **source_vocab['word2idx'])
# embedding = load_glove_matrix(word_index)
# nb_words = len(word_index)

model = seq2seq_attention.getModel()

for ep in range(config.EPOCH_NUM):
    print ("Epoch", ep)
    model.fit_generator(tr_gen, steps_per_epoch=step_per_epoch, epochs=1)
    model.save_weights(args.weights_path + "." + str(ep))
    model.save_weights(args.weights_path)

print ("Training is finished")

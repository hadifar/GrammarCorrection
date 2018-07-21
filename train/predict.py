#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import h5py
import numpy as np

import config
import model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="./data/trg_src_prepped.h5",
                    help='Path to HDF5 file')

parser.add_argument('--weights_path', type=str, default="./weights/KerasAttentionNMT.h5",
                    help='Path to Weights checkpoint')

args = parser.parse_args()

hf = h5py.File(args.dataset, 'r')

m = model.getModel(enc_seq_length=config.MAX_SEQ_LEN,
                   enc_vocab_size=config.MAX_VOCAB_SIZE,
                   dec_seq_length=config.MAX_SEQ_LEN,
                   dec_vocab_size=config.MAX_VOCAB_SIZE)

m.load_weights(args.weights_path)

target_vocab = json.loads(hf['target_vocab'].value)
source_vocab = json.loads(hf['source_vocab'].value)


def predict(sent):
    words = sent.split(' ')
    words = ['<start>'] + words + ['<end>']
    words_id = []

    for w in words:
        if w in target_vocab['word2idx']:
            words_id.append(target_vocab['word2idx'][w])
        else:
            words_id.append(target_vocab['word2idx']['<unk>'])
    words = words_id

    ret = ""

    m_input = [np.zeros((1, config.MAX_SEQ_LEN)), np.zeros((1, config.MAX_SEQ_LEN))]

    for i, w in enumerate(words):
        m_input[0][0, i] = w
    m_input[1][0, 0] = source_vocab['word2idx']['<start>']

    for w_i in range(1, config.MAX_SEQ_LEN):
        out = m.predict(m_input)
        out_w_i = out[0][w_i - 1].argmax()

        if out_w_i == 0:
            continue

        ret += source_vocab['idx2word'][str(out_w_i)] + " "
        m_input[1][0, w_i] = out_w_i

    return ret


while True:
    print ("Enter a sentence to correct typo: ")
    sent = raw_input()
    print (predict(sent).encode('utf-8'))

    print ("===============")

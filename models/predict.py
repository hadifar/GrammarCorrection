#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import h5py
import numpy as np

import config
import seq2seq_attention

parser = argparse.ArgumentParser()
parser.add_argument('--cache_dir', type=str, default="../data/",
                    help='Path to cache files')

parser.add_argument('--weights_path', type=str, default="../models/weights/KerasAttentionNMT.h5",
                    help='Path to Weights checkpoint')

args = parser.parse_args()
#
# hf = h5py.File(args.dataset, 'r')
# target_vocab = json.loads(hf['target_vocab'].value)
# source_vocab = json.loads(hf['source_vocab'].value)
word_index = np.load(open(args.cache_dir + config.CACHE_WORD_INDEX, 'rb'))
word_index = word_index.flatten()[0]

model = seq2seq_attention.getModel()

model.load_weights(args.weights_path)


def predict(sentence):
    words = sentence.split(' ')
    words = ['<start>'] + words + ['<end>']
    words_id = []

    for w in words:
        if w in word_index:
            words_id.append(word_index[w])
        else:
            words_id.append(word_index['<unk>'])
    words = words_id

    ret = ""

    m_input = [np.zeros((1, config.MAX_SEQ_LEN)), np.zeros((1, config.MAX_SEQ_LEN))]

    for i, w in enumerate(words):
        m_input[0][0, i] = w
    m_input[1][0, 0] = word_index['<start>']

    for w_i in range(1, config.MAX_SEQ_LEN):
        out = model.predict(m_input)
        out_w_i = out[0][w_i - 1].argmax()

        if out_w_i == 0:
            continue

        ret += word_index[str(out_w_i)] + " "
        m_input[1][0, w_i] = out_w_i

    return ret


while True:
    print ("Enter a sentence to correct typo: ")
    sent = raw_input()
    print ('your input: ' + sent)
    print (predict(sent).encode('utf-8'))

    print ("===============")

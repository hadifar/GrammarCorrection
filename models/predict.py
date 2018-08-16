#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np

import config
import seq2seq_attention
from models import general_helper

parser = argparse.ArgumentParser()
parser.add_argument('--cache_dir', default="../data/",
                    help='Path to cache files')

parser.add_argument('--weights_path', default="../models/weights/KerasAttentionNMT.h5",
                    help='Path to Weights checkpoint')

args = parser.parse_args()

word_index = np.load(open(args.cache_dir + config.CACHE_WORD_INDEX, 'rb'))
embedding = general_helper.load_embedding_matrix(word_index.item())
word_index = word_index.flatten()[0]
index_word = dict([(value, key) for (key, value) in word_index.items()])

model = seq2seq_attention.getModel(embedding, word_index)

model.load_weights(args.weights_path)


def predict(sentence):
    words = sentence.split(' ')
    words = ['<START>'] + words + ['<END>']
    words_id = []

    for w in words:
        if w in word_index:
            words_id.append(word_index[w])
        else:
            words_id.append(word_index['<UNK>'])
    words = words_id

    ret = ""

    m_input = [np.zeros((1, config.MAX_SEQ_LEN)), np.zeros((1, config.MAX_SEQ_LEN))]

    for i, w in enumerate(words):
        m_input[0][0, i] = w
    m_input[1][0, 0] = word_index['<START>']

    for w_i in range(1, config.MAX_SEQ_LEN):
        out = model.predict(m_input)
        out_w_i = out[0][w_i - 1].argmax()

        if out_w_i == 0:
            continue

        ret += index_word[out_w_i] + " "
        m_input[1][0, w_i] = out_w_i

    return ret


while True:
    print ("Enter a sentence to correct typo: ")
    sent = raw_input()
    print ('your input: ' + sent)
    print (predict(sent))
    print(20 * '-')

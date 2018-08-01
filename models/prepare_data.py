#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from keras_preprocessing.text import Tokenizer
from tensorflow import keras

import config

parser = argparse.ArgumentParser()
parser.add_argument('--text_A', default='../data/final-train/final-train.tok.src', help='source corpus', required=True)
parser.add_argument('--text_B', default='../data/final-train/final-train.tok.trg', help='target corpus', required=True)
parser.add_argument('--cache_dir', default='../data/', help='output directory to save caches', required=True)
args = parser.parse_args()


def tokenize_helper(text):
    text = text.replace('\n', '')
    return '<START> ' + text + ' <END>'


with open(args.text_A) as text_a, open(args.text_B) as text_b:
    tokenizer = Tokenizer(num_words=config.MAX_VOCAB_SIZE, filters='', oov_token='<UNK>')

    all_text_a = []
    all_text_b = []
    for line in text_a:
        all_text_a.append(tokenize_helper(line))
    for line in text_b:
        all_text_b.append(tokenize_helper(line))

    tokenizer.fit_on_texts(all_text_a + all_text_b)
    word_index = tokenizer.word_index

    source = tokenizer.texts_to_sequences(all_text_a)
    source = keras.preprocessing.sequence.pad_sequences(source, maxlen=config.MAX_SEQ_LEN, padding='post')

    target = tokenizer.texts_to_sequences(all_text_b)
    target = keras.preprocessing.sequence.pad_sequences(target, maxlen=config.MAX_SEQ_LEN, padding='post')

    np.save(open(args.cache_dir + config.CACHE_SOURCE, 'wb'), source)
    np.save(open(args.cache_dir + config.CACHE_TARGET, 'wb'), target)
    np.save(open(args.cache_dir + config.CACHE_WORD_INDEX, 'wb'), word_index)

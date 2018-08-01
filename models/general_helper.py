#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

import config


def load_glove_matrix(word_index):
    if os.path.isfile(config.EMBEDDING_CACHE):
        print('---- Load word vectors from cache.')
        embedding_matrix = np.load(open(config.EMBEDDING_CACHE, 'rb'))
        return embedding_matrix

    print('---- loading glove ...')
    embeddings_index = {}
    f = open(config.EMBEDDING_FILE, 'r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    nb_words = min(config.MAX_VOCAB_SIZE, len(word_index)) + 1

    embedding_matrix = np.zeros((nb_words, config.WORD_EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # check the words which not in embedding vectors
    not_found_words = []
    for word, i in word_index.items():
        if word not in embeddings_index:
            not_found_words.append(word)

    np.save(open(config.EMBEDDING_CACHE, 'wb'), embedding_matrix)
    return embedding_matrix

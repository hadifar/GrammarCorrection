#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
from gensim.models import KeyedVectors

import config


def load_glove_matrix(word_index):
    if os.path.isfile(config.EMBEDDING_CACHE):
        print('---- Load word vectors from cache.')
        embedding_matrix = np.load(open(config.EMBEDDING_CACHE, 'rb'))
        return embedding_matrix

    print('---- loading embedding ...')
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


def load_embedding_matrix(word_index):
    if os.path.isfile(config.EMBEDDING_CACHE):
        print('---- Load word vectors from cache.')
        embedding_matrix = np.load(open(config.EMBEDDING_CACHE, 'rb'))
        return embedding_matrix

    print('---- loading embedding ...')
    word2vec = KeyedVectors.load_word2vec_format(config.EMBEDDING_FILE)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))

    nb_words = len(word_index)

    embedding_matrix = np.zeros((nb_words, 300))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    # check the words which not in embedding vectors
    not_found_words = []
    for word, i in word_index.items():
        if word not in word2vec.vocab:
            not_found_words.append(word)

    np.save(open(config.EMBEDDING_CACHE, 'wb'), embedding_matrix)
    return embedding_matrix

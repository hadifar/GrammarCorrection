#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

import nltk
import numpy as np

import config


def tokenise(line, tokenizer_fn=nltk.word_tokenize, startEndTokens=False):
    p = tokenizer_fn(line)
    if startEndTokens:
        p = ['<start>'] + p + ['<end>']
    return p


def getSentencesMat(filePath, vocab, maxSentenceL=None,
                    padding='right', startEndTokens=False,
                    tokenizer_fn=nltk.word_tokenize):
    with open(filePath) as inputfile:

        tokenised = []
        file_size = 0
        for s in inputfile:
            file_size = file_size + 1
            tokenised.append(tokenise(s.rstrip('\n'), startEndTokens=startEndTokens, tokenizer_fn=tokenizer_fn))

        if maxSentenceL is None:
            maxSentenceL = max([len(s) for s in tokenised])

        sentencesMat = np.zeros((file_size, maxSentenceL)).astype('int64')

        for i, sen in enumerate(tokenised):
            ids = [vocab(w) for w in sen]
            if padding == 'right':
                ids = ids[:maxSentenceL]
            else:
                ids = ids[-1 * maxSentenceL:]

            if len(ids) < maxSentenceL:
                if padding == 'right':
                    ids = ids + [0] * (maxSentenceL - len(ids))
                else:
                    ids = [0] * (maxSentenceL - len(ids)) + ids

            sentencesMat[i] = np.array(ids)
        return sentencesMat


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.wordFreqs = {}
        self.idx = 1  # id 0 is reserved for padding
        self.add_word('<unk>')
        self.add_word('<start>')
        self.add_word('<end>')

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            self.wordFreqs[word] = 0
        self.wordFreqs[word] += 1

    def add_words(self, words):
        for w in words:
            self.add_word(w)

    def keepTopK(self, k):
        wordFreqs = self.wordFreqs.items()
        wordFreqs = sorted(wordFreqs, key=lambda x: x[1], reverse=True)
        wordFreqs = wordFreqs[:k]
        words = [w[0] for w in wordFreqs]

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

        for word in ['<unk>', '<start>', '<end>'] + words:
            if not word in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def getDicts(self):
        J = {"idx": self.idx, "idx2word": self.idx2word, "word2idx": self.word2idx, "wordFreqs": self.wordFreqs}
        return J

    def save(self, fname_prefix):
        open(fname_prefix + "word2idx.json", "wb").write(json.dumps(self.word2idx))
        open(fname_prefix + "idx2word.json", "wb").write(json.dumps(self.idx2word))
        J = {"idx": self.idx, "idx2word": self.idx2word, "word2idx": self.word2idx, "wordFreqs": self.wordFreqs}
        open(fname_prefix + "everything.json", "wb").write(json.dumps(J))


def load_glove_matrix(word_index):
    if os.path.isfile(config.EMBEDDING_CACHE):
        print('---- Load word vectors from cache.')
        embedding_matrix = np.load(open(config.EMBEDDING_CACHE, 'rb'))
        return embedding_matrix

    print('---- loading glove ...')
    embeddings_index = {}
    f = open(config.EMBEDDING_FILE,'r')
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

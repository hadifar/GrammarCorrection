# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

ROOT_DIR = '/Users/mac/PycharmProjects/GrammarCorrection/'
DATA_DIR = ROOT_DIR + 'data/'

CORPUS_SOURCE = DATA_DIR + 'final-train.tok.src'
CORPUS_TARGET = DATA_DIR + 'final-train.tok.trg'

CACHE_SOURCE = 'cache_source.npy'
CACHE_TARGET = 'cache_target.npy'
CACHE_WORD_INDEX = 'cache_word_index.npy'

MAX_SEQ_LEN = 40
MAX_VOCAB_SIZE = 60000  # recommended

BATCH_SIZE = 32
EPOCH_NUM = 200

USE_WORD_VECTOR = True
WORD_EMBEDDING_DIM = 300
EMBEDDING_CACHE = DATA_DIR + 'embedding/embedding_matrix.npy'
EMBEDDING_FILE = DATA_DIR + 'embedding/wiki.en.vec'

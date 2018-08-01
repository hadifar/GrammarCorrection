# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

ROOT_DIR = '/Users/mac/PycharmProjects/riminder/'
DATA_DIR = ROOT_DIR + 'data/'

CORPUS_SOURCE = DATA_DIR + 'nucle-final-train.tok.src' # todo: change this one to use whole corpus
CORPUS_TARGET = DATA_DIR + 'nucle-final-train.tok.trg'

CACHE_SOURCE = 'cache_source.npy'
CACHE_TARGET = 'cache_target.npy'
CACHE_WORD_INDEX = 'cache_word_index.npy'

MAX_SEQ_LEN = 40
MAX_VOCAB_SIZE = 10000  # todo: increase this one

BATCH_SIZE = 64
EPOCH_NUM = 10  # todo: increase this one

WORD_EMBEDDING_DIM = 300
EMBEDDING_CACHE = DATA_DIR + '/embedding/embedding_matrix.npy'
EMBEDDING_FILE = '/Users/mac/PycharmProjects/ensemble/embedding/fasttext.300d.txt'

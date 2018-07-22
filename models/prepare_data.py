#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import h5py

import config
from utils import getSentencesMat, Vocabulary

parser = argparse.ArgumentParser()
parser.add_argument('--text_A', type=str, help='source corpus')
parser.add_argument('--text_B', type=str, help='target corpus')
parser.add_argument('--out_file', type=str, default="./data/trg_src_prepped.h5", help='Output HDF5 file name')
args = parser.parse_args()

target_vocab = Vocabulary()
source_vocab = Vocabulary()

for src in open(args.text_A):
    source_vocab.add_words(src.rstrip('\n').split(' '))

for trg in open(args.text_B):
    target_vocab.add_words(trg.rstrip('\n').split(' '))

target_vocab.keepTopK(config.MAX_KEEP_WORD)
source_vocab.keepTopK(config.MAX_KEEP_WORD)

source_sent_mat = getSentencesMat(args.text_A, target_vocab,
                                  startEndTokens=True,
                                  tokenizer_fn=lambda x: x.split(' '),
                                  maxSentenceL=config.MAX_SENT_LEN)

target_sent_mat = getSentencesMat(args.text_B, source_vocab,
                                  startEndTokens=True,
                                  tokenizer_fn=lambda x: x.split(' '),
                                  maxSentenceL=config.MAX_SENT_LEN)

f = h5py.File(args.out_file, "w")

f.create_dataset("source_vocab", data=json.dumps(source_vocab.getDicts()))
f.create_dataset("target_vocab", data=json.dumps(target_vocab.getDicts()))

f.create_dataset('target_sent_mat', data=target_sent_mat)
f.create_dataset('source_sent_mat', data=source_sent_mat)

f.close()

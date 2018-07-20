#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import h5py
import config

from utils import getSentencesMat, Vocabulary

parser = argparse.ArgumentParser()
parser.add_argument('--text_A', type=str, help='corpus with typo (target)')
parser.add_argument('--text_B', type=str, help='corpus without typo (source)')
parser.add_argument('--out_file', type=str, default="./data/src_trg_prepared.h5", help='Output HDF5 file name')
args = parser.parse_args()

target_typo_sents = (open(args.text_A).read()).split("\n")[:-1]
source_wo_sents = (open(args.text_B).read()).split("\n")[:-1]

target_vocab = Vocabulary()
source_vocab = Vocabulary()

for trg in target_typo_sents:
    target_vocab.add_words(trg.split(' '))

for src in source_wo_sents:
    source_vocab.add_words(src.split(' '))

target_vocab.keepTopK(config.MAX_KEEP_WORD)
source_vocab.keepTopK(config.MAX_KEEP_WORD)

target_sent_mat = getSentencesMat(target_typo_sents, target_vocab,
                                  startEndTokens=True,
                                  tokenizer_fn=lambda x: x.split(' '),
                                  maxSentenceL=config.MAX_SENT_LEN)

source_sent_mat = getSentencesMat(source_wo_sents, source_vocab,
                                  startEndTokens=True,
                                  tokenizer_fn=lambda x: x.split(' '),
                                  maxSentenceL=config.MAX_SENT_LEN)

f = h5py.File(args.out_file, "w")

f.create_dataset("source_vocab", data=json.dumps(source_vocab.getDicts()))
f.create_dataset("target_vocab", data=json.dumps(target_vocab.getDicts()))

f.create_dataset("target_sents", data=json.dumps(target_typo_sents))
f.create_dataset("source_sents", data=json.dumps(source_wo_sents))

f.create_dataset('target_sent_mat', data=target_sent_mat)
f.create_dataset('source_sent_mat', data=source_sent_mat)

f.close()

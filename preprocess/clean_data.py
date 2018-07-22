#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import string
import argparse

import nltk

parser = argparse.ArgumentParser()
parser.add_argument('--input_src', type=str,
                    help='input clean source')
parser.add_argument('--input_trg', type=str,
                    help='input clean target')
parser.add_argument('--output_src', type=str,
                    help='input clean source')
parser.add_argument('--output_trg', type=str,
                    help='input clean target')

args = parser.parse_args()

input_path_src = args.input_src
input_path_trg = args.input_trg
output_path_src = args.output_src
output_path_trg = args.output_trg

DELETE_PUNCTUATION = set(string.punctuation) - {'.', ',', '-'}


# https://stackoverflow.com/a/30212799/1462770
def fullmatch(regex, string, flags=0):
    """Emulate python-3.4 re.fullmatch()."""
    return re.match("(?:" + regex + r")\Z", string, flags=flags)


def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1", text)


with open(input_path_src, 'r') as srcfile, \
        open(input_path_trg, 'r') as trgfile, \
        open(output_path_src, 'w') as src_outfile, \
        open(output_path_trg, 'w') as trg_outfile:
    all_src = srcfile.readlines()
    all_trg = trgfile.readlines()
    assert len(all_src) == len(all_trg)

    for src_line, trg_line in zip(all_src, all_trg):

        # remove non ascii characters
        src_line = re.sub(r'[^\x00-\x7f]', r' ', src_line)
        trg_line = re.sub(r'[^\x00-\x7f]', r' ', trg_line)

        # remove 3 or more successive characters with 1 character: eg. aaa -> a
        src_line = reduce_lengthening(src_line)
        trg_line = reduce_lengthening(trg_line)

        src_line = nltk.word_tokenize(src_line)
        trg_line = nltk.word_tokenize(trg_line)

        # remove punctuation
        src_line = [x for x in src_line if x not in DELETE_PUNCTUATION]
        trg_line = [x for x in trg_line if x not in DELETE_PUNCTUATION]

        src_line = " ".join(src_line)
        trg_line = " ".join(trg_line)

        if len(src_line.split()) > 1 and len(trg_line.split()) > 1:  # sentence with length of 2 or more
            src_outfile.write(" ".join(src_line.split()) + "\n")
            trg_outfile.write(" ".join(trg_line.split()) + '\n')

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import string
import sys

import nltk

if len(sys.argv) != 5:
    print "[USAGE] %s input_src input_trg output_src output_trg" % sys.argv[0]
    sys.exit()

input_path_src = sys.argv[1]
input_path_trg = sys.argv[2]
output_path_src = sys.argv[3]
output_path_trg = sys.argv[4]


# https://stackoverflow.com/a/30212799/1462770
def fullmatch(regex, string, flags=0):
    """Emulate python-3.4 re.fullmatch()."""
    return re.match("(?:" + regex + r")\Z", string, flags=flags)


with open(input_path_src, 'r') as srcfile, \
        open(input_path_trg, 'r') as trgfile, \
        open(output_path_src, 'w') as src_outfile, \
        open(output_path_trg, 'w') as trg_outfile:
    all_src = srcfile.readlines()
    all_trg = trgfile.readlines()
    assert len(all_src) == len(all_trg)

    for src_line, trg_line in zip(all_src, all_trg):

        src_line = re.sub(r'[^\x00-\x7f]', r' ', src_line)  # remove non ascii characters
        trg_line = re.sub(r'[^\x00-\x7f]', r' ', trg_line)

        src_line = nltk.word_tokenize(src_line)
        trg_line = nltk.word_tokenize(trg_line)

        src_line = [x for x in src_line if not fullmatch('[' + string.punctuation + ']+', x)]
        trg_line = [x for x in trg_line if not fullmatch('[' + string.punctuation + ']+', x)]

        src_line = " ".join(src_line)
        trg_line = " ".join(trg_line)

        if len(src_line.split()) > 1 and len(trg_line.split()) > 1:  # sentence with length of 2 or more
            src_outfile.write(" ".join(src_line.split()) + "\n")
            trg_outfile.write(" ".join(trg_line.split()) + '\n')

# with open('/Users/mac/PycharmProjects/riminder/data/concat-train/concat-train.tok.src', 'r') as srcfile, \
#     open('/Users/mac/PycharmProjects/riminder/data/concat-train/concat-train.tok.trg', 'r') as trgfile, \
#     open('/Users/mac/PycharmProjects/riminder/data/clean-train/clean-train.tok.src', 'w') as src_outfile, \
#     open('/Users/mac/PycharmProjects/riminder/data/clean-train/clean-train.tok.trg', 'w') as trg_outfile:

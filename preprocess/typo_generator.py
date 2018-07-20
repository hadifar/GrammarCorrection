# -*- coding: utf-8 -*-
# !/usr/bin/env python

from __future__ import unicode_literals

import random
import sys

import nltk
from pattern.en import pluralize, singularize, lexeme

if len(sys.argv) != 5:
    print "[USAGE] %s input_src input_trg output_src output_trg" % sys.argv[0]
    sys.exit()

input_path_src = sys.argv[1]
input_path_trg = sys.argv[2]
output_path_src = sys.argv[3]
output_path_trg = sys.argv[4]

DROPOUT_TOKENS = {"a", "an", "the", "'ll", "'s", "'m", "'ve"}

REPLACEMENTS = {"there": "their", "their": "there", "then": "than",
                "than": "then", 'on': 'in', 'in': 'on', 'the': 'a', 'a': 'the',
                'might': 'would', 'would': 'might',
                'could': 'might', 'can': 'may', 'may': 'can',
                'inside': 'in', 'besides': 'beside', 'beside': 'besides',
                'towards': 'toward', 'toward': 'towards', 'till': 'until', 'until': 'till',
                'to': 'through', 'through': 'to'
                }

MODAL = {'can': 'could', 'could': 'can',
         'may': 'might', 'might': 'may',
         'will': 'would', 'would': 'will'}

VERBS_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

NOUN_TAGS = ['NN', 'NNP', 'NNPS', 'NNS']


def change_pluralization(token):
    singularForm = singularize(token)
    pluralForm = pluralize(token)
    if token == singularForm:
        return pluralForm
    else:
        return singularForm


def change_tense(token):
    return random.choice(lexeme(token))


def pair_generator(x):
    target = []

    # x = x.lower() # this cause some error ignorance (Mec)
    x_split = nltk.word_tokenize(x)
    x_pos = nltk.pos_tag(x_split)

    # avoid too much error creation
    replace_flag = False
    plural_flag = False
    tense_flag = False
    modal_flag = False

    for token, pos in x_pos:

        dropout_token = (token in DROPOUT_TOKENS and
                         random.random() < 0.25)

        replace_token = (token in REPLACEMENTS and
                         random.random() < 0.25 and
                         not replace_flag)

        pos_plural_token = (pos in NOUN_TAGS and
                            random.random() < 0.25 and
                            not plural_flag)

        pos_tense_token = (pos in VERBS_TAGS and
                           random.random() < 0.25 and
                           not tense_flag)

        pos_modal_token = (pos in MODAL and
                           random.random() < 0.25 and
                           not modal_flag)

        if replace_token:
            target.append(REPLACEMENTS[token])
            replace_flag = True
        elif pos_plural_token:
            token = change_pluralization(token)
            target.append(token)
            plural_flag = True
        elif pos_tense_token:
            token = change_tense(token)
            target.append(token)
            tense_flag = True
        elif not dropout_token:
            target.append(token)
        elif pos_modal_token:
            target.append(MODAL[token])
            modal_flag = True

    return " ".join(x_split), " ".join(target)


with open(input_path_src, 'r') as srcfile, open(input_path_trg, 'r') as trgfile, \
        open(output_path_src, 'w') as src_outfile, open(output_path_trg, 'w') as trg_outfile:
    all_src = srcfile.readlines()
    all_trg = trgfile.readlines()

    assert len(all_src) == len(all_trg)

    for src_line, trg_line in zip(all_src, all_trg):

        src_line_gen, trg_line_gen = pair_generator(src_line)
        src_outfile.write(src_line_gen + '\n')
        trg_outfile.write(trg_line_gen + '\n')

        if src_line != trg_line:  # if already have error, keep original error as well
            src_outfile.write(src_line)
            trg_outfile.write(trg_line)

# with open('/Users/mac/PycharmProjects/riminder/data/clean-train/clean-train.tok.src', 'r') as srcfile, \
#         open('/Users/mac/PycharmProjects/riminder/data/clean-train/clean-train.tok.trg', 'r') as trgfile, \
#         open('/Users/mac/PycharmProjects/riminder/data/final-train/final-train.tok.src', 'w') as src_outfile, \
#         open('/Users/mac/PycharmProjects/riminder/data/final-train/final-train.tok.trg', 'w') as trg_outfile:

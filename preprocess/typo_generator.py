#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import argparse
import random

import nltk
from pattern.en import pluralize, singularize, lexeme

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

# todo: replace most common misspelling
# todo: try to do it in automatic way ! for now I just add some samples
# https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings
# https://en.wikipedia.org/wiki/Commonly_misspelled_English_words#A%E2%80%93B
MISSPELLING_TOKEN = {'absence': 'absense', 'acceptable': 'acceptible',
                     'accidentally': 'accidentaly', 'achieve': 'acheive',
                     'acknowledge': 'acknowlege', 'acquire': 'aquire', 'affect': 'effect',
                     'aggression': 'agression', 'almost': 'allmost', 'awful': 'awfull',
                     'because': 'becuase', 'believe': 'beleive', 'business': 'buisness'}

DROPOUT_TOKENS = {"a", "an", "the", "'ll", "'s", "'m", "'ve"}

REPLACEMENTS = {"there": "their", "their": "there", "then": "than",
                "than": "then", 'on': 'in', 'in': 'on', 'the': 'a', 'a': 'the',
                'might': 'would', 'would': 'might',
                'could': 'might', 'can': 'may', 'may': 'can',
                'inside': 'in', 'besides': 'beside', 'beside': 'besides',
                'towards': 'toward', 'toward': 'towards', 'till': 'until', 'until': 'till',
                'to': 'through', 'through': 'to'}

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


def pair_generator(target_sentece):
    generated_source = []

    # x = x.lower() # this cause some error ignorance (Mec)
    x_split = nltk.word_tokenize(target_sentece)
    x_pos = nltk.pos_tag(x_split)

    # avoid too much error creation
    replace_flag = False
    misspell_flag = False
    plural_flag = False
    tense_flag = False
    modal_flag = False

    for token, pos in x_pos:

        dropout_token = (token in DROPOUT_TOKENS and
                         random.random() < 0.25)

        misspell_token = (token in MISSPELLING_TOKEN and
                          random.random() < 0.25 and
                          not misspell_flag)

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
            generated_source.append(REPLACEMENTS[token])
            replace_flag = True
        elif misspell_token:
            generated_source.append(MISSPELLING_TOKEN[token])
            misspell_flag = True
        elif pos_plural_token:
            token = change_pluralization(token)
            generated_source.append(token)
            plural_flag = True
        elif pos_tense_token:
            token = change_tense(token)
            generated_source.append(token)
            tense_flag = True
        elif not dropout_token:
            generated_source.append(token)
        elif pos_modal_token:
            generated_source.append(MODAL[token])
            modal_flag = True

    return " ".join(generated_source), " ".join(x_split)


with open(input_path_src, 'r') as srcfile, open(input_path_trg, 'r') as trgfile, \
        open(output_path_src, 'w') as src_outfile, open(output_path_trg, 'w') as trg_outfile:
    all_src = srcfile.readlines()
    all_trg = trgfile.readlines()

    assert len(all_src) == len(all_trg)

    for src_line, trg_line in zip(all_src, all_trg):

        src_line_gen, trg_line_gen = pair_generator(trg_line)
        src_outfile.write(src_line_gen + '\n')
        trg_outfile.write(trg_line_gen + '\n')

        if src_line != trg_line:  # if already have error, keep original error too
            src_outfile.write(src_line)
            trg_outfile.write(trg_line)

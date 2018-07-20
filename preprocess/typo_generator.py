# -*- coding: utf-8 -*-
#
# Copyright 2018 Amir Hadifar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import random
import sys

import nltk
from pattern.en import pluralize, singularize, lexeme

if len(sys.argv) != 4:
    print "[USAGE] %s prepfile output_src output_tgt" % sys.argv[0]
    sys.exit()

input_path = sys.argv[1]
output_src_path = sys.argv[2]
output_tgt_path = sys.argv[3]


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

    for token, pos in x_pos:

        dropout_token = (token in DROPOUT_TOKENS and
                         random.random() < 0.25)

        replace_token = (token in REPLACEMENTS and
                         random.random() < 0.25)

        pos_plural_token = (pos in NOUN_TAGS and
                            random.random() < 0.25)

        pos_tense_token = (pos in VERBS_TAGS and
                           random.random() < 0.25)

        pos_modal_token = (pos in MODAL and
                           random.random() < 0.25)

        if replace_token:
            target.append(REPLACEMENTS[token])
        elif pos_plural_token:
            token = change_pluralization(token)
            target.append(token)
        elif pos_tense_token:
            token = change_tense(token)
            target.append(token)
        elif not dropout_token:
            target.append(token)
        elif pos_modal_token:
            target.append(MODAL[token])

    return " ".join(x_split), " ".join(target)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True,
                    help='input clean source')
parser.add_argument('--output_src', required=True,
                    help='output clean source')
parser.add_argument('--output_trg', required=True,
                    help='output clean target')

args = parser.parse_args()

input_path = args.input_path
output_src_path = args.output_src
output_trg_path = args.output_trg

words = []
corrected = []
sid = eid = 0
prev_sid = prev_eid = -1
pos = 0

with open(input_path) as input_file, \
        open(output_src_path, 'w') as output_src_file, \
        open(output_trg_path, 'w') as output_tgt_file:
    for line in input_file:
        line = line.strip()
        if line.startswith('S'):
            line = line[2:]
            words = line.split()
            corrected = ['<S>'] + words[:]
            output_src_file.write(line + '\n')
        elif line.startswith('A'):
            line = line[2:]
            info = line.split("|||")
            sid, eid = info[0].split()
            sid = int(sid) + 1
            eid = int(eid) + 1
            error_type = info[1]
            if error_type == "Um":
                continue
            for idx in range(sid, eid):
                corrected[idx] = ""
            if sid == eid:
                if sid == 0:
                    continue  # Originally index was -1, indicating no op
                if sid != prev_sid or eid != prev_eid:
                    pos = len(corrected[sid - 1].split())
                cur_words = corrected[sid - 1].split()
                cur_words.insert(pos, info[2])
                pos += len(info[2].split())
                corrected[sid - 1] = " ".join(cur_words)
            else:
                corrected[sid] = info[2]
                pos = 0
            prev_sid = sid
            prev_eid = eid
        else:
            target_sentence = ' '.join([word for word in corrected if word != ""])
            assert target_sentence.startswith('<S>'), '(' + target_sentence + ')'
            target_sentence = target_sentence[4:]
            output_tgt_file.write(target_sentence + '\n')
            prev_sid = -1
            prev_eid = -1
            pos = 0

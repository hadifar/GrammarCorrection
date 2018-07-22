#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str,
                    help='input clean source')
parser.add_argument('--output_src', type=str,
                    help='input clean source')
parser.add_argument('--output_trg', type=str,
                    help='input clean target')

args = parser.parse_args()

input_path = args.input_src
output_src_path = args.output_src
output_trg_path = args.output_trg

with open(input_path, 'r') as inputfile, \
        open(output_src_path, 'w') as outx, \
        open(output_trg_path, 'w') as outy:
    all_lines = inputfile.readlines()

    for line in all_lines:
        line = line.split('\t')
        if len(line) == 6:
            x, y = line[4], line[5]
            outx.write(x.replace('\n', '') + '\n')
            outy.write(y.replace('\n', '') + '\n')
        if len(line) == 7:
            x, y = line[4], line[6]
            outx.write(x.replace('\n', '') + '\n')
            outy.write(y.replace('\n', '') + '\n')
        if len(line) == 8:
            x, y = line[4], line[7]
            outx.write(x.replace('\n', '') + '\n')
            outy.write(y.replace('\n', '') + '\n')

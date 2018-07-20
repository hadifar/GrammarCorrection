#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

if len(sys.argv) != 4:
    print "[USAGE] %s lang8_file output_src output_tgt" % sys.argv[0]
    sys.exit()

input_path = sys.argv[1]
output_src_path = sys.argv[2]
output_tgt_path = sys.argv[3]

with open(input_path, 'r') as inputfile, \
        open(output_src_path, 'w') as outx, \
        open(output_tgt_path, 'w') as outy:

    all_lines = inputfile.readlines()

    for line in all_lines:
        line = line.split('\t')
        if len(line) == 5:
            x, y = line[4], line[4]
        elif len(line) > 5:
            x, y = line[4], line[5]
        else:
            continue

        outx.write(x.replace('\n', '') + '\n')
        outy.write(y.replace('\n', '') + '\n')

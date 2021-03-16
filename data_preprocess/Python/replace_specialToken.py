#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Zhichao Ouyang
# Time: 2021/3/11 16:19

import os
import re

def rules(line):
    # replace number by _num
    line = re.sub("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", "_num", line)

    # replace boolean by _bool
    line = line.replace('False', '_bool').replace('True', '_bool')

    # replace string by _str
    line = re.sub(r"'(.+?)'", "_str", line)
    line = re.sub(r'"(.+?)"', "_str", line)
    return line

def replace_special_token(origin_file, final_file):
    source_file = open(origin_file, 'r', encoding='utf-8')
    source_clean_file = open(final_file, 'w', encoding='utf-8')
    source_lines = source_file.readlines()
    for line in source_lines:
        line = rules(line)
        source_clean_file.write(line)

    source_file.close()
    source_clean_file.close()
    print('success')

replace_special_token('E:\\chao\\codeSum\\ASE2020\\transformer-ast\\python_datautils\\python_datautils\\data\\test\\final\\clean_test.token.source', 'E:\\chao\\codeSum\\ASE2020\\transformer-ast\\python_datautils\\python_datautils\\data\\test\\final\\special_test.token.source')
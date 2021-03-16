#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Zhichao Ouyang
# Time: 2021/3/11 16:19

"""
    get python format code
    delete Superfluous notes
"""
import json
import os

i = 0
with open('data/train.jsonl', 'r') as f:
    for jsonstr in f.readlines():
        data = json.loads(jsonstr)
        # lines = source.readline()
        code_strs = data['code']
        if code_strs.find('\"\"\"') >= 0:
            start = code_strs.index('\"\"\"')
            end = code_strs.index('\"\"\"', start + 3)
            # print('start:', start)
            # print('end:', end)
            tmp = code_strs
            idx = 0
            while tmp.find(":", idx) >= 0 and tmp.index(":", idx) < start:
                idx += 1
            code_strs = code_strs.replace(code_strs[idx:end + 3], '')
        elif code_strs.find("'''") >= 0:
            start = code_strs.index("'''")
            end = code_strs.index("'''", start + 3)
            # print('start:', start)
            # print('end:', end)
            tmp = code_strs
            idx = 0
            while tmp.find(":", idx) >= 0 and tmp.index(":", idx) < start:
                idx += 1
            code_strs = code_strs.replace(code_strs[idx:end + 3], '')
        code_strs.encode('utf-8').decode("utf-8")
        if i % 20000 == 0:
            path = 'E:\\chao\\codeSum\\ASE2020\\transformer-ast\\python_datautils\\python_datautils\\data\\train_' + str(int(i//20000)) + '\\'
            if not os.path.exists(path):
                os.makedirs(path)
                print(path)
        save2 = open(path + str(i) + '.py', 'w', encoding='utf-8')
        save2.write(code_strs)
        i += 1
        save2.close()
print('success')

print(i)

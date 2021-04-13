#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Zhichao Ouyang
# Time: 2021/3/11 16:19

import numpy as np

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
UNK_WORD = '<unk>'
MAX_LEN_CODE = 100
MAX_LEN_NL = 30


def build_vocab(train_path, vocab_path):
    train_code = open(train_path + '.code', encoding='utf-8')
    train_nl = open(train_path + '.nl', encoding='utf-8')
    code_word_count, nl_word_count = {}, {}
    for code_line in train_code:
        code_line = code_line.replace('\n', '').strip()
        words = code_line.split(' ')
        for word in words:
            if word.strip() is not '':
                if word not in code_word_count:
                    code_word_count[word] = 0
                code_word_count[word] += 1
    for nl_line in train_nl:
        nl_line = nl_line.replace('\n', '').strip()
        words = nl_line.split(' ')
        for word in words:
            if word.strip() is not '':
                if word not in nl_word_count:
                    nl_word_count[word] = 0
                nl_word_count[word] += 1

    code_word_count = list(code_word_count.items())
    code_word_count.sort(key=lambda k: k[1], reverse=True)
    write = open(vocab_path + '.code', 'w', encoding='utf-8')
    i = 0
    for word_pair in code_word_count:
        write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
        i += 1
        if i > 50000 or i == 50000:
            break
    write.close()
    nl_word_count = list(nl_word_count.items())
    nl_word_count.sort(key=lambda k: k[1], reverse=True)
    write = open(vocab_path + '.nl', 'w', encoding='utf-8')
    i = 0
    for word_pair in nl_word_count:
        write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
        i += 1
        if i > 50000 or i == 50000:
            break
    write.close()


def fix_length(nl, code):
    nl = np.concatenate(([BOS_WORD], nl))
    if len(nl) >= MAX_LEN_NL:
        nl = np.concatenate((nl[0:MAX_LEN_NL], [EOS_WORD]))
    else:
        nl = np.concatenate((nl, [EOS_WORD], [BLANK_WORD for _ in range(MAX_LEN_NL - len(nl))]))
    if len(code) > MAX_LEN_CODE:
        news = code[0:MAX_LEN_CODE]
    else:
        news = np.concatenate((code, [BLANK_WORD for _ in range(MAX_LEN_CODE - len(code))]))
    return nl, news


def get_w2i(dataName = 'Java'):
    code_vocab_file = open('code_sum_dataset/' + dataName + '/vocab.code', encoding='utf-8')
    nl_vocab_file = open('code_sum_dataset/' + dataName + '/vocab.nl', encoding='utf-8')
    code_w2i = {BLANK_WORD: 0, BOS_WORD: 1, EOS_WORD: 2, UNK_WORD: 3}
    nl_w2i = {BLANK_WORD: 0, BOS_WORD: 1, EOS_WORD: 2, UNK_WORD: 3}
    i = 4
    for v in code_vocab_file:
        v = v.split('\t')[0]
        code_w2i[v] = i
        i += 1
    i = 4
    for v in nl_vocab_file:
        v = v.split('\t')[0]
        nl_w2i[v] = i
        i += 1
    return code_w2i, nl_w2i


def word2idx(c, n, code_w2i, nl_w2i):
    code, nl = [], []
    for w in c:
        code.append(code_w2i.get(w, 3))
    for w in n:
        nl.append(nl_w2i.get(w, 3))
    return code, nl


def save_as_array(code_path, nl_path, ast_path, dataName = 'Java'):
    dataset = []
    code_lines = open(code_path, encoding='utf-8').readlines()
    nl_lines = open(nl_path, encoding='utf-8').readlines()
    ast_lines = open(ast_path, encoding='utf-8').readlines()
    code_w2i, nl_w2i = get_w2i(dataName)
    for i in range(len(code_lines)):
        c_words, n_words = code_lines[i].replace('\n', '').split(' '), nl_lines[i].replace('\n', '').split(' ')
        ast_vec = ast_lines[i].split(' ')
        ast_vec = [float(f_num) for f_num in ast_vec]
        n_words, c_words = fix_length(n_words, c_words)
        code, nl = word2idx(c_words, n_words, code_w2i, nl_w2i)
        dataset.append([np.array(code), np.array(ast_vec), np.array(nl)])
    file_name = code_path.split('/')[2].split('.')[0]
    np.save('code_sum_dataset/' + dataName + '/' + file_name + '.npy', np.array(dataset))
    print(file_name + ' dataset saved as numpy array!')


if __name__ == '__main__':
    dataName = ['Java', 'Python']
    for name in dataName:
        build_vocab(train_path='code_sum_dataset/' + name + '/train.token', vocab_path='code_sum_dataset/' + name + '/vocab')
        print(name + ' vocab success!!')
    type_list = ['test', 'valid', 'train']
    for name in dataName:
        for type in type_list:
            save_as_array('code_sum_dataset/' + name + '/' + type + '.token.code', 'code_sum_dataset/' + name + '/' + type + '.token.nl',
                        'code_sum_dataset/' + name + '/' + type + '.token.ast', dataName=name)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Zhichao Ouyang
# Time: 2021/3/11 16:19

import numpy as np
import math
import nltk
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.batch import Batch
from model.transformer_BASTS import make_model
from utils.label_smothing import LabelSmoothing
from utils.optimizer import NoamOpt
from utils.loss_compute import MultiGPULossCompute
from utils.train import run_epoch_, greedy_decode
import time
import random



def get_idx2word(dataName = 'Java'):
    code_vocab_file = open('code_sum_dataset/' + dataName + '/vocab.code', encoding='utf-8')
    code_vocab_dict = {0: BLANK_WORD, 1: BOS_WORD, 2: EOS_WORD, 3: UNK_WORD}
    i = 4
    for v in code_vocab_file:
        v = v.split('\t')[0]
        code_vocab_dict[i] = v
        i += 1
    nl_vocab_file = open('code_sum_dataset/' + dataName + '/vocab.nl', encoding='utf-8')
    nl_vocab_dict = {0: BLANK_WORD, 1: BOS_WORD, 2: EOS_WORD, 3: UNK_WORD}
    i = 4
    for v in nl_vocab_file:
        v = v.split('\t')[0]
        nl_vocab_dict[i] = v
        i += 1
    return code_vocab_dict, nl_vocab_dict


def load_data(set_type, dataName = 'Java'):
    if set_type == 'train' or set_type == 'valid':
        batch_size = BATCH_SIZE
    else:
        batch_size = 1
    train_set = np.load('code_sum_dataset/' + dataName + '/' + set_type + '.npy', allow_pickle=True)
    batches = []
    num_batch = math.ceil(len(train_set) / batch_size)  # 上取整
    for i in range(num_batch):
        b = train_set[batch_size*i:batch_size*(i+1)]
        batch_code = torch.LongTensor(np.stack(b[:, 0]))
        batch_ast = torch.FloatTensor(np.stack(b[:, 1]))
        batch_nl = torch.LongTensor(np.stack(b[:, 2]))
        batch = Batch(batch_code, batch_ast, batch_nl)
        batches.append(batch)
    return batches


def run(dataName = 'Java'):
    is_training = False
    test_batches = load_data('test')

    # multi gpu
    pad_idx = 0
    multi_gpu = False
    # devices = [0, 1, 2, 3]
    devices = [0]
    # if multi_gpu:
    #     model = make_model(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, N=6)
    #     model = model.cuda()
    #     criterion = LabelSmoothing(size=TRG_VOCAB_SIZE, padding_idx=pad_idx, smoothing=0.1)
    #     criterion = criterion.cuda()
    #     model_par = nn.DataParallel(model, device_ids=devices)
    #     model_par = model_par.cuda()
    # else:
    print('single gpu running!!!')
    model = make_model(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, N=6)
    model = model.cuda()
    criterion = LabelSmoothing(size=TRG_VOCAB_SIZE, padding_idx=pad_idx, smoothing=0.1)
    criterion = criterion.cuda()
    # model_par = nn.DataParallel(model, device_ids=devices)
    # model_par = model_par.cuda()

    # translate, output epoch1-100 results
    for epoch in range(0, 1, 5):
        if epoch == 0:
            model = torch.load("./BASTS_model_" + dataName + "/code_comment_ast.epoch" + str(epoch+1) + ".pt")
        else:
            model = torch.load("./BASTS_model_" + dataName + "/code_comment_ast.epoch" + str(epoch) + ".pt")

        if not is_training:
            print('begin test:' + str(epoch))
            _, nl_i2w = get_idx2word()
            output_file = open('./BASTS_model_' + dataName + '/BASTS_output_epoch' + str(epoch) + '.txt', 'w', encoding='utf-8')
            iter_num = 1
            for batch in tqdm(test_batches):
                if batch.src.size(0) == 0:
                    continue
                hypothesis, reference = [], []
                src = batch.src[:1]
                src = src.cuda()
                ast = batch.ast[:1]
                ast = ast.cuda()
                src_mask = (src != pad_idx)
                src_mask = src_mask.cuda()
                out = greedy_decode(model, src, ast, src_mask,
                                    max_len=MAX_LEN_NL, start_symbol=1)
                for i in range(1, out.size(1)):
                    sym = nl_i2w[out[0, i].item()]
                    if sym == "</s>": break
                    hypothesis.append(sym)
                    output_file.write(sym + ' ')
                output_file.write('\n')
                iter_num += 1
            output_file.close()
            print('end test:' + str(epoch))
            print('save files...')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = '<blank>'
    UNK_WORD = '<unk>'
    MAX_LEN_CODE = 100
    MAX_LEN_NL = 30
    BATCH_SIZE = 1  # the batch size of translate step is 1

    # Java dataset
    # SRC_VOCAB_SIZE = 44601   # the size of vocab.code, you can see the file
    # TRG_VOCAB_SIZE = 50004   # the size of vocab.nl, you can see the file
    # dataName = 'Java'  # Default dataset is java

    # Python
    SRC_VOCAB_SIZE = 50004   # the size of vocab.code, you can see the file
    TRG_VOCAB_SIZE = 50004   # the size of vocab.nl, you can see the file
    dataName = 'Python'
    run(dataName)

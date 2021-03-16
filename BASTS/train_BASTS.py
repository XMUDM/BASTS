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


def load_data(set_type, dataName = 'Java'):
    if set_type == 'train' or set_type == 'valid':
        batch_size = BATCH_SIZE
    else:
        batch_size = 1
    train_set = np.load('code_sum_dataset/' + dataName + '/' + set_type + '.npy', allow_pickle=True)
    batches = []
    num_batch = math.ceil(len(train_set) / batch_size)
    for i in range(num_batch-2):   # Round down
        b = train_set[batch_size*i:batch_size*(i+1)]
        batch_code = torch.LongTensor(np.stack(b[:, 0]))
        batch_ast = torch.FloatTensor(np.stack(b[:, 1]))
        batch_nl = torch.LongTensor(np.stack(b[:, 2]))
        batch = Batch(batch_code, batch_ast, batch_nl)
        batches.append(batch)
    return batches


def run(dataName = 'Java'):
    is_training = True
    if is_training:
        train_batches = load_data('train', dataName)
        val_batches = load_data('valid', dataName)

    # multi gpu, But sometimes we encounter bugs, so we use single gpu
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

    # train
    if is_training:
        if not os.path.isdir("./BASTS_model_"+dataName):
            os.mkdir("./BASTS_model_" + dataName)
        if not os.path.isdir("./BASTS_model_" + dataName + "/checkpoint"):
            os.mkdir("./BASTS_model_" + dataName + "/checkpoint")
        optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        start_epoch = -1   # if resume is false, start_epoch is -1
        RESUME = False
        if RESUME:
            path_checkpoint = "./BASTS_model_" + dataName + "/checkpoint/ckpt_best.pth"  # checkpoint path
            checkpoint = torch.load(path_checkpoint)  # Load checkpoint

            model.load_state_dict(checkpoint['net'])  # Load learned parameters

            optimizer.load_state_dict(checkpoint['optimizer'])  # Load optimizer parameters
            start_epoch = checkpoint['epoch']  # set the beginning of epoch
            print('resume success!!' + "ckpt_best.pth")
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                            optimizer)
        for epoch in range(start_epoch + 1, 101):
            print('begin ' + str(epoch + 1) + " training")
            start = time.time()
            model.train()
            run_epoch_(train_batches, model,
                       MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
            model.eval()
            loss = run_epoch_([val_batches[0]], model,
                              MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))

            print('epoch ', str(epoch+1), '   evaluate loss:', loss)
            torch.save(model, 'BASTS_model_' + dataName + '/code_comment_ast' + '.epoch' + str(epoch + 1) + '.pt')
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }

            torch.save(checkpoint, './BASTS_model_' + dataName + '/checkpoint/ckpt_best.pth')

def set_seed(SEED=2020):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'

    set_seed()
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = '<blank>'
    UNK_WORD = '<unk>'
    MAX_LEN_CODE = 100
    MAX_LEN_NL = 30
    BATCH_SIZE = 128

    # Java dataset
    # SRC_VOCAB_SIZE = 44601   # the size of vocab.code, you can see the file
    # TRG_VOCAB_SIZE = 50004   # the size of vocab.nl, you can see the file
    # dataName = 'Java'  # Default dataset is java

    # Python
    SRC_VOCAB_SIZE = 50004   # the size of vocab.code, you can see the file
    TRG_VOCAB_SIZE = 50004   # the size of vocab.nl, you can see the file
    dataName = 'Python'
    run(dataName)

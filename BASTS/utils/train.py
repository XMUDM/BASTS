#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Zhichao Ouyang
# Time: 2021/3/11 16:19

import torch
import time
from torch.autograd import Variable
from model.layers import subsequent_mask


def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        if batch is None:
            continue
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        # if i % 50 == 1:
        if i % 500 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / float(batch.ntokens), float(tokens) / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def run_epoch_(train_batches, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    i = 0
    for batch in train_batches:
        i += 1
        if batch is None:
            continue
        out = model.forward(batch.src.cuda(), batch.ast.cuda(), batch.trg.cuda(),
                            batch.src_mask.cuda(), batch.trg_mask.cuda())
        loss = loss_compute(out, batch.trg_y.cuda(), batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / float(batch.ntokens), float(tokens) / elapsed))
            start = time.time()
            tokens = 0
        # torch.cuda.empty_cache()
    return total_loss / total_tokens.item()


def greedy_decode(model, src, ast, src_mask, max_len, start_symbol):
    memory = model.encode(src, ast, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Zhichao Ouyang
# Time: 2021/3/11 16:19

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from model.attention import MultiHeadedAttention
from model.layers import PositionalEncoding, PositionwiseFeedForward, Embeddings
from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer
import seaborn
seaborn.set_context(context="talk")


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.ast_fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        # self.ast_fc2 = nn.Sequential(nn.Linear(1, 512), nn.Tanh())
        self.generator = generator

    def forward(self, src, ast, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, ast, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, ast, src_mask):
        # emb = self.src_embed(src) + self.ast_fc2(self.ast_fc1(ast).unsqueeze(-1))   # emb 是 （80， 100， 512） ast是（80，125）
        ast = ast.unsqueeze(1)
        # print(ast.shape)
        ast = ast.expand(ast.shape[0], src.shape[1], ast.shape[2])
        # print(ast.shape)
        emb = torch.cat([self.src_embed(src), ast], 2)
        # print("before", emb.size())
        emb = self.ast_fc1(emb)
        # print("after", emb.size())
        return self.encoder(emb, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)



#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-20 (yyyy-mm-dd)

"""
Just a basic feed-forward multilayer perceptron.
Domain strings have a fixed maximum length, so can just flatten the sequences to
yield a fixed size.

We don't need to use a deeper network because the DGA problem is pretty simple.
"""
import torch
import torch.nn as nn

from ..residual import TabularBasicBlock, linear_zero_bias_init, ExpansionFcn


class DgaFcn(nn.Module):
    def __init__(
        self, n_tokens, d_embedding, d_model, padding_idx, max_len, dropout=0.0
    ):
        super(DgaFcn, self).__init__()
        self.n_tokens = n_tokens
        self.d_embedding = d_embedding
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.max_len = max_len

        self.seq_fcn = nn.Sequential(
            nn.Embedding(
                self.n_tokens,
                self.d_embedding,
                padding_idx=self.padding_idx,
                max_norm=1.0,
                scale_grad_by_freq=True,
            ),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            ExpansionFcn(self.d_embedding, 4 * self.d_embedding, nn.GELU),
        )
        self.flat_fcn = nn.Sequential(
            nn.Flatten(1, 2),  # assume input is batch first
            TabularBasicBlock(
                self.max_len * self.d_embedding, self.d_model, nn.GELU, nn.BatchNorm1d
            ),
            TabularBasicBlock(self.d_model, self.d_model, nn.GELU, nn.BatchNorm1d),
            TabularBasicBlock(self.d_model, self.d_model, nn.GELU, nn.BatchNorm1d),
            linear_zero_bias_init(self.d_model, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.seq_fcn(x)
        return self.flat_fcn(x).squeeze()

    @torch.no_grad()
    def predict_proba(self, x):
        x = self.forward(x)
        return self.sigmoid(x)

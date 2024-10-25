#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-27 (yyyy-mm-dd)

"""
"""

import torch
import torch.nn as nn

from ..residual import TabularBasicBlock, linear_zero_bias_init

class ZScaleData(nn.Module):
    def __init__(self, center, scale, eps=1e-5):
        """
        This class centers and scales the data. It also accepts features with scale
        0.0 by simply removing those features from the output.
        :param center: torch.Tensor - this tensor is subtracted from the input
        :param scale: torch.Tensor - this tensor is divides the input
        :param eps: float - this is used to screen for features with near-zero scale
        """
        super(ZScaleData, self).__init__()
        assert isinstance(center, torch.Tensor)
        assert isinstance(scale, torch.Tensor)
        assert isinstance(eps, float) and eps >= 0.0
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.center.requires_grad_(False)
        self.scale.requires_grad_(False)
        self.eps = eps
        self.output_size = int((self.scale > self.eps).long().sum())

    def forward(self, x):
        x_scaled = (x - self.center) / self.scale
        # this awkward dance removes columns that have 0 variance, i.e. are constant.
        # We do this inside the model, so it's always applied without us having to
        # worry about a bunch of tedious pipeline steps.
        return x_scaled[:, self.scale > self.eps]


class EmberFcn(nn.Module):
    def __init__(self, n_features, d_model, center, scale, dropout=0.0):
        super(EmberFcn, self).__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.dropout = dropout

        self.center_and_scale = ZScaleData(center=center, scale=scale)

        self.fcn = nn.Sequential(
            self.center_and_scale,
            nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity(),
            linear_zero_bias_init(self.center_and_scale.output_size, self.d_model),
            TabularBasicBlock(self.d_model, self.d_model, nn.GELU, nn.BatchNorm1d),
            TabularBasicBlock(self.d_model, self.d_model, nn.GELU, nn.BatchNorm1d),
            TabularBasicBlock(self.d_model, self.d_model, nn.GELU, nn.BatchNorm1d),
            TabularBasicBlock(self.d_model, self.d_model, nn.GELU, nn.BatchNorm1d),
            linear_zero_bias_init(self.d_model, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.fcn(x).squeeze()

    @torch.no_grad()
    def predict_proba(self, x):
        x = self.forward(x)
        return self.sigmoid(x)


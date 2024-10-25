#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-19 (yyyy-mm-dd)


"""
Implements feed-forward residual networks for tabular data using linear layers with
shortcut connections.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual
Networks." (2016) preprint: https://arxiv.org/abs/1603.05027

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun in "Deep Residual Learning for Image
Recognition" (2015) preprint: https://arxiv.org/abs/1512.03385
"""
import torch.nn as nn


class AbstractResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        """
        This implements a revised Residual Block structure, described in Kaiming He,
        Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual
        Networks." (2016) preprint: https://arxiv.org/abs/1603.05027

        This is slightly different from the Residual Block described by Kaiming He,
        Xiangyu Zhang, Shaoqing Ren, Jian Sun in "Deep Residual Learning for Image
        Recognition" (2015) preprint: https://arxiv.org/abs/1512.03385 because the
        activation is moved inside the blocks, instead of being on the residual path.

        The 2016 paper reports a slightly tweaked structure that yields some
        improvement compared to the 2015 paper.

        :param in_features: int
        :param out_features: int
        :param activation: nn.Module that has the activation function
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.blocks = nn.Identity()  # Placeholder to be overridden later
        self.shortcut = (
            nn.Linear(self.in_features, self.out_features, bias=False)
            if self.should_apply_shortcut
            else nn.Identity()
        )

    def forward(self, x):
        block_out = self.blocks(x)
        # Below: self.shortcut is Identity() if should_apply_shortcut = False
        residual = self.shortcut(x)
        return block_out + residual

    @property
    def should_apply_shortcut(self):
        return self.in_features != self.out_features


class FcnBlock(AbstractResidualBlock):
    def __init__(
        self, in_features, expansion_features, activation, norm, *args, **kwargs
    ):
        """
        This is inspired by the fully-connected layer of the Transformer
        model in Attention Is All You Need, which is simply a way to do 1-dimensional
        convolution on an input.

        Using the GELU activation and adding a normalization layer are intended to
        improve speed of convergence.
        """
        super().__init__(
            in_features=in_features, out_features=in_features, *args, **kwargs
        )
        self.blocks = nn.Sequential(
            nn.Linear(in_features, expansion_features, bias=False),
            norm(expansion_features),
            activation(),
            linear_zero_bias_init(expansion_features, in_features),
        )


class TabularBasicBlock(AbstractResidualBlock):
    def __init__(self, in_features, out_features, activation, norm, *args, **kwargs):
        """
        Implements the nonlinear component for a residual network. Denote the input to
        layer L as x_L and the output of this layer as f(x_L). A residual network has
        the form:
        x_{L+1} = f(x_L) + x_L
        where x_{L+1} is the output of layer L (equiv. the input for layer L+1).

        This implements the block structure following the recommendations in
        "Identity Mappings in Deep Residual Networks" by Kaiming He, Xiangyu Zhang,
        Shaoqing Ren, Jian Sun preprint: https://arxiv.org/abs/1603.05027
        """
        super().__init__(
            in_features=in_features, out_features=out_features, *args, **kwargs
        )
        assert isinstance(in_features, int) and in_features > 0
        assert isinstance(out_features, int) and out_features > 0
        self.blocks = nn.Sequential(
            norm(in_features) if norm else nn.Identity(),
            activation(),
            nn.Linear(in_features, out_features, bias=False),
            norm(out_features) if norm else nn.Identity(),
            activation(),
            nn.Linear(out_features, out_features, bias=False),
        )


class ExpansionFcn(AbstractResidualBlock):
    def __init__(self, in_features, expansion_features, activation, *args, **kwargs):
        super().__init__(in_features=in_features, out_features=in_features)
        assert isinstance(in_features, int) and in_features > 0
        assert isinstance(expansion_features, int) and expansion_features > 0
        self.blocks = nn.Sequential(
            nn.Linear(in_features, expansion_features, bias=False),
            nn.LayerNorm(expansion_features),
            activation(),
            nn.Linear(expansion_features, in_features, bias=False),
        )


def linear_zero_bias_init(in_features, out_features, bias=True):
    """A convenience method to initialize a linear layer with biases equal to zero."""
    _linear = nn.Linear(in_features, out_features, bias=bias)
    nn.init.zeros_(_linear.bias)
    return _linear

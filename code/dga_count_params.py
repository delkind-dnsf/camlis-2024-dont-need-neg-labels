#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-29 (yyyy-mm-dd)

"""
"""
import numpy as np

from camlis24.dga.nn import DgaFcn
from camlis24.dga.tokenizer import NgramCoDec
from camlis24.nn_utils import count_params

if __name__ == "__main__":
    codec = NgramCoDec(reverse_input_str=True, ngram=3)
    embedding_size = 12
    print(f"embedding_size: {embedding_size}")
    net = DgaFcn(
        n_tokens=codec.n_token,
        d_embedding=embedding_size,
        padding_idx=codec.pad_index,
        max_len=75,
        d_model=256,
    )
    print(f"the model has {count_params(net):,} parameters")

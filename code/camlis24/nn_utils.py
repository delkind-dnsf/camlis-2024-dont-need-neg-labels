#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-19 (yyyy-mm-dd)

"""
Useful little functions for torch
"""

import random

import numpy as np
import torch


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([torch.numel(p) for p in model_parameters])


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()

    worker_seed = worker_info.seed + worker_id
    worker_seed %= 2**32 - 1
    np.random.seed(worker_seed)
    random.seed(worker_seed)  # unused
    worker_info.dataset.random_seed = np.random.default_rng(worker_seed)

#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-26 (yyyy-mm-dd)

"""
"""
import argparse
import pathlib

import numpy as np
import torch

from .tokenizer import NgramCoDec


def parse_dga_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--data_fname", type=pathlib.Path, required=True)
    p.add_argument("-s", "--save_point", type=pathlib.Path, required=True)
    p.add_argument("-r", "--seed", type=int, default=525178)  # rolled some dice
    p.add_argument("--corrupt_frac", type=float, default=0.4)
    p.add_argument("--n_epoch", type=int, default=64)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=6)

    p.add_argument("-D", "--dry_run", action="store_true", default=False)
    args = p.parse_args()

    if args.dry_run:
        args.num_workers = 1
        args.n_epoch = 1
        args.n_splits = 3
    return args


def get_dga_fcn_config(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.batch_size = 2048
    args.n_warmup_epoch = 1

    args.reverse_input_str = True
    args.ngram = 3
    args.codec = NgramCoDec(reverse_input_str=args.reverse_input_str, ngram=args.ngram)
    args.pad_index = args.codec.pad_index
    args.n_token = args.codec.n_token
    args.max_len = 75

    args.d_embedding = int(np.log2(args.codec.n_token) + 0.5)
    args.d_model = 256
    args.dropout = 0.0

    args.lr = 1e-4
    args.weight_decay = 0.0
    args.early_stop_patience = 3
    args.max_grad_norm = float("inf")

    args.lr_scheduler_step_size = 4
    args.lr_scheduler_gamma = 0.5

    args.bbe_delta = 0.1
    args.bbe_gamma = 0.0
    args.bbe_smoother = 1e-6
    args.hparams = dict(
        batch_size=args.batch_size,
        reverse_input_str=args.reverse_input_str,
        ngram=args.ngram,
        pad_index=args.pad_index,
        n_token=args.n_token,
        max_len=args.max_len,
        d_embedding=args.d_embedding,
        d_model=args.d_model,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        max_grad_norm=args.max_grad_norm,
        bbe_delta=args.bbe_delta,
        bbe_gamma=args.bbe_gamma,
        bbe_smoother=args.bbe_smoother,
    )
    args.net_config = dict(
        n_tokens=args.n_token,
        d_embedding=args.d_embedding,
        d_model=args.d_model,
        dropout=args.dropout,
        padding_idx=args.pad_index,
        max_len=args.max_len,
    )
    return args

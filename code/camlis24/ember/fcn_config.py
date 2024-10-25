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

import torch


def parse_ember_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--data_fname", type=pathlib.Path, required=True)
    p.add_argument("-s", "--save_point", type=pathlib.Path, required=True)
    p.add_argument("-r", "--seed", type=int, default=277052)  # rolled some dice
    p.add_argument("--corrupt_frac", type=float, default=0.4)
    p.add_argument("--n_epoch", type=int, default=128)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=6)

    p.add_argument("-U", "--exclude_unlabeled", action="store_false", default=True)
    p.add_argument("-D", "--dry_run", action="store_true", default=False)
    args = p.parse_args()

    if args.dry_run:
        args.num_workers = 1
        args.n_epoch = 1
        args.n_splits = 3
    return args


def get_ember_fcn_config(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.batch_size = 1024

    args.n_warmup_epoch = 4
    args.n_features = 2381
    args.d_model = 512
    args.dropout = 0.0

    args.lr = 3e-4
    args.weight_decay = 0.0
    args.early_stop_patience = 2 * args.n_warmup_epoch
    args.max_grad_norm = float("inf")

    args.lr_scheduler_step_size = args.n_warmup_epoch
    args.lr_scheduler_gamma = 0.75

    args.bbe_delta = 0.1
    args.bbe_gamma = 0.0
    args.bbe_smoother = 1e-6
    args.hparams = dict(
        batch_size=args.batch_size,
        n_features=args.n_features,
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
        n_features=args.n_features,
        d_model=args.d_model,
        dropout=args.dropout,
    )
    return args


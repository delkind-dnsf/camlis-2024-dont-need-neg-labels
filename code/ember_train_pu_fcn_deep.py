#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-19 (yyyy-mm-dd)


"""
Train a PN classifier that treats the PU data as if it were PN data.
"""
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from camlis24.ember.fcn_config import get_ember_fcn_config, parse_ember_args
from camlis24.ember.loader import read_ember
from camlis24.ember.nn import EmberFcn
from camlis24.ember.wrapper import ember_cross_val_wrapper
from camlis24.pipeline import PosUnlabEmberHarnessFcn

if __name__ == "__main__":
    args = parse_ember_args()
    train_val_df, test_df = read_ember(args.data_fname)

    if args.dry_run:
        # get a small, random, stratified slice of the data
        _, dry_run_ndx = train_test_split(
            range(len(train_val_df)),
            test_size=0.25,
            stratify=train_val_df["label"],
            random_state=666,
        )
        train_val_df = train_val_df.iloc[dry_run_ndx]

    args = get_ember_fcn_config(args)
    args.model_name = f"Ember-PU-deep-model-{args.corrupt_frac:.2f}"

    results_df = ember_cross_val_wrapper(
        train_val_df,
        test_df,
        pipeline_class=PosUnlabEmberHarnessFcn,
        args=args,
        net_class=EmberFcn,
    )
    args.tb_writer = SummaryWriter(log_dir=args.save_point.joinpath(args.model_name))

    print(results_df)

    metric_cols = ["accuracy", "brier_score", "cross_entropy", "roc_auc"]
    metric_dict = {f"hparam/{col}": results_df[col].mean() for col in metric_cols}
    print(metric_dict)
    args.tb_writer.add_hparams(args.hparams, metric_dict=metric_dict)
    # accuracy:       0.9278
    # brier_score:    0.0657
    # cross_entropy:  0.4977
    # roc_auc:        0.9634

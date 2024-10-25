#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-27 (yyyy-mm-dd)

"""
"""
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from camlis24.ember.fcn_config import get_ember_fcn_config, parse_ember_args
from camlis24.ember.loader import read_ember
from camlis24.ember.nn import EmberFcn
from camlis24.ember.wrapper import ember_cross_val_wrapper
from camlis24.pipeline import PosUnlabEmberHarnessFcn, PosNegHarnessFcn

if __name__ == "__main__":
    args = parse_ember_args()
    train_val_df, test_df = read_ember(args.data_fname)
    if args.dry_run:
        train_val_df = train_val_df.sample(
            int(0.25 * len(train_val_df)), random_state=args.seed
        )

    args = get_ember_fcn_config(args)

    print(f"Ember PU Models")
    pu_results = []
    corruption_iter = np.arange(0.05, 0.6, step=0.05)
    metric_cols = [
        "accuracy",
        "brier_score",
        "cross_entropy",
        "trim_cross_entropy",
        "roc_auc",
    ]
    for i, corrupt_frac in enumerate(corruption_iter):
        args.corrupt_frac = corrupt_frac
        args.model_name = f"Ember-PU-model-{args.corrupt_frac:.2f}"

        results_df = ember_cross_val_wrapper(
            train_val_df=train_val_df,
            test_df=test_df,
            pipeline_class=PosUnlabEmberHarnessFcn,
            args=args,
            net_class=EmberFcn,
        )
        args.tb_writer = SummaryWriter(
            log_dir=args.save_point.joinpath(args.model_name)
        )
        pu_results.append(results_df)

        results_df.drop(columns=["fold"], inplace=True)
        print(results_df)

        metric_dict = {f"hparam/{col}": results_df[col].mean() for col in metric_cols}
        print(metric_dict)
        args.tb_writer.add_hparams(args.hparams, metric_dict=metric_dict)

        if args.dry_run and i >= 1:
            break
    pu_results = pd.concat(pu_results)
    pu_results.to_csv(
        args.save_point.joinpath("ember_pu_results.csv"), index=False, sep="\t"
    )

    print(f"Ember Biased/PN Models")
    pn_results = []
    for i, corrupt_frac in enumerate(corruption_iter):
        args.corrupt_frac = corrupt_frac
        args.model_name = f"DGA-Biased-model-{args.corrupt_frac:.2f}"

        results_df = ember_cross_val_wrapper(
            train_val_df=train_val_df,
            test_df=test_df,
            pipeline_class=PosNegHarnessFcn,
            args=args,
            net_class=EmberFcn,
        )
        args.tb_writer = SummaryWriter(
            log_dir=args.save_point.joinpath(args.model_name)
        )
        pn_results.append(results_df)

        results_df.drop(columns=["fold"], inplace=True)
        print(results_df)

        metric_dict = {f"hparam/{col}": results_df[col].mean() for col in metric_cols}
        print(metric_dict)
        args.tb_writer.add_hparams(args.hparams, metric_dict=metric_dict)

        if args.dry_run and i >= 1:
            break

    pn_results = pd.concat(pn_results)
    pn_results.to_csv(
        args.save_point.joinpath("ember_biased_results.csv"), index=False, sep="\t"
    )

    all_results = pd.concat([pn_results, pu_results])
    all_results.to_csv(
        args.save_point.joinpath("ember_comparison_results.csv"), index=False, sep="\t"
    )

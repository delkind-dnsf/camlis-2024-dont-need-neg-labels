#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-27 (yyyy-mm-dd)

"""
"""
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
    all_results = []
    metric_cols = [
        "accuracy",
        "brier_score",
        "cross_entropy",
        "trim_cross_entropy",
        "roc_auc",
    ]
    args.corrupt_frac = 0.0
    args.exclude_unlabeled = True
    print(f"Ember PN Baseline Models")
    args.model_name = f"Ember-PN-Baseline-model-{args.corrupt_frac:.2f}"

    pn_results_df = ember_cross_val_wrapper(
        train_val_df=train_val_df,
        test_df=test_df,
        pipeline_class=PosNegHarnessFcn,
        args=args,
        net_class=EmberFcn,
    )
    args.tb_writer = SummaryWriter(log_dir=args.save_point.joinpath(args.model_name))

    metric_dict = {f"hparam/{col}": pn_results_df[col].mean() for col in metric_cols}
    print(metric_dict)
    args.tb_writer.add_hparams(args.hparams, metric_dict=metric_dict)

    all_results.append(pn_results_df)

    pn_results_df.to_csv(
        args.save_point.joinpath("ember_biased_results.csv"), index=False, sep="\t"
    )

    print(f"Ember PU Baseline Model")
    args.exclude_unlabeled = False
    args.model_name = f"Ember-PU-Baseline-model-{args.corrupt_frac:.2f}"
    pu_results_df = ember_cross_val_wrapper(
        train_val_df=train_val_df,
        test_df=test_df,
        pipeline_class=PosUnlabEmberHarnessFcn,
        args=args,
        net_class=EmberFcn,
    )
    args.tb_writer = SummaryWriter(log_dir=args.save_point.joinpath(args.model_name))

    metric_dict = {f"hparam/{col}": pu_results_df[col].mean() for col in metric_cols}
    print(metric_dict)
    args.tb_writer.add_hparams(args.hparams, metric_dict=metric_dict)

    pu_results_df.to_csv(
        args.save_point.joinpath("ember_pu_baseline_results.csv"), index=False, sep="\t"
    )
    all_results.append(pu_results_df)

    print(f"Ember PUN Biased Baseline Models")
    args.exclude_unlabeled = False
    args.model_name = f"Ember-PUN-Biased-Baseline-model-{args.corrupt_frac:.2f}"

    pun_results_df = ember_cross_val_wrapper(
        train_val_df=train_val_df,
        test_df=test_df,
        pipeline_class=PosNegHarnessFcn,
        args=args,
        net_class=EmberFcn,
    )
    args.tb_writer = SummaryWriter(log_dir=args.save_point.joinpath(args.model_name))

    metric_dict = {f"hparam/{col}": pun_results_df[col].mean() for col in metric_cols}
    print(metric_dict)
    args.tb_writer.add_hparams(args.hparams, metric_dict=metric_dict)

    all_results.append(pun_results_df)

    pun_results_df.to_csv(
        args.save_point.joinpath("ember_pun_biased_results.csv"), index=False, sep="\t"
    )

    all_results = pd.concat(all_results)
    all_results.to_csv(
        args.save_point.joinpath("ember_baseline_all_results.csv"),
        index=False,
        sep="\t",
    )

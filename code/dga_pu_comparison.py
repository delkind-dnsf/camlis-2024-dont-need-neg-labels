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

from camlis24.dga.fcn_config import get_dga_fcn_config, parse_dga_args
from camlis24.dga.nn import DgaFcn
from camlis24.dga.wrapper import dga_cross_val_wrapper
from camlis24.pipeline import PosNegHarnessFcn, PosUnlabDgaHarnessFcn

if __name__ == "__main__":
    args = parse_dga_args()
    df = pd.read_csv(args.data_fname)
    if args.dry_run:
        df = df.sample(int(0.25 * len(df)), random_state=args.seed)

    args = get_dga_fcn_config(args)

    print(f"DGA PU Models")
    pu_results = []
    corruption_iter = np.arange(0.05, 0.6, step=0.05)
    for i, corrupt_frac in enumerate(corruption_iter):
        args.corrupt_frac = corrupt_frac
        args.model_name = f"DGA-PU-model-{args.corrupt_frac:.2f}"

        results_df = dga_cross_val_wrapper(
            df=df,
            pipeline_class=PosUnlabDgaHarnessFcn,
            args=args,
            net_class=DgaFcn,
        )
        args.tb_writer = SummaryWriter(
            log_dir=args.save_point.joinpath(args.model_name)
        )
        pu_results.append(results_df)

        results_df.drop(columns=["fold"], inplace=True)
        print(results_df)

        metric_cols = ["accuracy", "brier_score", "cross_entropy", "roc_auc"]
        metric_dict = {f"hparam/{col}": results_df[col].mean() for col in metric_cols}
        print(metric_dict)
        args.tb_writer.add_hparams(args.hparams, metric_dict=metric_dict)

        if args.dry_run and i >= 1:
            break
    pu_results = pd.concat(pu_results)
    pu_results.to_csv(
        args.save_point.joinpath("dga_pu_results.csv"), index=False, sep="\t"
    )

    print(f"Biased/PN Models")
    pn_results = []
    for i, corrupt_frac in enumerate(corruption_iter):
        args.corrupt_frac = corrupt_frac
        args.model_name = f"DGA-PN-model-{args.corrupt_frac:.2f}"

        results_df = dga_cross_val_wrapper(
            df=df, pipeline_class=PosNegHarnessFcn, args=args, net_class=DgaFcn
        )
        args.tb_writer = SummaryWriter(
            log_dir=args.save_point.joinpath(args.model_name)
        )
        pn_results.append(results_df)

        results_df.drop(columns=["fold"], inplace=True)
        print(results_df)

        metric_cols = ["accuracy", "brier_score", "cross_entropy", "roc_auc"]
        metric_dict = {f"hparam/{col}": results_df[col].mean() for col in metric_cols}
        print(metric_dict)
        args.tb_writer.add_hparams(args.hparams, metric_dict=metric_dict)

        if args.dry_run and i >= 1:
            break

    pn_results = pd.concat(pn_results)
    pn_results.to_csv(
        args.save_point.joinpath("dga_pn_results.csv"), index=False, sep="\t"
    )

    all_results = pd.concat([pn_results, pu_results])
    all_results.to_csv(
        args.save_point.joinpath("dga_all_results.csv"), index=False, sep="\t"
    )

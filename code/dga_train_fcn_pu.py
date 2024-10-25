#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-19 (yyyy-mm-dd)


"""
Train a PN classifier that treats the PU data as if it were PN data.
"""
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from camlis24.dga.fcn_config import get_dga_fcn_config, parse_dga_args
from camlis24.dga.nn import DgaFcn
from camlis24.dga.wrapper import dga_cross_val_wrapper
from camlis24.pipeline import PosUnlabDgaHarnessFcn

if __name__ == "__main__":
    args = parse_dga_args()
    df = pd.read_csv(args.data_fname)
    if args.dry_run:
        df = df.sample(int(0.5 * len(df)), random_state=args.rng)

    args = get_dga_fcn_config(args)
    args.model_name = f"DGA-PU-model-{args.corrupt_frac:.2f}"

    results_df = dga_cross_val_wrapper(
        net_class=DgaFcn, pipeline_class=PosUnlabDgaHarnessFcn, df=df, args=args
    )
    args.tb_writer = SummaryWriter(log_dir=args.save_point.joinpath(args.model_name))

    results_df.drop(columns=["fold"], inplace=True)
    print(results_df)

    metric_cols = ["accuracy", "brier_score", "cross_entropy", "roc_auc"]
    metric_dict = {f"hparam/{col}": results_df[col].mean() for col in metric_cols}
    print(metric_dict)
    args.tb_writer.add_hparams(args.hparams, metric_dict=metric_dict)

    #    accuracy  brier_score  cross_entropy   roc_auc
    # 0  0.981436     0.014548       0.057399  0.998156
    # 1  0.981243     0.014769       0.059887  0.997925
    # 2  0.979806     0.015716       0.060472  0.997802
    # 3  0.979534     0.015850       0.060617  0.997789
    # 4  0.976157     0.018212       0.068146  0.997217
    # 5  0.975029     0.019196       0.071078  0.996881

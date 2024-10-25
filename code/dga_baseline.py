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

from camlis24.dga.fcn_config import get_dga_fcn_config, parse_dga_args
from camlis24.dga.nn import DgaFcn
from camlis24.dga.wrapper import dga_cross_val_wrapper
from camlis24.pipeline import PosNegHarnessFcn

if __name__ == "__main__":
    args = parse_dga_args()
    df = pd.read_csv(args.data_fname)
    if args.dry_run:
        df = df.sample(int(0.5 * len(df)), random_state=args.rng)

    args = get_dga_fcn_config(args)

    print(f"Baseline PN Models")
    args.corrupt_frac = 0.0
    args.model_name = f"DGA-Baseline-PN-model-{args.corrupt_frac:.2f}"

    pn_results = dga_cross_val_wrapper(
        df=df, pipeline_class=PosNegHarnessFcn, args=args, net_class=DgaFcn
    )
    args.tb_writer = SummaryWriter(log_dir=args.save_point.joinpath(args.model_name))

    pn_results.drop(columns=["fold"], inplace=True)
    print(pn_results)

    metric_cols = ["accuracy", "brier_score", "cross_entropy", "roc_auc"]
    metric_dict = {f"hparam/{col}": pn_results[col].mean() for col in metric_cols}
    print(metric_dict)
    args.tb_writer.add_hparams(args.hparams, metric_dict=metric_dict)

    pn_results.to_csv(
        args.save_point.joinpath("dga_baseline_pn_results.csv"), index=False, sep="\t"
    )

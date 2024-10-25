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
from camlis24.pipeline import PosNegHarnessFcn

if __name__ == "__main__":
    args = parse_dga_args()
    df = pd.read_csv(args.data_fname)
    if args.dry_run:
        df = df.sample(int(0.5 * len(df)), random_state=args.rng)

    args = get_dga_fcn_config(args)
    args.model_name = f"DGA-PN-model-{args.corrupt_frac:.2f}"

    results_df = dga_cross_val_wrapper(
        df=df, pipeline_class=PosNegHarnessFcn, args=args, net_class=DgaFcn
    )
    args.tb_writer = SummaryWriter(log_dir=args.save_point.joinpath(args.model_name))

    results_df.drop(columns=["fold"], inplace=True)
    print(results_df)

    metric_cols = ["accuracy", "brier_score", "cross_entropy", "roc_auc"]
    metric_dict = {f"hparam/{col}": results_df[col].mean() for col in metric_cols}
    print(metric_dict)
    args.tb_writer.add_hparams(args.hparams, metric_dict=metric_dict)

    #    accuracy  brier_score  cross_entropy   roc_auc
    # 0  0.939188     0.109812       0.345381  0.996767
    # 1  0.941783     0.108122       0.341975  0.996828
    # 2  0.917543     0.114178       0.356996  0.995641
    # 3  0.929061     0.110557       0.348600  0.995974
    # 4  0.921894     0.112758       0.353735  0.995586
    # 5  0.944516     0.110372       0.345938  0.997051

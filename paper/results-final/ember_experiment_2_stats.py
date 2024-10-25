#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-10-03 (yyyy-mm-dd)

"""
"""

import pathlib

import pandas as pd
from scipy.stats import ttest_rel

if __name__ == "__main__":
    results_dir = pathlib.Path(__file__).parent
    fname_experiment = results_dir.joinpath("ember_comparison_results.csv")
    fname_baseline = results_dir.joinpath("ember_baseline_all_results.csv")
    df_xpmt = pd.read_csv(fname_experiment, sep="\t")
    df_base = pd.read_csv(fname_baseline, sep="\t")

    df_xpmt["accuracy"] = 100.0 * df_xpmt["accuracy"]
    df_xpmt["error_rate"] = 100.0 - df_xpmt["accuracy"]
    df_xpmt["method"] = df_xpmt["model"].apply(
        lambda x: "(TED)ⁿ" if "-PU-" in x else "Biased PN"
    )
    df_xpmt["corrupt_frac"] = [f"{x:.2f}" for x in df_xpmt["corrupt_frac"]]
    print(df_xpmt)

    metrics = ["brier_score", "cross_entropy", "trim_cross_entropy", "error_rate"]

    results = []
    for c in sorted(set(df_xpmt["corrupt_frac"])):
        for m in metrics:
            case = df_xpmt.loc[
                (df_xpmt["corrupt_frac"] == c) & (df_xpmt["method"] == "(TED)ⁿ"), m
            ]
            control = df_xpmt.loc[
                (df_xpmt["corrupt_frac"] == c) & (df_xpmt["method"] == "Biased PN"), m
            ]
            output = ttest_rel(case, control, alternative="less")
            results.append(
                {
                    "corrupt_frac": c,
                    "metric": m,
                    "statistic": (
                        f"{output.statistic:.2f} *"
                        if output.pvalue < 0.05
                        else f"{output.statistic:.2f}"
                    ),
                    "pvalue": output.pvalue,
                    "signif": output.pvalue < 0.05,
                    "test": "paired t-test",
                }
            )
    results = pd.DataFrame(results)
    results.sort_values(by=["metric", "corrupt_frac"], inplace=True)
    print(
        results.pivot(index=["corrupt_frac"], columns=["metric"], values=["statistic"])
    )

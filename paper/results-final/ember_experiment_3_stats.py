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
    df_base = pd.read_csv(fname_baseline, sep="\t")
    df_base = df_base.loc[
        df_base["model"].isin(
            ["Ember-PU-Baseline-model-0.00", "Ember-PUN-Biased-Baseline-model-0.00"]
        )
    ]
    print(df_base)

    df_base["accuracy"] = 100.0 * df_base["accuracy"]
    df_base["error_rate"] = 100.0 - df_base["accuracy"]
    df_base["method"] = df_base["model"].apply(
        lambda x: "(TED)ⁿ" if "-PU-" in x else "Biased PN"
    )

    metrics = ["brier_score", "cross_entropy", "trim_cross_entropy", "error_rate"]

    results = []
    for m in metrics:
        case = df_base.loc[df_base["method"] == "(TED)ⁿ", m]
        control = df_base.loc[df_base["method"] == "Biased PN", m]
        output = ttest_rel(case, control, alternative="less")
        results.append(
            {
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
    print(results)
    print(results.pivot(columns="metric", values="statistic"))

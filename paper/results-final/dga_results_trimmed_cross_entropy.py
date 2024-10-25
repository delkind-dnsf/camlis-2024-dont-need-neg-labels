#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-09-18 (yyyy-mm-dd)

"""
"""
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

from plot_prep import dnsf_colors, prep_data, jitter

if __name__ == "__main__":
    results_dir = pathlib.Path(__file__).parent
    fname_experiment = results_dir.joinpath("dga_all_results.csv")
    fname_baseline = results_dir.joinpath("dga_baseline_pn_results.csv")
    df_base_grp, df_xpmt = prep_data(
        fname_baseline=fname_baseline, fname_experiment=fname_experiment
    )
    sns.set_theme(style="ticks", palette=dnsf_colors, rc={"figure.dpi": 200})

    g = sns.scatterplot(
        x=jitter(df_xpmt["Positives among Unlabeled (%)"], 0.5),
        y=df_xpmt["Trimmed Cross-Entropy"],
        hue=df_xpmt["method"],
        palette=dnsf_colors,
        alpha=0.5,
    )
    g.hlines(
        y=[
            df_base_grp.loc[
                df_base_grp.index == "DGA-Baseline-PN-model-0.00",
                "trim_cross_entropy_lo",
            ],
            df_base_grp.loc[
                df_base_grp.index == "DGA-Baseline-PN-model-0.00",
                "trim_cross_entropy_hi",
            ],
        ],
        xmin=df_xpmt["Positives among Unlabeled (%)"].min(),
        xmax=df_xpmt["Positives among Unlabeled (%)"].max(),
        colors="black",
        ls="--",
        lw=1,
    )
    g.set_title("DGA dataset - Trimmed Cross-Entropy (5%)")
    plt.show()

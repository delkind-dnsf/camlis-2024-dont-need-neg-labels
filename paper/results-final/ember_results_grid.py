#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-10-02 (yyyy-mm-dd)

"""
"""

import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

from plot_prep import dnsf_colors, prep_data, jitter

if __name__ == "__main__":
    results_dir = pathlib.Path(__file__).parent
    fname_experiment = results_dir.joinpath("ember_comparison_results.csv")
    fname_baseline = results_dir.joinpath("ember_baseline_all_results.csv")
    df_base_grp, df_xpmt = prep_data(
        fname_baseline=fname_baseline, fname_experiment=fname_experiment
    )
    sns.set_theme(style="ticks", palette=dnsf_colors, rc={"figure.dpi": 200})

    print(df_base_grp.columns)
    print(df_base_grp[["roc_auc_mean", "roc_auc_se", "roc_auc_lo", "roc_auc_hi"]])

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(2, 2, 1)
    g = sns.scatterplot(
        x=jitter(df_xpmt["Positives among Unlabeled (%)"], 0.5),
        y=df_xpmt["Brier Score"],
        hue=df_xpmt["method"],
        palette=dnsf_colors,
        alpha=0.5,
    )
    g.hlines(
        y=[
            df_base_grp.loc[
                df_base_grp.index == "Ember-PN-Baseline-model-0.00", "brier_score_lo"
            ],
            df_base_grp.loc[
                df_base_grp.index == "Ember-PN-Baseline-model-0.00", "brier_score_hi"
            ],
        ],
        xmin=df_xpmt["Positives among Unlabeled (%)"].min(),
        xmax=df_xpmt["Positives among Unlabeled (%)"].max(),
        colors="black",
        ls="--",
        lw=1,
    )
    ax = fig.add_subplot(2, 2, 2)
    g = sns.scatterplot(
        x=jitter(df_xpmt["Positives among Unlabeled (%)"], 0.5),
        y=df_xpmt["Error Rate (%)"],
        hue=df_xpmt["method"],
        palette=dnsf_colors,
        alpha=0.5,
    )
    g.hlines(
        y=[
            df_base_grp.loc[
                df_base_grp.index == "Ember-PN-Baseline-model-0.00", "error_rate_lo"
            ],
            df_base_grp.loc[
                df_base_grp.index == "Ember-PN-Baseline-model-0.00", "error_rate_hi"
            ],
        ],
        xmin=df_xpmt["Positives among Unlabeled (%)"].min(),
        xmax=df_xpmt["Positives among Unlabeled (%)"].max(),
        colors="black",
        ls="--",
        lw=1,
    )
    ax = fig.add_subplot(2, 2, 3)
    g = sns.scatterplot(
        x=jitter(df_xpmt["Positives among Unlabeled (%)"], 0.5),
        y=df_xpmt["Cross-Entropy"],
        hue=df_xpmt["method"],
        palette=dnsf_colors,
        alpha=0.5,
    )
    g.hlines(
        y=[
            df_base_grp.loc[
                df_base_grp.index == "Ember-PN-Baseline-model-0.00", "cross_entropy_lo"
            ],
            df_base_grp.loc[
                df_base_grp.index == "Ember-PN-Baseline-model-0.00", "cross_entropy_hi"
            ],
        ],
        xmin=df_xpmt["Positives among Unlabeled (%)"].min(),
        xmax=df_xpmt["Positives among Unlabeled (%)"].max(),
        colors="black",
        ls="--",
        lw=1,
    )
    ax = fig.add_subplot(2, 2, 4)
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
                df_base_grp.index == "Ember-PN-Baseline-model-0.00",
                "trim_cross_entropy_lo",
            ],
            df_base_grp.loc[
                df_base_grp.index == "Ember-PN-Baseline-model-0.00",
                "trim_cross_entropy_hi",
            ],
        ],
        xmin=df_xpmt["Positives among Unlabeled (%)"].min(),
        xmax=df_xpmt["Positives among Unlabeled (%)"].max(),
        colors="black",
        ls="--",
        lw=1,
    )
    fig.suptitle("Experiment 2 (Ember)", fontsize=16)
    plt.show()

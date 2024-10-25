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
import pandas as pd
from plot_prep import dnsf_colors, prep_data

if __name__ == "__main__":
    results_dir = pathlib.Path(__file__).parent
    fname_baseline = results_dir.joinpath("ember_baseline_all_results.csv")
    fname_experiment = results_dir.joinpath("ember_comparison_results.csv")

    df_base_raw = pd.read_csv(fname_baseline, sep="\t")
    df_base_raw_grp, _ = prep_data(
        fname_baseline=fname_baseline, fname_experiment=fname_experiment
    )

    df_base_raw = df_base_raw.loc[
        df_base_raw["model"].isin(
            ["Ember-PU-Baseline-model-0.00", "Ember-PUN-Biased-Baseline-model-0.00"]
        )
    ]

    df_base_raw["method"] = df_base_raw["model"].apply(
        lambda x: "(TED)ⁿ" if "-PU-" in x else "Biased PN"
    )
    df_base_raw["accuracy"] = 100.0 * df_base_raw["accuracy"]
    df_base_raw["error_rate"] = 100.0 - df_base_raw["accuracy"]

    df_base_raw.rename(
        columns={
            "error_rate": "Error Rate (%)",
            "accuracy": "Accuracy (%)",
            "brier_score": "Brier Score",
            "cross_entropy": "Cross-Entropy",
            "trim_cross_entropy": "Trimmed Cross-Entropy",
            "roc_auc": "ROC AUC",
        },
        inplace=True,
    )

    sns.set_theme(style="ticks", palette=dnsf_colors, rc={"figure.dpi": 200})

    # Draw a nested boxplot to show bills by day and time

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(2, 2, 1)
    g = sns.boxplot(
        x=df_base_raw["method"],
        y=df_base_raw["Brier Score"],
        hue=df_base_raw["method"],
        palette=(dnsf_colors[1], dnsf_colors[0]),
    )
    plt.axhline(
        df_base_raw_grp.loc[
            df_base_raw_grp.index == "Ember-PN-Baseline-model-0.00", "brier_score_lo"
        ].iloc[0],
        c="black",
        linestyle="--",
    )
    plt.axhline(
        df_base_raw_grp.loc[
            df_base_raw_grp.index == "Ember-PN-Baseline-model-0.00", "brier_score_hi"
        ].iloc[0],
        c="black",
        linestyle="--",
    )
    ax = fig.add_subplot(2, 2, 2)
    g = sns.boxplot(
        x=df_base_raw["method"],
        y=df_base_raw["Error Rate (%)"],
        hue=df_base_raw["method"],
        palette=(dnsf_colors[1], dnsf_colors[0]),
    )
    plt.axhline(
        df_base_raw_grp.loc[
            df_base_raw_grp.index == "Ember-PN-Baseline-model-0.00", "error_rate_lo"
        ].iloc[0],
        c="black",
        linestyle="--",
    )
    plt.axhline(
        df_base_raw_grp.loc[
            df_base_raw_grp.index == "Ember-PN-Baseline-model-0.00", "error_rate_hi"
        ].iloc[0],
        c="black",
        linestyle="--",
    )
    ax = fig.add_subplot(2, 2, 3)
    g = sns.boxplot(
        x=df_base_raw["method"],
        y=df_base_raw["Cross-Entropy"],
        hue=df_base_raw["method"],
        palette=(dnsf_colors[1], dnsf_colors[0]),
    )
    plt.axhline(
        df_base_raw_grp.loc[
            df_base_raw_grp.index == "Ember-PN-Baseline-model-0.00", "cross_entropy_lo"
        ].iloc[0],
        c="black",
        linestyle="--",
    )
    plt.axhline(
        df_base_raw_grp.loc[
            df_base_raw_grp.index == "Ember-PN-Baseline-model-0.00", "cross_entropy_hi"
        ].iloc[0],
        c="black",
        linestyle="--",
    )
    ax = fig.add_subplot(2, 2, 4)
    g = sns.boxplot(
        x=df_base_raw["method"],
        y=df_base_raw["Trimmed Cross-Entropy"],
        hue=df_base_raw["method"],
        palette=(dnsf_colors[1], dnsf_colors[0]),
    )
    plt.axhline(
        df_base_raw_grp.loc[
            df_base_raw_grp.index == "Ember-PN-Baseline-model-0.00",
            "trim_cross_entropy_lo",
        ].iloc[0],
        c="black",
        linestyle="--",
    )
    plt.axhline(
        df_base_raw_grp.loc[
            df_base_raw_grp.index == "Ember-PN-Baseline-model-0.00",
            "trim_cross_entropy_hi",
        ].iloc[0],
        c="black",
        linestyle="--",
    )
    fig.suptitle("Experiment 3 (Ember)", fontsize=16)
    plt.show()

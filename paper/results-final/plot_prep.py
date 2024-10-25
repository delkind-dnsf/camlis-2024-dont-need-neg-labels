#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-09-25 (yyyy-mm-dd)

"""
"""

import numpy as np
import pandas as pd
from scipy import stats

dnsf_colors = ["#F300A2", "#3400F8", "#00BEFB", "#000000"]


def jitter(x, scale=0.005, seed=42):
    noise = np.random.default_rng(seed).normal(loc=0, scale=scale, size=x.shape)
    return x + noise


def prep_data(fname_baseline, fname_experiment):
    df_xpmt = pd.read_csv(fname_experiment, sep="\t")
    df_base = pd.read_csv(fname_baseline, sep="\t")

    df_xpmt["accuracy"] = 100.0 * df_xpmt["accuracy"]
    df_base["accuracy"] = 100.0 * df_base["accuracy"]

    df_xpmt["error_rate"] = 100.0 - df_xpmt["accuracy"]
    df_base["error_rate"] = 100.0 - df_base["accuracy"]

    df_xpmt.rename(
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

    metric_cols = [
        "accuracy",
        "error_rate",
        "brier_score",
        "cross_entropy",
        "trim_cross_entropy",
        "roc_auc",
    ]
    info_cols = ["alpha_true", "corrupt_frac", "model"]

    df_base_grp = (
        df_base[metric_cols + info_cols]
        .groupby("model")
        .agg(
            accuracy_mean=pd.NamedAgg("accuracy", "mean"),
            error_rate_mean=pd.NamedAgg("error_rate", "mean"),
            brier_score_mean=pd.NamedAgg("brier_score", "mean"),
            cross_entropy_mean=pd.NamedAgg("cross_entropy", "mean"),
            trim_cross_entropy_mean=pd.NamedAgg("trim_cross_entropy", "mean"),
            roc_auc_mean=pd.NamedAgg("roc_auc", "mean"),
            accuracy_std=pd.NamedAgg("accuracy", lambda x: np.std(x, ddof=1)),
            error_rate_std=pd.NamedAgg("error_rate", lambda x: np.std(x, ddof=1)),
            brier_score_std=pd.NamedAgg("brier_score", lambda x: np.std(x, ddof=1)),
            cross_entropy_std=pd.NamedAgg("cross_entropy", lambda x: np.std(x, ddof=1)),
            trim_cross_entropy_std=pd.NamedAgg(
                "trim_cross_entropy", lambda x: np.std(x, ddof=1)
            ),
            roc_auc_std=pd.NamedAgg("roc_auc", lambda x: np.std(x, ddof=1)),
            n=pd.NamedAgg("model", "count"),
        )
    )

    n = df_base_grp["n"].min()
    assert (
        n == df_base_grp["n"].max()
    )  # n should be the same for all models, but this checks just in case something changed. Safety first.

    alpha = 0.05
    q = stats.t.ppf(q=1.0 - alpha, df=n - 1)
    for c in metric_cols:
        df_base_grp[f"{c}_se"] = df_base_grp[f"{c}_std"] / np.sqrt(df_base_grp["n"])
        df_base_grp[f"{c}_lo"] = df_base_grp[f"{c}_mean"] - q * df_base_grp[f"{c}_se"]
        df_base_grp[f"{c}_hi"] = df_base_grp[f"{c}_mean"] + q * df_base_grp[f"{c}_se"]

    print(df_xpmt)
    df_xpmt["method"] = df_xpmt["model"].apply(
        lambda x: "(TED)ⁿ" if "-PU-" in x else "Biased PN"
    )
    df_xpmt["Positives among Unlabeled (%)"] = 100.0 * df_xpmt["alpha_true"]

    print(df_xpmt[["model", "alpha_true", "Positives among Unlabeled (%)"]].head(30))

    return df_base_grp, df_xpmt

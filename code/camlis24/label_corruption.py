#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-19 (yyyy-mm-dd)

"""
Apply the PU label corruption to a dataset.
"""

import numpy as np


def pu_label_corruption(df, corrupt_frac, random_seed: int):
    assert isinstance(corrupt_frac, float) and 0.0 <= corrupt_frac < 1.0
    df["prng"] = np.random.default_rng(random_seed).uniform(size=len(df))
    # move a fraction of positives in with the unlabeled class
    df["pu_label"] = df.loc[df["label"] == 1, "prng"].apply(
        lambda x: 0 if x < corrupt_frac else 1
    )
    df.fillna({"pu_label": df["label"]}, inplace=True)
    df["pu_label"] = [max(lab, 0) for lab in df["pu_label"]]  # unlabeled are -1
    df["pu_label"] = df["pu_label"].astype(int)
    df.drop(columns=["prng"], inplace=True)
    return df

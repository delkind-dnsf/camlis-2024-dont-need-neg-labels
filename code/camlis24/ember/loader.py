#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-27 (yyyy-mm-dd)

"""
Data loading utilities for the Ember dataset.
"""

import ember
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EmberCollate:
    def __call__(self, inputs):
        x, y, df_dict = zip(*inputs)
        x = torch.FloatTensor(np.array(x))
        y = torch.FloatTensor(y)
        df = pd.DataFrame(df_dict)
        return x, y, df


class PuEmberDfDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        subset: str,
        exclude_unlabeled: bool,
        data_directory: str = "data/ember2018/",
    ):
        if subset not in ["train", "test"]:
            raise ValueError(
                f'Invalid partition: got {subset} but expected "train" or "test"'
            )
        assert min(df["idx"]) >= 0
        self.df = df
        self.exclude_unlabeled = exclude_unlabeled
        if self.exclude_unlabeled:
            self.df = df.loc[df["label"] != -1]
        self.x, self.real_y = ember.read_vectorized_features(
            data_directory, subset=subset, feature_version=2
        )
        assert self.df["idx"].max() < self.x.shape[0]
        if subset == "test":
            assert len(set(self.df["idx"])) == self.x.shape[0]
            assert self.df["idx"].min() == 0

    def fetch(self, idx, col):
        target_col = self.df.columns.get_loc(col)
        return self.df.iloc[idx, target_col]

    def __getitem__(self, idx):
        subset_ind = self.fetch(idx, "idx")
        pu_label = self.fetch(idx, "pu_label")
        x = self.x[subset_ind, :]
        df_real_label = self.fetch(idx, "label")

        # This is a debugging check to assure we don't have indexing errors
        # memmap_real_label = self.real_y[subset_ind]
        # assert memmap_real_label == df_real_label

        df_dict = {"label": df_real_label, "pu_label": pu_label, "idx": subset_ind}
        # assert pu_label >= 0  # -1 is the code for unlabeled data in EMBER; exclude!
        return x, float(pu_label), df_dict

    def __len__(self):
        return len(self.df)

    def get_center_scale(self, eps=1e-5):
        # Loading the memmap array into main memory consumes a lot of memory, but
        # is probably the best way to extract this data.
        # I guess we could try playing games with sparse arrays, but this works on my
        # machine. ¯\_(ツ)_/¯
        # The memory should be freed after return, so I'm not worried about memory
        # consumption from loading the memmap array into RAM.
        arr_buff = np.asarray(self.x)
        # down-select to only the indices in the training set
        subset_ind = [self.fetch(i, "idx") for i in range(len(self))]
        arr_buff = arr_buff[subset_ind, :]
        # compute the stats
        center = arr_buff.mean(axis=0)
        scale = np.std(arr_buff, axis=0, ddof=1)

        # report some FYIs about the scale data
        n_zeros = np.isclose(scale, 0.0).sum()
        n_small = np.where(scale < eps, 1, 0).sum()
        if n_zeros > 0 and n_zeros == n_small:
            print(f"There are {n_zeros:,} scale values that are (numerically) 0.0.")
        elif n_small > n_zeros > 0:
            print(
                f"There are {n_small:,} scale values smaller than {eps} (including {n_zeros:,} zeros)."
            )

        scale = np.maximum(scale, eps)  # ensure scale > 0.0
        assert not np.isnan(scale).any()
        assert not np.isnan(center).any()
        assert (scale > 0.0).all()
        return torch.FloatTensor(center), torch.FloatTensor(scale)


def read_ember(data_fname):
    x, y, x_test, y_test = ember.read_vectorized_features(data_fname)

    train_df_ = pd.DataFrame({"idx": range(x.shape[0]), "label": y})
    test_df_ = pd.DataFrame({"idx": range(x_test.shape[0]), "label": y_test})
    return train_df_, test_df_

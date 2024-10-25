#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-20 (yyyy-mm-dd)

"""
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class IntCollatePad:
    def __init__(self, max_len, pad_index=0):
        """
        Creates a callable that transforms the data to tensors for our model:
        - unicode strings are mapped to integers
        - the integers are placed in torch.LongTensors
        - the list of LongTensors is padded to uniform length with pad_index
        - the labels are placed in LongTensors

        :param pad_index: int - integer to represent the padding token; default=0.
        """
        assert isinstance(max_len, int) and max_len > 0
        assert isinstance(pad_index, int) and pad_index >= 0
        self.max_len = max_len
        self.pad_index = pad_index
        self.batch_first = True

    def enc_str2tensor(self, enc_str):
        """
        Convert np.array of ints into torch.LongTensors, and then pad to max_len
        (if too short) or truncate to max_len (if too long).
        :param enc_str: list of np.array of ints
        :return: torch.LongTensor
        """
        # enc_str is np.array of integers encoding the characters of the domain str
        ints = list(map(torch.LongTensor, enc_str))
        ints = nn.utils.rnn.pad_sequence(
            ints, batch_first=self.batch_first, padding_value=self.pad_index
        )
        B, T = ints.shape  # batch first
        if T < self.max_len:
            #  pad too-short sequences
            ints = nn.functional.pad(
                ints, pad=(0, self.max_len - T), mode="constant", value=self.pad_index
            )
        elif T > self.max_len:
            # otherwise truncate too-long sequences
            ints = ints[:, : self.max_len]
        return ints

    def __call__(self, data_in):
        enc_str, label, df_dict = zip(*data_in)
        enc_str = self.enc_str2tensor(enc_str)
        str_x = enc_str
        label = torch.FloatTensor(label)
        df_ = pd.DataFrame.from_dict(df_dict)
        return str_x, label, df_


class PuDgaDataset(Dataset):
    def __init__(self, df, codec):
        self.df = df
        self.codec = codec

    def fetch(self, idx, col):
        target_col = self.df.columns.get_loc(col)
        return self.df.iloc[idx, target_col]

    def __getitem__(self, idx):
        pu_label = self.fetch(idx, "pu_label")
        true_label = self.fetch(idx, "label")
        domain_str = self.fetch(idx, "domain")
        domain_ints = self.codec.encode(domain_str)
        df_dict = {"pu_label": pu_label, "domain": domain_str, "label": true_label}
        return domain_ints, pu_label, df_dict

    def __len__(self):
        return len(self.df)

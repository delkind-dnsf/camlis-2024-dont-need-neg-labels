#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-26 (yyyy-mm-dd)

"""
"""

import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import dok_array, csr_array

from camlis24.dga.tokenizer import NgramCoDec

if __name__ == "__main__":
    fname = sys.argv[1]
    df = pd.read_csv(fname)
    ngram = 3
    codec = NgramCoDec(ngram=3, reverse_input_str=False)
    print(f"With {ngram}-grams, there are {codec.n_token:,} tokens.")
    X = dok_array((len(df), codec.n_token), dtype=np.int64)
    start_time = datetime.now()
    for i, dom in enumerate(tqdm.tqdm(df["domain"])):
        ints = codec.encode(dom)
        for j in ints:
            X[i, j] = 1

    X = csr_array(X)
    ngram_freq = X.sum(axis=0)

    token_freq_df = pd.DataFrame(
        {"token": [codec.imap(i) for i in range(codec.n_token)], "freq": ngram_freq}
    )
    token_freq_df = token_freq_df.sort_values(by="freq", ascending=False)
    print(token_freq_df.head(32))

    threshold = 10
    n_rare = np.where(ngram_freq < threshold, 1, 0).sum()
    print(
        f"There are {n_rare:,} {ngram}-grams that appear less that {threshold} times, leaving {codec.n_token - n_rare:,} frequently-seen tokens."
    )

    # hist, edges = np.histogram(ngram_freq, bins=np.unique(ngram_freq))
    # np.histogram(ngram_freq, bins=100)
    # plt.hist(ngram_freq)
    # plt.show()

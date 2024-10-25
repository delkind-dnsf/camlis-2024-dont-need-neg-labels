#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-19 (yyyy-mm-dd)

"""
Implements the PU methods from

Garg, Saurabh, et al.
"Mixture proportion estimation and PU learning: a modern approach."
Advances in Neural Information Processing Systems 34 (2021): 8532-8544.

After an extensive literature review, we chose this paper because it is, to my
knowledge, the only paper that
(1) scales to large data;
(2) provides tight inequality bounds on the mixture proportion estimate (MPE);
(3) obtains a good fit to the data (not over-fit like uPU, does not under-fit like nnPU, etc.);
(4) does not make heroic assumptions about the data.

Moreover, alternative model estimation methods require knowing the MPE, but we don't
know it a priori. Additionally, it is hard for us to estimate the MPE because (1) all
methods require feature vectors. The best feature vectors require training an
autoregressive model on the domain string. The worse alternative methods uses query
behavior data, but this violates the irreducibility assumption required for consistency
of the estimator; the estimate could be arbitrarily poor. (2) all other MPE methods
train a model to get a statistic and then discard the model, or make pairwise
comparisons. Both are somewhere between expensive and infeasible for data at our scale.

By contrast, this is an "all-in-one" method that simultaneously estimates the MPE and
a classifier that takes advantage of the MPE in a way that is mutually-reinforcing.
"""
import numpy as np
from scipy.optimize import minimize_scalar


class SmoothedEcdf:
    def __init__(self, smoother=0.0):
        self._data = []
        self._locked = False
        assert isinstance(smoother, float) and 0.0 <= smoother
        self._smoother = smoother
        self._ecdf_interp_points = None

    @property
    def data(self):
        if not self._locked:
            data = np.concatenate(self._data)
            if self._smoother > 0.0:
                data = np.append(data, [0.0, 1.0])
            self._data = np.sort(data, axis=None)
            ecdf_counts = np.arange(self._data.size)
            # sorted data -> we know the cumulative counts a priori (it's the index)
            #  ECDF answers the question "what proportion of points are less than k?"
            # smooth the counts to bound away from 0.0 and 1.0.
            self._ecdf_interp_points = ecdf_counts + self._smoother
            self._ecdf_interp_points /= self._data.size + 2.0 * self._smoother
            self._locked = True
        return self._data

    @property
    def bins(self):
        return np.array([np.min(self.data), np.max(self.data)])

    def __len__(self):  # pylint: disable=invalid-length-returned
        return self.data.size

    def append(self, a: np.array):
        if self._locked:
            raise ValueError("SmoothedEcdf is locked & you can't append to it.")
        self._data.append(np.sort(a.flatten()))
        return self

    def ecdf(self, z):
        return np.interp(z, xp=self.data, fp=self._ecdf_interp_points)

    def sf(self, z):
        return 1.0 - self.ecdf(z)


def bbe_estimate(p_hist: SmoothedEcdf, u_hist: SmoothedEcdf, delta, gamma):
    """
    Implements the best-bin estimator (BBE, Algorithm 1) from Garg, Saurabh, et al.
    "Mixture proportion estimation and PU learning: a modern approach."
    Advances in Neural Information Processing Systems 34 (2021): 8532-8544.

    :param p_hist: HistogramEcdf - a class that implements the empirical survival
    function for the positive data, sample sizes and bins.
    The empirical survival function is 1 - ecdf(z), where ecdf(z) is the empirical
    cumulative distribution function evaluated at z.
    :param u_hist: HistogramEcdf - Same as p_hist, but for the unlabeled data.
    :param delta: float - Hyperparameter - the probability of the UBC inequality
    :param gamma: float - Hyperparameter - increases the UCB penalty, probably doesn't
    need to be tuned
    :return: tuple of floats - (the mixture proportion estimate, the threshold corresponding
    to the MPE)
    """
    assert isinstance(gamma, float) and 0.0 <= gamma < 1.0
    assert isinstance(delta, float) and 0.0 < delta < 1.0 and not np.isclose(delta, 0.0)
    # Derive the constant used for the upper confidence bound
    c = np.sqrt(np.log(4.0 / delta))
    ucb = (1.0 + gamma) * (
        c / np.sqrt(2.0 * len(p_hist)) + c / np.sqrt(2.0 * len(u_hist))
    )

    # Minimize the upper confidence bound on q_hat_u(c) / q_hat_p(c)
    # print(
    #     f"Quantiles of predictions for positives:\t\n{np.quantile(p_hist.data, np.arange(11)/10.0)}"
    # )
    # print(
    #     f"Quantiles of predictions for unlabeled:\t\n{np.quantile(u_hist.data, np.arange(11)/10.0)}"
    # )

    optim_result = minimize_scalar(
        lambda c: (u_hist.sf(c) + ucb) / p_hist.sf(c), bounds=(0.0, 1.0)
    )
    c_hat = float(optim_result.x)
    # print(f"BBE info: u_hist.sf(ĉ):{u_hist.sf(c_hat)}\tp_hist.sf(ĉ):{p_hist.sf(c_hat)}")
    # Compute the mixture proportion estimate alpha_hat at the optimal value c_hat.
    alpha_hat = float(u_hist.sf(c_hat) / p_hist.sf(c_hat))
    # Theorem 1 of Saurabh Garg et al. "Mixture Proportion Estimation and PU Learning: A Modern Approach" requires that
    # this inequality be satisfied for the population CDF for the bbe estimate of class prior to be valid. We don't
    # know the population CDF, but we can *estimate* it using the ecdf.
    theorem_1_rhs = 2.0 * np.log(4.0 / delta) / p_hist.sf(c_hat)
    min_sample_size = min(len(p_hist), len(u_hist))
    if min_sample_size < theorem_1_rhs:
        print(
            f"We don't not satisfy Theorem 1 in Garg et al.! We have {min_sample_size:,} < {theorem_1_rhs} but Theorem 1 requires the opposite."
        )
    return alpha_hat, c_hat

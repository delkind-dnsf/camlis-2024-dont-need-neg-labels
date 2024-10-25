#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-20 (yyyy-mm-dd)

"""
Implements methods to compute FPR and AUC from PU data.

"Recovering True Classifier Performance in Positive-Unlabeled Learning"
Shantanu Jain, Martha White, Predrag Radivojac
Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17)
2017
https://ojs.aaai.org/index.php/AAAI/article/view/10937
"""

from sklearn.metrics import roc_auc_score


def pu_adjust_auc(auc_raw, class_prior):
    return (auc_raw - class_prior / 2.0) / (1.0 - class_prior)


def pu_auc(scores, labels, class_prior):
    """Convenience function to compute ROC AUC(PU)"""
    auc_raw = roc_auc_score(y_true=labels, y_score=scores)
    return pu_adjust_auc(auc_raw=auc_raw, class_prior=class_prior)

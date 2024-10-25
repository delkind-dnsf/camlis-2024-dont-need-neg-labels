# Where we’re going, we don’t need (negative) labels

This repository collects the code, paper, and supporting materials for David J. Elkind's CAMLIS 2024 poster.

# Abstract

Weakly supervised methods can train highly effective machine learning models for computer security, even when benign labels for data are entirely missing. Typically, computer security data only contains labels for the "bad stuff" (malware, botnet domains), whereas labels for the "good stuff" are either missing entirely, or only cover trivial cases (Microsoft software, Alexa Top 1 Million Domains). The standard supervised classification approach requires all samples to have a label. However, it can be very expensive and time-consuming to obtain a large, diverse sample benign labels. Using two publicly-available datasets, EMBER and Namgung's DGA corpus, we show that weakly-supervised learning methods out-perform conventional supervised learning when one class is unlabeled (a mix of positive and negative data in some unknown proportion). Moreover, we show that applying conventional supervised learning approaches to unlabeled data creates a "backdoor" in the machine learning model. We show that the weakly-supervised learning approach minimizes this vulnerability.

# About the Author

David J. Elkind is the Chief Data Scientist at DNS Filter. Previously, he was a senior data scientist at CrowdStrike. He holds a MS in mathematics and statistics from Georgetown University.

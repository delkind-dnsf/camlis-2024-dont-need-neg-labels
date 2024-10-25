#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-19 (yyyy-mm-dd)

"""
"""
import ember

if __name__ == "__main__":
    ember.create_vectorized_features("data/ember2018/")
    ember.create_metadata("data/ember2018/")

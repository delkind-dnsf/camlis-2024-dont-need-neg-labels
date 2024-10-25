#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-20 (yyyy-mm-dd)

"""
"""
import itertools

import numpy as np


class NgramCoDec:
    def __init__(self, ngram, reverse_input_str):
        """
        This is a general class for producing n-gram encodings for user-supplied n.
        :param ngram: positive integer - the order of n-grams to compute (e.g. 2-grams
        or 3-grams)
        :param reverse_input_str: bool - set to True to reverse the input string, or
        False to leave the input string as-is.
        """
        assert isinstance(ngram, int) and ngram >= 1
        assert isinstance(reverse_input_str, bool)
        self._ngram = ngram
        self._reverse_input_str = reverse_input_str
        # The explicit design here is that anything not in self.fqdn_alphabet is mapped
        # to the unknown character token. This design reserves the padding token for
        # padding alone (not both padding and unknown token)
        self.pad_index = 0
        self.unk_index = 1
        self.start_index = 2
        self.stop_index = 3
        self._mapper = {
            "~": self.pad_index,
            "?": self.unk_index,
            "^": self.start_index,
            "$": self.stop_index,
        }
        self._special_keys = list(self._mapper.keys())
        assert all(k not in self.fqdn_alphabet for k in self._mapper.keys())
        assert all(len(k) == 1 for k in self._mapper.keys())
        initial_len = len(self._mapper)
        n_gram_lists = self.ngram * [self.fqdn_alphabet]
        assert len(n_gram_lists) == self.ngram
        for i, char in enumerate(itertools.product(*n_gram_lists)):
            self._mapper.update({"".join(char): initial_len + i})
        self._inverse_mapper = {idx: str(char) for char, idx in self._mapper.items()}

    def map(self, character):
        assert isinstance(character, str)
        index = self._mapper.get(character, self.unk_index)
        # TODO - need to chunk this data
        return index

    def imap(self, index):
        assert isinstance(index, int)
        character = self._inverse_mapper.get(index, "%UNK")
        return character

    def ngram_iterator(self, x):
        for item in zip(*[x[i:] for i in range(self.ngram)]):
            yield "".join(item)

    def encode(self, fqdn):
        fqdn = fqdn.lower()
        if self.reverse_input_str:
            fqdn = fqdn[::-1]
        encoding = [self.map(c) for c in self.ngram_iterator(fqdn)]
        # Append the start & stop symbols to the fqdn.
        # This is standard practice in text-processing; it's semantically important
        # to understand start/stop of a sequence of characters.
        if self.reverse_input_str:
            encoding = [self.stop_index] + encoding + [self.start_index]
        else:
            encoding = [self.start_index] + encoding + [self.stop_index]
        return np.array(encoding)

    def decode(self, int_seq, strip=True):
        # padding characters are excluded from decoding
        exclude = [self.pad_index]
        if strip:
            exclude += [self.start_index, self.stop_index]
        int_seq = filter(lambda x: x not in exclude, int_seq)
        decoded = [self.imap(i.item()) for i in int_seq]
        decoded = [c[0] for c in decoded[:-1]] + [decoded[-1]]
        output = "".join(decoded)
        if self.reverse_input_str:
            output = output[::-1]
        return output

    @property
    def special_keys(self):
        return self._special_keys

    @property
    def special_indices(self):
        return [self.map(c) for c in self.special_keys]

    @property
    def fqdn_alphabet(self):
        # I don't know why, but sometimes domains have _ as a character, even though
        # that's outside the DNS spec, as far as I can tell!
        # All other characters in FQDN are mapped to a special %UNK token
        return "abcdefghijklmnopqrstuvwxyz0123456789.-_"

    @property
    def n_token(self):
        return len(self._mapper)

    @property
    def ngram(self):
        return self._ngram

    @property
    def reverse_input_str(self):
        return self._reverse_input_str

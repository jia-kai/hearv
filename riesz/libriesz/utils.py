#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: utils.py
# $Date: Wed Jan 07 01:01:59 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np
import matplotlib.pyplot as plt

import logging
import os
import cPickle as pickle

logger = logging.getLogger(__name__)

class CachedResult(object):
    enabled = True
    _cache_dir = None

    def __init__(self, cache_type, getter, cache_dir=None):
        """:param getter: callabel that map key to value"""
        self._type = cache_type
        self._getter = getter
        self._cache_dir = cache_dir

    def __call__(self, key):
        if not self.enabled:
            return self._getter(key)

        cache_fpath = '{}-{}.pkl'.format(key, self._type)
        if self._cache_dir:
            cache_fpath = os.path.join(self._cache_dir, cache_fpath)
        if os.path.isfile(cache_fpath):
            logger.info('load cache from {}'.format(cache_fpath))
            with open(cache_fpath) as fin:
                obj = pickle.load(fin)
            return obj
        obj = self._getter(key)
        with open(cache_fpath, 'w') as fout:
            pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)
        return obj

def plot_val_with_fft(data, sample_rate=1.0, cut_low=None, cut_high=2000,
                      output=None, show=True):
    sample_rate = float(sample_rate)
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.plot(np.arange(len(data)) / sample_rate, data)
    ax = fig.add_subplot(2, 1, 2)
    fft = np.fft.fft(data)
    freq = sample_rate / len(data) * np.arange(len(data))
    if cut_low is not None:
        cut_low = max(int(len(data) * cut_low / sample_rate - 1), 0)
    else:
        cut_low = 0
    if cut_high is not None:
        cut_high = int(len(data) * cut_high / sample_rate)
    else:
        cut_high = float('inf')
    cut_high = min(cut_high, len(data) / 2)
    assert cut_low < cut_high
    fft = fft[cut_low:cut_high]
    freq = freq[cut_low:cut_high]
    ax.set_xlabel('freq')
    ax.set_ylabel('amplitude')
    ax.plot(freq, np.abs(fft))
    if output:
        fig.savefig(output)
    if show:
        plt.show()

def get_env_config(name, default):
    v = os.getenv(name, '')
    if v:
        logger.info('set {}={} by environment var'.format(name, v))
        return v
    return default

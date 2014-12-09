#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: utils.py
# $Date: Mon Dec 08 00:21:00 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import cv2
import numpy as np
import matplotlib.pyplot as plt

import logging
import os
import cPickle as pickle

logger = logging.getLogger(__name__)

def get_riesz_pyr_with_cache(fpath, use_quat=False):
    from .pyramid_quat import RieszQuatPyramidBuilder, RieszQuatPyramid
    from .pyramid import RieszPyramidBuilder, RieszPyramid
    if use_quat:
        builder = RieszQuatPyramidBuilder()
        pyrclass = RieszQuatPyramid
    else:
        builder = RieszPyramidBuilder()
        pyrclass = RieszPyramid
    pyr_fpath = fpath[:fpath.rfind('.')] + '-rieszpyr.pkl'
    if os.path.isfile(pyr_fpath):
        logger.info('load cached pyr from {}'.format(pyr_fpath))
        with open(pyr_fpath) as fin:
            obj = pickle.load(fin)
        assert isinstance(obj, pyrclass)
        return obj
    logger.info('computing riesz pyramid: fpath={} builder={} pyr={}'.format(
        fpath, type(builder).__name__, pyrclass.__name__))
    img = cv2.imread(fpath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    assert img is not None, 'failed to read {}'.format(fpath)
    img = img.T
    obj = builder(img)
    assert isinstance(obj, pyrclass)
    with open(pyr_fpath, 'w') as fout:
        pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)
    return obj

def plot_val_with_fft(data, sample_rate=1.0, cut_low=None, cut_high=None,
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

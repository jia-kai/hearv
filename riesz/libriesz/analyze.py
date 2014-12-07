# -*- coding: utf-8 -*-
# $File: analyze.py
# $Date: Sun Dec 07 17:22:32 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .config import floatX
from .utils import plot_val_with_fft

import matplotlib.pyplot as plt
import numpy as np

from abc import ABCMeta, abstractmethod
import logging
logger = logging.getLogger(__name__)

class Motion1DAnalyserBase(object):
    __metaclass__ = ABCMeta
    __slots__ = ['pyr_list']

    @property
    def nr_level(self):
        return self.pyr_list[0].nr_level

    @property
    def nr_frame(self):
        return len(self.pyr_list)

    @abstractmethod
    def local_motion_map(self, frame, level):
        """get local motion map for every pixel at specific level and frame
            index
        :param frame: int, frame index
        :param level: int, level index
        :return: motion map, amplitude map"""

    def __init__(self, pyr_list):
        self.pyr_list = pyr_list
        assert len(pyr_list) >= 2 and \
            all(i.img_shape == pyr_list[0].img_shape for i in pyr_list)


def next_pow2(n):
    v = 1
    while v < n:
        v *= 2
    return v

def avg_spectrum(frame, motion1d):
    """get average spectrum across all levels
    :return: amp, relative freq to sample_rate"""
    padded_width = None
    fft_sum = None
    weight_sum = None
    expect_pad_len = None
    for level in range(motion1d.nr_level):
        seg_size = 2 ** (motion1d.nr_level - level)
        motion, amp = motion1d.local_motion_map(frame, level)
        assert motion.shape == amp.shape and motion.ndim == 2
        amps = np.square(amp)
        padded_arr = np.zeros(shape=next_pow2(motion.shape[1]), dtype=floatX)
        if expect_pad_len is None:
            expect_pad_len = padded_arr.size
            final_fft_len = padded_arr.size / (2 ** motion1d.nr_level)
        else:
            assert expect_pad_len == padded_arr.size
        expect_pad_len /= 2
        for y in range(0, motion.shape[0], seg_size):
            cur_amps = amps[y:y+seg_size]
            amps_sum = np.sum(cur_amps, axis=0)
            cur_weight = np.sum(amps_sum) / cur_amps.size
            cur_motion = np.sum(motion[y:y+seg_size] * cur_amps, axis=0)
            cur_motion /= amps_sum
            if False:
                for i in range(y, y + seg_size, max(seg_size / 4, 1)):
                    plot_val_with_fft(motion[i], show=False)
                plot_val_with_fft(cur_motion)
            padded_arr[:cur_motion.size] = cur_motion
            fft = np.abs(np.fft.fft(padded_arr)[:final_fft_len]) * cur_weight
            if fft_sum is None:
                fft_sum = fft
                weight_sum = cur_weight
            else:
                fft_sum += fft
                weight_sum += cur_weight

    assert final_fft_len == expect_pad_len
    fft = fft_sum / weight_sum
    k = final_fft_len * 2 * (2 ** (motion1d.nr_level - 1))
    freq = np.arange(len(fft), dtype=floatX) / k
    assert freq[-1] < 0.5
    return fft, freq

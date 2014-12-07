# -*- coding: utf-8 -*-
# $File: analyze.py
# $Date: Sun Dec 07 15:44:02 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .fastmath import smooth_pca
from .config import floatX

import matplotlib.pyplot as plt
import numpy as np

import logging
logger = logging.getLogger(__name__)

class PCAMotion1DAnalyser(object):
    __slots__ = ['pyr_list', 'level_local_pca']

    def __init__(self, pyr_list):
        assert len(pyr_list) >= 2 and \
            all(i.img_shape == pyr_list[0].img_shape for i in pyr_list)
        self.pyr_list = pyr_list
        self.level_local_pca = []
        for i in range(self.nr_level):
            self._init_pt_pca(i)

    @property
    def nr_level(self):
        return self.pyr_list[0].nr_level

    @property
    def nr_frame(self):
        return len(self.pyr_list)

    def show_pca(self, level=0, show=True):
        plt.figure()
        pca = self.level_local_pca[level]
        for dx in [0, 1]:
            for dy in [0, 1]:
                x = []
                y = []
                h, w = self.pyr_list[0].levels[level].shape[:2]
                coord = h / 2 + dy, w / 2 + dx
                for i in self.pyr_list:
                    camp, cx, cy = i.levels[level][coord]
                    x.append(cx)
                    y.append(cy)
                x = np.array(x)
                y = np.array(y)
                pca_x, pca_y = pca[coord]
                plt.subplot(2, 2, dy * 2 + dx + 1)
                plt.scatter(x, y)
                x0 = np.mean(x)
                y0 = np.mean(y)
                plt.scatter([x0], [y0], s=[5])
                plt.plot([x0, x0 + pca_x], [y0, y0 + pca_y])
        if show:
            plt.show()

    def local_motion_map(self, frame, level):
        """:return: motion map, amplitude map"""
        v0 = self.pyr_list[frame].levels[level]
        v1 = self.pyr_list[0].levels[level]
        pca = self.level_local_pca[level]
        md = np.sum((v0[:, :, 1:] - v1[:, :, 1:]) * pca, axis=2)
        return md, (v0[:, :, 0] + v1[:, :, 0]) / 2

    def _init_pt_pca(self, level):
        """compute the PCA of motion of each point in a list of pyramid at given
        level
        :return: array(h, w, 2), last dim is pca (x, y)"""
        pyr_list = self.pyr_list
        logger.info('computing point-wise PCA for {} pyrmids at level {}'.format(
            len(pyr_list), level))
        a = np.concatenate([i.levels[level][:, :, 1:2] for i in pyr_list],
                           axis=2)
        b = np.concatenate([i.levels[level][:, :, 2:3] for i in pyr_list],
                           axis=2)
        a -= np.mean(a, axis=2, keepdims=True)
        b -= np.mean(b, axis=2, keepdims=True)
        w00 = np.sum(np.square(a), axis=2)
        w01 = np.sum(a * b, axis=2)
        w11 = np.sum(np.square(b), axis=2)
        eq_b = -(w00 + w11)
        eq_c = w00 * w11 - w01 * w01
        eq_delta = np.sqrt(np.square(eq_b) - 4 * eq_c)
        eq_x = (-eq_b + eq_delta) / 2
        x = w11 - eq_x
        y = -w01
        k = 1 / np.sqrt(np.square(x) + np.square(y))
        x *= k
        y *= k
        h, w = a.shape[:2]
        result = np.concatenate(
            [x.reshape(h, w, 1), y.reshape(h, w, 1)], axis=2)
        assert np.isfinite(np.sum(result))
        smooth_pca(result)
        self.level_local_pca.append(result)

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
            cur_motion = np.sum(motion[y:y+seg_size] * cur_amps, axis=0) / \
                amps_sum
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

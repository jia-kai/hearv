# -*- coding: utf-8 -*-
# $File: analyze.py
# $Date: Fri Dec 12 00:16:27 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .config import floatX
from .utils import plot_val_with_fft

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window

from abc import ABCMeta, abstractmethod
import logging
logger = logging.getLogger(__name__)

class Motion1DAnalyserBase(object):
    """analyze local 1D motion"""
    __metaclass__ = ABCMeta

    start_idx = 0
    """actual index of first frame in pyr_list"""

    ref = None
    """reference frame"""

    pyr_list = None
    """current active frame"""

    def __init__(self, pyr_list):
        self.pyr_list = pyr_list
        assert len(pyr_list) >= 2 and \
            all(i.img_shape == pyr_list[0].img_shape for i in pyr_list)
        self.ref = self.pyr_list[0]
        self.__cached_motion = [None] * len(pyr_list)

    @property
    def nr_level(self):
        return self.pyr_list[0].nr_level

    @property
    def nr_frame(self):
        return len(self.pyr_list) + self.start_idx

    @abstractmethod
    def _local_motion_map(self, frame_idx, level):
        """:param frame_idx: the actual idx in self.pyr_list"""
        pass

    def local_motion_map(self, frame_idx, level):
        """get local motion map for every pixel at specific level and frame
            index
        :param frame: int, frame index
        :param level: int, level index
        :return: motion, amp"""
        frame_idx -= self.start_idx
        assert frame_idx >= 0 and frame_idx < len(self.pyr_list)
        if self.__cached_motion[frame_idx]:
            d = self.__cached_motion[frame_idx]
        else:
            d = self.__cached_motion[frame_idx] = dict()
        v = d.get(level)
        if v is None:
            v = self._local_motion_map(frame_idx, level)
            d[level] = v
        motion, amp = v
        assert motion.shape == amp.shape and motion.ndim == 2
        assert np.min(motion) < 0 and np.min(amp) > 0
        return motion, amp

    def add_frame(self, new_pyr):
        """move forward by adding a new pyramid"""
        assert type(new_pyr) is type(self.pyr_list[0])
        del self.pyr_list[0]
        del self.__cached_motion[0]
        self.pyr_list.append(new_pyr)
        self.__cached_motion.append(None)
        self.start_idx += 1


def next_pow2(n):
    v = 1
    while v < n:
        v *= 2
    return v

def find_optimal_fft_size(val):
    while True:
        p = 2
        v0 = val
        while val > 1:
            while val % p == 0:
                val /= p
            p += 1
        if p - 1 <= 5:
            return v0
        val = v0 + 1

class AvgSpectrum(object):
    _nr_adj_frame = None
    _sample_rate = None
    _target_duration = None

    _last_level_nr_sample = None

    window_type = 'hann'
    vert_group_size = 6
    """size of vertical group"""

    _real_sample_rate = None
    """actual sample rate of last level in the pyramid"""

    def __init__(self, nr_adj_frame, sample_rate, target_duration=None):
        """:param nr_adj_frame: number of adjacent frames to be used
        :param sample_rate: sample rate for columns in one frame
        :param target_duration: target duration for each frame"""
        self._nr_adj_frame = int(nr_adj_frame)
        self._sample_rate = float(sample_rate)
        self._target_duration = target_duration
        assert self._nr_adj_frame > 0

    def __call__(self, motion_ana, frame_idx):
        # nr_sample in last level
        ll_nr_sample = motion_ana.local_motion_map(
            frame_idx, motion_ana.nr_level - 1)[0].shape[1]
        sample_rate = self._sample_rate / (2 ** (motion_ana.nr_level - 1))
        if self._real_sample_rate is None:
            self._real_sample_rate = sample_rate
        assert self._real_sample_rate == sample_rate
        if self._target_duration:
            tgt_ll_nr_sample = int(float(self._target_duration) * sample_rate)
        else:
            tgt_ll_nr_sample = ll_nr_sample
        assert tgt_ll_nr_sample >= ll_nr_sample

        if self._last_level_nr_sample is None:
            if False:
                padded_width = next_pow2(tgt_ll_nr_sample * self._nr_adj_frame)
                if padded_width / 2 >= ll_nr_sample * self._nr_adj_frame:
                    padded_width /= 2
                if self._last_level_nr_sample is None:
                    self._last_level_nr_sample = padded_width
                assert self._last_level_nr_sample == padded_width
            else:
                padded_width = find_optimal_fft_size(tgt_ll_nr_sample)
                self._last_level_nr_sample = padded_width
            logger.info('padding ratio: {}/{}={}'.format(
                padded_width, ll_nr_sample,
                float(padded_width) / ll_nr_sample))

        padded_width = self._last_level_nr_sample
        padded_width *= 2 ** (motion_ana.nr_level - 1)
        spec_sum = None
        weight_sum = None
        for level in range(motion_ana.nr_level):
            frames = [motion_ana.local_motion_map(frame_idx + i, level)
                      for i in range(self._nr_adj_frame)]
            for spec, weight in self._analyze_one_scale(frames, padded_width):
                weight = float(weight)
                spec *= weight
                if spec_sum is None:
                    spec_sum = spec.copy()
                    weight_sum = weight
                else:
                    spec_sum += spec
                    weight_sum += weight
            padded_width /= 2
        amp = spec_sum / weight_sum
        assert len(amp) == self._last_level_nr_sample / 2
        freq = np.arange(len(amp), dtype=floatX)
        freq *= self._real_sample_rate / (len(amp) * 2)
        return amp, freq

    def _analyze_one_scale(self, frames, nr_sample):
        """:return: spectrum, weight"""
        motion = [i[0] for i in frames]
        amps = [np.square(i[1]) for i in frames]
        signal = np.empty(shape=nr_sample, dtype=floatX)
        window = None
        for y in range(0, frames[0][0].shape[1], self.vert_group_size):
            all_weight = []
            signal.fill(0)
            for fidx in range(len(frames)):
                cur_amps = amps[fidx][y:y+self.vert_group_size]
                amps_sum = np.sum(cur_amps, axis=0)
                assert amps_sum.ndim == 1
                all_weight.append(np.sum(amps_sum) / cur_amps.size)
                x0 = fidx * nr_sample / len(frames)
                x1 = x0 + amps_sum.size
                xnext = (fidx + 1) * nr_sample / len(frames)
                assert xnext >= x1
                cur_motion = np.sum(motion[fidx][y:y+self.vert_group_size] *
                                    cur_amps, axis=0) / amps_sum
                while x1 < xnext:
                    signal[x1 - cur_motion.size:x1] = cur_motion
                    x1 += cur_motion.size
                x1 -= cur_motion.size
                if x1 - x0 > cur_motion.size:
                    logger.warn('duplicate signal')
                if window is None:
                    window = get_window(self.window_type, x1 - x0)
                signal[x0:x1] *= window
                #plot_val_with_fft(cur_motion, self._sample_rate, show=False)
                #plot_val_with_fft(signal, self._sample_rate)
            cur_weight = np.mean(all_weight)
            fft_amp = np.abs(np.fft.fft(signal)[:self._last_level_nr_sample/2])
            yield fft_amp, cur_weight


def avg_spectrum(frame_idx, motion1d, padding_size=32, window_type='hann'):
    """get average spectrum across all levels; 
    :return: amp, relative freq to sample_rate"""
    padded_width = None
    fft_sum = None
    weight_sum = None
    expect_pad_len = None
    for level in range(motion1d.nr_level):
        remain_size = 2 ** (motion1d.nr_level - level - 1)
        seg_size = 6
        motion, amp = motion1d.local_motion_map(frame_idx, level)
        assert motion.shape == amp.shape and motion.ndim == 2
        amps = np.square(amp)
        del amp
        padded_arr = np.zeros(
            shape=next_pow2(motion.shape[1]) + padding_size * remain_size,
            dtype=floatX)
        window = get_window(window_type, padded_arr.size)
        if expect_pad_len is None:
            expect_pad_len = padded_arr.size
            final_fft_len = padded_arr.size / (2 ** motion1d.nr_level)
            assert final_fft_len * remain_size * 2 == expect_pad_len
        else:
            assert expect_pad_len == padded_arr.size
        expect_pad_len /= 2
        for y in range(0, motion.shape[0], seg_size):
            cur_amps = amps[y:y+seg_size]
            amps_sum = np.sum(cur_amps, axis=0)
            cur_weight = np.sum(amps_sum) / cur_amps.size
            cur_motion = np.sum(motion[y:y+seg_size] * cur_amps, axis=0)
            cur_motion *= window[:cur_motion.size]
            cur_motion /= amps_sum
            assert cur_motion.size == motion.shape[1]
            padded_arr[:cur_motion.size] = cur_motion
            if False:
                for i in range(y, y + seg_size, max(seg_size / 3, 1)):
                    plot_val_with_fft(motion[i], show=False)
                plot_val_with_fft(padded_arr)
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

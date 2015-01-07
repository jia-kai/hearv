# -*- coding: utf-8 -*-
# $File: analyze.py
# $Date: Wed Jan 07 01:30:31 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .config import floatX
from .utils import plot_val_with_fft, get_env_config

import matplotlib.pyplot as plt
import numpy as np

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

    def __call__(self, frame_idx, level):
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
    EPS = np.finfo(floatX).tiny

    _nr_adj_frame = None
    _sample_rate = None
    _target_duration = None

    _last_level_nr_sample = None

    vert_group_size = int(get_env_config('VERT_GROUP_SIZE', 150))
    """size of vertical group"""

    vert_group_exp_decay = int(get_env_config('VERT_GROUP_EXP_DECAY', 0))

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
        ll_nr_sample = motion_ana(
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
                padded_width = find_optimal_fft_size(
                    tgt_ll_nr_sample * self._nr_adj_frame)
                self._last_level_nr_sample = padded_width
            nopadding_width = ll_nr_sample * self._nr_adj_frame
            logger.info('padding ratio: {}/{}={}'.format(
                padded_width, nopadding_width,
                float(padded_width) / nopadding_width))

        padded_width = self._last_level_nr_sample
        padded_width *= 2 ** (motion_ana.nr_level - 1)
        spec_sum = None
        weight_sum = None
        vg_orig = self.vert_group_size
        for level in range(motion_ana.nr_level):
            frames = [motion_ana(frame_idx + i, level)
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
            if self.vert_group_exp_decay:
                self.vert_group_size /= 2
        self.vert_group_size = vg_orig
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
        for y in range(0, amps[0].shape[0], self.vert_group_size):
            all_weight = []
            signal.fill(0)
            x0 = 0
            for fidx in range(len(frames)):
                cur_amps = amps[fidx][y:y+self.vert_group_size]
                amps_sum = np.sum(cur_amps, axis=0)
                assert amps_sum.ndim == 1
                all_weight.append(np.sum(amps_sum) / cur_amps.size)
                x1 = x0 + amps_sum.size
                cur_motion = np.sum(motion[fidx][y:y+self.vert_group_size] *
                                    cur_amps, axis=0) / (amps_sum + self.EPS)
                signal[x0:x1] = cur_motion
                x0 = x1
                #plot_val_with_fft(cur_motion, self._sample_rate, show=False)
            if window is None:
                window = RawSpectrogram.make_window(x0)
            signal[:x0] *= window
            #plot_val_with_fft(signal, self._sample_rate)
            cur_weight = np.mean(all_weight)
            fft_amp = np.abs(np.fft.fft(signal)[:self._last_level_nr_sample/2])
            yield fft_amp, cur_weight

class RawSpectrogram(object):
    _window = None
    _step = None
    def __init__(self, length, step, win_type='hamming'):
        step = int(step)
        assert step > 0
        self._step = step
        self._window = self.make_window(length, win_type)

    @classmethod
    def make_window(cls, length, win_type='hamming'):
        length = int(length)
        if win_type == 'hamming':
            a = 0.54
            b = -0.46
        else:
            assert win_type == 'hanning'
            a = 0.5
            b = -0.5

        x = np.arange(length, dtype=floatX) * 2 + 1
        x *= np.pi / length
        w = a + np.cos(x) * b
        w /= np.sqrt(4 * a * a + 2 * b * b)

        if length % 4 == 0:
            # check for normalization
            step = length / 4
            for i in range(length):
                p = i
                while p >= step:
                    p -= step
                s = 0
                while p < length:
                    s += w[p] * w[p]
                    p += step
                assert abs(s - 1) < 1e-5
        else:
            logger.warn(
                '\n\n!!!length not multiple of 4 for scaled window, '
                'do not use recon')

        return w

    def __call__(self, signal, nr_time=None):
        """return: 2D matrix indexed by (time, frequency)"""
        assert signal.ndim == 1 and signal.size >= self._window.size
        if nr_time is None:
            nr_time = (signal.size - self._window.size) / self._step + 1
        result = np.empty((nr_time, self._window.size), dtype=floatX)
        pos = 0
        for i in range(nr_time):
            p = i * self._step
            sub = signal[p:p+self._window.size]
            result[i] = np.abs(np.fft.fft(sub * self._window))
        return result

    @property
    def win_size(self):
        return self._window.size


class AdjSTFT(object):
    """short-time fourier transform for adjacent frams"""

    EPS = np.finfo(floatX).tiny

    _sample_rate = None
    _step_factor = None
    """step size relative to window length"""

    _nr_spec_per_frame = None
    """number of time points for spectrogram for each frame"""

    _get_spectrogram = None
    """list of :class:`RawSpectrogram` objects to calculate spectrogram for
        each scale"""

    _ll_nr_sample = None
    """number of samples in last level"""

    _ll_sample_rate = None
    """sample rate in last level"""

    vert_group_size = 6
    """size of vertical group"""

    def __init__(self, sample_rate, target_duration):
        """:param sample_rate: sample rate for level0
        :param target_duration: duration for each frame"""
        self._sample_rate = float(sample_rate)
        self._step_factor = 0.25 / (target_duration * self._sample_rate)

    def __call__(self, motion_ana, frame_idx):
        if self._get_spectrogram is None:
            self._get_spectrogram = []
            self._step_factor *= motion_ana(frame_idx, 0)[0].shape[0]
            self._nr_spec_per_frame = int(1 / self._step_factor)
            logger.info('stretch={} nr_spec_per_frame={}'.format(
                0.25 / self._step_factor, self._nr_spec_per_frame))
            for level in range(motion_ana.nr_level):
                win_size = motion_ana(frame_idx, level)[0].shape[1]
                step = int(win_size * self._step_factor)
                logger.info('level {}: spectrogram '
                            'win_size={} step={}'.format(level, win_size, step))
                assert step * self._nr_spec_per_frame <= win_size
                self._get_spectrogram.append(RawSpectrogram(win_size, step))
            self._ll_nr_sample = self._get_spectrogram[-1].win_size
            self._ll_sample_rate = self._sample_rate / (
                2 ** (motion_ana.nr_level - 1))
            logger.info('usable sample rate: {}'.format(self._ll_sample_rate))

        spec_sum = None
        weight_sum = None
        for level in range(motion_ana.nr_level):
            frames = [motion_ana(frame_idx + i, level) for i in range(2)]
            for spec, weight in self._analyze_one_scale(
                    frames, self._get_spectrogram[level]):
                weight = float(weight)
                spec *= weight
                if spec_sum is None:
                    spec_sum = spec.copy()
                    weight_sum = weight
                else:
                    spec_sum += spec
                    weight_sum += weight

        spec_sum /= weight_sum
        N = self._ll_nr_sample / 2
        assert spec_sum.shape[1] == N
        freq = np.arange(N, dtype=floatX)
        freq *= self._ll_sample_rate / (N * 2)
        return spec_sum, freq

    def _analyze_one_scale(self, frames, spectrogram):
        assert len(frames) == 2
        motion = [i[0] for i in frames]
        amps = [np.square(i[1]) for i in frames]
        signal = np.empty(shape=spectrogram.win_size*2, dtype=floatX)
        for y in range(0, amps[0].shape[0], self.vert_group_size):
            all_weight = []
            signal.fill(0)
            x0 = 0
            for fidx in range(len(frames)):
                cur_amps = amps[fidx][y:y+self.vert_group_size]
                amps_sum = np.sum(cur_amps, axis=0)
                assert amps_sum.ndim == 1
                all_weight.append(np.sum(amps_sum) / cur_amps.size)
                x1 = x0 + amps_sum.size
                cur_motion = np.sum(motion[fidx][y:y+self.vert_group_size] *
                                    cur_amps, axis=0) / (amps_sum + self.EPS)
                signal[x0:x1] = cur_motion
                x0 = x1
            cur_weight = np.mean(all_weight)
            cur_spec = spectrogram(signal, nr_time=self._nr_spec_per_frame)
            yield cur_spec[:, :self._ll_nr_sample/2], cur_weight

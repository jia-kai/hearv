# -*- coding: utf-8 -*-
# $File: recon.py
# $Date: Thu Dec 11 23:46:59 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .config import floatX
from .utils import plot_val_with_fft

import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, get_window

import logging
logger = logging.getLogger(__name__)

class AudioRecon(object):
    _loc = 0
    """signal writing location"""

    _tot_err = 0
    """total error for sample number"""

    _sample_per_frame = 0
    _frame_length = 0
    _prev_amp = None

    filter_cutoff = 2000
    filter_width = 2000

    amp_smooth = 0.3

    window_name = 'triang'
    _window = None

    _init_phase = None
    _frame_num = 0

    def __init__(self, fps, sample_rate, target_energy, **kwarg):
        self._frame_length = 1.0 / fps
        self._sample_rate = sample_rate
        self._sample_per_frame = sample_rate / fps
        assert self._sample_per_frame > 1
        self._signal = np.zeros(shape=self._sample_per_frame, dtype=floatX)
        self._target_energy = target_energy

        for k, v in kwarg.iteritems():
            assert getattr(self, k, None) is not None
            setattr(self, k, v)

    def add(self, freq, amp):
        if self._prev_amp is None:
            self._prev_amp = np.zeros_like(amp)
        assert self._prev_amp.shape == amp.shape and freq.shape == amp.shape
        self._prev_amp *= self.amp_smooth
        amp *= 1 - self.amp_smooth
        self._prev_amp += amp
        amp = self._prev_amp
        res = freq[1] - freq[0]
        duration = 1 / res
        assert duration >= self._frame_length
        assert freq[-1] <= self._sample_rate / 2

        # get nr_sample and nr_sample_ext
        nr_sample = int(self._sample_rate * duration)
        if nr_sample % 2:
            nr_sample += 1

        spectrum = np.zeros(shape=nr_sample, dtype=complex)
        assert freq[0] == 0
        assert len(amp) * 2 < spectrum.size
        # discard DC
        amp = amp[1:]
        spectrum[1:len(amp)+1] = amp
        s0 = spectrum[1:spectrum.size/2]
        if self._init_phase is None:
            self._init_phase = np.random.normal(scale=np.pi * 2, size=s0.size)
            self._phase_shift = np.arange(s0.size, dtype=floatX)
            self._phase_shift *= -np.pi * 2 / s0.size
        s0 *= np.exp((self._init_phase + self._phase_shift * self._loc) * 1j)
        spectrum[-len(s0):] = np.conjugate(s0)[::-1]

        recon = np.real(np.fft.ifft(spectrum))
        if self._window is None:
            win_size = int(self._sample_per_frame * 2)
            assert recon.size <= win_size
            self._window = get_window(self.window_name, win_size)[:recon.size]
        #self._plot_window(recon)
        signal = np.tile(recon, 60)
        from scipy.io import wavfile
        wavfile.write('/tmp/singal.wav', 44100,
                      (signal / np.max(np.abs(signal)) * 20000).astype('int16'))
        plot_val_with_fft(signal, self._sample_rate)
        recon *= self._window
        self._ensure_signal_size(recon.size)
        self._signal[self._loc:self._loc+recon.size] += recon
        self._frame_num += 1
        self._loc = int(self._frame_num * self._sample_per_frame)
        #print nr_sample, self._sample_per_frame
        #plot_val_with_fft(recon, self._sample_rate, show=False)
        #plot_val_with_fft(self._signal[:self._loc], self._sample_rate)

    def _plot_window(self, signal):
        plot_val_with_fft(signal, self._sample_rate, cut_high=1500, cut_low=100,
                          show=False)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(signal)
        plt.subplot(3, 1, 2)
        plt.plot(self._window)
        plt.subplot(3, 1, 3)
        signal = signal * self._window
        plt.plot(signal)
        plot_val_with_fft(signal, self._sample_rate, cut_high=1500, cut_low=100)

    def _ensure_signal_size(self, nr_sample):
        if self._loc + nr_sample >= self._signal.size:
            s1 = np.zeros(self._signal.size * 2, dtype=floatX)
            s1[:self._signal.size] = self._signal
            self._signal = s1
    
    def get_signal(self):
        ripple_db = 60.0
        nyq_rate = self._sample_rate / 2.0
        N, beta = kaiserord(ripple_db, self.filter_width / nyq_rate)
        logger.info('filter size: {}'.format(N))
        taps = firwin(N, self.filter_cutoff / nyq_rate,
                      window=('kaiser', beta))
        signal = self._signal[:self._loc]
        signal = lfilter(taps, 1.0, signal)
        signal *= self._target_energy / np.sqrt(np.mean(np.square(signal)))
        return np.clip(signal, -0.98, 0.98)

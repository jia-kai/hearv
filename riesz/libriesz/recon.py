# -*- coding: utf-8 -*-
# $File: recon.py
# $Date: Mon Jan 05 20:45:51 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .config import floatX
from .utils import plot_val_with_fft
from .analyze import RawSpectrogram

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import kaiserord, lfilter, firwin, get_window
from scipy.interpolate import interp1d

import logging
logger = logging.getLogger(__name__)

def rms(s):
    return np.sqrt(np.mean(np.square(s)))

class AudioReconBase(object):
    _loc = 0
    """signal writing location"""

    _tot_err = 0
    """total error for sample number"""

    _sample_per_frame = 0
    _frame_length = 0
    _prev_amp = None
    _prev_recon_rms = None

    filter_cutoff = 2000
    filter_width = 2000

    amp_smooth = 0.3
    recon_rms_smooth = 0.9

    window_name = 'triang'
    _window = None

    _init_phase = None

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
        signal *= self._target_energy / rms(signal)
        return np.clip(signal, -0.98, 0.98)

    def _get_phase(self, size):
        if self._init_phase is None:
            self._init_phase = np.random.uniform(low=0, high=np.pi * 2, size=size)
            self._phase_shift = np.arange(size, dtype=floatX)
            self._phase_shift *= -np.pi * 2 / size
        return np.exp((self._init_phase + self._phase_shift * self._loc) * 1j)

    def _smoothed_amp(self, amp):
        if self._prev_amp is None:
            self._prev_amp = amp
            return amp
        assert self._prev_amp.shape == amp.shape
        self._prev_amp *= self.amp_smooth
        amp *= 1 - self.amp_smooth
        self._prev_amp += amp
        return self._prev_amp

    def _smoothed_recon(self, recon):
        cur_rms = rms(recon)
        if self._prev_recon_rms is None:
            self._prev_recon_rms = cur_rms
            return recon
        a = self.recon_rms_smooth
        b = 1 - a
        self._prev_recon_rms = self._prev_recon_rms * a + cur_rms * b
        recon *= self._prev_recon_rms / cur_rms
        return recon

class AudioReconOverlap(AudioReconBase):
    _frame_num = 0
    def add(self, freq, amp):
        amp = self._smoothed_amp(amp)
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
        s0 *= self._get_phase(s0.size)
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

class AudioReconFreqInterpolate(AudioReconBase):
    def add(self, freq, amp):
        amp = self._smoothed_amp(amp)
        freq_orig = freq
        amp_orig = amp
        res = freq[1] - freq[0]
        duration = 1 / res
        assert freq[-1] <= self._sample_rate / 2
        sample_rate = freq[-1] * 2

        nr_sample = amp.size * 2

        # discard DC
        amp = amp[1:]

        spectrum = np.zeros(shape=nr_sample, dtype=complex)
        assert len(amp) + 1 == spectrum.size / 2
        s0 = spectrum[1:len(amp)+1]
        s0[:] = amp
        s0 *= self._get_phase(s0.size)
        spectrum[-len(s0):] = np.conjugate(s0)[::-1]

        recon = self._smoothed_recon(np.real(np.fft.ifft(spectrum)))
        
        if True:
            signal = np.tile(recon, 60)
            N = len(amp)
            plot_val_with_fft(signal[:N], sample_rate, show=False)
            plot_val_with_fft(signal[N/4:N/4+N], sample_rate, show=False)
            from scipy.io import wavfile
            wavfile.write('/tmp/signal.wav', int(sample_rate),
                          (signal / np.max(np.abs(signal)) *
                           20000).astype('int16'))
            plot_val_with_fft(signal, sample_rate)
        self._ensure_signal_size(recon.size)
        self._signal[self._loc:self._loc+nr_sample] += recon
        self._loc += nr_sample / 2
        #plot_val_with_fft(recon, self._sample_rate, show=False)
        #plot_val_with_fft(self._signal[:self._loc+nr_sample/2],
        #                  self._sample_rate)


class AudioRecon(object):
    EPS = np.finfo(floatX).tiny

    _target_frame_duration = None
    _frame_freq_axis = None
    _signal = None
    _signal_end = None
    _sample_rate = None
    _window = None
    _proc_logger = None

    max_nr_iter = 200

    _prev_amp = None
    _all_amp = None

    amp_smooth = None

    def __init__(self, target_frame_duration, proc_logger=None):
        self._target_frame_duration = float(target_frame_duration)
        self._all_amp = list()
        self._proc_logger = proc_logger

    def add(self, freq, amp):
        if self._frame_freq_axis is None:
            self._frame_freq_axis = freq
            duration = 1 / freq[1]
            self._sample_rate = freq[-1] * 2
            logger.info('duration: sampled={} target={}'.format(
                duration, self._target_frame_duration))
            assert duration < self._target_frame_duration
        else:
            assert np.all(freq == self._frame_freq_axis)

        time = len(self._all_amp) * self._target_frame_duration
        loc = time * self._sample_rate
        amp = self._smoothed_amp(amp)
        self._all_amp.append(amp.copy())
        recon = self._recon_single(amp)
        N = recon.size
        self._ensure_signal_size(loc + N)
        self._signal[loc:loc+N] = recon
        self._signal_end = loc + N
        if self._proc_logger:
            self._proc_logger.add_frame(
                freq, amp,
                np.arange(loc, loc+N) / self.sample_rate, recon)
        #plot_val_with_fft(self._signal, self._sample_rate)

    def get_signal(self):
        return self._global_opt()

    def _recon_single(self, amp):
        nr_sample = len(amp) * 2
        assert nr_sample % 4 == 0
        if self._window is None:
            self._window = RawSpectrogram.make_window(nr_sample)
        spectrum = np.empty(nr_sample, dtype=complex)
        spectrum[0] = 0
        spectrum[len(amp)] = 0
        s0 = spectrum[1:len(amp)]
        s0[:] = amp[1:]
        s0 *= np.exp(np.random.uniform(0, np.pi*2, len(s0)) * 1j)
        spectrum[-len(s0):] = np.conjugate(s0)[::-1]
        return np.real(np.fft.ifft(spectrum))

    def _global_opt(self):
        N = self._window.size
        step = N / 4
        signal = self._signal[:self._signal_end]
        time_grid = range(0, signal.size - N + 1, step)
        all_amp = self._get_interpolated_all_amp(time_grid)
        all_amp = [np.concatenate((i, [0], i[1:][::-1])) for i in all_amp]
        for iter_idx in range(self.max_nr_iter):
            if (iter_idx <= 8 or iter_idx % 25 == 0) and \
                    self._proc_logger:
                self._proc_logger.add_global_opt(
                    iter_idx, np.arange(len(signal)) / self.sample_rate,
                    signal)
            err = 0
            signal_next = np.zeros_like(signal)
            for i, amp in zip(time_grid, all_amp):
                cur = np.fft.fft(signal[i:i+N] * self._window)
                err += np.sqrt(np.sum(np.square(np.abs(cur) - amp)))
                cur *= amp / (np.abs(cur) + self.EPS)
                recon = np.real(np.fft.ifft(cur)) * self._window
                signal_next[i:i+N] += recon
            signal = signal_next
            if iter_idx % 50 == 0:
                logger.info('err at iter {}: {}'.format(
                    iter_idx, err / len(time_grid)))

        return signal

    def _get_interpolated_all_amp(self, time_grid):
        OFFSET = 1
        def tolin(v):
            return np.log(v + OFFSET)
        def fromlin(v):
            return np.exp(v) - OFFSET
        def add(k):
            rst.append(fromlin(prev_lin * (1 - k) + cur_lin * k))
        assert len(self._all_amp) >= 2
        time_grid = np.array(time_grid) / self._sample_rate
        assert time_grid[0] == 0
        frame_duration = self._target_frame_duration
        assert time_grid[-1] < (len(self._all_amp) - 1) * frame_duration
        amp_iter = iter(self._all_amp)
        rst = [next(amp_iter)]
        prev_lin = tolin(rst[0])
        prev_time = 0
        cur_lin = tolin(next(amp_iter))
        cur_time = frame_duration

        def plot():
            N = 10
            x = self._frame_freq_axis[:N]
            plt.plot(x, fromlin(prev_lin[:N]), label='prev')
            plt.plot(x, fromlin(cur_lin[:N]), label='next')
            plt.plot(x, rst[-1][:N], label='interp')
            plt.legend(loc='best')
            plt.show()
        plot = lambda: None

        for t in time_grid[1:]:
            if t > cur_time:
                prev_lin, prev_time = cur_lin, cur_time
                cur_lin = tolin(next(amp_iter))
                cur_time += frame_duration
            assert t > prev_time and t <= cur_time
            add((t - prev_time) / frame_duration)
            plot()

        assert len(rst) == len(time_grid)

        return rst

    def _smoothed_amp(self, amp):
        if not self.amp_smooth:
            return amp
        if self._prev_amp is None:
            self._prev_amp = amp
            return amp
        assert self._prev_amp.shape == amp.shape
        self._prev_amp *= self.amp_smooth
        amp *= 1 - self.amp_smooth
        self._prev_amp += amp
        return self._prev_amp

    def _ensure_signal_size(self, tsize):
        if self._signal is None:
            self._signal = np.zeros(tsize, dtype=floatX)
            return
        if tsize >= self._signal.size:
            s1 = np.zeros(tsize * 2, dtype=floatX)
            s1[:self._signal.size] = self._signal
            self._signal = s1

    @property
    def sample_rate(self):
        return self._sample_rate

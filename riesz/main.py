#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: main.py
# $Date: Sat Jan 03 22:36:01 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from libriesz.utils import CachedResult, plot_val_with_fft
from libriesz.analyze import AvgSpectrum
from libriesz.recon import AudioRecon
from libriesz.denoise import Denoise

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.io import wavfile

import json
import os
import argparse
import logging
logger = logging.getLogger(__name__)

class CachedRieszPyramid(CachedResult):
    _type = 'riesz-pyr'

    def __init__(self):
        from libriesz.pyramid import RieszPyramidBuilder
        self.builder = RieszPyramidBuilder()

    def _getter(self, fpath):
        logger.info('computing riesz pyramid: {}'.format(fpath))
        img = cv2.imread(fpath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert img is not None, 'failed to read {}'.format(fpath)
        img = img.T
        return self.builder(img)

class CachedFrameSpectrum(CachedResult):
    _type = 'spectrum'

    def __init__(self, motion_ana, args):
        self._cache_dir = os.path.dirname(args.img[0])
        self.avg_spectrum = AvgSpectrum(
            args.nr_adj_frame, 1 / args.line_delay)
        self.motion_ana = motion_ana

    def _getter(self, fidx):
        return self.avg_spectrum(self.motion_ana, fidx)

class ReconProcJsonLogger(object):
    def __init__(self, fpath, video_link):
        self.frames = []
        self.glbl = []
        self.frame_mag_range = [float('inf'), float('-inf')]
        self.glbl_sig_range = [float('inf'), float('-inf')]
        self.fpath = fpath
        self.video_link = video_link

    def __del__(self):
        logger.info('save recon process json to {}'.format(self.fpath))
        with open(self.fpath, 'w') as fout:
            json.dump(self.get_json_obj(), fout)

    def _update_range(self, t, v):
        min0, max0 = t
        t[:] = map(float, [min(np.min(v), min0), max(np.max(v), max0)])


    def add_frame(self, freq, mag, t, y):
        self.frame_freq = freq
        self.frames.append({
            'spectrum': mag.tolist(),
            'signal': zip(t.tolist(), y.tolist())
        })
        self._update_range(self.frame_mag_range, mag)

    def add_global_opt(self, iter_idx, t, y):
        self.glbl_time = t
        self.glbl.append({
            'iter': iter_idx,
            'signal': y.tolist()
        })
        self._update_range(self.glbl_sig_range, y)

    def get_json_obj(self):
        return {
            'frames': self.frames,
            'global': self.glbl,
            'frame_config': {
                'mag_range': self.frame_mag_range,
                'spectrum_freq': self.frame_freq.tolist()
            },
            'global_config': {
                'time': self.glbl_time.tolist(),
                'signal_range': self.glbl_sig_range,
            },
            'video_link': self.video_link
        }

def main():
    parser = argparse.ArgumentParser(
        description='analyze audio from video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--line_delay', type=float, default=15.93e-6)
    parser.add_argument('--cut_low', type=float, default=150,
                        help='low freq cutoff')
    parser.add_argument('--cut_high', type=float, default=1000,
                        help='high freq cutoff')
    parser.add_argument('--frame_win_length', type=int, default=1,
                        help='frame window length for analysing')
    parser.add_argument('--fps', type=float, default=59.940)
    parser.add_argument('--recon', help='path for reconstruction output')
    parser.add_argument('--recon_nr_iter', type=int)
    parser.add_argument('--target_energy', type=float, default=0.2,
                        help='target energy for reconstruction output')
    parser.add_argument('--disp', action='store_true',
                        help='display spectrum for each frame (enabled when'
                        ' recon not supplied)')
    parser.add_argument('--disp_all', action='store_true',
                        help='display all spectrums in a single plot')
    parser.add_argument('--nr_adj_frame', type=int, default=1,
                        help='number of adjacent frames to be used for '
                        'spectrum analysis')
    parser.add_argument('--frame_duration_scale', type=float, default=1,
                        help='scale frame duration for spectrum analysis')
    parser.add_argument('--disable_spectrum_cache', action='store_true')
    parser.add_argument('--force_spectrum_cache', action='store_true')
    parser.add_argument('--noise_frame', type=int, default=5,
                        help='number of frames used for noise sampling')
    parser.add_argument('--json_out_video_link',
                        help='video_link used in --json_out')
    parser.add_argument('--json_out',
                        help='output processing details to json file')
    parser.add_argument('img', nargs='+')
    args = parser.parse_args()

    win_length = args.frame_win_length + args.nr_adj_frame + 1
    get_riesz_pyr = CachedRieszPyramid()
    pyr_list = (map(get_riesz_pyr, args.img[:win_length]))
    motion_ana = pyr_list[0].make_motion_analyser(pyr_list)
    avg_spectrum = CachedFrameSpectrum(motion_ana, args)
    if args.disable_spectrum_cache:
        avg_spectrum.enabled = False

    proc_logger = None
    if args.json_out:
        assert args.recon and args.json_out_video_link
        proc_logger = ReconProcJsonLogger(
            args.json_out, args.json_out_video_link)

    if args.recon:
        recon = AudioRecon(1 / args.fps, proc_logger)
        if args.recon_nr_iter:
            recon.max_nr_iter = args.recon_nr_iter
    else:
        recon = None

    sample_rate = 1.0 / args.line_delay

    if args.disp_all:
        fig_all = plt.figure()
        ax_all = fig_all.add_subplot(111)
        ax_all.set_xlabel('freq')
        ax_all.set_ylabel('amp')

    noise_spec = []
    assert args.noise_frame >= 0

    for i in range(1, len(args.img) - args.nr_adj_frame + 1):
        logger.info('frame {}'.format(i))
        amp, freq = avg_spectrum(i)
        if i + 1 < len(args.img) and not args.force_spectrum_cache:
            motion_ana.add_frame(get_riesz_pyr(args.img[i + 1]))

        if i - 1 < args.noise_frame:
            logger.info('used as noise sample')
            noise_spec.append(amp)
            continue
        elif i - 1 == args.noise_frame:
            cut_low = min(np.nonzero(freq >= args.cut_low)[0])
            cut_high = min(np.nonzero(freq >= args.cut_high)[0])
            if noise_spec:
                denoise = Denoise(noise_spec)
                if False:
                    plt.figure()
                    plt.plot(freq, denoise._avg_spec)
                    plt.show()
            else:
                denoise = lambda v: v
        amp = denoise(amp)
        amp[:cut_low] = 0
        amp[cut_high:] = 0
        if i == 1:
            logger.info('freq_resolution={}'.format(i, freq[1] - freq[0]))
        if args.disp:
            logger.info('disp spectrum')
            plt.figure()
            plt.plot(freq[cut_low:cut_high], amp[cut_low:cut_high])
            plt.show()
        if args.disp_all:
            ax_all.plot(freq[cut_low:cut_high], amp[cut_low:cut_high],
                        label='frame{}'.format(i))
        if recon:
            recon.add(freq, amp)

    if args.disp_all:
        ax_all.legend(loc='best')
        ax_all.grid(True, which='both')
        plt.show()

    if recon:
        signal = recon.get_signal()
        signal = signal / np.max(np.abs(signal))
        wavfile.write(args.recon, int(recon.sample_rate),
                      (signal * 20000).astype('int16'))
        plot_val_with_fft(signal, sample_rate=recon.sample_rate)

if __name__ == '__main__':
    main()

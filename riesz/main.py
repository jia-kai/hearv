#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: main.py
# $Date: Tue Dec 09 10:41:35 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from libriesz.utils import get_riesz_pyr_with_cache, plot_val_with_fft
from libriesz.analyze import avg_spectrum
from libriesz.recon import AudioRecon

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import argparse
import logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='analyze audio from video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--line_delay', type=float, default=15e-6)
    parser.add_argument('--cut_low', type=float, default=100)
    parser.add_argument('--cut_high', type=float, default=2000)
    parser.add_argument('--frame_win_length', type=int, default=10,
                        help='frame window length for analysing')
    parser.add_argument('--fps', type=float, default=59.940)
    parser.add_argument('--recon', help='path for reconstruction output')
    parser.add_argument('--recon_sr', type=int, default=44100,
                        help='reconstruction sample rate')
    parser.add_argument('--target_energy', type=float, default=0.2,
                        help='target energy for reconstruction output')
    parser.add_argument('--disp', action='store_true',
                        help='display spectrum for each frame (enabled when'
                        ' recon not supplied)')
    parser.add_argument('img', nargs='+')
    args = parser.parse_args()

    win_length = args.frame_win_length
    pyr_list = (map(get_riesz_pyr_with_cache, args.img[:win_length]))
    motion1d = pyr_list[0].make_motion_analyser(pyr_list)

    if args.recon:
        recon = AudioRecon(args.fps, args.recon_sr, args.target_energy,
                           filter_cutoff=args.cut_high)
    else:
        recon = None

    sample_rate = 1.0 / args.line_delay
    for i in range(1, len(args.img)):
        logger.info('frame {}'.format(i))
        if i >= win_length:
            motion1d.move_forward(get_riesz_pyr_with_cache(args.img[i]))
        amp, freq = avg_spectrum(min(i, win_length - 1), motion1d)
        freq *= sample_rate
        cut_low = min(np.nonzero(freq >= args.cut_low)[0])
        amp[:cut_low] = 0
        avg = np.mean(sorted(amp)[len(amp)/4:-len(amp)/4])
        amp = np.clip(amp - avg, 0, np.max(amp))
        amp = np.power(60, amp)
        if i == 1:
            logger.info('freq_resolution={}'.format(i, freq[1] - freq[0]))
        if not recon or args.disp:
            logger.info('disp spectrum')
            cut_high = min(np.nonzero(freq >= args.cut_high)[0])
            plt.figure()
            plt.plot(freq[cut_low:cut_high], amp[cut_low:cut_high])
            plt.show()
        if recon:
            recon.add(freq, amp)

    if recon:
        signal = recon.get_signal()
        wavfile.write(args.recon, args.recon_sr,
                      (signal * 32767).astype('int16'))
        plot_val_with_fft(signal, sample_rate=args.recon_sr, cut_high=2500)

if __name__ == '__main__':
    main()

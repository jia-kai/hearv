#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: main.py
# $Date: Sun Dec 07 18:47:40 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from libriesz.utils import get_riesz_pyr_with_cache, plot_val_with_fft
from libriesz.analyze import avg_spectrum

import matplotlib.pyplot as plt
import numpy as np

import argparse
import logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--line_delay', type=float, default=15e-6)
    parser.add_argument('--cut_low', type=float, default=100)
    parser.add_argument('--cut_high', type=float, default=2000)
    parser.add_argument('img', nargs='+')
    args = parser.parse_args()

    pyr_list = (map(get_riesz_pyr_with_cache, args.img))
    motion1d = pyr_list[0].make_motion_analyser(pyr_list)

    sample_rate = 1.0 / args.line_delay
    for i in range(1, motion1d.nr_frame):
        amp, freq = avg_spectrum(i, motion1d)
        freq *= sample_rate
        cut_low = min(np.nonzero(freq >= args.cut_low)[0])
        cut_high = min(np.nonzero(freq >= args.cut_high)[0])
        amp = amp[cut_low:cut_high]
        freq = freq[cut_low:cut_high]
        logger.info('spectrum for frame {}, freq_diff={}'.format(
            i, freq[1] - freq[0]))
        plt.plot(freq, amp)
        plt.show()

if __name__ == '__main__':
    main()

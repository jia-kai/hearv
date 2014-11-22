#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: disp_freq.py
# $Date: Sat Nov 22 23:57:12 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import matplotlib.pyplot as plt
import numpy as np

import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', help='array json fpath')
    parser.add_argument('--sample_rate', type=float, default=59.940)
    parser.add_argument('--fl', type=float, default=5,
                        help='low cutoff')
    parser.add_argument('--dmin', type=int, help='min index of data')
    parser.add_argument('--dnr', type=int, help='number of data points used')
    parser.add_argument('--no_shift_mean', action='store_true',
                        help='do not shift mean value to zero')
    parser.add_argument('--clip', type=float,
                        help='clip all samples to be within range [-x, x]')
    args = parser.parse_args()

    with open(args.fpath) as fin:
        vals = np.array(json.load(fin))

    if not args.no_shift_mean:
        vals -= np.mean(vals)
    if args.clip:
        vals = np.clip(vals, -args.clip, args.clip)
    if args.dmin:
        vals = vals[args.dmin:]
    if args.dnr:
        vals = vals[:args.dnr]

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(vals)

    fft = np.fft.fft(vals)[:len(vals) / 2]
    freq = args.sample_rate / len(vals) * np.arange(1, len(fft) + 1)
    if args.fl > 0:
        fl = min(np.nonzero(freq >= args.fl)[0])
        fft = fft[fl:]
        freq = freq[fl:]
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(freq, np.abs(fft))
    
    plt.show()

if __name__ == '__main__':
    main()

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: main.py
# $Date: Sun Dec 07 15:32:41 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from libriesz.utils import get_riesz_pyr_with_cache, plot_val_with_fft
from libriesz.analyze import PCAMotion1DAnalyser, avg_spectrum

import matplotlib.pyplot as plt
import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', nargs='+')
    args = parser.parse_args()

    motion1d = PCAMotion1DAnalyser(map(get_riesz_pyr_with_cache, args.img))

    for i in range(1, motion1d.nr_frame):
        amps, freq = avg_spectrum(i, motion1d)
        plt.plot(freq * 1.0 / 18e-6, amps)
        plt.show()

if __name__ == '__main__':
    main()

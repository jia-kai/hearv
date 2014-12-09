#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: fft.py
# $Date: Sun Dec 07 23:56:43 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np
import matplotlib.pyplot as plt

def main():
    N = 128
    sample_rate = N
    freq = 10.48
    k = np.pi * 2 * freq / sample_rate
    x = np.sin(np.arange(N, dtype='float64') * k)
    #x = np.pad(x, (0, N * 3), mode='constant')
    xfreq = np.arange(N/2) * (float(sample_rate) / len(x))

    fv = np.fft.fft(x)[:N/2]
    amp = np.abs(fv)
    phase = np.angle(fv)
    plt.subplot(2, 1, 1)
    plt.plot(xfreq, amp)
    plt.subplot(2, 1, 2)
    plt.plot(xfreq, phase)
    plt.show()

main()

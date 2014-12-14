#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: fft.py
# $Date: Sun Dec 14 20:51:29 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np
import matplotlib.pyplot as plt

def main():
    sample_rate = 128
    N = 128
    freq = 10.5
    k = np.pi * 2 * freq / sample_rate
    x = np.sin(np.arange(N, dtype='float64') * k)
    #x = np.pad(x, (0, N * 3), mode='constant')
    xfreq = np.arange(N/2) * (float(sample_rate) / len(x))

    fv = np.fft.fft(x)
    amp = np.abs(fv[:N/2])
    phase = np.angle(fv[:N/2])
    print amp
    print phase
    mid = fv[N/2-3:N/2+4]
    print amp[0], np.abs(mid), mid
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(xfreq, amp)
    plt.subplot(2, 1, 2)
    plt.plot(xfreq, phase)

    N = len(x)
    fv[:N/2] *= np.exp(np.random.uniform(0, np.pi*2, N/2) * 1j)
    fv[N/2:] = np.conjugate(fv[:N/2])
    recon = np.fft.ifft(fv)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(xfreq, np.angle(fv[:len(xfreq)]))
    plt.subplot(2, 1, 2)
    plt.plot(np.real(recon))
    plt.show()


main()

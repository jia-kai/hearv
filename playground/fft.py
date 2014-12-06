#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: fft.py
# $Date: Sat Dec 06 20:26:53 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np
import matplotlib.pyplot as plt

def main():
    N = 1024
    x = np.sign(np.random.uniform(low=-1, high=1, size=N))

    fv = np.fft.fft(x)[:N/2]
    amp = np.abs(fv)
    phase = np.angle(fv)
    plt.subplot(2, 1, 1)
    plt.plot(amp)
    plt.subplot(2, 1, 2)
    plt.plot(phase)
    plt.show()

main()

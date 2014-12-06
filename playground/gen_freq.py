#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: gen_freq.py
# $Date: Fri Dec 05 23:50:30 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100
AMP = 0.9

def gen_single(freq, time):
    assert freq * 2 < SAMPLE_RATE
    nr_sample = int(SAMPLE_RATE * time)
    k = np.pi * 2 / SAMPLE_RATE * freq
    return np.sin(np.arange(nr_sample, dtype='float32') * k) * AMP

def main():
    data = []
    data.extend(gen_single(500, 0.5))
    for freq in range(100, 2001, 50):
        data.extend(gen_single(freq, 0.5))
    data = np.array(data) * 32767
    data = data.astype('int16')
    wavfile.write('/tmp/100-2000.wav', SAMPLE_RATE, data)

if __name__ == '__main__':
    main()

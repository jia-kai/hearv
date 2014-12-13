#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: gen_freq.py
# $Date: Fri Dec 12 17:28:13 2014 +0800
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
    data.extend(gen_single(0, 2))
    scale = np.array([-9, -7, -5, -4, -2, 0, 2, 3])
    freq = 880 * np.power(2, scale / 12.0)
    print map(int, freq)
    song = [1, 5, 6, 5, 4, 3, 2, 1]
    song = [freq[i - 1] for i in song]
    print map(int, song)
    for f in song:
        data.extend(gen_single(f, 0.5) * np.log(f / 440))
    data /= np.max(np.abs(data))
    data = np.array(data) * 32767
    data = data.astype('int16')
    wavfile.write('/tmp/song.wav', SAMPLE_RATE, data)

if __name__ == '__main__':
    main()

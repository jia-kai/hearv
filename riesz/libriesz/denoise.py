#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: denoise.py
# $Date: Sun Dec 14 22:50:34 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np

class Denoise(object):
    _avg_spec = None

    def __init__(self, noise_spec, thresh=1.3):
        tot = np.array(noise_spec)
        assert tot.ndim == 2
        self._avg_spec = np.mean(tot, axis=0) * thresh

    def __call__(self, spec):
        #avg = np.mean(sorted(amp)[len(amp)/4:-len(amp)/4])
        #amp = np.clip(amp - avg, 0, np.max(amp))
        #amp = np.power(2000, amp)
        #amp *= 60 / np.max(amp) # XXX
        spec -= self._avg_spec
        spec[spec < 0] = 0
        return spec

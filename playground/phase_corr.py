#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: phase_corr.py
# $Date: Sat Nov 15 21:30:40 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


class PhaseCorr(object):
    LUMIN_THRESH = 20
    MIN_WIDTH = 200
    WIN_HEIGHT = 10

    _roi_list = None
    """list of (row0, col0, col1) tuples, where each ROI is
    [row0:row0+WIN_HEIGHT, col0:col1]"""

    _roi_row = None

    _img_ref = None

    def set_ref(self, ref):
        self._img_ref = ref
        self._roi_list = list()
        row0 = -1
        col0 = float('-inf')
        col1 = float('inf')
        for r in range(ref.shape[0]):
            cols, = np.nonzero(ref[r] >= self.LUMIN_THRESH)
            if len(cols) >= self.MIN_WIDTH:
                col0 = max(col0, min(cols))
                col1 = min(col1, max(cols))
                if r - row0 == self.WIN_HEIGHT:
                    self._roi_list.append((row0 + 1, col0, col1 + 1))
                    row0 = r
                    col0 = float('-inf')
                    col1 = float('inf')
            else:
                row0 = r

        self._roi_row = [i[0] for i in self._roi_list]

        #self._disp_roi()
        
        self._img_ref = self._img_ref.astype('float32') / 255.0

    def _disp_roi(self):
        img = np.zeros_like(self._img_ref)
        for r, c0, c1 in self._roi_list:
            img[r:r+self.WIN_HEIGHT, c0:c1] = 255
        cv2.imshow('roi', img)
        cv2.waitKey(-1)

    def add_img(self, img, visualize=True):
        assert img.shape == self._img_ref.shape
        img = img.astype('float32') / 255.0
        rst = []
        for row, c0, c1 in self._roi_list:
            dx, dy = cv2.phaseCorrelate(
                self._img_ref[row:row+self.WIN_HEIGHT, c0:c1],
                img[row:row+self.WIN_HEIGHT, c0:c1])
            rst.append((dx, dy))

        if visualize:
            x, y = map(np.array, zip(*rst))
            phase = np.arctan2(x, y)
            amph = np.sqrt(x ** 2 + y ** 2)
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(self._roi_row, amph)
            plt.subplot(2, 1, 2)
            plt.plot(self._roi_row, phase)

        return rst

def main():
    pc = PhaseCorr()
    pc.set_ref(cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE))

    for fpath in sys.argv[2:]:
        print fpath
        pc.add_img(cv2.imread(fpath, cv2.IMREAD_GRAYSCALE))
    plt.show()

if __name__ ==  '__main__':
    main()

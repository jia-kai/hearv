#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: riesz.py
# $Date: Fri Nov 21 01:09:28 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import pyximport
pyximport.install()

import cv2
import numpy as np
import sys
from scipy import signal
from fastmath import smooth_orient

floatX = 'float64'

class RieszPyramid(object):
    EPS = 1e-10
    riesz_kernel = np.array(
        [[-0.12, -0.34, -0.12],
         [0, 0, 0],
         [0.12, 0.34, 0.12]], dtype=floatX)

    min_pyr_img_size = 10
    min_pyr_scale = None
    max_pyr_scale = None

    _img_ref = None
    _lap_pyr_ref = None
    _riesz_pyr_ref = None

    _pyr_amps_phasediff = None
    """list of (sqr(amp), phase_diff) images for each scale in lap pyr,
    all scaled to original image size"""

    def __init__(self, img_ref, min_scale=1, max_scale=3):
        self.min_pyr_scale = int(min_scale)
        self.max_pyr_scale = int(max_scale)
        self._img_ref = img_ref.copy()
        self._lap_pyr_ref = self.build_lap_pyr(img_ref)
        self._riesz_pyr_ref = map(self.get_riesz_triple, self._lap_pyr_ref)

    def set_image(self, img):
        assert img.shape == self._img_ref.shape
        pyr = self.build_lap_pyr(img)
        tsize_cv = img.shape[:2][::-1]
        self._pyr_amps_phasediff = list()
        for riesz0, band1 in zip(self._riesz_pyr_ref, pyr):
            riesz1 = self.get_riesz_triple(band1)
            assert riesz0.shape == riesz1.shape
            pd = self.get_phase_diff(riesz0, riesz1)
            pd = cv2.resize(pd, tsize_cv)
            amp = (riesz0[:, :, 0] + riesz1[:, :, 0]) / 2
            amp = cv2.resize(amp, tsize_cv)
            amps = np.square(amp)
            self._pyr_amps_phasediff.append((amps, pd))

    def get_avg_phase_diff(self, subslice=slice(None, None, None)):
        """average phase diff within a block"""
        val = []
        for amps, pd in self._pyr_amps_phasediff:
            amps = amps[subslice]
            pd = pd[subslice]
            val.append(np.sum(amps * pd) / np.sum(amps))
        return float(np.mean(val))

    def build_lap_pyr(self, img):
        """:return: list of images, layers in the Laplacian Pyramid"""
        im_min = np.min(img)
        im_max = np.max(img)
        assert im_min >= 0 and im_max <= 255 and im_max >= 10, (im_min, im_max)
        img = img.astype(floatX) / 255.0
        rst = []
        while min(img.shape[:2]) >= self.min_pyr_img_size and \
                len(rst) <= self.max_pyr_scale:
            next_img = cv2.pyrDown(img)
            recon = cv2.pyrUp(next_img, dstsize=img.shape[:2][::-1])
            rst.append(img - recon)
            img = next_img
        return rst[self.min_pyr_scale:]

    def get_riesz_triple(self, img):
        """:param img: 2D input image of shape (h, w)
        :return: 3D image of shape (h, w, 3), where the 3 channels correspond to
        amplitude, orientation and phase
        """
        assert img.ndim == 2
        img = img.astype(floatX)
        r1 = signal.convolve2d(img, self.riesz_kernel, mode='valid')
        r2 = signal.convolve2d(img, self.riesz_kernel.T, mode='valid')
        kh, kw = self.riesz_kernel.shape
        assert kh % 2 == 1 and kw % 2 == 1 and kh == kw
        img = img[kh/2:-(kh/2), kw/2:-(kw/2)]
        amp = np.sqrt(np.square(img) + np.square(r1) + np.square(r2))
        phase = np.arccos(img / (amp + self.EPS))
        t = amp * np.sin(phase) + self.EPS
        orient = np.arctan2(r2 / t, r1 / t)
        rst = np.concatenate(map(lambda a: np.expand_dims(a, axis=2),
                                 (amp, orient, phase)), axis=2)
        assert np.all(np.isfinite(rst)), \
            np.transpose(np.nonzero(1 - np.isfinite(rst)))
        smooth_orient(rst)
        return rst

    def get_phase_diff(self, riesz0, riesz1):
        assert riesz0.shape == riesz1.shape
        if riesz0.ndim == 2:
            assert riesz0.shape[1] == 3
            rst = riesz1[:, 2] - riesz0[:, 2]
        else:
            rst = riesz1[:, :, 2] - riesz0[:, :, 2]
        rst = np.mod(rst, np.pi * 2)
        rst[rst >= np.pi] -= np.pi * 2
        return rst

    def disp_refimg_riesz(self):
        import matplotlib.pyplot as plt
        for idx, img in enumerate(self._riesz_pyr_ref):
            row = img[img.shape[0] / 2].T
            plt.subplot(3, 1, 1)
            plt.title('amp')
            plt.plot(row[0])
            plt.subplot(3, 1, 2)
            plt.title('orient')
            plt.plot(row[1])
            plt.subplot(3, 1, 3)
            plt.title('phase')
            plt.plot(row[2])
            plt.show()

class Slicer(object):
    def __getitem__(self, idx):
        return idx
slicer = Slicer()

def normalize_disp(img):
    img = cv2.normalize(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img.astype('uint8')

def imshow(name, img, wait=False):
    cv2.imshow(name, normalize_disp(img))
    if wait:
        if chr(cv2.waitKey(-1) & 0xFF) == 'q':
            sys.exit()

def test_motion():
    SIZE = 500
    k = np.pi / 40
    shift = 0.05 * k
    x0 = np.arange(SIZE) * k
    def make(x):
        x = np.tile(x, SIZE).reshape(SIZE, SIZE)
        y = np.tile(x0, SIZE).reshape(SIZE, SIZE).T
        val = (np.sin(x + y) + 1) / 2
        #return val * 255
        return normalize_disp(val)
    img0 = make(x0)
    img1 = make(x0 + shift)
    pyr = RieszPyramid(img0)
    #pyr.disp_refimg_riesz()
    pyr.set_image(img1)
    get = pyr.get_avg_phase_diff()
    print shift, get, shift / get
    #imshow('img0', img0)
    #imshow('img1', img1, True)

def main():
    test_motion()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img0')
    parser.add_argument('img1')
    parser.add_argument('-o', '--output', help='output plot')
    args = parser.parse_args()

    import matplotlib
    if args.output:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img0 = cv2.imread(args.img0, cv2.IMREAD_GRAYSCALE)
    pyr = RieszPyramid(img0)
    pyr.disp_refimg_riesz()
    pyr.set_image(cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE))

    HEIGHT = 10
    pdiff = []
    for row in range(HEIGHT / 2, img0.shape[0] - HEIGHT * 3 / 2):
        pdiff.append(pyr.get_avg_phase_diff(slicer[row:row+HEIGHT]))
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(pdiff)
    ax = fig.add_subplot(2, 1, 2)
    fft = np.fft.fft(pdiff)
    sample_rate = 1.0 / 16e-6
    freq = sample_rate / len(pdiff) * np.arange(1, len(fft) + 1)
    cut_low = min(np.nonzero(freq >= 50)[0])
    cut_high = min(np.nonzero(freq >= 1000)[0])
    fft = fft[cut_low:cut_high]
    freq = freq[cut_low:cut_high]
    ax.plot(freq, np.abs(fft))
    if args.output:
        fig.savefig(args.output)
    else:
        plt.show()


if __name__ == '__main__':
    main()

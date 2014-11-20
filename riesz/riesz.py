#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: riesz.py
# $Date: Thu Nov 20 22:23:16 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import pyximport
pyximport.install()

import cv2
import numpy as np
import sys
from scipy import signal
from fastmath import smooth_orient

EPS = 1e-10

RIESZ_KERNEL = np.array(
    [[-0.12, -0.34, -0.12],
     [0, 0, 0],
     [0.12, 0.34, 0.12]], dtype='float32')

def build_lap_pyr(img, min_size=10, max_nr_level=None):
    """:return: list of images, layers in the Laplacian Pyramid"""
    if max_nr_level is None:
        max_nr_level = float('inf')
    im_min = np.min(img)
    im_max = np.max(img)
    assert im_min >= 0 and im_max <= 255 and im_max >= 10, (im_min, im_max)
    img = img.astype('float32') / 255.0
    rst = []
    while min(img.shape[:2]) >= min_size and len(rst) < max_nr_level:
        next_img = cv2.pyrDown(img)
        recon = cv2.pyrUp(next_img, dstsize=img.shape[:2][::-1])
        rst.append(img - recon)
        img = next_img
    return rst

def get_riesz_triple(img):
    """:param img: 2D input image of shape (h, w)
    :return: 3D image of shape (h, w, 3), where the 3 channels correspond to
    amplitude, orientation and phase
    """
    assert img.ndim == 2
    img = img.astype('float32')
    r1 = signal.convolve2d(img, RIESZ_KERNEL, mode='valid')
    r2 = signal.convolve2d(img, RIESZ_KERNEL.T, mode='valid')
    kh, kw = RIESZ_KERNEL.shape
    img = img[kh/2:-(kh/2), kw/2:-(kw/2)]
    amp = np.sqrt(np.square(img) + np.square(r1) + np.square(r2)) + EPS
    phase = np.arccos(img / amp)
    t = amp * np.sin(phase) + EPS
    orient = np.arctan2(r2 / t, r1 / t)
    rst = np.concatenate(map(lambda a: np.expand_dims(a, axis=2),
                             (amp, orient, phase)), axis=2)
    assert np.all(np.isfinite(rst)), \
        np.transpose(np.nonzero(1 - np.isfinite(rst)))
    smooth_orient(rst)
    return rst

def get_phase_diff(riesz0, riesz1):
    assert riesz0.shape == riesz1.shape
    if riesz0.ndim == 2:
        assert riesz0.shape[1] == 3
        rst = riesz1[:, 2] - riesz0[:, 2]
    else:
        rst = riesz1[:, :, 2] - riesz0[:, :, 2]
    rst = np.mod(rst, np.pi * 2)
    rst[rst >= np.pi] -= np.pi * 2
    return rst

def get_avg_phase_diff(riesz0, riesz1):
    amp = (riesz0[:, :, 0] + riesz1[:, :, 0]) / 2
    pd = get_phase_diff(riesz0, riesz1)
    return np.sum(pd * amp) / np.sum(amp)

def normalize_disp(img):
    img = cv2.normalize(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img.astype('uint8')

def imshow(name, img, wait=False):
    cv2.imshow(name, normalize_disp(img))
    if wait:
        if chr(cv2.waitKey(-1) & 0xFF) == 'q':
            sys.exit()

def test_motion():
    import matplotlib.pyplot as plt
    SIZE = 500
    k = np.pi / 40
    shift = 0.05 * k
    x0 = np.arange(SIZE) * k
    def make(x):
        x = np.tile(x, SIZE).reshape(SIZE, SIZE)
        y = np.tile(x0, SIZE).reshape(SIZE, SIZE).T
        val = (np.sin(x + y) + 1) / 2
        return val * 255
        return normalize_disp(val)
    img0 = make(x0)
    img1 = make(x0 + shift)
    riesz0 = get_riesz_triple(build_lap_pyr(img0, max_nr_level=1)[0])
    riesz1 = get_riesz_triple(build_lap_pyr(img1, max_nr_level=1)[0])
    if True:
        row = riesz0[30]
        row1 = riesz1[30]
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.title('amp')
        plt.plot(row[:, 0])
        plt.subplot(4, 1, 2)
        plt.title('orientation')
        plt.plot(row[:, 1])
        plt.subplot(4, 1, 3)
        plt.title('phase')
        plt.plot(row[:, 2])
        plt.subplot(4, 1, 4)
        plt.title('phase_diff')
        plt.plot(get_phase_diff(row, row1))
        plt.figure()
        plt.title('row_orient_avg')
        plt.plot(np.mean(riesz0[:, :, 1], axis=1))
        plt.show()
    riesz_diff = riesz0 - riesz1
    print riesz0[100, 30:70]
    print riesz_diff[100, 30:70]
    print shift, get_avg_phase_diff(riesz0, riesz1)

    imshow('img0', img0)
    imshow('img1', img1)
    imshow('phase0', riesz0[:, :, 2])
    imshow('sign(orit0)', np.sign(riesz0[:, :, 1]))
    imshow('sign(phase_diff)', np.sign(riesz_diff[:, :, 2]))
    imshow('riesz_diff', riesz_diff, True)

def main():
    test_motion()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img')
    parser.add_argument('--level', type=int,
                        help='max level of Laplacian pyramid')
    args = parser.parse_args()

    img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    cv2.imshow('orig', img)
    lap_pyr = build_lap_pyr(img, max_nr_level=args.level)
    for idx, i in enumerate(lap_pyr):
        riesz = get_riesz_triple(i)
        cv2.imshow('img{}'.format(idx), normalize_disp(i))
        cv2.imshow('phase{}'.format(idx), normalize_disp(riesz[:, :, 2]))
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()

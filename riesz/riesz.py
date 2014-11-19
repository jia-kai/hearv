#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: riesz.py
# $Date: Thu Nov 20 00:12:30 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import cv2
import numpy as np
import sys
from scipy import signal

EPS = 1e-10

RIESZ_KERNEL = np.array(
    [[-0.12, -0.34, -0.12],
     [0, 0, 0],
     [0.12, 0.34, 0.12]], dtype='float32')

def build_lap_pyr(img, min_size=10, max_nr_level=None):
    """:return: list of images, layers in the Laplacian Pyramid"""
    if max_nr_level is None:
        max_nr_level = float('inf')
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
    r1 = signal.convolve2d(img, RIESZ_KERNEL, mode='same')
    r2 = signal.convolve2d(img, RIESZ_KERNEL.T, mode='same')
    amp = np.sqrt(np.square(img) + np.square(r1) + np.square(r2)) + EPS
    phase = np.arccos(img / amp)
    t = amp * np.sin(phase) + EPS
    orient = np.arctan2(r2 / t, r1 / t)
    return np.concatenate(map(lambda a: np.expand_dims(a, axis=2),
                              (amp, orient, phase)), axis=2)

def get_phase_diff(riesz0, riesz1):
    oriet_sign = (np.sign(riesz0[:, :, 1]) + np.sign(riesz1[:, :, 1])) / 2
    return (riesz1[:, :, 2] - riesz0[:, :, 2]) * oriet_sign

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
    make = lambda x: np.tile(np.sin(x), SIZE).reshape(SIZE, SIZE)
    img0 = make(x0)
    img1 = make(x0 + shift)
    riesz0 = get_riesz_triple(build_lap_pyr(img0, max_nr_level=1)[0])
    riesz1 = get_riesz_triple(build_lap_pyr(img1, max_nr_level=1)[0])
    riesz_diff = riesz0 - riesz1
    print shift, np.mean(get_phase_diff(riesz0, riesz1))

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

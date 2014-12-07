# -*- coding: utf-8 -*-
# $File: pyramid_quat.py
# $Date: Sun Dec 07 14:21:10 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .config import floatX

import cv2
import numpy as np
import sys
from scipy import signal

class RieszPyramid(object):
    __slots__ = ['levels', 'img_shape']

    def __init__(self, levels, img_shape):
        self.levels = levels
        self.img_shape = img_shape

    @property
    def nr_level(self):
        return len(self.levels)

class RieszQuatPyramidBuilder(object):
    """quaternionic pyramid: amplitude, phi cos(theta), phi sin(theta)"""
    EPS = np.finfo(floatX).tiny
    riesz_kernel = np.array(
        [[-0.12, -0.34, -0.12],
         [0, 0, 0],
         [0.12, 0.34, 0.12]], dtype=floatX)

    min_pyr_img_size = 10
    min_pyr_scale = 0
    max_pyr_scale = 3
    spatial_blur = 0.5
    spatial_ksize = (3, 3)

    def __call__(self, img):
        """:return: RieszPyramid"""
        levels = []
        for idx, bandi in enumerate(self._build_lap_pyr(img)):
            levels.append(self._get_riesz_triple(bandi))
        return RieszPyramid(levels, img.shape)

    def _build_lap_pyr(self, img):
        """:return: list of images, layers in the Laplacian Pyramid"""
        im_min = np.min(img)
        im_max = np.max(img)
        assert im_min >= 0 and im_max <= 255 and im_max >= 10, (im_min, im_max)
        img = img.astype(floatX) / 255.0
        img = cv2.GaussianBlur(img, (0, 0), 2)
        assert img.ndim == 2
        rst = []
        while len(rst) <= self.max_pyr_scale:
            assert min(img.shape) >= self.min_pyr_img_size
            next_img = cv2.pyrDown(img)
            recon = cv2.pyrUp(next_img, dstsize=img.shape[::-1])
            rst.append(img - recon)
            img = next_img
        return rst[self.min_pyr_scale:]

    def _get_riesz_triple(self, img):
        """:param img: 2D input image of shape (h, w)
        :return: 3D image of shape (h, w, 3), where the 3 channels correspond to
        amplitude, phase * cos(orient), phase * sin(orient)
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
        t = np.sqrt(np.square(r1) + np.square(r2)) + self.EPS
        v0 = phase * r1 / t
        v1 = phase * r2 / t
        if self.spatial_blur:
            amp_blur = cv2.GaussianBlur(
                amp, self.spatial_ksize, self.spatial_blur)
            def blur(v):
                a = cv2.GaussianBlur(
                    amp * v, self.spatial_ksize, self.spatial_blur)
                return a / amp_blur
            v0 = blur(v0)
            v1 = blur(v1)
        rst = np.concatenate(map(lambda a: np.expand_dims(a, axis=2),
                                 (amp, v0, v1)), axis=2)
        assert np.all(np.isfinite(rst))
        return rst

    @classmethod
    def disp_riesz(cls, img, show=True):
        import matplotlib.pyplot as plt
        assert img.ndim == 3 and img.shape[2] == 3
        row = img[img.shape[0] / 2].T
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title('$A$')
        plt.plot(row[0])
        plt.subplot(3, 1, 2)
        plt.title(r'$\varphi\cos(\theta)$')
        plt.plot(row[1])
        plt.subplot(3, 1, 3)
        plt.title(r'$\varphi\sin(\theta)$')
        plt.plot(row[2])
        if show:
            plt.show()

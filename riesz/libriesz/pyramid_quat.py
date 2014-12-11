# -*- coding: utf-8 -*-
# $File: pyramid_quat.py
# $Date: Wed Dec 10 23:34:22 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .config import floatX
from .pyramid import RieszPyramidBuilderBase
from .analyze import Motion1DAnalyserBase
from .fastmath import smooth_pca

import cv2
import numpy as np
from scipy import signal

import logging
logger = logging.getLogger(__name__)

class RieszQuatPyramid(object):
    __slots__ = ['levels', 'img_shape']

    @classmethod
    def make_motion_analyser(cls, pyr_list):
        return PCAMotion1DAnalyser(pyr_list)

    def __init__(self, levels, img_shape):
        self.levels = levels
        self.img_shape = img_shape

    @property
    def nr_level(self):
        return len(self.levels)

class RieszQuatPyramidBuilder(RieszPyramidBuilderBase):
    """quaternionic pyramid: amplitude, phi cos(theta), phi sin(theta)"""
    def __call__(self, img):
        """:return: RieszQuatPyramid"""
        levels = []
        for idx, bandi in enumerate(self._build_lap_pyr(img)):
            levels.append(self._get_riesz_triple(bandi))
        return RieszQuatPyramid(levels, img.shape)

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


class PCAMotion1DAnalyser(Motion1DAnalyserBase):

    def __init__(self, pyr_list):
        super(PCAMotion1DAnalyser, self).__init__(pyr_list)
        for i in pyr_list:
            assert isinstance(i, RieszQuatPyramid)
        self.pyr_list = pyr_list
        self.level_local_pca = []
        for i in range(self.nr_level):
            self._init_pt_pca(i)

    def show_pca(self, level=0, show=True):
        plt.figure()
        pca = self.level_local_pca[level]
        for dx in [0, 1]:
            for dy in [0, 1]:
                x = []
                y = []
                h, w = self.pyr_list[0].levels[level].shape[:2]
                coord = h / 2 + dy, w / 2 + dx
                for i in self.pyr_list:
                    camp, cx, cy = i.levels[level][coord]
                    x.append(cx)
                    y.append(cy)
                x = np.array(x)
                y = np.array(y)
                pca_x, pca_y = pca[coord]
                plt.subplot(2, 2, dy * 2 + dx + 1)
                plt.scatter(x, y)
                x0 = np.mean(x)
                y0 = np.mean(y)
                plt.scatter([x0], [y0], s=[5])
                plt.plot([x0, x0 + pca_x], [y0, y0 + pca_y])
        if show:
            plt.show()

    def local_motion_map(self, frame, level):
        v0 = self.pyr_list[frame].levels[level]
        v1 = self.pyr_list[0].levels[level]
        pca = self.level_local_pca[level]
        md = np.sum((v0[:, :, 1:] - v1[:, :, 1:]) * pca, axis=2)
        return md, (v0[:, :, 0] + v1[:, :, 0]) / 2

    def _init_pt_pca(self, level):
        """compute the PCA of motion of each point in a list of pyramid at given
        level
        :return: array(h, w, 2), last dim is pca (x, y)"""
        pyr_list = self.pyr_list
        logger.info('computing point-wise PCA for {} pyrmids at level {}'.format(
            len(pyr_list), level))
        a = np.concatenate([i.levels[level][:, :, 1:2] for i in pyr_list],
                           axis=2)
        b = np.concatenate([i.levels[level][:, :, 2:3] for i in pyr_list],
                           axis=2)
        a -= np.mean(a, axis=2, keepdims=True)
        b -= np.mean(b, axis=2, keepdims=True)
        w00 = np.sum(np.square(a), axis=2)
        w01 = np.sum(a * b, axis=2)
        w11 = np.sum(np.square(b), axis=2)
        eq_b = -(w00 + w11)
        eq_c = w00 * w11 - w01 * w01
        eq_delta = np.sqrt(np.square(eq_b) - 4 * eq_c)
        eq_x = (-eq_b + eq_delta) / 2
        x = w11 - eq_x
        y = -w01
        k = 1 / np.sqrt(np.square(x) + np.square(y))
        x *= k
        y *= k
        h, w = a.shape[:2]
        result = np.concatenate(
            [x.reshape(h, w, 1), y.reshape(h, w, 1)], axis=2)
        assert np.isfinite(np.sum(result))
        smooth_pca(result)
        self.level_local_pca.append(result)

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: riesz.py
# $Date: Sat Dec 06 00:38:08 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import pyximport
pyximport.install()

import cv2
import numpy as np
import sys
from scipy import signal
from fastmath import smooth_orient, regularize_orient

floatX = 'float64'

class RieszPyramidBuilder(object):
    EPS = 1e-10
    riesz_kernel = np.array(
        [[-0.12, -0.34, -0.12],
         [0, 0, 0],
         [0.12, 0.34, 0.12]], dtype=floatX)
    hori_pyrdown_kernel = np.array([[1, 4, 6, 4, 1]], dtype=floatX) / 16.0
    pyrdown_kernel = np.outer(hori_pyrdown_kernel, hori_pyrdown_kernel)

    min_pyr_img_size = 10
    min_pyr_scale = 1
    max_pyr_scale = 3
    spatial_blur = 0.5
    spatial_ksize = (3, 3)

    def __call__(self, img, pyr_ref=None):
        """:return: Riesz pyramid, largest image first"""
        lap_pyr = self._build_lap_pyr(img)
        if pyr_ref:
            assert len(lap_pyr) == len(pyr_ref)
        riesz_pyr = []
        for idx, bandi in enumerate(lap_pyr):
            rv = self._get_riesz_triple(bandi)
            if pyr_ref is None:
                smooth_orient(rv)
            else:
                ref = pyr_ref[idx]
                assert ref.shape == rv.shape
                regularize_orient(rv, ref)
            riesz_pyr.append(rv)
        return riesz_pyr

    def _build_lap_pyr(self, img):
        """:return: list of images, layers in the Laplacian Pyramid"""
        im_min = np.min(img)
        im_max = np.max(img)
        assert im_min >= 0 and im_max <= 255 and im_max >= 10, (im_min, im_max)
        img = img.astype(floatX) / 255.0
        img = cv2.GaussianBlur(img, (0, 0), 2)
        assert img.ndim == 2
        rst = []
        lb = self.pyrdown_kernel.shape[0] / 2
        while len(rst) <= self.max_pyr_scale:
            assert min(img.shape) >= self.min_pyr_img_size
            lowpass = signal.convolve2d(img, self.pyrdown_kernel, mode='valid')
            bandpass = img[lb:-lb, lb:-lb] - lowpass
            next_img = lowpass[::2, ::2]
            rst.append(bandpass)

            if False:
                recon = cv2.pyrUp(cv2.pyrDown(img), dstsize=img.shape[::-1])
                imshow('cv-lap', normalize_disp(img - recon), False)
                imshow('lap', normalize_disp(bandpass), False)
                imshow('cv-down', cv2.pyrDown(img), False)
                imshow('down', next_img, True)

            img = next_img
        return rst[self.min_pyr_scale:]

    def _get_riesz_triple(self, img):
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
        if self.spatial_blur:
            amp_blur = cv2.GaussianBlur(
                amp, self.spatial_ksize, self.spatial_blur)
            def blur(v):
                a = cv2.GaussianBlur(
                    amp * v, self.spatial_ksize, self.spatial_blur)
                return a / amp_blur
            v0 = blur(phase * np.cos(orient))
            v1 = blur(phase * np.sin(orient))
            phase = np.sqrt(np.square(v0) + np.square(v1))
            orient = np.arctan2(v1, v0)
        rst = np.concatenate(map(lambda a: np.expand_dims(a, axis=2),
                                 (amp, orient, phase)), axis=2)
        assert np.all(np.isfinite(rst))
        return rst

    @classmethod
    def disp_riesz(cls, img, show=True):
        import matplotlib.pyplot as plt
        assert img.ndim == 3 and img.shape[2] == 3
        row = img[img.shape[0] / 2].T
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title('amp')
        plt.plot(row[0])
        plt.subplot(3, 1, 2)
        plt.title('orient')
        plt.plot(row[1])
        plt.subplot(3, 1, 3)
        plt.title('phase')
        plt.plot(row[2])
        if show:
            plt.show()

    @classmethod
    def pyrdown_hori(cls, img):
        img = signal.convolve2d(img, cls.hori_pyrdown_kernel, mode='valid')
        return img[:, ::2]

    @classmethod
    def get_row_phase_diff(cls, pyr0, pyr1, y):
        """get phase diff of each location in a row
        :param y: row index on the last level of pyramid"""
        assert len(pyr0) == len(pyr1)
        min_h, min_w = pyr0[-1].shape[:2]
        all_pd = []
        all_amps = []
        for rimg0, rimg1 in zip(pyr0, pyr1):
            assert rimg0.shape == rimg1.shape
            y_begin = y * rimg0.shape[0] / min_h
            y_end = (y + 1) * rimg0.shape[0] / min_h
            rimg0 = rimg0[y_begin:y_end]
            rimg1 = rimg1[y_begin:y_end]
            pd = rimg1[:, :, 2] - rimg0[:, :, 2]
            if False:
                cls.disp_riesz(rimg0, False)
                cls.disp_riesz(rimg1, False)
                diff = rimg1 - rimg0
                diff[:, :, 2] = pd
                cls.disp_riesz(diff)
            amps = (np.square(rimg0[:, :, 0]) + np.square(rimg1[:, :, 0])) / 2
            while (pd.shape[1] + 1) / 2 >= min_w:
                pd = cls.pyrdown_hori(pd)
                amps = cls.pyrdown_hori(amps)
            pd = pd[:, :min_w]
            amps = amps[:, :min_w]
            all_pd.extend(pd)
            all_amps.extend(amps)

        pd = np.array(all_pd)
        amps = np.array(all_amps)
        return np.sum(pd * amps, axis=0) / np.sum(amps, axis=0), amps.mean()
        
    @classmethod
    def get_avg_phase_diff_freq(cls, pyr0, pyr1):
        val = []
        wsum = 0
        min_h, min_w = pyr0[-1].shape[:2]
        for y in range(pyr1[-1].shape[0]):
            pd, w = cls.get_row_phase_diff(pyr0, pyr1, y)
            fft = np.fft.fft(pd) * w
            wsum += w
            assert len(fft) == min_w, (len(fft), min_w)
            val.append(np.abs(fft[:min_w / 2]))
        return np.sum(val, axis=0) / wsum

def normalize_disp(img):
    img = cv2.normalize(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img.astype('uint8')

def imshow(name, img, wait=False):
    cv2.imshow(name, normalize_disp(img))
    if wait:
        if chr(cv2.waitKey(-1) & 0xFF) == 'q':
            sys.exit()

def test_motion(plot=False):
    SIZE = 500
    k = np.pi / 40
    v0 = np.arange(SIZE) * k
    sigma = 1
    def make(v1):
        x = np.tile(v1, SIZE).reshape(SIZE, SIZE)
        y = np.tile(v0, SIZE).reshape(SIZE, SIZE).T
        #val = (np.sin(x + y) + 1) / 2
        val = (np.sin(x) + np.sin(y) + 2) / 4
        if sigma:
            val += np.random.normal(scale=sigma/255.0, size=val.shape)
        #return val * 255
        return normalize_disp(val)
    pyr_builder = RieszPyramidBuilder()
    if not plot:
        img0 = make(v0)
        pyr0 = pyr_builder(img0)
        shift = 0.01 * k
        img1 = make(v0 + shift)
        cv2.imwrite('/tmp/img0.png', img0)
        cv2.imwrite('/tmp/img1.png', img1)
        pyr1 = pyr_builder(img1, pyr0)

        plot_val_with_fft(pyr_builder.get_row_phase_diff(pyr0, pyr1, 15))

        imshow('img0', img0)
        imshow('img1', img1, True)
    else:
        import matplotlib.pyplot as plt
        for sigma in [0, 1.5, 3, 6, 10]:
            x = np.arange(-0.5, 0.5, 0.01) * k
            y = []
            img0 = make(v0)
            pyr = RieszPyramid(img0)
            for shift in x:
                pyr.set_image(make(v0 + shift))
                y.append(pyr.get_avg_phase_diff())
                print sigma, shift, y[-1], shift / y[-1]
            plt.plot(x / k, np.array(y) / k,
                     label=r'$\sigma$={}/255'.format(sigma))
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlabel('ground truth shift (pixels)')
        plt.ylabel('phase diff')
        plt.savefig('/tmp/fig.png')
        plt.show()

def plot_val_with_fft(data, sample_rate=1.0, cut_low=None, cut_high=None,
                      output=None, show=True):
    import matplotlib.pyplot as plt
    sample_rate = float(sample_rate)
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.plot(data)
    ax = fig.add_subplot(2, 1, 2)
    fft = np.fft.fft(data)
    freq = sample_rate / len(data) * np.arange(len(data))
    if cut_low is not None:
        cut_low = max(int(len(data) * cut_low / sample_rate - 1), 0)
    else:
        cut_low = 0
    if cut_high is not None:
        cut_high = int(len(data) * cut_high / sample_rate)
    else:
        cut_high = float('inf')
    cut_high = min(cut_high, len(data) / 2)
    assert cut_low < cut_high
    fft = fft[cut_low:cut_high]
    freq = freq[cut_low:cut_high]
    ax.set_xlabel('freq')
    ax.set_ylabel('amplitude')
    ax.plot(freq, np.abs(fft))
    if output:
        fig.savefig(output)
    if show:
        plt.show()

def main():
    #test_motion(False)
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img0')
    parser.add_argument('img1', nargs='+')
    parser.add_argument('-o', '--output', help='output plot')
    parser.add_argument('--do', help='data output')
    parser.add_argument('--update', type=int,
                        help='number of frames to update ref image')
    args = parser.parse_args()

    import matplotlib
    if args.output:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img0 = cv2.imread(args.img0, cv2.IMREAD_GRAYSCALE).T
    pyr_builder = RieszPyramidBuilder()
    pyr0 = pyr_builder(img0)
    #pyr.disp_refimg_riesz()
    line_delay = 14e-6 * img0.shape[1] / pyr0[-1].shape[1]
    for img1 in args.img1:
        img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE).T
        pyr1 = pyr_builder(img1, pyr_ref=pyr0)
        
        amp = pyr_builder.get_avg_phase_diff_freq(pyr0, pyr1)
        freq = 1.0 / line_delay * np.arange(len(amp)) / (len(amp) * 2)
        fl = min(np.nonzero(freq >= 100)[0])
        freq = freq[fl:]
        amp = amp[fl:]
        fh = min(np.nonzero(freq > 2000)[0])
        freq = freq[:fh]
        amp = amp[:fh]
        plt.plot(freq, amp)
        plt.show()

if __name__ == '__main__':
    main()

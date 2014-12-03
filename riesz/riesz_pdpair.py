#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: riesz.py
# $Date: Thu Dec 04 00:02:45 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import pyximport
pyximport.install()

import cv2
import numpy as np
import sys
from scipy import signal
from fastmath import smooth_orient, regularize_orient, calc_min_phase_diff

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
    min_pyr_scale = 0
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
            """
            if pyr_ref is None:
                smooth_orient(rv)
            else:
                ref = pyr_ref[idx]
                assert ref.shape == rv.shape
                regularize_orient(rv, ref)
                """
            riesz_pyr.append(rv)
        return riesz_pyr

    def _build_lap_pyr(self, img):
        """:return: list of images, layers in the Laplacian Pyramid"""
        im_min = np.min(img)
        im_max = np.max(img)
        assert im_min >= 0 and im_max <= 255 and im_max >= 10, (im_min, im_max)
        img = img.astype(floatX) / 255.0
        #img = cv2.GaussianBlur(img, (5, 5), 2)
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
        t = phase / (amp * np.sin(phase) + self.EPS)
        v0 = r1 * t
        v1 = r2 * t
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

    @classmethod
    def pyrdown_hori(cls, img):
        img = signal.convolve2d(img, cls.hori_pyrdown_kernel, mode='valid')
        return img[:, ::2]

    @classmethod
    def get_avg_phase_diff_colidx(cls, pyr0, pyr1):
        """get phase diff of each column
        :return: W * 2 matrix"""
        assert len(pyr0) == len(pyr1)
        min_h, min_w = pyr0[-1].shape[:2]
        all_pd0 = []
        all_pd1 = []
        all_amps = []
        for rimg0, rimg1 in zip(pyr0, pyr1):
            assert rimg0.shape == rimg1.shape
            amps = (np.square(rimg0[:, :, 0]) + np.square(rimg1[:, :, 0])) / 2
            pd0 = rimg0[:, :, 1] - rimg1[:, :, 1]
            pd1 = rimg0[:, :, 2] - rimg1[:, :, 2]
            if False:
                cls.disp_riesz(rimg0, False)
                cls.disp_riesz(rimg1, False)
                cls.disp_riesz(rimg1 - rimg0, True)
            while (pd0.shape[1] + 1) / 2 >= min_w:
                pd0 = cls.pyrdown_hori(pd0)
                pd1 = cls.pyrdown_hori(pd1)
                amps = cls.pyrdown_hori(amps)
            all_pd0.extend(pd0[:, :min_w])
            all_pd1.extend(pd1[:, :min_w])
            all_amps.extend(amps[:, :min_w])

        pd0 = np.array(all_pd0)
        pd1 = np.array(all_pd1)
        amps = np.array(all_amps)
        asum = np.sum(amps, axis=0)
        return (np.sum(pd0 * amps, axis=0) / asum,
                np.sum(pd1 * amps, axis=0) / asum)
        

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
    sigma = 0
    def make(v1):
        x = np.tile(v1, SIZE).reshape(SIZE, SIZE)
        y = np.tile(v0, SIZE).reshape(SIZE, SIZE).T
        #val = (np.sin(x + y) + 1) / 2
        val = (np.sin(x) + np.sin(y) + 2) / 4
        if sigma:
            val += np.random.normal(scale=sigma/255.0, size=val.shape)
        return val * 255
        return normalize_disp(val)
    pyr_builder = RieszPyramidBuilder()
    if not plot:
        img0 = make(v0)
        pyr0 = pyr_builder(img0)
        shift = 0.01 * k
        img1 = make(v0 + shift)
        cv2.imwrite('/tmp/img0.png', img0)
        cv2.imwrite('/tmp/img1.png', img1)
        pyr1 = pyr_builder(img1)

        tot_d0 = []
        tot_d1 = []
        for i in range(pyr0[-1].shape[0]):
            d0, d1 = pyr_builder.get_row_phase_diff(pyr0, pyr1, i)
            tot_d0.append(d0)
            tot_d1.append(d1)
        print 

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
    test_motion(False)
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

    img0 = cv2.imread(args.img0, cv2.IMREAD_GRAYSCALE)
    pyr = RieszPyramid(img0)
    #pyr.disp_refimg_riesz()
    vals = []
    for img1 in args.img1:
        img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        pyr.set_image(img1)
        cur = float(pyr.get_avg_phase_diff())
        vals.append(cur)
        print cur
        if args.update and len(vals) % args.update == 0:
            print 'update ref'
            pyr = RieszPyramid(img1)

        HEIGHT = 1
        pdiff = []
        for row in range(HEIGHT / 2, img0.shape[0] - HEIGHT * 3 / 2):
            pdiff.append(pyr.get_avg_phase_diff(slicer[row:row+HEIGHT]))
    print vals
    if args.do:
        with open(args.do, 'w') as fout:
            json.dump(vals, fout)

if __name__ == '__main__':
    main()

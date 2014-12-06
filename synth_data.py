#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: synth_data.py
# $Date: Sun Dec 07 00:04:23 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import cv2
import numpy as np
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def gen_frame(start_time, args):
    # f(x, y) = sin(x + shift(t0 + dy)) + sin(y)
    start_line = start_time / args.line_delay
    dx_pxl = None
    for freq, amp in zip(args.freq, args.amp):
        prd_nr_line = 1.0 / freq / args.line_delay
        k = np.pi * 2 / prd_nr_line
        cur = np.sin(start_line + np.arange(args.img_size) * k) * amp
        if dx_pxl is None:
            dx_pxl = cur
        else:
            dx_pxl += cur
    pat_w = np.pi * 2 / args.pattern_period
    x0 = np.arange(args.img_size)
    v0 = np.add.outer(dx_pxl, x0)
    v1 = np.tile(x0, args.img_size).reshape(args.img_size, args.img_size).T
    val = np.sin(v0 * pat_w) + np.sin(v1 * pat_w)
    if args.noise:
        val += np.random.normal(scale=args.noise*4/255.0, size=val.shape)
    val = cv2.normalize(val, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    if args.transpose:
        val = val.T
    return val.astype('uint8')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=720)
    parser.add_argument('--nr_frame', type=int, required=True)
    parser.add_argument('--noise', type=float, default=1.0,
                        help='noise added to individual pixels')
    parser.add_argument('--line_delay', type=float, default=18e-6)
    parser.add_argument('--fps', type=float, default=60.0)
    parser.add_argument('--pattern_period', type=float, default=20,
                        help='period of patterns on image')
    parser.add_argument('--freq', type=float, default=[500], nargs='*',
                        help='target signal frequency')
    parser.add_argument('--amp', type=float, default=[0.05], nargs='*',
                        help='target signal amplitude, in pixels;'
                        ' must match given freq')
    parser.add_argument('-o', '--output', required=True,
                        help='output basename')
    parser.add_argument('--transpose', action='store_true',
                        help='transpose output image')
    args = parser.parse_args()

    assert len(args.freq) == len(args.amp)
    for freq, amp in zip(args.freq, args.amp):
        logger.info('add signal: f={} A={}'.format(freq, amp))
        assert 1.0 / args.line_delay >= freq * 2

    frame_time = 1 / args.fps
    sp_time = args.img_size * args.line_delay
    logger.info('sample_time={:.3e}  frame_duration={:.3e}; k={:.3f}'.format(
        sp_time, frame_time, sp_time / frame_time))
    assert frame_time >= sp_time

    for i in range(args.nr_frame):
        frame = gen_frame(i * frame_time, args)
        fpath = '{}-{:02}.png'.format(args.output, i)
        cv2.imwrite(fpath, frame)
    with open('{}-args.txt'.format(args.output), 'w') as fout:
        fout.write(str(args))

if __name__ == '__main__':
    main()

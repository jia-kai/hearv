#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: get_rolling.py
# $Date: Sat Nov 15 23:46:54 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import cv2
import sys
import numpy as np
import argparse

THRESH = 10

def get_height(fpath):
    print fpath
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    first_row = -1
    in_block = False
    max_len = 0
    for r in range(img.shape[0]):
        if np.mean(img[r]) >= THRESH:
            if not in_block:
                in_block = True
                first_row = r
        else:
            if in_block:
                in_block = False
                print '[{}, {}): len={}'.format(first_row, r, r - first_row)
                max_len = max(max_len, r - first_row)
    return max_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shutter_time')
    parser.add_argument('img_fpath', nargs='+',
                        help='in the format ".../<shutter_type>-num"')
    args = parser.parse_args()

    with open(args.shutter_time) as fin:
        shutter_time = [1 / float(i) for i in fin.readlines()]

    height = list()
    for fpath in args.img_fpath:
        val = get_height(fpath)
        t = int(fpath.split('/')[-1].split('-')[0])
        if t >= len(height):
            assert t == len(height)
            height.append([])
        assert t == len(height) - 1
        height[-1].append(val)

    assert len(shutter_time) == len(height)
    height = map(np.mean, height)

    line_delay = []

    print 'shutter time, height, line delay'
    for i, (t, h) in enumerate(zip(shutter_time, height)):
        ld = '--'
        if i:
            de = shutter_time[i - 1] - shutter_time[i]
            dh = height[i - 1] - height[i]
            ld = de / dh
            line_delay.append(ld)
            ld = '{:.3e}'.format(ld)
        print int(1 / t), '{:.2f}'.format(h), ld

    print 'avg line_delay:', np.mean(line_delay)


if __name__ == '__main__':
    main()

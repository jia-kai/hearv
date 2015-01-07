#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: plot_param.py
# $Date: Wed Jan 07 02:13:41 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import matplotlib.pyplot as plt

import numpy as np

def read_file(name):
    xall = []
    yall = []
    with open('data/params/{}.txt'.format(name)) as fin:
        for line in fin:
            x, y = map(float, line.split())
            xall.append(x)
            yall.append(y)
    return np.array(xall), np.array(yall)

def setup(x):
    plt.figure()
    plt.ylabel('SNR')
    plt.xlabel(x)
    plt.grid()

def save(name):
    plt.savefig('data/params/{}.png'.format(name))

def main():
    setup(r'$g_\theta$')
    x, y = read_file('vg')
    plt.plot(x, y, 'o-')
    save('vg')

    setup(r'$\theta_m$')
    x, y = read_file('lvl')
    plt.plot(x, y, label=r'$g_\theta=150$')
    x, y = read_file('lvl-vg10')
    plt.plot(x, y * 2.5, label=r'$g_\theta=10$ (SNR scaled)')
    plt.xticks(map(int, x))
    plt.legend(loc='best')
    save('lvl')

if __name__ == '__main__':
    main()

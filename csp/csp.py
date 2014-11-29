#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: csp.py
# $Date: Sat Nov 29 16:36:32 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import os

import pymatlab
import numpy as np

session = pymatlab.session_factory()
session.run("addpath('{}/textureSynth')".format(os.path.dirname(__file__)))
session.run("addpath('{}/textureSynth/matlabPyrTools')".format(os.path.dirname(__file__)))

def build_csp(img, nr_orient=4, nr_scale=4):
    pass

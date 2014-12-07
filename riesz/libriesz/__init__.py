#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Sun Dec 07 13:34:00 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import pyximport
pyximport.install()

import logging
logging.basicConfig(
    format='\033[32m[%(asctime)s %(lineno)d@%(filename)s:%(name)s]\033[0m'
    ' %(message)s',
    datefmt='%d %H:%M:%S', level=logging.INFO)

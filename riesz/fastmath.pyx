# -*- coding: utf-8 -*-
# $File: fastmath.pyx
# $Date: Fri Nov 21 00:42:25 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t
cdef DTYPE_t MAX_DIFF = np.pi / 2 + 1e-4

cdef _update_row(np.ndarray[DTYPE_t, ndim=2] row, DTYPE_t tgt_val):
    cdef DTYPE_t mean, mean_orig, cur_max_diff
    cur_max_diff = np.max(np.abs(row[:-1, 1] - row[1:, 1]))
    assert cur_max_diff <= MAX_DIFF, cur_max_diff
    mean = np.mean(row[:, 1])
    if abs(mean - tgt_val) <= MAX_DIFF:
        return
    mean_orig = mean
    cdef int chg_sign = 0
    while abs(mean - tgt_val) > MAX_DIFF:
        if mean > tgt_val:
            mean -= np.pi
        else:
            mean += np.pi
        chg_sign ^= 1
    row[:, 1] += mean - mean_orig
    if chg_sign:
        np.negative(row[:, 2], row[:, 2])

@cython.boundscheck(False)
def smooth_orient(np.ndarray[DTYPE_t, ndim=3] riesz):
    cdef int row, col
    cdef int nr_row = riesz.shape[0]
    cdef int nr_col = riesz.shape[1]
    cdef DTYPE_t vprev, vnow

    # independently smooth each row
    for row in range(nr_row):
        if not row:
            vprev = 0
        else:
            vprev = riesz[row, 0, 1]
        for col in range(row > 0, nr_col):
            vnow = riesz[row, col, 1]
            if abs(vnow - vprev) > MAX_DIFF:
                phase = riesz[row, col, 2]
                while abs(vnow - vprev) > MAX_DIFF:
                    if vnow < vprev:
                        vnow += np.pi
                    else:
                        vnow -= np.pi
                    phase = -phase
                riesz[row, col, 1] = vnow
                riesz[row, col, 2] = phase
            vprev = vnow
        _update_row(riesz[row], 0)

    vprev = np.mean(riesz[0, :, 1])
    for row in range(nr_row):
        _update_row(riesz[row], vprev)
        vprev = np.mean(riesz[row, :, 1])

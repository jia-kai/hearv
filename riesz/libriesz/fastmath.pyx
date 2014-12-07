# -*- coding: utf-8 -*-
# $File: fastmath.pyx
# $Date: Sun Dec 07 15:59:35 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t
cdef DTYPE_t MAX_DIFF = np.pi / 2 + 1e-4

@cython.boundscheck(False)
def smooth_pca(np.ndarray[DTYPE_t, ndim=3] pca):
    cdef DTYPE_t tgt_x, tgt_y, x, y, smooth_coeff = 0.95

    # smooth each row
    for row_idx in range(pca.shape[0]):
        tgt_x = tgt_y = 0
        row = pca[row_idx]
        for col_idx in range(pca.shape[1]):
            x, y = row[col_idx]
            if x * tgt_x + y * tgt_y < 0:
                x = -x
                y = -y
                row[col_idx] = x, y
            tgt_x = tgt_x * smooth_coeff + x * (1 - smooth_coeff)
            tgt_y = tgt_y * smooth_coeff + y * (1 - smooth_coeff)

    # smooth across row
    tgt_x = tgt_y = 0
    for row_idx in range(pca.shape[0]):
        row = pca[row_idx]
        x, y = np.mean(row, axis=0)
        if x * tgt_x + y * tgt_y < 0:
            x = -x
            y = -y
            np.negative(row, row)
        tgt_x = tgt_x * smooth_coeff + x * (1 - smooth_coeff)
        tgt_y = tgt_y * smooth_coeff + y * (1 - smooth_coeff)


cdef DTYPE_t _update_row(np.ndarray[DTYPE_t, ndim=2] row, DTYPE_t tgt_val):
    cdef DTYPE_t mean, mean_orig, cur_max_diff
    mean = np.mean(row[:, 1])
    if abs(mean - tgt_val) <= MAX_DIFF:
        return mean
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
    return mean

@cython.boundscheck(False)
def smooth_orient(np.ndarray[DTYPE_t, ndim=3] riesz):
    cdef int row, col
    cdef int nr_row = riesz.shape[0]
    cdef int nr_col = riesz.shape[1]
    cdef DTYPE_t vprev, vnow, smooth_coeff = 0.95

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
            vprev = vprev * smooth_coeff + vnow * (1 - smooth_coeff)
        _update_row(riesz[row], 0)

    vprev = np.mean(riesz[0, :, 1])
    for row in range(nr_row):
        vnow = _update_row(riesz[row], vprev)
        vprev = vprev * smooth_coeff + vnow * (1 - smooth_coeff)

@cython.boundscheck(False)
def regularize_orient(np.ndarray[DTYPE_t, ndim=3] img,
                      np.ndarray[DTYPE_t, ndim=3] ref):
    cdef int row
    cdef DTYPE_t vimg, vref, phase
    for row in range(img.shape[0]):
        r0 = img[row]
        r1 = ref[row]
        for col in range(img.shape[1]):
            vimg = r0[col, 1]
            vref = r1[col, 1]
            if abs(vimg - vref) > MAX_DIFF:
                phase = r0[col, 2]
                while abs(vimg - vref) > MAX_DIFF:
                    if vimg < vref:
                        vimg += np.pi
                    else:
                        vimg -= np.pi
                    phase = -phase
                r0[col, 1] = vimg
                r0[col, 2] = phase

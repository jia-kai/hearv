# -*- coding: utf-8 -*-
# $File: fastmath.pyx
# $Date: Sun Dec 07 14:29:23 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t

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

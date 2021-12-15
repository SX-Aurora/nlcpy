#
# * The source code in this file is developed independently by NEC Corporation.
#
# # NLCPy License #
#
#     Copyright (c) 2020-2021 NEC Corporation
#     All rights reserved.
#
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither NEC Corporation nor the names of its contributors may be
#       used to endorse or promote products derived from this software
#       without specific prior written permission.
#
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#     ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#     WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#     ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#     (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#     SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# distutils: language = c++

import nlcpy
import numpy

from libc.stdint cimport *

from nlcpy import veo
from nlcpy.core cimport core
from nlcpy.core cimport broadcast
from nlcpy.core cimport dtype as _dtype
from nlcpy.core cimport internal
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport vememory
from nlcpy.request cimport request

import time

cdef CblasRowMajor = <int64_t>101
cdef CblasColMajor = <int64_t>102
cdef CblasNoTrans = <int64_t>111
cdef CblasTrans = <int64_t>112


cpdef ndarray cblas_dot(ndarray x, ndarray y, out=None):
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('mismatch number of dimensions')
    if x.size != y.size:
        raise ValueError('mismatch number of array elements')
    if out is not None and out.size != 1:
        raise ValueError('mismatch output shapes')

    dtype_out = numpy.result_type(x.dtype, y.dtype)

    if out is None:
        out = nlcpy.ndarray(shape=(), dtype=dtype_out)
    out.fill(0)

    if x.dtype == numpy.float32 and y.dtype == numpy.float32:
        request._push_request(
            "wrapper_cblas_sdot",
            "cblas_op",
            (x, y, out),
        )
    elif x.dtype == numpy.float64 and y.dtype == numpy.float64:
        request._push_request(
            "wrapper_cblas_ddot",
            "cblas_op",
            (x, y, out),
        )
    elif x.dtype == numpy.complex64 and y.dtype == numpy.complex64:
        request._push_request(
            "wrapper_cblas_cdotu_sub",
            "cblas_op",
            (x, y, out),
        )
    elif x.dtype == numpy.complex128 and y.dtype == numpy.complex128:
        request._push_request(
            "wrapper_cblas_zdotu_sub",
            "cblas_op",
            (x, y, out),
        )
    elif dtype_out not in (
        numpy.int8, numpy.int16,
        numpy.uint8, numpy.uint16, numpy.float16
    ):
        request._push_request(
            "nlcpy_dot",
            "linalg_op",
            (x, y, out),
        )
    else:
        raise TypeError('output dtype \'%s\' is not supported' % dtype_out)

    return out


cpdef ndarray cblas_gemm(ndarray x, ndarray y, out=None, order='K', dtype=None):
    order_char = internal._normalize_order(order)
    if order_char == b'F':
        order_out = 'F'
    elif order_char == b'C':
        order_out = 'C'
    elif order_char == b'A':
        if x._f_contiguous and not x._c_contiguous and \
                y._f_contiguous and not y._c_contiguous:
            order_out = 'F'
        else:
            order_out = 'C'
    elif order_char == b'K':
        order_out = 'C'
    else:
        raise ValueError('unknown order was detected.')

    if not x._c_contiguous and not x._f_contiguous:
        x = x.copy(order=order_out)
    if not y._c_contiguous and not y._f_contiguous:
        y = y.copy(order=order_out)

    if (max(x.ndim, y.ndim) > 2):
        raise NotImplementedError('not supported for ndim > 2.')
    if (x.ndim == 2):
        m = <int>x.shape[0]
        k1= <int>x.shape[1]
    elif (x.ndim==1):
        m = <int>1
        k1= <int>x.shape[0]
    else:
        raise ValueError("matmul: Input operand 0 does not have enough "
                         "dimensions (has 0, gufunc core with signature"
                         " (n?,k),(k,m?)->(n?,m?) requires 1)")

    if (y.ndim == 2):
        k2= <int>y.shape[0]
        n = <int>y.shape[1]
    elif (y.ndim == 1):
        k2= <int>y.shape[0]
        n = <int>1
    else:
        raise ValueError("matmul: Input operand 1 does not have enough "
                         "dimensions (has 0, gufunc core with signature"
                         " (n?,k),(k,m?)->(n?,m?) requires 1)")

    # bool gemm is not supported
    if x.dtype == numpy.dtype("bool") or \
        y.dtype == numpy.dtype("bool") or \
            dtype == numpy.dtype("bool"):
        raise NotImplementedError("not supported for boolean arguments.")

    # shape check
    if (k1!=k2):
        raise ValueError(
            'shapes ({},{}) and ({},{}) not aligned: {} (dim 1) = {} (dim 0)'
            .format(m, k1, k2, n, k1, k2))

    values = [x, y]
    if (x._f_contiguous and not x._c_contiguous):
        x_is_c_contiguous=<int>0
    else:
        x_is_c_contiguous=<int>1

    if (y._f_contiguous and not y._c_contiguous):
        y_is_c_contiguous=<int>0
    else:
        y_is_c_contiguous=<int>1

    if (x.ndim==1 and y.ndim==1):
        shape = nlcpy.ndarray(()).shape
    elif (x.ndim==1):
        shape = [n]
    elif (y.ndim==1):
        shape = [m]
    else:
        shape = [m, n]

    if out is None:
        z = ndarray(shape=shape, dtype=dtype, order=order_out)
    else:
        raise NotImplementedError('out is not implemented yet.')
    if z.ve_adr == 0:
        raise MemoryError()
    # quick return
    if m == 0 or n == 0:
        return z
    if k1 == 0:
        return nlcpy.zeros(shape, dtype=dtype)

    values = [x, y, z]

    if values[0].dtype is numpy.dtype('float64') and \
        values[1].dtype is numpy.dtype('float64') and \
            dtype is numpy.dtype('float64'):
        # use dgemm
        a = x
        b = y
        c = z
        alpha = numpy.float64(1.0)
        beta = numpy.float64(0.0)

        if order_out is 'C':
            if x_is_c_contiguous == 1 and y_is_c_contiguous == 1:
                args = (
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, k1, b, n,
                    beta, c, n
                )
            elif x_is_c_contiguous == 1 and y_is_c_contiguous == 0:
                args = (
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, k1, b, k1,
                    beta, c, n
                )
            elif x_is_c_contiguous == 0 and y_is_c_contiguous == 1:
                args = (
                    CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, m, b, n,
                    beta, c, n
                )
            else:
                args = (
                    CblasRowMajor,
                    CblasTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, m, b, k1,
                    beta, c, n
                )
        else:
            if x_is_c_contiguous == 0 and y_is_c_contiguous == 0:
                args = (
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, m, b, k1,
                    beta, c, m
                )
            elif x_is_c_contiguous == 0 and y_is_c_contiguous == 1:
                args = (
                    CblasColMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, m, b, n,
                    beta, c, m
                )
            elif x_is_c_contiguous == 1 and y_is_c_contiguous == 0:
                args = (
                    CblasColMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, k1, b, k1,
                    beta, c, m
                )
            else:
                args = (
                    CblasColMajor,
                    CblasTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, k1, b, n,
                    beta, c, m
                )
        request._push_request(
            "wrapper_cblas_dgemm",
            "cblas_op",
            args,
        )
        return z

    elif values[0].dtype is numpy.dtype('float32') and \
        values[1].dtype is numpy.dtype('float32') and \
            dtype is numpy.dtype('float32'):
        # use sgemm
        a = x
        b = y
        c = z
        alpha = numpy.float32(1.0)
        beta = numpy.float32(0.0)

        if order_out is 'C':
            if x_is_c_contiguous == 1 and y_is_c_contiguous == 1:
                args = (
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, k1, b, n,
                    beta, c, n
                )
            elif x_is_c_contiguous == 1 and y_is_c_contiguous == 0:
                args = (
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, k1, b, k1,
                    beta, c, n
                )
            elif x_is_c_contiguous == 0 and y_is_c_contiguous == 1:
                args = (
                    CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, m, b, n,
                    beta, c, n
                )
            else:
                args = (
                    CblasRowMajor,
                    CblasTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, m, b, k1,
                    beta, c, n
                )
        else:
            if x_is_c_contiguous == 0 and y_is_c_contiguous == 0:
                args = (
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, m, b, k1,
                    beta, c, m
                )
            elif x_is_c_contiguous == 0 and y_is_c_contiguous == 1:
                args = (
                    CblasColMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, m, b, n,
                    beta, c, m
                )
            elif x_is_c_contiguous == 1 and y_is_c_contiguous == 0:
                args = (
                    CblasColMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, k1, b, k1,
                    beta, c, m
                )
            else:
                args = (
                    CblasColMajor,
                    CblasTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, k1, b, n,
                    beta, c, m
                )
        request._push_request(
            "wrapper_cblas_sgemm",
            "cblas_op",
            args,
        )
        return z

    elif values[0].dtype is numpy.dtype('complex128') and \
        values[1].dtype is numpy.dtype('complex128') and \
            dtype is numpy.dtype('complex128'):
        # use zgemm
        a = x
        b = y
        c = z
        alpha = numpy.complex128(1+0j)
        beta = numpy.complex128(0+0j)

        if order_out is 'C':
            if x_is_c_contiguous == 1 and y_is_c_contiguous == 1:
                args = (
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, k1, b, n,
                    beta, c, n
                )
            elif x_is_c_contiguous == 1 and y_is_c_contiguous == 0:
                args = (
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, k1, b, k1,
                    beta, c, n
                )
            elif x_is_c_contiguous == 0 and y_is_c_contiguous == 1:
                args = (
                    CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, m, b, n,
                    beta, c, n
                )
            else:
                args = (
                    CblasRowMajor,
                    CblasTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, m, b, k1,
                    beta, c, n
                )
        else:
            if x_is_c_contiguous == 0 and y_is_c_contiguous == 0:
                args = (
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, m, b, k1,
                    beta, c, m
                )
            elif x_is_c_contiguous == 0 and y_is_c_contiguous == 1:
                args = (
                    CblasColMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, m, b, n,
                    beta, c, m
                )
            elif x_is_c_contiguous == 1 and y_is_c_contiguous == 0:
                args = (
                    CblasColMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, k1, b, k1,
                    beta, c, m
                )
            else:
                args = (
                    CblasColMajor,
                    CblasTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, k1, b, n,
                    beta, c, m
                )
        request._push_request(
            "wrapper_cblas_zgemm",
            "cblas_op",
            args,
        )
        return z

    elif values[0].dtype is numpy.dtype('complex64') and \
        values[1].dtype is numpy.dtype('complex64') and \
            dtype is numpy.dtype('complex64'):
        # use cgemm
        a = x
        b = y
        c = z
        alpha = numpy.complex64(1+0j)
        beta = numpy.complex64(0+0j)

        if order_out is 'C':
            if x_is_c_contiguous == 1 and y_is_c_contiguous == 1:
                args = (
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, k1, b, n,
                    beta, c, n
                )
            elif x_is_c_contiguous == 1 and y_is_c_contiguous == 0:
                args = (
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, k1, b, k1,
                    beta, c, n
                )
            elif x_is_c_contiguous == 0 and y_is_c_contiguous == 1:
                args = (
                    CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, m, b, n,
                    beta, c, n
                )
            else:
                args = (
                    CblasRowMajor,
                    CblasTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, m, b, k1,
                    beta, c, n
                )
        else:
            if x_is_c_contiguous == 0 and y_is_c_contiguous == 0:
                args = (
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, m, b, k1,
                    beta, c, m
                )
            elif x_is_c_contiguous == 0 and y_is_c_contiguous == 1:
                args = (
                    CblasColMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, m, b, n,
                    beta, c, m
                )
            elif x_is_c_contiguous == 1 and y_is_c_contiguous == 0:
                args = (
                    CblasColMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m, n, k1,
                    alpha, a, k1, b, k1,
                    beta, c, m
                )
            else:
                args = (
                    CblasColMajor,
                    CblasTrans,
                    CblasTrans,
                    m, n, k1,
                    alpha, a, k1, b, n,
                    beta, c, m
                )
        request._push_request(
            "wrapper_cblas_cgemm",
            "cblas_op",
            args,
        )
        return z

    # == general case bellow ==
    args = (values[0], values[1], values[2],)
    request._push_request(
        "nlcpy_matmul",
        "linalg_op",
        args,
    )
    return z

#
# * The source code in this file is developed independently by NEC Corporation.
#
# # NLCPy License #
#
#     Copyright (c) 2020 NEC Corporation
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
from nlcpy.request cimport request

import time

cdef int64_t CblasRowMajor = 101
cdef int64_t CblasColMajor = 102
cdef int64_t CblasNoTrans = 111
cdef int64_t CblasTrans = 112


cpdef ndarray cblas_dot(ndarray x, ndarray y, out=None):
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('mismatch number of dimensions')
    if x.size != y.size:
        raise ValueError('mismatch number of array elements')
    if out is not None and out.size != 1:
        raise ValueError('mismatch output shapes')

    dtype_out = numpy.result_type(x.dtype, y.dtype)

    if dtype_out in (
        numpy.int8, numpy.int16,
        numpy.uint8, numpy.uint16, numpy.float16
    ):
        raise TypeError('output dtype \'%s\' is not supported' % dtype_out)

    if out is None:
        out = nlcpy.ndarray(shape=(), dtype=dtype_out)
    out.fill(0)

    if x.dtype == y.dtype == out.dtype and \
       x.dtype in (numpy.float32, numpy.float64, numpy.complex64, numpy.complex128):
        if x.dtype == numpy.float32:
            name = "wrapper_cblas_sdot"
        elif x.dtype == numpy.float64:
            name = "wrapper_cblas_ddot"
        elif x.dtype == numpy.complex64:
            name = "wrapper_cblas_cdotu_sub"
        elif x.dtype == numpy.complex128:
            name = "wrapper_cblas_zdotu_sub"
        fpe = request._get_fpe_flag()
        request._push_and_flush_request(
            name,
            (x, y, out,
             veo.OnStack(fpe, inout=veo.INTENT_OUT)),
            callback=None,
        )
    else:
        request._push_request(
            "nlcpy_dot",
            "linalg_op",
            (x, y, out),
        )

    return out


cpdef ndarray cblas_gemm(ndarray x, ndarray y, out=None, order='K', dtype=None):
    cdef int64_t cblas_order, transA, transB, m, n, k1, k2, lda, ldb, ldc
    cdef uint64_t x_is_c_contiguous, y_is_c_contiguous
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

    if (max(x.ndim, y.ndim) > 2):
        raise NotImplementedError('not supported for ndim > 2.')
    if (x.ndim == 0 or y.ndim == 0):
        raise ValueError("matmul: Input operand 1 does not have enough "
                         "dimensions (has 0, gufunc core with signature"
                         " (n?,k),(k,m?)->(n?,m?) requires 1)")

    # bool gemm is not supported
    if x.dtype == numpy.dtype("bool") or \
        y.dtype == numpy.dtype("bool") or \
            dtype == numpy.dtype("bool"):
        raise NotImplementedError("not supported for boolean arguments.")

    # shape check
    if (x.shape[-1] != y.shape[0]):
        m = x.shape[0] if x.ndim == 2 else 1
        n = y.shape[1] if y.ndim == 2 else 1
        raise ValueError(
            'shapes ({},{}) and ({},{}) not aligned: {} (dim 1) = {} (dim 0)'
            .format(m, x.shape[-1], y.shape[0], n, x.shape[-1], y.shape[0]))

    if (x.ndim==1 and y.ndim==1):
        if x.size > 10000 or x.dtype.kind == 'c' and x.size > 2000:
            return cblas_dot(x, y, out=out)
        shape = nlcpy.ndarray(()).shape
    elif (x.ndim==1):
        shape = y.shape[1]
    elif (y.ndim==1):
        shape = x.shape[0]
    else:
        shape = [x.shape[0], y.shape[1]]

    if out is None:
        z = ndarray(shape=shape, dtype=dtype, order=order_out)
    else:
        raise NotImplementedError('out is not implemented yet.')
    if z.ve_adr == 0:
        raise MemoryError()
    # quick return
    if x.size == 0 or y.shape[0] == 0:
        z.fill(0)
        return z

    if x.dtype == y.dtype and (dtype is None or dtype == x.dtype) \
       and x.dtype in (numpy.float32, numpy.float64, numpy.complex64, numpy.complex128):
        if x.dtype == numpy.float32:
            name = "wrapper_cblas_sgemm"
        elif x.dtype == numpy.float64:
            name = "wrapper_cblas_dgemm"
        elif x.dtype == numpy.complex64:
            name = "wrapper_cblas_cgemm"
        elif x.dtype == numpy.complex128:
            name = "wrapper_cblas_zgemm"
        if (x.ndim == 2):
            if x.strides[0] == x.strides[1]:
                x_is_c_contiguous = 1
                lda = x.shape[1]
            elif x.strides[1] == x.itemsize:
                x_is_c_contiguous = 1
                lda = x.strides[0] / x.itemsize
            elif x.strides[0] == x.itemsize:
                x_is_c_contiguous = 0
                lda = x.strides[1] / x.itemsize
            else:
                x = x.copy(order=order_out)
                x_is_c_contiguous = 0 if order_out == 'F' else 1
                lda = x.shape[1] if x_is_c_contiguous else x.shape[0]
            m = x.shape[0]
            k1= x.shape[1]
        else:
            x_is_c_contiguous = 1
            if x.strides[0] != x.itemsize:
                x = x.copy()
            m = 1
            k1= x.shape[0]
            lda = k1

        if (y.ndim == 2):
            if y.strides[0] == y.strides[1]:
                y_is_c_contiguous = 1
                ldb = y.shape[1]
            elif y.strides[1] == y.itemsize:
                y_is_c_contiguous = 1
                ldb = y.strides[0] / y.itemsize
            elif y.strides[0] == y.itemsize:
                y_is_c_contiguous = 0
                ldb = y.strides[1] / y.itemsize
            else:
                y = y.copy(order=order_out)
                y_is_c_contiguous = 0 if order_out == 'F' else 1
                ldb = y.shape[1] if y_is_c_contiguous else y.shape[0]
            k2= y.shape[0]
            n = y.shape[1]
        else:
            y_is_c_contiguous = 1
            if y.strides[0] != y.itemsize:
                y = y.copy()
            k2= y.shape[0]
            n = 1
            ldb = 1

        if order_out is 'C':
            cblas_order = CblasRowMajor
            ldc = n
            transA = CblasNoTrans if x_is_c_contiguous else CblasTrans
            transB = CblasNoTrans if y_is_c_contiguous else CblasTrans
        else:
            cblas_order = CblasColMajor
            ldc = m
            transA = CblasTrans if x_is_c_contiguous else CblasNoTrans
            transB = CblasTrans if y_is_c_contiguous else CblasNoTrans
        fpe = request._get_fpe_flag()
        args = (
            cblas_order,
            transA,
            transB,
            m,
            n,
            k1,
            x,
            lda,
            y,
            ldb,
            z,
            ldc,
            veo.OnStack(fpe, inout=veo.INTENT_OUT),
        )
        request._push_and_flush_request(
            name,
            args,
            callback=None,
        )
        return z

    # == general case bellow ==
    args = (x, y, z)
    request._push_request(
        "nlcpy_matmul",
        "linalg_op",
        args,
    )
    return z

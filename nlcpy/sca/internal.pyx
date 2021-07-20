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

from nlcpy import veo
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport internal
from nlcpy.sca cimport utility
from nlcpy.sca cimport kernel

from libcpp.vector cimport vector
from libc.stdint cimport *

import numpy
cimport numpy as cnp


cdef inline ndarray _get_fact_1d(fact, ls, le, dtype):
    n_fact = abs(le - ls) + 1
    if type(fact) is ndarray:
        if fact.size != n_fact:
            raise ValueError('input fact has invalid length.')
    elif type(fact) in (list, tuple, numpy.ndarray):
        if len(fact) != n_fact:
            raise ValueError('input fact has invalid length.')
        a_fact = nlcpy.asarray(fact, dtype=dtype)
    elif type(fact) in (int, float):
        a_fact = nlcpy.empty((n_fact,), dtype=dtype)
        a_fact[...] = fact
    else:
        raise TypeError('fact type must be ndarray/list/tuple/scalar')

    return a_fact


cdef inline ndarray _get_fact_2da(fact, ls1, le1, ls2, le2, dtype):
    n_fact = abs(le1 + ls1) + abs(ls2 + le2) + 1
    if type(fact) is ndarray:
        if fact.size != n_fact:
            raise ValueError('input fact has invalid length.')
    elif type(fact) in (list, tuple, numpy.ndarray):
        if len(fact) != n_fact:
            raise ValueError('input fact has invalid length.')
        a_fact = nlcpy.asarray(fact, dtype=dtype)
    elif type(fact) in (int, float):
        a_fact = nlcpy.empty((n_fact,), dtype=dtype)
        a_fact[...] = fact
    else:
        raise TypeError('fact type must be ndarray/list/tuple/scalar')

    return a_fact


cdef inline ndarray _get_fact_3da(fact, ls1, le1, ls2, le2, ls3, le3, dtype):
    n_fact = abs(le1 + ls1) + abs(ls2 + le2) + abs(ls3 + le3) + 1
    if type(fact) is ndarray:
        if fact.size != n_fact:
            raise ValueError('input fact has invalid length.')
    elif type(fact) in (list, tuple, numpy.ndarray):
        if len(fact) != n_fact:
            raise ValueError('input fact has invalid length.')
        a_fact = nlcpy.asarray(fact, dtype=dtype)
    elif type(fact) in (int, float):
        a_fact = nlcpy.empty((n_fact,), dtype=dtype)
        a_fact[...] = fact
    else:
        raise TypeError('fact type must be ndarray/list/tuple/scalar')

    return a_fact


cdef inline vector[Py_ssize_t] _estimate_out_shape(ndim, nx, ny, nz, nw):
    cdef vector[Py_ssize_t] out_shape
    out_shape.resize(ndim, 0)

    out_shape[ndim-1] = nx
    if ndim > 1:
        out_shape[ndim-2] = ny
    if ndim > 2:
        out_shape[ndim-3] = nz
    if ndim > 3:
        out_shape[ndim-4] = nw

    return out_shape


cdef inline tuple _get_leading_dimensions(ndarray x):
    cdef int64_t mx, my, mz

    sx = x._strides[x.ndim - 1]
    mx = 0
    my = 0
    mz = 0

    if x.ndim >= 2:
        mx = <int64_t>(x._strides[x.ndim-2] // sx)

    if x.ndim >= 3:
        my = <int64_t>(x._strides[x.ndim-3] // (sx * mx))

    if x.ndim >= 4:
        mz = <int64_t>(x._strides[x.ndim-4] // (sx * mx * my))

    return (mx, my, mz)

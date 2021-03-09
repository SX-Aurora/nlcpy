#
# * The source code in this file is based on the soure code of CuPy.
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
# # CuPy License #
#
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

# distutils: language = c++

from libcpp.vector cimport vector
from libc.stdint cimport *

from nlcpy.core.core cimport ndarray
from nlcpy.core.core cimport MemoryLocation
from nlcpy.core cimport internal
from nlcpy.core cimport vememory

import numpy
import sys
cimport cython

cdef Py_ssize_t PY_SSIZE_T_MAX = sys.maxsize

cpdef tuple _broadcast_core(arrays):
    cdef Py_ssize_t i, j, s, smin, smax, a_ndim, a_shape, nd
    cdef vector[Py_ssize_t] shape, strides
    cdef ndarray a
    cdef list ret

    ret = list(arrays)
    nd = 0
    # find maximum number of dimensions
    for i, x in enumerate(ret):
        if not isinstance(x, ndarray):
            ret[i] = None
            continue
        a = x
        nd = max(nd, <Py_ssize_t>a._shape.size())

    # set the broadcasted shapes and values
    shape.reserve(nd)
    for i in range(nd):
        smin = PY_SSIZE_T_MAX
        smax = 0
        for a in ret:
            if a is None:
                continue
            a_ndim = <Py_ssize_t>a._shape.size()
            if i >= nd - a_ndim:
                s = a._shape[i - (nd - a_ndim)]
                smin = min(smin, s)
                smax = max(smax, s)
        if smin == 0 and smax > 1:
            raise ValueError(
                'shape mismatch: objects cannot be broadcast to a '
                'single shape')
        shape.push_back(0 if smin == 0 else smax)

    for i, a in enumerate(ret):
        if a is None:
            ret[i] = arrays[i]
            continue
        if internal.vector_equal(a._shape, shape):
            continue

        strides.assign(nd, <Py_ssize_t>0)
        a_ndim = <Py_ssize_t>a._shape.size()
        for j in range(a_ndim):
            # a_shape = a._shape[j]
            if a._shape[j] == shape[j + nd - a_ndim]:
                strides[j + nd - a_ndim] = a._strides[j]
            elif a._shape[j] != 1:
                raise ValueError(
                    'operands could not be broadcast together with shapes '
                    '{}'.format(
                        ', '.join([str(x.shape) if isinstance(x, ndarray)
                                  else '()' for x in arrays])))

        ret[i] = a._view(shape, strides, True, True)
    return ret, tuple(shape)


cpdef ndarray broadcast_to(ndarray array, shape):
    if numpy.isscalar(shape):
        shape = (shape,)
    cdef int i, j, ndim = array._shape.size(), length = len(shape)
    cdef Py_ssize_t sh, a_sh

    if array._memloc == MemoryLocation.on_VH:
        raise NotImplementedError(
            'broadcast_to with _memloc=\'on_VH\' not yet implemented.')

    if ndim > length:
        raise ValueError(
            'input operand has more dimensions than allowed by the axis '
            'remapping')
    cdef vector[Py_ssize_t] strides, _shape = shape
    strides.assign(length, 0)
    for i in range(ndim):
        j = i + length - ndim
        sh = _shape[j]
        a_sh = array._shape[i]
        if sh == a_sh:
            strides[j] = array._strides[i]
        elif a_sh != 1:
            raise ValueError(
                'operands could not be broadcast together with shape {} and '
                'requested shape {}'.format(array.shape, shape))

    vh_view = None
    if array._memloc in {MemoryLocation.on_VH, MemoryLocation.on_VE_VH}:
        vh_view = numpy.broadcast_to(array.vh_data, shape)
    view = array._view(_shape, strides, True, True, True, vh_view=vh_view)
    return view

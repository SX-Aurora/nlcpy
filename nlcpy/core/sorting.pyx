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

import numpy

from nlcpy.core.core cimport ndarray
from nlcpy.core cimport internal
from nlcpy.core.core cimport MemoryLocation
from nlcpy.core cimport core
from nlcpy.core cimport manipulation
from nlcpy.core cimport broadcast
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.core cimport dtype as _dtype
from nlcpy.request cimport request

import nlcpy
from nlcpy import veo
from nlcpy.core import error

cimport cython
cimport cpython

cdef _ndarray_sort(ndarray self, int axis, kind=None, order=None):
    cdef int ndim = self._shape.size()

    if ndim == 0:
        raise ValueError('Sorting arrays with the rank of zero is not '
                         'supported')
    # axis check
    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise AxisError('Axis out of range')
    # move axis dimension to inner
    if axis == ndim - 1:
        data = self
    else:
        data = manipulation._rollaxis(self, axis, start=ndim)
    # move the largest dimension(exclusive axis dim) to outer
    if data.ndim > 2:
        outer_dim = numpy.argmax(data.shape[:-1])
        data = manipulation._rollaxis(data, outer_dim, start=0)

    request._push_request(
        "nlcpy_sort",
        "sorting_op",
        (data,),
    )


cdef ndarray _ndarray_argsort(ndarray self, axis, kind=None, order=None):
    cdef int ndim = self._shape.size()

    if ndim == 0:
        return nlcpy.array([0])
    #     raise ValueError('Sorting arrays with the rank of zero is not '
    #                      'supported')

    # axis check
    if axis is None:
        self = self.ravel()
        axis = -1
    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise AxisError('Axis out of range')

    # move axis dimension to inner
    if axis == ndim - 1:
        data = self
    else:
        data = manipulation._rollaxis(self, axis, start=ndim)
    # move the largest dimension(exclusive axis dim) to outer
    if data.ndim > 2:
        outer_dim = numpy.argmax(data.shape[:-1])
        data = manipulation._rollaxis(data, outer_dim, start=0)
    else:
        outer_dim = 0

    idx = nlcpy.empty(data.shape, dtype='i8')
    request._push_request(
        "nlcpy_argsort",
        "sorting_op",
        (data, idx),
    )

    # restore outer axis
    if outer_dim != 0:
        idx = manipulation._moveaxis(idx, 0, outer_dim)
    # restore inner axis
    if axis != ndim - 1:
        idx = manipulation._moveaxis(idx, -1, axis)
    # return c contiguous array to match strides to numpy.argsort
    if not idx._c_contiguous:
        return idx.copy()
    else:
        return idx

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

from libcpp.vector cimport vector
from libcpp cimport vector
from libc.stdint cimport *

from nlcpy.core.core cimport ndarray

cdef ndarray _ndarray_reshape(ndarray self, tuple shape, order)

cdef _ndarray_shape_setter(ndarray self, newshape)

cpdef _copyto(ndarray dst, src, casting, where)

cdef ndarray _ndarray_ravel(ndarray self, order)

cdef ndarray _ndarray_resize(ndarray self, tuple shape, refcheck)

cpdef ndarray _reshape(ndarray self,
                       const vector[Py_ssize_t] &shape_spec)

cpdef ndarray _T(ndarray self)

cpdef ndarray _transpose(ndarray self, const vector[Py_ssize_t] &axes)

cpdef ndarray _rollaxis(ndarray a, Py_ssize_t axis, Py_ssize_t start=*)

cpdef ndarray _ndarray_swapaxes(ndarray a, Py_ssize_t axis1, Py_ssize_t axis2)

cpdef ndarray _reduced_view(ndarray self)

cdef _fill_kernel(ndarray a, value)

cpdef ndarray _ndarray_concatenate(op, axis, ret)

cdef ndarray _ndarray_transpose(ndarray self, axes)
cdef ndarray _ndarray_flatten(ndarray self, order)

cpdef ndarray _moveaxis(ndarray a, source, destination)

cpdef ndarray _expand_dims(ndarray a, axis)
cpdef ndarray _ndarray_squeeze(ndarray self, axis)
cpdef ndarray _ndarray_repeat(ndarray self, repeats, axis)

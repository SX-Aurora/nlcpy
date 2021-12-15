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
from libc.stdint cimport *

from nlcpy.core cimport vememory

cimport numpy

cpdef enum MemoryLocation:
    on_VE = 0b001
    on_VH = 0b010
    on_VE_VH = 0b100


cdef class ndarray:
    cdef:
        readonly Py_ssize_t size
        public vector[Py_ssize_t] _shape
        public vector[Py_ssize_t] _strides
        readonly bint _c_contiguous
        readonly bint _f_contiguous
        readonly bint _is_view
        readonly bint _owndata
        readonly object dtype
        readonly uint64_t itemsize
        readonly uint64_t ve_adr
        readonly numpy.ndarray vh_data
        readonly ndarray base
        readonly int _memloc  # this flag indicates whether memory locates in VE or VH.

    cpdef sort(self, axis=*, kind=*, order=*)
    cpdef ndarray argsort(self, axis=*, kind=*, order=*)
    cpdef tolist(self)
    cpdef ndarray view(self, dtype=*, object type=*)
    cpdef ndarray copy(self, order=*)
    cpdef ndarray astype(self, dtype, order=*, casting=*, subok=*, copy=*)
    cpdef fill(self, value)

    cpdef ndarray take(self, indices, axis=*, out=*)
    cpdef ndarray diagonal(self, offset=*, axis1=*, axis2=*)
    cpdef ndarray argmax(self, axis=*, out=*)
    cpdef ndarray argmin(self, axis=*, out=*)
    cpdef nonzero(self)
    cpdef all(self, axis=*, out=*, keepdims=*)
    cpdef any(self, axis=*, out=*, keepdims=*)

    cpdef ndarray max(self, axis=*, out=*, keepdims=*, initial=*, where=*)
    cpdef ndarray min(self, axis=*, out=*, keepdims=*, initial=*, where=*)

    cpdef get(self, order=*)
    cpdef _set_contiguous_strides(
        self,
        int64_t itemsize,
        bint is_c_contiguous
    )
    cdef ndarray _view(
        self,
        const vector[Py_ssize_t]& shape,
        const vector[Py_ssize_t]& strides,
        bint update_c_contiguity,
        bint update_f_contiguity,
        bint mem_check=*,
        object vh_view=*,
        object dtype=*,
        object type=*,
        int64_t offset=*
    )
    cpdef _set_contiguous_strides(
        self,
        int64_t itemsize,
        bint is_c_contiguous
    )
    cpdef _update_contiguity(self)
    cpdef _set_shape_and_strides(
        self,
        const vector[Py_ssize_t]& shape,
        const vector[Py_ssize_t]& strides,
        bint update_c_contiguity,
        bint update_f_contiguity
    )
    cpdef _update_c_contiguity(self)
    cpdef _update_f_contiguity(self)


cpdef ndarray array(obj, dtype=*, bint copy=*, order=*, bint subok=*,
                    Py_ssize_t ndmin=*)
cpdef int _update_order_char(ndarray a, int order_char)
cpdef set_boundary_size(size=*)
cpdef int64_t get_boundary_size()
cpdef ndarray argument_conversion(x)
cpdef tuple argument_conversion2(x, y)
cpdef check_fpe_flags(fpe_flags)
cpdef determine_contiguous_property(self, order)

#
# * The source code in this file is based on the soure code of CuPy.
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
from nlcpy.core cimport scalar
from nlcpy.core cimport dtype as _dtype
from nlcpy.request cimport request

import nlcpy
from nlcpy import veo

cimport cython
cimport cpython

cdef ndarray _ndarray_real_getter(ndarray self):
    if self.dtype.kind == 'c':
        view = self._view(
            self._shape,
            self._strides,
            True,
            True,
            False,
            vh_view=None,
            dtype=_dtype.get_dtype(self.dtype.char.lower()),
            offset=0,
        )

        view.base = self.base if self.base is not None else self
        return view
    return self


cdef ndarray _ndarray_real_setter(ndarray self, value):
    # dst check
    if self.dtype.kind == 'c':
        dst = self.real
    else:
        dst = self
    # src check
    if numpy.isscalar(value):
        src = scalar.convert_scalar(value)
    src = nlcpy.asarray(value)
    if src.dtype.kind == 'c':
        src = src.real

    src = broadcast.broadcast_to(src, dst.shape)

    request._push_request(
        "nlcpy_copy",
        "creation_op",
        (src, dst),
    )

cdef ndarray _ndarray_imag_getter(ndarray self):
    if self.dtype.kind == 'c':
        view = self._view(
            self._shape,
            self._strides,
            True,
            True,
            False,
            vh_view=None,
            dtype=_dtype.get_dtype(self.dtype.char.lower()),
            offset=self.itemsize // 2,
        )
        return view
    new_array = ndarray(self.shape, dtype=self.dtype)
    new_array.fill(0)
    return new_array


cdef ndarray _ndarray_imag_setter(ndarray self, value):
    # dst check
    if self.dtype.kind == 'c':
        dst = self.imag
    else:
        raise TypeError('array does not have imaginary part to set')
    # src check
    if numpy.isscalar(value):
        src = scalar.convert_scalar(value)
    src = nlcpy.asarray(value)
    if src.dtype.kind == 'c':
        src = src.imag

    src = broadcast.broadcast_to(src, dst.shape)

    request._push_request(
        "nlcpy_copy",
        "creation_op",
        (src, dst),
    )

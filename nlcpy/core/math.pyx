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
from numpy.core._exceptions import UFuncTypeError

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
            type=None,
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
            type=None,
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


cpdef ndarray _ndarray_clip(ndarray self, a_min, a_max,
                            out, dtype, order, where, casting):
    if a_min is None and a_max is None:
        raise ValueError('array_clip: must set either max or min')
    if order is None or order in 'kK':
        order = 'C'
    elif order in 'aA':
        if self._f_contiguous:
            order='F'
        else:
            order='C'

    if a_min is None:
        _a_min = None
    elif type(a_min) is nlcpy.ndarray:
        if a_min.ndim == 0:
            _a_min = a_min.get()
        else:
            _a_min = a_min
    elif type(a_min) in (list, tuple):
        _a_min = numpy.array(a_min)
    else:
        _a_min = a_min

    if a_max is None:
        _a_max = None
    elif type(a_max) is nlcpy.ndarray:
        if a_max.ndim == 0:
            _a_max = a_max.get()
        else:
            _a_max = a_max
    elif type(a_max) in (list, tuple):
        _a_max = numpy.array(a_max)
    else:
        _a_max = a_max

    if dtype is None:
        dtype = self.dtype
        if a_max is not None and a_min is not None:
            dtype = numpy.result_type(self, _a_max, _a_min)
        elif a_max is not None:
            dtype = numpy.result_type(self, _a_max)
        elif a_min is not None:
            dtype = numpy.result_type(self, _a_min)
        else:
            dtype = self.dtype
    else:
        dtype = nlcpy.dtype(dtype)

    dtypes = (dtype,) if out is None else (out.dtype, dtype)
    for _dtype in dtypes:
        msg1 = None
        if not numpy.can_cast(self, _dtype, casting):
            msg1 = 0
            msg2 = self.dtype
        if _a_min is not None:
            if not numpy.can_cast(_a_min, _dtype, casting):
                msg1 = 1
                msg2 = _a_min.dtype
        if _a_max is not None:
            if not numpy.can_cast(_a_max, _dtype, casting):
                msg1 = 2
                msg2 = _a_max.dtype
        if msg1 is not None:
            raise UFuncTypeError(
                "Cannot cast ufunc 'clip' input {} from dtype('{}') to dtype('{}')"
                " with casting rule '{}'".format(msg1, msg2, _dtype, casting))
    if out is not None and not numpy.can_cast(dtype, out.dtype, casting):
        raise UFuncTypeError(
            "Cannot cast ufunc 'clip' output from dtype('{}') to dtype('{}')"
            " with casting rule '{}'".format(dtype, out.dtype, casting))

    self = nlcpy.asanyarray(self, dtype=dtype)
    if a_min is not None:
        a_min = nlcpy.asanyarray(a_min, dtype=dtype)
    if a_max is not None:
        a_max = nlcpy.asanyarray(a_max, dtype=dtype)

    if where is True:
        where = nlcpy.array(())
        if a_min is None:
            self, a_max = nlcpy.broadcast_arrays(self, a_max)
            a_min = nlcpy.array((), dtype=dtype)
        elif a_max is None:
            self, a_min = nlcpy.broadcast_arrays(self, a_min)
            a_max = nlcpy.array((), dtype=dtype)
        else:
            self, a_min, a_max = nlcpy.broadcast_arrays(self, a_min, a_max)
    else:
        if a_min is None:
            self, a_max, where = nlcpy.broadcast_arrays(self, a_max, where)
            a_min = nlcpy.array((), dtype=dtype)
        elif a_max is None:
            self, a_min, where = nlcpy.broadcast_arrays(self, a_min, where)
            a_max = nlcpy.array((), dtype=dtype)
        else:
            self, a_min, a_max, where = nlcpy.broadcast_arrays(self, a_min, a_max, where)

    if out is not None and self.shape != out.shape:
        raise ValueError(
            "non-broadcastable output operand with shape {} doesn't match "
            "the broadcast shape {}".format(out.shape, self.shape))

    self = nlcpy.asanyarray(self)
    a_min = nlcpy.asanyarray(a_min)
    a_max = nlcpy.asanyarray(a_max)
    where = nlcpy.asanyarray(where)
    work = nlcpy.zeros(self.shape, dtype=dtype, order=order)
    if out is None:
        out = work

    if self.size > 0:
        request._push_request(
            "nlcpy_clip",
            "math_op",
            (self, out, work, a_min, a_max, where),
        )
    return out

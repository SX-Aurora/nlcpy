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

import unittest

import numpy

from nlcpy import testing


def _calc_out_shape(shape, axis, keepdims):
    if axis is None:
        axis = list(range(len(shape)))
    elif isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis)

    shape = numpy.array(shape)

    if keepdims:
        shape[axis] = 1
    else:
        shape[axis] = -1
        shape = filter(lambda x: x != -1, shape)
    return tuple(shape)


@testing.parameterize(
    *testing.product(
        {'f': ['all', 'any'],
         'x': [numpy.arange(24) - 10,
               numpy.arange(24).reshape(4, 6) - 10,
               numpy.arange(24).reshape(2, 3, 4) - 10,
               numpy.zeros((2, 3, 4)),
               numpy.ones((2, 3, 4)),
               # numpy.zeros((0, 3, 4)),
               # numpy.ones((0, 3, 4))
               ],
         'axis': [None, (0, 1, 2), 0, 1, 2, (0, 1)],
         'keepdims': [False, True]}))
class TestAllAny(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(accept_error=numpy.AxisError)
    def test_without_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        return getattr(xp, self.f)(x, self.axis, None, self.keepdims)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(accept_error=numpy.AxisError)
    def test_ndarray_without_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        return getattr(x, self.f)(
            axis=self.axis, out=None, keepdims=self.keepdims)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(accept_error=IndexError)
    def test_with_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        out_shape = _calc_out_shape(x.shape, self.axis, self.keepdims)
        out = xp.empty(out_shape, dtype=x.dtype)
        getattr(xp, self.f)(x, self.axis, out, self.keepdims)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(accept_error=IndexError)
    def test_ndarray_with_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        out_shape = _calc_out_shape(x.shape, self.axis, self.keepdims)
        out = xp.empty(out_shape, dtype=x.dtype)
        getattr(x, self.f)(
            axis=self.axis, out=out, keepdims=self.keepdims)
        return out


@testing.parameterize(
    *testing.product(
        {'f': ['all', 'any'],
         'x': [numpy.array([numpy.nan]),
               numpy.array([numpy.nan, 0]),
               numpy.array([[numpy.nan, 0]]),
               numpy.array([[[numpy.nan]]]),
               numpy.array([[[numpy.nan, 0]]]),
               numpy.array([[[numpy.nan, 1]]]),
               numpy.array([[[numpy.nan, 0, 1]]])],
         'axis': [None, (0, 1, 2), 0, 1, 2, (0, 1)],
         'keepdims': [False, True]}))
class TestAllAnyWithNaN(unittest.TestCase):

    @testing.for_dtypes(
        (numpy.float64, numpy.float32, numpy.bool_))
    @testing.numpy_nlcpy_array_equal(accept_error=numpy.AxisError)
    def test_without_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        return getattr(xp, self.f)(x, self.axis, None, self.keepdims)

    @testing.for_dtypes(
        (numpy.float64, numpy.float32, numpy.bool_))
    @testing.numpy_nlcpy_array_equal(accept_error=IndexError)
    def test_with_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        out_shape = _calc_out_shape(x.shape, self.axis, self.keepdims)
        out = xp.empty(out_shape, dtype=x.dtype)
        getattr(xp, self.f)(x, self.axis, out, self.keepdims)
        return out

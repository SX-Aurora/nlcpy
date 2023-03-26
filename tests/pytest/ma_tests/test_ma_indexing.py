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

import unittest
import warnings

import numpy

from nlcpy import testing


@testing.parameterize(
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': (1, 0, 2)},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': (-1, 0, -2)},
    {'shape': (2, 3, 4), 'transpose': (2, 0, 1), 'indexes': (1, 0, 2)},
    {'shape': (2, 3, 4), 'transpose': (2, 0, 1), 'indexes': (-1, 0, -2)},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(None), slice(None, 1), slice(2))},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(None), slice(None, -1), slice(-2))},
    {'shape': (2, 3, 4), 'transpose': (2, 0, 1),
     'indexes': (slice(None), slice(None, 1), slice(2))},
    {'shape': (2, 3, 5), 'transpose': None,
     'indexes': (slice(None, None, -1), slice(1, None, -1), slice(4, 1, -2))},
    {'shape': (2, 3, 5), 'transpose': (2, 0, 1),
     'indexes': (slice(4, 1, -2), slice(None, None, -1), slice(1, None, -1))},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': (Ellipsis, 2)},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': (1, Ellipsis)},
    {'shape': (2, 3, 4, 5), 'transpose': None, 'indexes': (1, Ellipsis, 3)},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (1, None, slice(2), None, 2)},
    {'shape': (2, 3), 'transpose': None, 'indexes': (None,)},
    {'shape': (2,), 'transpose': None, 'indexes': (slice(None,), None)},
    {'shape': (), 'transpose': None, 'indexes': (None,)},
    {'shape': (), 'transpose': None, 'indexes': (None, None)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(10, -9, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-9, -10, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-1, -10, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-1, -11, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-11, -11, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(10, -9, -3),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-1, -11, -3),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(1, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(0, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-1, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-4, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-6, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-10, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-11, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-12, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, 1, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, 0, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -1, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -4, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -6, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -10, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -11, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -12, -1),)},
    # reversing indexing on empty dimension
    {'shape': (0,), 'transpose': None, 'indexes': (slice(None, None, -1),)},
    {'shape': (0, 0), 'transpose': None,
     'indexes': (slice(None, None, -1), slice(None, None, -1))},
    {'shape': (0, 0), 'transpose': None,
     'indexes': (None, slice(None, None, -1))},
    {'shape': (0, 0), 'transpose': None,
     'indexes': (slice(None, None, -1), None)},
    {'shape': (0, 1), 'transpose': None,
     'indexes': (slice(None, None, -1), None)},
    {'shape': (1, 0), 'transpose': None,
     'indexes': (None, slice(None, None, -1))},
    {'shape': (1, 0, 1), 'transpose': None,
     'indexes': (None, slice(None, None, -1), None)},
    #
    {'shape': (2, 0), 'transpose': None,
     'indexes': (1, slice(None, None, None))},
)
class TestArrayIndexingParameterized(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(accept_error=IndexError)
    def test_getitem(self, xp, dtype):
        data = testing.shaped_arange(self.shape, xp, dtype)
        mask = testing.shaped_random(self.shape, xp, numpy.bool_)
        fill_value = testing.shaped_random(self.shape, xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        if self.transpose:
            a = a.transpose(self.transpose)
        res = a[self.indexes]
        return res


@testing.parameterize(
    {'shape': (), 'transpose': None, 'indexes': (slice(0, 1, 0),)},
    {'shape': (2, 3), 'transpose': None, 'indexes': (0, 0, 0)},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': -3},
    {'shape': (2, 3, 4), 'transpose': (2, 0, 1), 'indexes': -5},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': 3},
    {'shape': (2, 3, 4), 'transpose': (2, 0, 1), 'indexes': 5},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(0, 1, 0), )},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice((0, 0), None, None), )},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(None, (0, 0), None), )},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(None, None, (0, 0)), )},
)
@testing.with_requires('numpy>=1.12.0')
class TestArrayInvalidIndex(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_raises()
    def test_invalid_getitem(self, xp, dtype):
        data = testing.shaped_arange(self.shape, xp, dtype)
        mask = testing.shaped_random(self.shape, xp, numpy.bool_)
        fill_value = testing.shaped_random(self.shape, xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        if self.transpose:
            a = a.transpose(self.transpose)
        a[self.indexes]


class TestArrayIndex(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_setitem_constant(self, xp, dtype):
        shape = (2, 3, 4)
        data = xp.zeros(shape, dtype=dtype)
        mask = testing.shaped_random(shape, xp, numpy.bool_)
        fill_value = testing.shaped_random(shape, xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        a[:] = 1
        return a

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_setitem_partial_constant(self, xp, dtype):
        shape = (2, 3, 4)
        data = xp.zeros(shape, dtype=dtype)
        mask = testing.shaped_random(shape, xp, numpy.bool_)
        fill_value = testing.shaped_random(shape, xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        a[1, 1:3] = 1
        return a

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_setitem_copy(self, xp, dtype):
        shape = (2, 3, 4)
        data = xp.zeros(shape, dtype=dtype)
        mask = testing.shaped_random(shape, xp, numpy.bool_)
        fill_value = testing.shaped_random(shape, xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)

        data = testing.shaped_arange(shape, xp, dtype=dtype)
        mask = testing.shaped_random(shape, xp, numpy.bool_)
        fill_value = testing.shaped_random(shape, xp)
        b = xp.ma.array(data, mask=mask, fill_value=fill_value)

        a[:] = b
        return a

    @testing.for_all_dtypes_combination(('src_type', 'dst_type'))
    @testing.numpy_nlcpy_array_equal()
    def test_setitem_different_type(self, xp, src_type, dst_type):
        shape = (2, 3, 4)
        data = xp.zeros(shape, dtype=dst_type)
        mask = testing.shaped_random(shape, xp, numpy.bool_)
        fill_value = testing.shaped_random(shape, xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)

        data = testing.shaped_arange(shape, xp, dtype=src_type)
        mask = testing.shaped_random(shape, xp, numpy.bool_)
        fill_value = testing.shaped_random(shape, xp)
        b = xp.ma.array(data, mask=mask, fill_value=fill_value)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            a[:] = b
        return a

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_setitem_partial_copy(self, xp, dtype):
        shape_a = (2, 3, 4)
        data = xp.zeros(shape_a, dtype=dtype)
        mask = testing.shaped_random(shape_a, xp, numpy.bool_)
        fill_value = testing.shaped_random(shape_a, xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)

        shape_b = (3, 2)
        data = testing.shaped_arange(shape_b, xp, dtype=dtype)
        mask = testing.shaped_random(shape_b, xp, numpy.bool_)
        fill_value = testing.shaped_random(shape_b, xp)
        b = xp.ma.array(data, mask=mask, fill_value=fill_value)
        a[1, ::-1, 1:4:2] = b
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_setitem(self, xp):
        data = testing.shaped_arange([10], xp)
        mask = [i % 2 for i in range(10)]
        fill_value = testing.shaped_random([10], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        a[:3] = 5
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_setitem_masked(self, xp):
        data = testing.shaped_arange([10], xp)
        mask = [i % 2 for i in range(10)]
        fill_value = testing.shaped_random([10], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        a[:3] = xp.ma.masked
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_setitem_masked2(self, xp):
        data = testing.shaped_arange([10], xp)
        fill_value = testing.shaped_random([10], xp)
        a = xp.ma.array(data, fill_value=fill_value)
        a[:3] = xp.ma.masked
        return a

    @testing.numpy_nlcpy_raises()
    def test_setitem_to_masked(self, xp):
        a = xp.ma.masked
        a[()] = 5

    @testing.numpy_nlcpy_array_equal()
    def test_setitem_ma_to_nomask_ma(self, xp):
        data = testing.shaped_arange([10], xp)
        fill_value = testing.shaped_random([10], xp)
        a = xp.ma.array(data, fill_value=fill_value)
        b = xp.ma.array(5, mask=True, fill_value=-5)
        a[:3] = b
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_setitem_nomask_ma_to_nomask_ma(self, xp):
        data = testing.shaped_arange([10], xp)
        fill_value = testing.shaped_random([10], xp)
        a = xp.ma.array(data, fill_value=fill_value)
        b = xp.ma.array(5, mask=False, fill_value=-5)
        a[:3] = b
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_getitem_scalar_expected(self, xp):
        a = xp.ma.array(1, fill_value=2)
        return a[()]

    @testing.numpy_nlcpy_array_equal()
    def test_getitem_scalar_expected2(self, xp):
        a = xp.ma.array(1, mask=True, fill_value=2)
        return a[()]

    @testing.numpy_nlcpy_array_equal()
    def test_T(self, xp):
        shape = (2, 3, 4)
        data = xp.zeros(shape, dtype='F')
        mask = testing.shaped_random(shape, xp, numpy.bool_)
        fill_value = testing.shaped_random(shape, xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, hard_mask=True)
        b = a.T
        a[0] = 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_T_vector(self, xp):
        shape = (4,)
        data = xp.zeros(shape)
        mask = testing.shaped_random(shape, xp, numpy.bool_)
        fill_value = testing.shaped_random(shape, xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return a.T

    @testing.numpy_nlcpy_array_equal()
    @testing.with_requires('numpy<1.22')
    def test_setitem_with_masked_array_index(self, xp):
        data = testing.shaped_random([3, 2, 4], xp)
        mask = testing.shaped_random([3, 2, 4], xp, dtype=numpy.bool_)
        a = xp.ma.array(data, mask=mask)
        b = xp.ma.array([2, 0, 0, 1], mask=[0, 1, 1, 0])
        a[b] = -1
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_setitem_with_boolean_masked_array_index(self, xp):
        data = testing.shaped_random([3, 2, 4], xp)
        mask = testing.shaped_random([3, 2, 4], xp, dtype=numpy.bool_)
        a = xp.ma.array(data, mask=mask, hard_mask=True)
        b = xp.array([True, False, False, True])
        a[b] = -1
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_take_by_scalar(self, xp):
        data = testing.shaped_random([2, 4, 3], xp)
        mask = testing.shaped_arange([2, 4, 3], xp) % 2
        a = xp.ma.array(data, mask=mask)
        return a.take(2, axis=1)

    @testing.numpy_nlcpy_array_equal()
    def test_take_by_scalar2(self, xp):
        data = testing.shaped_random([2, 4, 3], xp)
        mask = testing.shaped_arange([2, 4, 3], xp) % 2
        a = xp.ma.array(data, mask=mask)
        return a.take(1, axis=0)

    @testing.numpy_nlcpy_array_equal()
    def test_take_by_ndarray(self, xp):
        data = testing.shaped_random([3, 4, 3], xp)
        mask = testing.shaped_arange([3, 4, 3], xp) % 2
        a = xp.ma.array(data, mask=mask)
        idx = xp.array([[1, 3], [2, 0]])
        return a.take(idx, axis=1)

    @testing.numpy_nlcpy_array_equal()
    def test_take_no_axis(self, xp):
        data = testing.shaped_random([3, 4, 3], xp)
        mask = testing.shaped_arange([3, 4, 3], xp) % 2
        a = xp.ma.array(data, mask=mask)
        idx = xp.array([[10, 5], [3, 20]])
        return a.take(idx)

    @testing.numpy_nlcpy_array_equal()
    def test_take_by_masked_array(self, xp):
        data = testing.shaped_random([3, 4, 3], xp)
        mask = testing.shaped_arange([3, 4, 3], xp) % 2
        a = xp.ma.array(data, mask=mask)
        data = testing.shaped_random([2, 2], xp) % 3
        mask = testing.shaped_random([2, 2], xp) % 3
        idx = xp.ma.array(data, mask=mask, dtype='l')
        return a.take(idx, axis=2)

    @testing.numpy_nlcpy_array_equal()
    def test_take_with_ndarray_out(self, xp):
        data = testing.shaped_random([3, 4, 3], xp)
        mask = testing.shaped_arange([3, 4, 3], xp) % 2
        a = xp.ma.array(data, mask=mask)
        idx = xp.array([[1, 3], [2, 0]])
        out = xp.empty([3, 2, 2, 3], dtype=numpy.float32)
        a.take(idx, axis=1, out=out)
        return out

    @testing.numpy_nlcpy_array_equal()
    def test_take_with_masked_array_out(self, xp):
        data = testing.shaped_random([2, 4, 3], xp)
        mask = testing.shaped_arange([2, 4, 3], xp) % 2
        a = xp.ma.array(data, mask=mask)
        idx = xp.array([[1, 3], [2, 0]])
        data = xp.empty([2, 2, 2, 3])
        mask = xp.zeros([2, 2, 2, 3])
        mask[0] = 1
        out = xp.ma.array(data, mask=mask, dtype=numpy.float32)
        a.take(idx, axis=1, out=out)
        return out

    @testing.numpy_nlcpy_array_equal()
    def test_take_with_nomask_array_out(self, xp):
        data = testing.shaped_random([2, 4, 3], xp)
        a = xp.ma.array(data)
        idx = xp.array([[1, 3], [2, 0]])
        data = xp.empty([2, 2, 2, 3])
        mask = testing.shaped_arange([2, 2, 2, 3], xp) % 2
        out = xp.ma.array(data, mask=mask, dtype=numpy.float32)
        a.take(idx, axis=1, out=out)
        return out

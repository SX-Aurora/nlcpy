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

import nlcpy
from nlcpy import testing


class TestBasic(unittest.TestCase):
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_empty(self, xp, dtype, order):
        a = xp.empty((2, 3, 4), dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_empty_scalar(self, xp, dtype, order):
        a = xp.empty(None, dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_empty_int(self, xp, dtype, order):
        a = xp.empty(3, dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_empty_like(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order=order)
        b.fill(0)
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_empty_like_contiguity(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order=order)
        b.fill(0)
        if order in ['f', 'F']:
            self.assertTrue(b.flags.f_contiguous)
        else:
            self.assertTrue(b.flags.c_contiguous)
        return b

    @testing.numpy_nlcpy_raises()
    def test_empty_like_invalid_order(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order='Q')
        return b

    @testing.numpy_nlcpy_raises()
    def test_empty_like_subok(self):
        a = testing.shaped_arange((2, 3, 4), nlcpy)
        b = nlcpy.empty_like(a, subok=True)
        return b

    @testing.for_CF_orders()
    def test_empty_zero_sized_array_strides(self, order):
        a = numpy.empty((1, 0, 2), dtype='d', order=order)
        b = nlcpy.empty((1, 0, 2), dtype='d', order=order)
        self.assertEqual(b.strides, a.strides)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_eye(self, xp, dtype):
        return xp.eye(5, 4, 1, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_identity(self, xp, dtype):
        return xp.identity(4, dtype)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_zeros(self, xp, dtype, order):
        return xp.zeros((2, 3, 4), dtype=dtype, order=order)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_zeros_scalar(self, xp, dtype, order):
        return xp.zeros(None, dtype=dtype, order=order)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_zeros_int(self, xp, dtype, order):
        return xp.zeros(3, dtype=dtype, order=order)

    @testing.for_CF_orders()
    def test_zeros_strides(self, order):
        a = numpy.zeros((2, 3), dtype='d', order=order)
        b = nlcpy.zeros((2, 3), dtype='d', order=order)
        self.assertEqual(b.strides, a.strides)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_zeros_like(self, xp, dtype, order):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.zeros_like(a, order=order)

    @testing.numpy_nlcpy_raises()
    def test_zeros_like_subok(self):
        a = nlcpy.ndarray((2, 3, 4))
        return nlcpy.zeros_like(a, subok=True)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_ones(self, xp, dtype):
        return xp.ones((2, 3, 4), dtype=dtype)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_ones_like(self, xp, dtype, order):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.ones_like(a, order=order)

    @testing.numpy_nlcpy_raises()
    def test_ones_like_subok(self):
        a = nlcpy.ndarray((2, 3, 4))
        return nlcpy.ones_like(a, subok=True)


@testing.parameterize(
    *testing.product({
        'shape': [4, (4, ), (4, 2), (4, 2, 3), (5, 4, 2, 3), (5, 4, 2, 3, 2)],
    })
)
class TestBasicReshape(unittest.TestCase):

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_empty_like_reshape(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order=order, shape=self.shape)
        b.fill(0)
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    def test_empty_like_reshape_nlcpy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), nlcpy, dtype)
        b = nlcpy.empty_like(a, shape=self.shape)
        b.fill(0)
        c = nlcpy.empty(self.shape, order=order, dtype=dtype)
        c.fill(0)
        testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_empty_like_reshape_contiguity(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order=order, shape=self.shape)
        b.fill(0)
        if order in ['f', 'F']:
            self.assertTrue(b.flags.f_contiguous)
        else:
            self.assertTrue(b.flags.c_contiguous)
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    def test_empty_like_reshape_contiguity_nlcpy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), nlcpy, dtype)
        b = nlcpy.empty_like(a, order=order, shape=self.shape)
        b.fill(0)
        c = nlcpy.empty(self.shape)
        c.fill(0)
        if order in ['f', 'F']:
            self.assertTrue(b.flags.f_contiguous)
        else:
            self.assertTrue(b.flags.c_contiguous)
        testing.assert_array_equal(b, c)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    def test_zeros_like_reshape_nlcpy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), nlcpy, dtype)
        b = nlcpy.zeros_like(a, shape=self.shape)
        c = nlcpy.zeros(self.shape, order=order, dtype=dtype)
        testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_ones_like_reshape(self, xp, dtype, order):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.ones_like(a, order=order, shape=self.shape)

    @testing.for_all_dtypes()
    def test_ones_like_reshape_nlcpy_only(self, dtype):
        a = testing.shaped_arange((2, 3, 4), nlcpy, dtype)
        b = nlcpy.ones_like(a, shape=self.shape)
        c = nlcpy.ones(self.shape, dtype=dtype)
        testing.assert_array_equal(b, c)

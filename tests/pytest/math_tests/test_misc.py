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

import numpy
from numpy.core._exceptions import UFuncTypeError

from nlcpy import testing


@testing.parameterize(*(
    testing.product({
        'shape': [(2,), (2, 3), (2, 3, 4)],
    })
))
class TestMisc(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = testing.shaped_arange(self.shape, xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(atol=1e-5)
    def check_binary(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = testing.shaped_arange(self.shape, xp, dtype)
        b = testing.shaped_reverse_arange(self.shape, xp, dtype)
        return getattr(xp, name)(a, b)

    @testing.for_dtypes(['?', 'i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_allclose(atol=1e-5)
    def check_unary_negative(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = xp.array(self.shape, dtype=dtype) - 3
        if numpy.dtype(dtype).kind == 'c':
            a += (a * 1j).astype(dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes(['f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_array_equal()
    def check_binary_nan(self, name, xp, dtype):
        a = xp.array([-3, numpy.NAN, -1, numpy.NAN, 0, numpy.NAN, 2],
                     dtype=dtype)
        b = xp.array([numpy.NAN, numpy.NAN, 1, 0, numpy.NAN, -1, -2],
                     dtype=dtype)
        return getattr(xp, name)(a, b)

    @testing.with_requires('numpy>=1.11.2')
    def test_sqrt(self):
        # numpy.sqrt is broken in numpy<1.11.2
        self.check_unary('sqrt')

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose(atol=1e-5)
    def test_cbrt(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return xp.cbrt(a)

    def test_square(self):
        self.check_unary('square')

    def test_absolute(self):
        self.check_unary('absolute')

    def test_absolute_negative(self):
        self.check_unary_negative('absolute')

    def test_sign(self):
        self.check_unary('sign', no_bool=True)

    def test_sign_negative(self):
        self.check_unary_negative('sign', no_bool=True)

    def test_maximum(self):
        self.check_binary('maximum')

    def test_maximum_nan(self):
        self.check_binary_nan('maximum')

    def test_minimum(self):
        self.check_binary('minimum')

    def test_minimum_nan(self):
        self.check_binary_nan('minimum')

    def test_fmax(self):
        self.check_binary('fmax')

    def test_fmax_nan(self):
        self.check_binary_nan('fmax')

    def test_fmin(self):
        self.check_binary('fmin')

    def test_fmin_nan(self):
        self.check_binary_nan('fmin')


@testing.parameterize(*(
    testing.product({
        'shape': [(10,), (3, 4), (3, 4, 2), (2, 4, 5, 3)],
    })
))
class TestClip(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_clip1(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        return xp.clip(a, 3, 6)

    @testing.numpy_nlcpy_array_equal()
    def test_clip2(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        return xp.clip(a, None, 6)

    @testing.numpy_nlcpy_array_equal()
    def test_clip3(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        return xp.clip(a, 3, None)

    @testing.numpy_nlcpy_array_equal()
    def test_clip_with_out(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        out = xp.zeros(self.shape)
        return xp.clip(a, 3, None, out=out)

    @testing.numpy_nlcpy_array_equal()
    def test_clip_with_where(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        where = testing.shaped_random(self.shape, xp, bool)
        return xp.clip(a, 3, None, where=where)[where]

    @testing.numpy_nlcpy_array_equal()
    def test_clip_with_where_with_out(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        where = testing.shaped_random(self.shape, xp, bool)
        out = xp.zeros(self.shape)
        return xp.clip(a, 3, None, where=where, out=out)

    @testing.numpy_nlcpy_array_equal()
    def test_clip_broadcast1(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        amin = xp.ones((2,) + self.shape)
        return xp.clip(a, amin, 6)

    @testing.numpy_nlcpy_array_equal()
    def test_clip_broadcast2(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        amax = xp.ones((2,) + self.shape)
        return xp.clip(a, 3, amax)

    @testing.numpy_nlcpy_array_equal()
    def test_clip_broadcast3(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        return xp.clip(a, 3, 6)

    @testing.numpy_nlcpy_array_equal()
    def test_clip_broadcast4(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        amin = xp.ones((1, 1, 4) + self.shape)
        amax = xp.ones((1, 3, 1) + self.shape)
        where = xp.ones((2, 1, 1) + self.shape, dtype='?')
        return xp.clip(a, amin, amax, where=where)

    @testing.numpy_nlcpy_array_equal()
    def test_clip_amin_larger_than_amax(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        return xp.clip(a, 5, 4)


class TestClipOrder(unittest.TestCase):

    @testing.for_orders('CF', name='order1')
    @testing.for_orders('CF', name='order2')
    @testing.for_orders('CF', name='order3')
    @testing.for_orders('CF', name='order4')
    @testing.for_orders('CF', name='order5')
    @testing.for_orders('CFKA', name='order6')
    @testing.numpy_nlcpy_array_equal()
    def test_clip_order(self, xp, order1, order2, order3, order4, order5, order6):
        a = xp.asarray(testing.shaped_arange([2, 3, 4], xp), order=order1)
        amin = xp.asarray(testing.shaped_random(a.shape, xp), order=order2)
        amax = xp.asarray(testing.shaped_random(a.shape, xp), order=order3)
        out = xp.zeros(a.shape, order=order4)
        where = xp.asarray(testing.shaped_random(a.shape, xp, dtype=bool), order=order5)
        return xp.clip(a, amin, amax, out=out, order=order6, where=where)


class TestClipFailure(unittest.TestCase):

    @testing.numpy_nlcpy_raises()
    def test_clip_both_min_max_none(self, xp):
        xp.clip(1, None, None)

    @testing.numpy_nlcpy_raises()
    def test_clip_out_shape_mismatch(self, xp):
        xp.clip(xp.empty([2, 3, 4]), 1, 2, out=xp.empty([3, 4]))

    @testing.numpy_nlcpy_raises()
    def test_clip_shape_mismatch(self, xp):
        xp.clip(xp.empty([2, 3, 4]), xp.empty([3]), 2, out=xp.empty([3, 4]))


@testing.parameterize(*(
    testing.product({
        'casting': ['no', 'safe', 'unsafe', 'same_kind', 'equiv'],
    })
))
class TestClipDtype(unittest.TestCase):

    @testing.for_all_dtypes(name='dt_a')
    @testing.for_all_dtypes(name='dt_amin')
    @testing.for_all_dtypes(name='dt_amax')
    @testing.numpy_nlcpy_array_equal()
    def test_clip(self, xp, dt_a, dt_amin, dt_amax):
        a = testing.shaped_arange([3, 4], xp, dtype=dt_a)
        amin = xp.array(3, dtype=dt_amin)
        amax = xp.array(10, dtype=dt_amax)
        # to avoid casting error
        try:
            return xp.clip(a, amin, amax, casting=self.casting)
        except UFuncTypeError:
            return 0

    @testing.for_all_dtypes(name='dt_a')
    @testing.for_all_dtypes(name='dt_amin')
    @testing.for_all_dtypes(name='dt_amax')
    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_clip_with_dtype(self, xp, dt_a, dt_amin, dt_amax, dtype):
        a = testing.shaped_arange([3, 4], xp, dtype=dt_a)
        amin = xp.array(3, dtype=dt_amin)
        amax = xp.array(10, dtype=dt_amax)
        try:
            return xp.clip(a, amin, amax, dtype=dtype, casting=self.casting)
        except UFuncTypeError:
            return 0

    @testing.for_all_dtypes(name='dt_a')
    @testing.for_all_dtypes(name='dt_amin')
    @testing.for_all_dtypes(name='dt_amax')
    @testing.for_all_dtypes(name='dt_out')
    @testing.numpy_nlcpy_array_equal()
    def test_clip_with_out(self, xp, dt_a, dt_amin, dt_amax, dt_out):
        a = testing.shaped_arange([3, 4], xp, dtype=dt_a)
        amin = xp.array(3, dtype=dt_amin)
        amax = xp.array(10, dtype=dt_amax)
        out = xp.zeros(a.shape, dtype=dt_out)
        try:
            return xp.clip(a, amin, amax, out=out, casting=self.casting)
        except UFuncTypeError:
            return 0

    @testing.for_all_dtypes(name='dt_a')
    @testing.for_all_dtypes(name='dt_amin')
    @testing.for_all_dtypes(name='dt_amax')
    @testing.for_all_dtypes(name='dt_out')
    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_clip_with_out_dtype(self, xp, dt_a, dt_amin, dt_amax, dt_out, dtype):
        a = testing.shaped_arange([3, 4], xp, dtype=dt_a)
        amin = xp.array(3, dtype=dt_amin)
        amax = xp.array(10, dtype=dt_amax)
        out = xp.zeros(a.shape, dtype=dt_out)
        try:
            return xp.clip(a, amin, amax, out=out, dtype=dtype, casting=self.casting)
        except UFuncTypeError:
            return 0


@testing.parameterize(*(
    testing.product({
        'casting': ['no', 'safe', 'unsafe', 'same_kind', 'equiv'],
    })
))
class TestClipDtypeFailure(unittest.TestCase):

    @testing.for_all_dtypes(name='dt_a')
    @testing.for_all_dtypes(name='dt_amin')
    @testing.for_all_dtypes(name='dt_amax')
    @testing.numpy_nlcpy_raises()
    def test_clip(self, xp, dt_a, dt_amin, dt_amax):
        a = testing.shaped_arange([3, 4], xp, dtype=dt_a)
        amin = xp.array(3, dtype=dt_amin)
        amax = xp.array(10, dtype=dt_amax)
        xp.clip(a, amin, amax, casting=self.casting)
        raise Exception  # to avoid normal case

    @testing.for_all_dtypes(name='dt_a')
    @testing.for_all_dtypes(name='dt_amin')
    @testing.for_all_dtypes(name='dt_amax')
    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_raises()
    def test_clip_with_dtype(self, xp, dt_a, dt_amin, dt_amax, dtype):
        a = testing.shaped_arange([3, 4], xp, dtype=dt_a)
        amin = xp.array(3, dtype=dt_amin)
        amax = xp.array(10, dtype=dt_amax)
        xp.clip(a, amin, amax, dtype=dtype, casting=self.casting)
        raise Exception

    @testing.for_all_dtypes(name='dt_a')
    @testing.for_all_dtypes(name='dt_amin')
    @testing.for_all_dtypes(name='dt_amax')
    @testing.for_all_dtypes(name='dt_out')
    @testing.numpy_nlcpy_raises()
    def test_clip_with_out(self, xp, dt_a, dt_amin, dt_amax, dt_out):
        a = testing.shaped_arange([3, 4], xp, dtype=dt_a)
        amin = xp.array(3, dtype=dt_amin)
        amax = xp.array(10, dtype=dt_amax)
        out = xp.zeros(a.shape, dtype=dt_out)
        xp.clip(a, amin, amax, out=out, casting=self.casting)
        raise Exception

    @testing.for_all_dtypes(name='dt_a')
    @testing.for_all_dtypes(name='dt_amin')
    @testing.for_all_dtypes(name='dt_amax')
    @testing.for_all_dtypes(name='dt_out')
    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_raises()
    def test_clip_with_out_dtype(self, xp, dt_a, dt_amin, dt_amax, dt_out, dtype):
        a = testing.shaped_arange([3, 4], xp, dtype=dt_a)
        amin = xp.array(3, dtype=dt_amin)
        amax = xp.array(10, dtype=dt_amax)
        out = xp.zeros(a.shape, dtype=dt_out)
        xp.clip(a, amin, amax, out=out, dtype=dtype, casting=self.casting)
        raise Exception

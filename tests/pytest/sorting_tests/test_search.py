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


class TestSearch(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.argmax()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_external_argmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.argmax(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose(accept_error=ValueError)
    def test_argmax_nan(self, xp, dtype):
        a = xp.array([float('nan'), -1, 1], dtype)
        return a.argmax()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.argmax(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_external_argmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.argmax(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmax_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmax_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmax_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=2)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmax_tie(self, xp, dtype):
        a = xp.array([0, 5, 2, 3, 4, 5], dtype)
        return a.argmax()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_raises(accept_error=ValueError)
    def test_argmax_zero_size(self, xp, dtype):
        a = testing.shaped_random((0, 1), xp, dtype)
        return a.argmax()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_raises(accept_error=ValueError)
    def test_argmax_zero_size_axis0(self, xp, dtype):
        a = testing.shaped_random((0, 1), xp, dtype)
        return a.argmax(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmax_zero_size_axis1(self, xp, dtype):
        a = testing.shaped_random((0, 1), xp, dtype)
        return a.argmax(axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.argmin()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose(accept_error=ValueError)
    def test_argmin_nan(self, xp, dtype):
        a = xp.array([float('nan'), -1, 1], dtype)
        return a.argmin()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_external_argmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.argmin(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.argmin(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_external_argmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.argmin(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmin_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmin_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmin_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=2)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmin_tie(self, xp, dtype):
        a = xp.array([0, 1, 2, 3, 0, 5], dtype)
        return a.argmin()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_raises(accept_error=ValueError)
    def test_argmin_zero_size(self, xp, dtype):
        a = testing.shaped_random((0, 1), xp, dtype)
        return a.argmin()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_raises(accept_error=ValueError)
    def test_argmin_zero_size_axis0(self, xp, dtype):
        a = testing.shaped_random((0, 1), xp, dtype)
        return a.argmin(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose()
    def test_argmin_zero_size_axis1(self, xp, dtype):
        a = testing.shaped_random((0, 1), xp, dtype)
        return a.argmin(axis=1)


@testing.parameterize(
    {'cond_shape': (2, 3, 4), 'x_shape': (2, 3, 4), 'y_shape': (2, 3, 4)},
    {'cond_shape': (4,), 'x_shape': (2, 3, 4), 'y_shape': (2, 3, 4)},
    {'cond_shape': (2, 3, 4), 'x_shape': (2, 3, 4), 'y_shape': (3, 4)},
    {'cond_shape': (3, 4), 'x_shape': (2, 3, 4), 'y_shape': (4,)},
)
class TestWhereTwoArrays(unittest.TestCase):

    @testing.for_all_dtypes_combination(
        names=['cond_type', 'x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose()
    def test_where_two_arrays(self, xp, cond_type, x_type, y_type):
        m = testing.shaped_random(self.cond_shape, xp, xp.bool_)
        # Almost all values of a matrix `shaped_random` makes are not zero.
        # To make a sparse matrix, we need multiply `m`.
        cond = testing.shaped_random(self.cond_shape, xp, cond_type) * m
        x = testing.shaped_random(self.x_shape, xp, x_type, seed=0)
        y = testing.shaped_random(self.y_shape, xp, y_type, seed=1)
        return xp.where(cond, x, y)


@testing.parameterize(
    {'cond_shape': (2, 3, 4)},
    {'cond_shape': (4,)},
    {'cond_shape': (2, 3, 4)},
    {'cond_shape': (3, 4)},
)
class TestWhereCond(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_list_equal()
    def test_where_cond(self, xp, dtype):
        m = testing.shaped_random(self.cond_shape, xp, xp.bool_)
        cond = testing.shaped_random(self.cond_shape, xp, dtype) * m
        return xp.where(cond)


class TestWhereError(unittest.TestCase):

    @testing.numpy_nlcpy_raises()
    def test_one_argument(self, xp):
        cond = testing.shaped_random((3, 4), xp, dtype=xp.bool_)
        x = testing.shaped_random((2, 3, 4), xp, xp.int32)
        xp.where(cond, x)


@testing.parameterize(
    {'array': numpy.random.randint(0, 2, (20,))},
    {'array': numpy.random.randn(3, 2, 4)},
    {'array': numpy.empty((0,))},
    {'array': numpy.empty((0, 2))},
    {'array': numpy.empty((0, 2, 0))},
)
class TestNonzero(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_list_equal()
    def test_nonzero(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        return xp.nonzero(array)


@testing.parameterize(
    {'array': numpy.array(0)},
    {'array': numpy.array(1)},
)
@testing.with_requires('numpy>=1.17.0')
class TestNonzeroZeroDimension(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_list_equal()
    def test_nonzero(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        return xp.nonzero(array)

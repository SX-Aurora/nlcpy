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
import nlcpy

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

    @testing.numpy_nlcpy_array_equal()
    def test_argmax_array_none(self, xp):
        return xp.argmax(None)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_array_equal()
    def test_argmax_f_order(self, xp, dtype):
        a = testing.shaped_random((3, 4), xp, dtype).copy(order='F')
        return xp.argmax(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_array_equal()
    def test_argmax_axis_out(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp, dtype)
        z = xp.empty(3, dtype='i8')
        xp.argmax(x, axis=1, out=z)
        return z

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_array_equal()
    def test_argmax_axis_out_f_order(self, xp, dtype):
        x = testing.shaped_random((3, 4, 5), xp, dtype)
        z = xp.empty((3, 4), dtype='i8', order='F')
        xp.argmax(x, axis=2, out=z)
        return z

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_array_equal()
    def test_argmax_axis_array_0dim(self, xp, dtype):
        x = xp.array(0)
        return xp.argmax(x, axis=0)

    @testing.numpy_nlcpy_raises()
    def test_argmax_out_not_ndarray(self, xp):
        xp.argmax(0, out=0)

    def test_argmax_out_not_integer_type(self):
        with self.assertRaises(TypeError):
            nlcpy.argmax(0, out=nlcpy.empty(1, dtype='f4'))

    @testing.numpy_nlcpy_raises()
    def test_argmax_axis_not_int(self, xp):
        xp.argmax(0, axis=1.1)

    @testing.numpy_nlcpy_raises()
    def test_argmax_axis_out_of_bounds(self, xp):
        xp.argmax([1, 2], axis=1)

    @testing.numpy_nlcpy_raises()
    def test_argmax_out_shape_mismatch(self, xp):
        xp.argmax([1, 2], out=xp.empty(2, dtype='i8'))

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

    @testing.numpy_nlcpy_array_equal()
    def test_argmin_array_none(self, xp):
        return xp.argmin(None)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_array_equal()
    def test_argmin_f_order(self, xp, dtype):
        a = testing.shaped_random((3, 4), xp, dtype).copy(order='F')
        return xp.argmin(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_array_equal()
    def test_argmin_axis_out(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp, dtype)
        z = xp.empty(3, dtype='i8')
        xp.argmin(x, axis=1, out=z)
        return z

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_array_equal()
    def test_argmin_axis_out_f_order(self, xp, dtype):
        x = testing.shaped_random((3, 4, 5), xp, dtype)
        z = xp.empty((3, 4), dtype='i8', order='F')
        xp.argmin(x, axis=2, out=z)
        return z

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_array_equal()
    def test_argmin_axis_array_0dim(self, xp, dtype):
        x = xp.array(0)
        return xp.argmin(x, axis=0)

    @testing.numpy_nlcpy_raises()
    def test_argmin_out_not_ndarray(self, xp):
        xp.argmin(0, out=0)

    def test_argmin_out_not_integer_type(self):
        with self.assertRaises(TypeError):
            nlcpy.argmin(0, out=nlcpy.empty(1, dtype='f4'))

    @testing.numpy_nlcpy_raises()
    def test_argmin_axis_not_int(self, xp):
        xp.argmin(0, axis=1.1)

    @testing.numpy_nlcpy_raises()
    def test_argmin_axis_out_of_bounds(self, xp):
        xp.argmin([1, 2], axis=1)

    @testing.numpy_nlcpy_raises()
    def test_argmin_out_shape_mismatch(self, xp):
        xp.argmin([1, 2], out=xp.empty(2, dtype='i8'))


@testing.parameterize(
    {'cond_shape': (10,), 'x_shape': (10,), 'y_shape': (10,)},
    {'cond_shape': (5, 6), 'x_shape': (5, 6), 'y_shape': (5, 6)},
    {'cond_shape': (2, 3, 4), 'x_shape': (2, 3, 4), 'y_shape': (2, 3, 4)},
    {'cond_shape': (2, 3, 4, 5), 'x_shape': (2, 3, 4, 5), 'y_shape': (2, 3, 4, 5)},
    {'cond_shape': (2, 3, 1, 5), 'x_shape': (2, 1, 4, 5), 'y_shape': (2, 3, 4, 1)},
    {'cond_shape': (1, 3, 1, 5), 'x_shape': (2, 1, 1, 5), 'y_shape': (1, 1, 4, 1)},
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
    {'shape': (10,)},
    {'shape': (10, 11)},
    {'shape': (10, 11, 12)},
    {'shape': (10, 11, 12, 5)},
)
class TestWhereOneArray(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_where_condition_zero_0(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype, seed=0)
        return xp.where(0, x, 1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_where_condition_zero_1(self, xp, dtype):
        y = testing.shaped_random(self.shape, xp, dtype, seed=0)
        return xp.where(0, 1, y)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_where_condition_one_0(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype, seed=0)
        return xp.where(1, x, 1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_where_condition_one_1(self, xp, dtype):
        y = testing.shaped_random(self.shape, xp, dtype, seed=0)
        return xp.where(1, 1, y)


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
    {'array': numpy.random.randint(0, 2, (4, 5))},
    {'array': numpy.random.randint(0, 2, (5, 7, 8))},
    {'array': numpy.random.randint(0, 2, (4, 3, 6, 5))},
    {'array': numpy.random.randint(0, 2, (2, 6, 3, 4, 5))},
    {'array': numpy.random.randn(3, 2, 4)},
    {'array': numpy.empty((0,))},
    {'array': numpy.empty((0, 2))},
    {'array': numpy.empty((0, 2, 0))},
)
class TestNonzero(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_list_equal()
    def test_nonzero(self, xp, dtype, order):
        array = xp.array(self.array, dtype=dtype, order=order)
        return xp.nonzero(array)

    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_list_equal()
    def test_nonzero_ndarray(self, xp, dtype, order):
        array = xp.array(self.array, dtype=dtype, order=order)
        return array.nonzero()


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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            return xp.nonzero(array)


class TestNonzeroNone(unittest.TestCase):

    def test_nonzero_none(self):
        assert nlcpy.nonzero(None)[0].shape == (0,)


@testing.parameterize(
    {'array': numpy.random.randint(0, 2, (20,))},
    {'array': numpy.random.randn(3, 2, 4)},
    {'array': numpy.empty((0,))},
    {'array': numpy.empty((0, 2))},
    {'array': numpy.empty((0, 2, 0))},
)
class TestArgwhere(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_argwhere0(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        return xp.argwhere(array)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_argwhere1(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        return xp.argwhere(array > 1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_argwhere_not_contiguous(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)[..., ::2]
        return xp.argwhere(array > 1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_argwhere_f_order(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype, order='F')
        return xp.argwhere(array > 1)


@testing.parameterize(
    {'value': 0},
    {'value': 3},
)
@testing.with_requires('numpy>=1.18')
class TestArgwhereZeroDimension(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_argwhere(self, xp, dtype):
        array = xp.array(self.value, dtype=dtype)
        return xp.argwhere(array)

    @testing.numpy_nlcpy_array_equal()
    def test_argwhere_scalar(self, xp):
        return xp.argwhere(self.value)


@testing.parameterize(
    {'value': 0},
    {'value': 3},
    {'value': [[1, 3], [2, 4]]},
    {'value': ((1, 3), [2, 4])},
    {'value': numpy.array(0)},
    {'value': numpy.array(3)},
    {'value': numpy.array(3) > 1},
    {'value': numpy.array(3) > 5},
    {'value': numpy.array([3])},
    {'value': numpy.array([3]) > 1},
    {'value': numpy.array([3]) > 5},
    {'value': numpy.array([[1, 3], [2, 4]])},
    {'value': []},
    {'value': None},
)
class TestArgwhereNotNdarray(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_argwhere_not_ndarray(self, xp):
        return xp.argwhere(self.value)


@testing.parameterize(
    {'func': 'nanargmin'},
    {'func': 'nanargmax'},
)
class TestNanArgFunc(unittest.TestCase):

    @testing.for_dtypes('fdFD')
    @testing.numpy_nlcpy_array_equal()
    def test_nanargfunc_with_nan1(self, xp, dtype):
        a = testing.shaped_random([10], xp, dtype)
        a[0] = xp.nan
        return getattr(xp, self.func)(a)

    @testing.for_dtypes('fdFD')
    @testing.numpy_nlcpy_array_equal()
    def test_nanargfunc_with_nan2(self, xp, dtype):
        a = testing.shaped_random([10], xp, dtype)
        a[:2] = xp.nan
        return getattr(xp, self.func)(a)

    @testing.for_dtypes('fdFD')
    @testing.numpy_nlcpy_array_equal()
    def test_nanargfunc_with_nan3(self, xp, dtype):
        a = testing.shaped_random([10], xp, dtype)
        a[8:] = xp.nan
        return getattr(xp, self.func)(a)

    @testing.for_dtypes('fdFD')
    @testing.numpy_nlcpy_array_equal()
    def test_nanargfunc_with_nan4(self, xp, dtype):
        a = testing.shaped_random([10], xp, dtype)
        a[2:4] = xp.nan
        return getattr(xp, self.func)(a)

    @testing.for_dtypes('FD')
    @testing.numpy_nlcpy_array_equal()
    def test_nanargfunc_with_nan_imag(self, xp, dtype):
        a = testing.shaped_random([10], xp, dtype)
        a[2:4] = a[2:4] + 1j * xp.nan
        return getattr(xp, self.func)(a)

    @testing.for_dtypes('FD')
    @testing.numpy_nlcpy_array_equal()
    def test_nanargfunc_with_nan_real_imag(self, xp, dtype):
        a = testing.shaped_random([10], xp, dtype)
        a[2:4] = xp.nan + 1j * xp.nan
        return getattr(xp, self.func)(a)


@testing.parameterize(*(
    testing.product({
        'shape': ((3, 3), (10, 10), (3, 4, 5), (6, 4, 5), (3, 0, 2)),
        'axis': (None, -1, 0, 1),
        'func': ('nanargmin', 'nanargmax'),
    }))
)
class TestNanArgFuncWithoutNan(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_nanargfunc(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if self.axis is None and a.size == 0 or \
           self.axis is not None and a.shape[self.axis] == 0:
            return 0
        return getattr(xp, self.func)(a, axis=self.axis)


@testing.parameterize(
    {'func': 'nanargmin'},
    {'func': 'nanargmax'},
)
class TestNanArgFuncFailure(unittest.TestCase):

    @testing.numpy_nlcpy_raises()
    def test_nanargfunc_all_nan1(self, xp):
        return getattr(xp, self.func)([xp.nan, xp.nan, xp.nan])

    @testing.numpy_nlcpy_raises()
    def test_nanargfunc_all_nan2(self, xp):
        return getattr(xp, self.func)([[xp.nan, xp.nan], [0, 0]], axis=1)

    @testing.numpy_nlcpy_raises()
    def test_nanargfunc_on_empty(self, xp):
        a = testing.shaped_random([3, 0, 2], xp)
        return getattr(xp, self.func)(a, axis=1)

    @testing.numpy_nlcpy_raises()
    def test_nanargfunc_invalid_axis(self, xp):
        a = testing.shaped_random([2, 3], xp)
        return getattr(xp, self.func)(a, axis=2)

    @testing.numpy_nlcpy_raises()
    def test_nanargfunc_invalid_axis2(self, xp):
        a = testing.shaped_random([2, 3], xp)
        return getattr(xp, self.func)(a, axis=-3)

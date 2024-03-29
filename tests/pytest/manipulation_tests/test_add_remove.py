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


@testing.parameterize(*(
    testing.product({
        'params': [
            ([6, ], [2], None),
            ([3, 2], [1], None),
            ([3, 2], [3, 1], 1),
            ([10, 10, 2], [10, 10, 1], 2),
            ([3, 4, 5], [3, 1, 5], 1),
            ([3, 4, 2, 5], [3, 1, 2, 5], 1),
        ],
    })
))
class TestAppend(unittest.TestCase):
    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_all_dtypes(name='dtype_v')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_v')
    @testing.numpy_nlcpy_array_equal()
    def test_append(self, xp, dtype_a, dtype_v, order_a, order_v):
        arr = testing.shaped_arange(self.params[0], xp, dtype_a, order_a)
        values = testing.shaped_arange(self.params[1], xp, dtype_v, order=order_v)
        axis = self.params[2]
        return xp.append(arr, values, axis)


class TestAppendFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_append_incompatible_axis1(self, xp):
        return xp.append([1], 1, 1)

    @testing.numpy_nlcpy_raises()
    def test_append_incompatible_axis2(self, xp):
        return xp.append(xp.zeros([2, 2]), 1, "axis")

    @testing.numpy_nlcpy_raises()
    def test_append_incompatible_axis3(self, xp):
        return xp.append(xp.zeros([2, 2]), 1, (0, 1))

    @testing.numpy_nlcpy_raises()
    def test_append_incompatible_axis4(self, xp):
        return xp.append(xp.zeros([2, 2]), 1, [0, 1])

    @testing.numpy_nlcpy_raises()
    def test_append_incompatible_axis5(self, xp):
        return xp.append(xp.zeros([2, 2]), 1, xp.array([0, 1]))

    @testing.numpy_nlcpy_raises()
    def test_append_shape_mismatch(self, xp):
        return xp.append(xp.zeros([2, 2]), xp.zeros([3, 3]), 0)


@testing.parameterize(
    {'obj': 0},
    {'obj': 1},
    {'obj': 2},
    {'obj': -2},
    {'obj': (0, 1)},
    {'obj': (-2, 3)},
    {'obj': ((1, ), (2, ))},
    {'obj': [0, 1]},
    {'obj': [3, 2]},
    {'obj': [-2, 3]},
    {'obj': slice(2, 5, 2)},
    {'obj': slice(-10, 10, 2)},
    {'obj': slice(0, 0)},
    {'obj': ()},
    {'obj': []},
)
class TestDelete_OBJ(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_delete_1d(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return xp.delete(a, self.obj, 0)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_delete_ND(self, xp, dtype, order):
        a = testing.shaped_arange((5, 5, 5, 5, 5), xp, dtype, order)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return xp.delete(a, self.obj, 0)


class TestDeleteAxisNone(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_delete_axis_None(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2, 2), xp, dtype)
        return xp.delete(a, [7, 8, 9], None)


@testing.parameterize(
    {'axis': 0},
    {'axis': 1},
    {'axis': 2},
    {'axis': 3},
    {'axis': 4},
    {'axis': -1},
    {'axis': -2},
    {'axis': -3},
    {'axis': -4},
    {'axis': -5},
)
class TestDelete_axis(unittest.TestCase):
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_delete_N_axis(self, xp, dtype, order):
        a = testing.shaped_arange((5, 5, 5, 5, 5), xp, dtype, order)
        return xp.delete(a, [0, 2, 4], self.axis)

    @testing.numpy_nlcpy_array_equal()
    def test_delete_N_axis_not_contiguous(self, xp):
        a = xp.moveaxis(testing.shaped_arange((5, 5, 5, 5, 5), xp), 0, 1)
        return xp.delete(a, [0, 2, 4], self.axis)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_delete_all(self, xp, dtype, order):
        a = testing.shaped_arange((5, 5, 5, 5, 5), xp, dtype, order)
        return xp.delete(a, [0, 1, 2, 3, 4], self.axis)


class TestDelete_Failure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_delete_failure_axis1(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        xp.delete(a, 0, 5)

    @testing.numpy_nlcpy_raises()
    def test_delete_failure_axis2(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        xp.delete(a, 0, [0, 1])

    @testing.numpy_nlcpy_raises()
    def test_delete_failure_axis3(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        axis = testing.shaped_arange((2,), xp)
        xp.delete(a, 0, axis)

    @testing.numpy_nlcpy_raises()
    def test_delete_failure_obj_plus(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        xp.delete(a, 5, 0)

    @testing.numpy_nlcpy_raises()
    def test_delete_failure_obj_minus(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        xp.delete(a, -6, 0)

    @testing.numpy_nlcpy_raises()
    def test_delete_bool_obj_size_mismatch(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        xp.delete(a, True, 0)

    @testing.for_dtypes('fdFD')
    @testing.numpy_nlcpy_raises()
    def test_delete_non_int_obj(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp)
        obj = xp.array(1, dtype=dtype)
        xp.delete(a, obj, 0)

    @testing.numpy_nlcpy_raises()
    def test_delete_index_out_of_bound_with_tuple_obj(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        xp.delete(a, (1, 10), 0)

    @testing.numpy_nlcpy_raises()
    def test_delete_0d(self, xp, dtype):
        return xp.delete(100, 0, 0)


@testing.parameterize(*(
    testing.product({
        'params': [
            ([6, ], [2, 2], [2], None),
            ([3, 2], 0, [1], None),
            ([4, 2], 1, [1], 0),
            ([3, 2], [1], [3, 1], 1),
            ([2, 4], [1, 3], [1], 1),
            ([5, 4, 3], 2, [3, ], 1),
            ([3, 5, 4], -1, [3, 3, 5], -1),
            ([10, 10, 2], 1, [10, 10, 1], 2),
            ([3, 4, 5], [2, 0], [4, 5], 0),
            ([3, 4, 5], [2, 2, 0, -1], [4, 5], 0),
            ([3, 4, 5], [-5, -6, -7, -8], [3, 1, 5], 1),
            ([3, 4, 2, 5], [-5, -6, -7, -6], [3, 1, 2, 5], 1),
        ],
    })
))
class TestInsert(unittest.TestCase):
    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_all_dtypes(name='dtype_v')
    @testing.for_dtypes('?il', name='dtype_o')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_v')
    @testing.numpy_nlcpy_array_equal()
    def test_insert(self, xp, dtype_a, dtype_o, dtype_v, order_a, order_v):
        arr = testing.shaped_arange(self.params[0], xp, dtype_a, order_a)
        obj = xp.array(self.params[1], dtype=dtype_o)
        values = testing.shaped_arange(self.params[2], xp, dtype_v, order=order_v)
        axis = self.params[3]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            warnings.simplefilter('ignore', FutureWarning)
            return xp.insert(arr, obj, values, axis)


@testing.parameterize(*(
    testing.product({
        'params': [
            ([6], slice(2, 4), [2], None),
            ([20], slice(2, 25, 3), [1], None),
            ([3, 2], 0, [1], None),
            ([4, 2], 1, [1], 0),
            ([3, 5, 4], -1, [3, 3, 5], -1),
            ([3, 4, 5], [2, 2, 0, -1], [4, 5], 0),
            ([3, 4, 5], (), [4, 5], 0),
            ([3, 4, 5], [], [4, 5], 0),
        ],
    })
))
class TestInsertNotArrayObj(unittest.TestCase):
    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_all_dtypes(name='dtype_v')
    @testing.numpy_nlcpy_array_equal()
    def test_insert_not_array_obj(self, xp, dtype_a, dtype_v):
        arr = testing.shaped_arange(self.params[0], xp, dtype_a)
        obj = self.params[1]
        values = testing.shaped_arange(self.params[2], xp, dtype_v)
        axis = self.params[3]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            return xp.insert(arr, obj, values, axis)


class TestInsertNonIntegerObj(unittest.TestCase):
    @testing.for_dtypes('il', name='dtype_o')
    @testing.numpy_nlcpy_array_equal()
    def test_insert_obj_is_uint(self, xp, dtype_o):
        arr = testing.shaped_arange([3, 4, 5], xp)
        obj = testing.shaped_arange([3], xp, dtype_o)
        values = testing.shaped_arange([4, 5], xp)
        return xp.insert(arr, obj, values, 0)


class TestInsertFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_insert_incompatible_axis1(self, xp):
        return xp.insert([1], 1, 1, 1)

    @testing.numpy_nlcpy_raises()
    def test_insert_incompatible_axis2(self, xp):
        return xp.insert(xp.zeros([2, 2]), 1, 1, "axis")

    @testing.numpy_nlcpy_raises()
    def test_insert_incompatible_axis3(self, xp):
        return xp.insert(xp.zeros([2, 2]), 1, 1, (0, 1))

    @testing.numpy_nlcpy_raises()
    def test_insert_incompatible_axis4(self, xp):
        return xp.insert(xp.zeros([2, 2]), 1, 1, [0, 1])

    @testing.numpy_nlcpy_raises()
    def test_insert_incompatible_axis5(self, xp):
        return xp.insert(xp.zeros([2, 2]), 1, 1, xp.array([0, 1]))

    @testing.numpy_nlcpy_raises()
    def test_insert_incompatible_scalar_index1(self, xp):
        return xp.insert([1], -3, 1)

    @testing.numpy_nlcpy_raises()
    def test_insert_incompatible_scalar_index2(self, xp):
        return xp.insert([1], 3, 1)

    @testing.numpy_nlcpy_raises()
    def test_insert_incompatible_scalar_index3(self, xp):
        return xp.insert([1], xp.array(-1).astype(dtype='L'), 1)

    @testing.numpy_nlcpy_raises()
    def test_insert_incompatible_list_index1(self, xp):
        return xp.insert([1], [-5, 0], 1)

    @testing.numpy_nlcpy_raises()
    def test_insert_incompatible_list_index2(self, xp):
        return xp.insert([1], [0, 2], 1)

    @testing.numpy_nlcpy_raises()
    def test_insert_obj_is_2d(self, xp):
        return xp.insert([1], [[0], [0]], 1)

    @testing.for_dtypes('fdFD', name='dtype_o')
    @testing.numpy_nlcpy_raises()
    def test_insert_obj_dtype_is_not_int1(self, xp, dtype_o):
        return xp.insert([1], xp.array(0, dtype=dtype_o), 1)

    @testing.for_dtypes('IL', name='dtype_o')
    @testing.numpy_nlcpy_raises()
    def test_insert_obj_dtype_is_not_int2(self, xp, dtype_o):
        return xp.insert([1], xp.array([0, 0], dtype=dtype_o), 1)

    @testing.numpy_nlcpy_raises()
    def test_insert_shape_mismatch(self, xp):
        return xp.insert(xp.zeros([2, 2]), 0, xp.zeros([3, 3]), 1)

    @testing.numpy_nlcpy_raises()
    def test_insert_scalar_arr(self, xp):
        return xp.insert(arr=0, obj=0, values=1, axis=0)

    @testing.for_dtypes('fdFD', name='dtype_o')
    @testing.numpy_nlcpy_raises()
    def test_insert_obj_is_real(self, xp, dtype_o):
        arr = testing.shaped_arange([3, 4, 5], xp)
        obj = testing.shaped_arange([3], xp, dtype_o)
        values = testing.shaped_arange([4, 5], xp)
        return xp.insert(arr, obj, values, 0)


@testing.parameterize(*(
    testing.product({
        'shape': [(100,), (10, 10), (10, 6, 4), (4, 8, 6, 10)],
        'axis': [None, 0, 1, -1],
        'index': [False, True],
        'inverse': [False, True],
        'count': [False, True],
    })
))
class TestUnique(unittest.TestCase):
    @testing.for_dtypes('?ilILfd')
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_unique(self, xp, dtype, order):
        if len(self.shape) == 1 and self.axis == 1:
            return 0
        shape = list(self.shape)
        idx = 0 if self.axis is None else self.axis
        shape[idx] = int(shape[idx] / 2)
        a = xp.asarray(testing.shaped_random(shape, xp, dtype), order=order)
        a = xp.tile(a, 2)
        return xp.unique(
            a, axis=self.axis, return_index=self.index,
            return_inverse=self.inverse, return_counts=self.count)


@testing.parameterize(*(
    testing.product({
        'index': [False, True],
        'inverse': [False, True],
        'count': [False, True],
    })
))
class TestUnique2(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_unique_empty(self, xp):
        return xp.unique((), self.index, self.inverse, self.count, None)

    @testing.for_dtypes('?ilILfd')
    @testing.numpy_nlcpy_array_equal()
    def test_unique_empty2(self, xp, dtype):
        return xp.unique(
            xp.array([], dtype=dtype), self.index, self.inverse, self.count, None)

    @testing.for_dtypes('?ilILfd')
    @testing.numpy_nlcpy_array_equal()
    def test_unique_scalar(self, xp, dtype):
        return xp.unique(
            xp.array(1, dtype=dtype), self.index, self.inverse, self.count, None)


class TestUnique3(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_unique_empty_array(self, xp, dtype):
        return xp.unique(xp.empty([2, 0, 3], dtype=dtype), axis=0)


class TestUniqueFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_unique_axis_error(self, xp):
        xp.unique(xp.arange(3), axis=2)

    @testing.numpy_nlcpy_raises()
    def test_unique_axis_error2(self, xp):
        xp.unique(xp.arange(3), axis=-2)

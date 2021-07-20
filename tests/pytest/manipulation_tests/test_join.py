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


class DummyError(Exception):
    pass


class TestJoin(unittest.TestCase):

    # Test for concatenate
    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate1(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=2)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_large_2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        d = testing.shaped_arange((2, 3, 5), xp, dtype)
        e = testing.shaped_arange((2, 3, 2), xp, dtype)
        return xp.concatenate((a, b, c, d, e) * 2, axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_large_3(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 1), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 1), xp, dtype)
        return xp.concatenate((a, b) * 10, axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_large_4(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 4), xp, dtype)
        return xp.concatenate((a, b) * 10, axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_large_5(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 4), xp, 'i')
        return xp.concatenate((a, b) * 10, axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_f_contiguous(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((2, 3, 2), xp, dtype).T
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_large_f_contiguous(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((2, 3, 2), xp, dtype).T
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        d = testing.shaped_arange((2, 3, 2), xp, dtype).T
        e = testing.shaped_arange((2, 3, 2), xp, dtype)
        return xp.concatenate((a, b, c, d, e) * 2, axis=-1)

    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_many_multi_dptye(self, xp):
        a = testing.shaped_arange((2, 1), xp, 'i')
        b = testing.shaped_arange((2, 1), xp, 'f')
        return xp.concatenate((a, b) * 1024, axis=1)

    def test_concatenate_wrong_ndim(self):
        a = nlcpy.empty((2, 3))
        b = nlcpy.empty((2,))
        with self.assertRaises(ValueError):
            nlcpy.concatenate((a, b))

    def test_concatenate_wrong_shape(self):
        a = nlcpy.empty((2, 3, 4))
        b = nlcpy.empty((3, 3, 4))
        c = nlcpy.empty((4, 4, 4))
        with self.assertRaises(ValueError):
            nlcpy.concatenate((a, b, c))

    # Test for stack

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_stack_simple(self, xp, dtype):
        arrays = [testing.shaped_random((2, 3, 4), xp, dtype) for i in range(5)]
        return xp.stack(arrays)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_stack_0d(self, xp, dtype):
        arrays = [testing.shaped_random((), xp, dtype) for i in range(5)]
        return xp.stack(arrays)

    @testing.for_all_dtypes(name='a_dtype')
    @testing.for_all_dtypes(name='b_dtype')
    @testing.for_all_dtypes(name='c_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_stack_diff_dtype(self, xp, a_dtype, b_dtype, c_dtype):
        a = testing.shaped_random((3, 5), xp, a_dtype)
        b = testing.shaped_random((3, 5), xp, b_dtype)
        c = testing.shaped_random((3, 5), xp, c_dtype)
        return xp.stack((a, b, c))

    @testing.for_all_dtypes(name='dtype')
    @testing.for_all_axis(-4, 4)
    @testing.numpy_nlcpy_array_equal()
    def test_stack_axis(self, xp, dtype, axis):
        arrays = [testing.shaped_random((2, 3, 4), xp, dtype) for i in range(5)]
        return xp.stack(arrays, axis=axis)

    @testing.for_all_dtypes(name='in_dtype')
    @testing.for_all_dtypes(name='out_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_stack_out1(self, xp, in_dtype, out_dtype):
        if numpy.can_cast(in_dtype, out_dtype, casting='same_kind'):
            arrays = [testing.shaped_random((2, 3, 4), xp, in_dtype) for i in range(5)]
            out = xp.empty((5, 2, 3, 4), dtype=out_dtype)
            return xp.stack(arrays, out=out)
        else:
            return 0

    @testing.for_all_dtypes(name='in_dtype')
    @testing.for_all_dtypes(name='out_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_stack_out2(self, xp, in_dtype, out_dtype):
        if numpy.can_cast(in_dtype, out_dtype, casting='same_kind'):
            arrays = [testing.shaped_random((2, 3, 4), xp, in_dtype) for i in range(5)]
            out = xp.empty((2, 5, 3, 4), dtype=out_dtype)
            axis = 1
            return xp.stack(arrays, axis=axis, out=out)
        else:
            return 0

    @testing.numpy_nlcpy_raises()
    def test_stack_err__empty_arrays(self, xp):
        arrays = ()
        return xp.stack(arrays)

    @testing.numpy_nlcpy_raises()
    def test_stack_err_axis1(self, xp):
        arrays = [testing.shaped_random((2, 3, 4), xp, 'float64') for i in range(5)]
        axis = 4
        return xp.stack(arrays, axis=axis)

    @testing.numpy_nlcpy_raises()
    def test_stack_err_axis2(self, xp):
        arrays = [testing.shaped_random((2, 3, 4), xp, 'float64') for i in range(5)]
        axis = -5
        return xp.stack(arrays, axis=axis)

    @testing.numpy_nlcpy_raises()
    def test_stack_err_axis3(self, xp):
        arrays = [testing.shaped_random((2, 3, 4), xp, 'float64') for i in range(5)]
        axis = (2,)
        return xp.stack(arrays, axis=axis)

    @testing.numpy_nlcpy_raises()
    def test_stack_err_axis4(self, xp):
        arrays = [testing.shaped_random((2, 3, 4), xp, 'float64') for i in range(5)]
        axis = 1.2
        return xp.stack(arrays, axis=axis)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_raises()
    def test_stack_err_diff_shape(self, xp, dtype):
        arrays = [testing.shaped_random((2, 3, 4), xp, dtype) for i in range(4)]
        arrays += (testing.shaped_random((2, 3, 5), xp, dtype),)
        return xp.stack(arrays)

    @testing.numpy_nlcpy_raises()
    def test_stack_err_out_shape(self, xp):
        arrays = [testing.shaped_random((2, 3, 4), xp, 'float64') for i in range(5)]
        out = xp.empty((5, 3, 3, 4), dtype='float64')
        return xp.stack(arrays, out=out)

    @testing.for_all_dtypes(name='in_dtype')
    @testing.for_all_dtypes(name='out_dtype')
    @testing.numpy_nlcpy_raises()
    def test_stack_err_out_dtype(self, xp, in_dtype, out_dtype):
        if not numpy.can_cast(in_dtype, out_dtype, casting='same_kind'):
            arrays = [testing.shaped_random((2, 3, 4), xp, in_dtype) for i in range(5)]
            out = xp.empty((5, 2, 3, 4), dtype=out_dtype)
            return xp.stack(arrays, out=out)
        else:
            raise DummyError()

    # Test for vstack

    @testing.for_all_dtypes(name='a_dtype')
    @testing.for_all_dtypes(name='b_dtype')
    @testing.for_all_dtypes(name='c_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_vstack_0d_a(self, xp, a_dtype, b_dtype, c_dtype):
        a = testing.shaped_random((), xp, a_dtype)
        b = testing.shaped_random((3, 1), xp, b_dtype)
        c = testing.shaped_random((5, 1), xp, c_dtype)
        return xp.vstack((a, b, c))

    @testing.for_all_dtypes(name='a_dtype')
    @testing.for_all_dtypes(name='b_dtype')
    @testing.for_all_dtypes(name='c_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_vstack_1d_a(self, xp, a_dtype, b_dtype, c_dtype):
        a = testing.shaped_random((2,), xp, a_dtype)
        b = testing.shaped_random((3, 2), xp, b_dtype)
        c = testing.shaped_random((5, 2), xp, c_dtype)
        return xp.vstack((a, b, c))

    @testing.for_all_dtypes(name='a_dtype')
    @testing.for_all_dtypes(name='b_dtype')
    @testing.for_all_dtypes(name='c_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_vstack_2d(self, xp, a_dtype, b_dtype, c_dtype):
        a = testing.shaped_random((2, 3), xp, a_dtype)
        b = testing.shaped_random((3, 3), xp, b_dtype)
        c = testing.shaped_random((4, 3), xp, c_dtype)
        return xp.vstack((a, b, c))

    @testing.for_all_dtypes(name='a_dtype')
    @testing.for_all_dtypes(name='b_dtype')
    @testing.for_all_dtypes(name='c_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_vstack_large(self, xp, a_dtype, b_dtype, c_dtype):
        a = testing.shaped_random((2, 3, 4), xp, a_dtype)
        b = testing.shaped_random((3, 3, 4), xp, b_dtype)
        c = testing.shaped_random((4, 3, 4), xp, c_dtype)
        return xp.vstack((a, b, c) * 5)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_vstack_f_contiguous_a(self, xp, dtype):
        a = testing.shaped_random((4, 3, 2), xp, dtype).T  # a.shape = (2, 3, 4)
        b = testing.shaped_random((3, 3, 4), xp, dtype)
        c = testing.shaped_random((4, 3, 4), xp, dtype)
        return xp.vstack((a, b, c))

    @testing.numpy_nlcpy_raises()
    def test_vstack_wrong_arg(self, xp):
        xp.vstack(1)

    @testing.numpy_nlcpy_raises()
    def test_vstack_wrong_ndim(self, xp):
        a = testing.shaped_random((2, 2, 2), xp)
        b = testing.shaped_random((2, 2), xp)
        xp.vstack((a, b))

    @testing.numpy_nlcpy_raises()
    def test_vstack_wrong_shape(self, xp):
        a = testing.shaped_random((2, 2), xp)
        b = testing.shaped_random((2, 3), xp)
        xp.vstack((a, b))

    # Test for hstack

    @testing.for_all_dtypes(name='a_dtype')
    @testing.for_all_dtypes(name='b_dtype')
    @testing.for_all_dtypes(name='c_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_hstack_0d_a(self, xp, a_dtype, b_dtype, c_dtype):
        a = testing.shaped_random((), xp, a_dtype)
        b = testing.shaped_random((3,), xp, b_dtype)
        c = testing.shaped_random((5,), xp, c_dtype)
        return xp.hstack((a, b, c))

    @testing.for_all_dtypes(name='a_dtype')
    @testing.for_all_dtypes(name='b_dtype')
    @testing.for_all_dtypes(name='c_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_hstack_1d(self, xp, a_dtype, b_dtype, c_dtype):
        a = testing.shaped_random((2,), xp, a_dtype)
        b = testing.shaped_random((3,), xp, b_dtype)
        c = testing.shaped_random((5,), xp, c_dtype)
        return xp.hstack((a, b, c))

    @testing.for_all_dtypes(name='a_dtype')
    @testing.for_all_dtypes(name='b_dtype')
    @testing.for_all_dtypes(name='c_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_hstack_2d(self, xp, a_dtype, b_dtype, c_dtype):
        a = testing.shaped_random((3, 2), xp, a_dtype)
        b = testing.shaped_random((3, 3), xp, b_dtype)
        c = testing.shaped_random((3, 4), xp, c_dtype)
        return xp.hstack((a, b, c))

    @testing.for_all_dtypes(name='a_dtype')
    @testing.for_all_dtypes(name='b_dtype')
    @testing.for_all_dtypes(name='c_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_hstack_large(self, xp, a_dtype, b_dtype, c_dtype):
        a = testing.shaped_random((2, 2, 4), xp, a_dtype)
        b = testing.shaped_random((2, 3, 4), xp, b_dtype)
        c = testing.shaped_random((2, 4, 4), xp, c_dtype)
        return xp.hstack((a, b, c) * 5)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_hstack_f_contiguous_a(self, xp, dtype):
        a = testing.shaped_random((4, 2, 2), xp, dtype).T  # a.shape = (2, 2, 4)
        b = testing.shaped_random((2, 3, 4), xp, dtype)
        c = testing.shaped_random((2, 4, 4), xp, dtype)
        return xp.hstack((a, b, c))

    @testing.numpy_nlcpy_raises()
    def test_hstack_wrong_arg(self, xp):
        xp.hstack(1)

    @testing.numpy_nlcpy_raises()
    def test_hstack_wrong_ndim(self, xp):
        a = testing.shaped_random((2, 2, 2), xp)
        b = testing.shaped_random((2, 2), xp)
        xp.hstack((a, b))

    @testing.numpy_nlcpy_raises()
    def test_hstack_wrong_shape(self, xp):
        a = testing.shaped_random((2, 2), xp)
        b = testing.shaped_random((3, 2), xp)
        xp.hstack((a, b))


class TestBlock(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_block_scalar(self, xp):
        return xp.block([i for i in range(3)])

    @testing.numpy_nlcpy_array_equal()
    def test_block_scalar_large(self, xp):
        return xp.block([i for i in range(20)])

    @testing.numpy_nlcpy_array_equal()
    def test_block_list(self, xp):
        return xp.block([[i, i + 1, i + 2] for i in range(0, 3 * 3, 3)])

    @testing.numpy_nlcpy_array_equal()
    def test_block_list_large(self, xp):
        return xp.block([[i, i + 1, i + 2] for i in range(0, 3 * 20, 3)])

    @testing.numpy_nlcpy_array_equal()
    def test_block_scalar_and_array(self, xp):
        return xp.block([*[i for i in range(3)], xp.arange(2)])

    @testing.numpy_nlcpy_array_equal()
    def test_block_scalar_and_array_large(self, xp):
        return xp.block([*[i for i in range(20)], xp.arange(2)])

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_all_dtypes(name='dtype_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_array_equal()
    def test_block_array(self, xp, dtype_a, dtype_b, order_a, order_b):
        a = xp.arange(16).reshape(4, 4)
        b = xp.arange(12).reshape(4, 3)
        a = xp.asarray(a, dtype=dtype_a, order=order_a)
        b = xp.asarray(b, dtype=dtype_b, order=order_b)
        return xp.block([a, b])

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_all_dtypes(name='dtype_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_array_equal()
    def test_block_array_large(self, xp, dtype_a, dtype_b, order_a, order_b):
        a = xp.arange(16).reshape(4, 4)
        b = xp.arange(12).reshape(4, 3)
        a = xp.asarray(a, dtype=dtype_a, order=order_a)
        b = xp.asarray(b, dtype=dtype_b, order=order_b)
        return xp.block([a, b] * 10)

    @testing.numpy_nlcpy_array_equal()
    def test_block_array_in_list(self, xp):
        a = [[xp.array([i, i + 1, i + 2])] for i in range(0, 2 * 2, 2)]
        return xp.block([*a, [-1, -2, -3]])

    @testing.numpy_nlcpy_array_equal()
    def test_block_array_in_list_large(self, xp):
        a = [[xp.array([i, i + 1, i + 2])] for i in range(0, 2 * 20, 2)]
        return xp.block([*a, [-1, -2, -3]])

    @testing.numpy_nlcpy_array_equal()
    def test_block_deep(self, xp):
        return xp.block(xp.arange(8).reshape(2, 1, 2, 1, 2, 1).tolist())

    @testing.numpy_nlcpy_array_equal()
    def test_block_deep_large(self, xp):
        return xp.block(xp.arange(720).reshape(1, 2, 3, 4, 5, 6).tolist())

    @testing.numpy_nlcpy_array_equal()
    def test_block_deep2(self, xp):
        a = xp.arange(9).reshape(3, 3)
        b = xp.arange(6).reshape(3, 2)
        c = xp.arange(6).reshape(2, 3)
        d = xp.arange(4).reshape(2, 2)
        return xp.block([[a, b], [c, d]])

    @testing.numpy_nlcpy_array_equal()
    def test_block_deep2_large(self, xp):
        a = xp.arange(9).reshape(3, 3)
        b = xp.arange(6).reshape(3, 2)
        c = xp.arange(6).reshape(2, 3)
        d = xp.arange(4).reshape(2, 2)
        return xp.block([[a, b], [c, d]] * 10)


class TestBlockFailure(unittest.TestCase):

    @testing.numpy_nlcpy_raises()
    def test_block_shape_mismatch(self, xp):
        xp.block([xp.ones([2, 2]), 1])

    @testing.numpy_nlcpy_raises()
    def test_block_tuple_input(self, xp):
        xp.block((xp.zeros([1, 2]), xp.ones([1, 2]), xp.arange(2)))

    @testing.numpy_nlcpy_raises()
    def test_block_list_depth_mismatch(self, xp):
        xp.block([xp.ones([2, 3, 4]).tolist(), xp.ones([3, 4]).tolist()])

    @testing.numpy_nlcpy_raises()
    def test_block_list_bottom_index_none(self, xp):
        xp.block([xp.ones([2, 3, 0]).tolist(), xp.ones([2, 3, 0]).tolist()])

    @testing.numpy_nlcpy_raises()
    def test_block_array_shape_mismatch(self, xp):
        xp.block([xp.ones([3, 4]).tolist(), xp.ones([2, 3]).tolist()])

    @testing.numpy_nlcpy_raises()
    def test_block_with_empty_list(self, xp):
        return xp.block([[i for i in range(3)], []])

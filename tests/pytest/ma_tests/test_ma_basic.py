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


class TestMaskedArray(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=['dt1', 'dt2', 'dt3', 'dt4'], full=True)
    @testing.numpy_nlcpy_array_equal()
    def test_creation_dtype(self, xp, dt1, dt2, dt3, dt4):
        data = testing.shaped_random([10], xp, dtype=dt1)
        mask = testing.shaped_arange([10], xp, dtype=dt2)
        fill_value = testing.shaped_random([10], xp, dtype=dt3)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, dtype=dt4)
        data[:] = 1
        return a

    @testing.for_orders('CF', name='order1')
    @testing.for_orders('CF', name='order2')
    @testing.for_orders('CF', name='order3')
    @testing.for_orders('CF', name='order4')
    @testing.numpy_nlcpy_array_equal()
    def test_creation_order(self, xp, order1, order2, order3, order4):
        data = testing.shaped_arange([4, 5], xp, order=order1)
        mask = testing.shaped_arange([4, 5], xp, order=order2)
        fill_value = testing.shaped_arange([4, 5], xp, order=order3)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, order=order4)
        data[0] += 1
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_creation_mask_true(self, xp):
        data = testing.shaped_arange([4, 5], xp)
        fill_value = testing.shaped_arange([15], xp)
        a = xp.ma.array(data, mask=True, fill_value=fill_value, hard_mask=True)
        data[0] += 1
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_creation_mask_false(self, xp):
        data = testing.shaped_arange([4, 5], xp)
        fill_value = testing.shaped_arange([15], xp)
        a = xp.ma.array(data, mask=False, fill_value=fill_value, hard_mask=True)
        data[0] += 1
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_creation_shape1(self, xp):
        data = testing.shaped_arange([4, 5], xp)
        mask = testing.shaped_arange([20], xp)
        fill_value = testing.shaped_arange([15], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        data[0] += 1
        return a

    @testing.numpy_nlcpy_raises()
    def test_creation_shape2(self, xp):
        data = testing.shaped_arange([4, 5], xp)
        mask = testing.shaped_arange([21], xp)
        fill_value = testing.shaped_arange([15], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        data[0] += 1
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_creation_ndmin(self, xp):
        data = testing.shaped_arange([4, 5], xp)
        a = xp.ma.array(data, ndmin=4)
        data[0] += 1
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_scalar1(self, xp):
        return xp.ma.array(1, mask=0, fill_value=3)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_scalar2(self, xp):
        return xp.ma.array(1, mask=2, fill_value=3)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_tuple(self, xp):
        data = (0, 1, 2, 3)
        mask = (0, 1, 0, 2)
        fill_value = (3, 2, 1, 0)
        return xp.ma.array(data, mask=mask, fill_value=fill_value)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_tuple2(self, xp):
        data = ((0, 1), (2, 3))
        mask = ((0, 1), (0, 2))
        fill_value = (3, 2, 1, 0, -1)
        return xp.ma.array(data, mask=mask, fill_value=fill_value)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_list(self, xp):
        data = [0, 1, 2, 3]
        mask = [0, 1, 0, 2]
        fill_value = [3, 2, 1, 0]
        return xp.ma.array(data, mask=mask, fill_value=fill_value)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_list_no_mask(self, xp):
        data = [0, 1, 2, 3]
        fill_value = [3, 2, 1, 0]
        return xp.ma.array(data, fill_value=fill_value)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_ma_list_no_mask(self, xp):
        data = [xp.ma.array(data=testing.shaped_random([2, 3], xp),
                            mask=testing.shaped_random([2, 3], xp, dtype=numpy.bool_),
                            fill_value=testing.shaped_random([2, 3], xp))
                for _ in range(5)]
        return xp.ma.array(data)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_ma_list_no_mask2(self, xp):
        data = [xp.ma.array(data=testing.shaped_random([2, 3], xp),
                            fill_value=testing.shaped_random([2, 3], xp))
                for _ in range(5)]
        return xp.ma.array(data)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_numpy_array(self, xp):
        data = testing.shaped_random([4, 5], numpy)
        mask = testing.shaped_random([4, 5], numpy, dtype=numpy.bool_)
        fill_value = testing.shaped_random([4, 5], numpy)
        return xp.ma.array(data, mask=mask, fill_value=fill_value)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_maskedarray(self, xp):
        data = testing.shaped_random([4, 5], xp)
        mask = testing.shaped_random([4, 5], xp, dtype=numpy.bool_)
        fill_value = testing.shaped_random([4, 5], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return xp.ma.array(a)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_maskedarray_copy(self, xp):
        data = testing.shaped_random([4, 5], xp)
        mask = testing.shaped_random([4, 5], xp, dtype=numpy.bool_)
        fill_value = testing.shaped_random([4, 5], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return xp.ma.array(a, copy=True)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_maskedarray_mask_resize(self, xp):
        data = testing.shaped_random([4, 5], xp)
        mask = [1]
        fill_value = testing.shaped_random([4, 5], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return xp.ma.array(a)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_maskedarray_with_arguments(self, xp):
        data = testing.shaped_random([4, 5], xp)
        mask = testing.shaped_random([4, 5], xp)
        fill_value = testing.shaped_random([4, 5], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)

        mask2 = testing.shaped_random([20], xp, dtype=numpy.bool_)
        fill_value2 = testing.shaped_random([3, 4], xp)
        b = xp.ma.array(
            a, mask=mask2, fill_value=fill_value2, dtype=nlcpy.float32, order='F')
        return b

    @testing.numpy_nlcpy_raises()
    def test_creation_from_maskedarray_shape_mismatch(self, xp):
        data = testing.shaped_random([4, 5], xp)
        mask = testing.shaped_random([4, 5], xp)
        fill_value = testing.shaped_random([4, 5], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)

        mask2 = testing.shaped_random([15], xp, dtype=numpy.bool_)
        return xp.ma.array(a, mask=mask2)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_from_numpy_maskedarray(self, xp):
        data = testing.shaped_random([4, 5], numpy, dtype=numpy.uint32)
        mask = testing.shaped_random([4, 5], numpy, dtype=numpy.bool_)
        fill_value = testing.shaped_random([4, 5], numpy)
        a = numpy.ma.array(data, mask=mask, fill_value=fill_value, hard_mask=True)
        return xp.ma.array(a)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_not_keepmask_not_shrink(self, xp):
        data = testing.shaped_random([4, 5], xp)
        mask = testing.shaped_random([4, 5], xp, dtype=numpy.bool_)
        fill_value = testing.shaped_random([4, 5], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return xp.ma.array(a, keep_mask=False, shrink=False)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_not_keepmask_shrink(self, xp):
        data = testing.shaped_random([4, 5], xp)
        mask = testing.shaped_random([4, 5], xp, dtype=numpy.bool_)
        fill_value = testing.shaped_random([4, 5], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return xp.ma.array(a, keep_mask=False, shrink=True)

    @testing.numpy_nlcpy_array_equal()
    def test_creation_not_keepmask_copy(self, xp):
        data = testing.shaped_random([4, 5], xp)
        mask = testing.shaped_random([4, 5], xp, dtype=numpy.bool_)
        fill_value = testing.shaped_random([4, 5], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)

        mask2 = testing.shaped_random([4, 5], xp, dtype=numpy.bool_)
        b = xp.ma.array(a, mask=mask2, keep_mask=False, copy=True)
        a[0] += 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_creation_not_keepmask_not_copy(self, xp):
        data = testing.shaped_random([4, 5], xp, dtype=numpy.uint64)
        mask = testing.shaped_random([4, 5], xp, dtype=numpy.bool_)
        fill_value = testing.shaped_random([4, 5], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)

        mask2 = testing.shaped_random([4, 5], xp, dtype=numpy.bool_)
        b = xp.ma.array(a, mask=mask2, keep_mask=False, copy=False)
        a[0] += numpy.uint64(1)
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_set_mask1(self, xp):
        data = testing.shaped_random([10], xp)
        a = xp.ma.array(data)
        a.mask = xp.ma.masked
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_set_mask2(self, xp):
        data = testing.shaped_random([10], xp)
        a = xp.ma.array(data)
        a.mask = xp.ma.nomask
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_set_mask3(self, xp):
        data = testing.shaped_random([10], xp)
        a = xp.ma.array(data)
        mask = testing.shaped_random([10], xp, dtype=numpy.bool_)
        a.mask = mask
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_set_mask4(self, xp):
        data = testing.shaped_random([5], xp)
        mask = [0, 0, 0, 1, 1]
        a = xp.ma.array(data, mask=mask, hard_mask=False)
        mask = [0, 0, 1, 0, 1]
        a.mask = mask
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_set_mask5(self, xp):
        data = testing.shaped_random([5], xp)
        mask = [0, 0, 0, 1, 1]
        a = xp.ma.array(data, mask=mask, hard_mask=True)
        mask = xp.array([0, 0, 1, 0, 1], dtype=numpy.bool_)
        a.mask = mask
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_set_mask6(self, xp):
        data = testing.shaped_random([5], xp, dtype=numpy.uint32)
        mask = [0, 0, 0, 1, 1]
        a = xp.ma.array(data, mask=mask, hard_mask=False)
        a.mask = 1
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_set_shape(self, xp):
        data = testing.shaped_random([6], xp, dtype=numpy.uint32)
        mask = [0, 0, 0, 1, 1, 1]
        a = xp.ma.array(data, mask=mask, hard_mask=True)
        a.shape = (2, 3)
        data[0] += 1
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_reshape(self, xp):
        data = testing.shaped_random([6], xp, dtype=numpy.uint32)
        mask = [0, 0, 0, 1, 1, 1]
        a = xp.ma.array(data, mask=mask, hard_mask=True)
        b = a.reshape((2, 3), order='F')
        a[0] += 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_reshape_external(self, xp):
        data = testing.shaped_random([6], xp, dtype=numpy.uint32)
        mask = [0, 0, 0, 1, 1, 1]
        a = xp.ma.array(data, mask=mask, hard_mask=True)
        b = xp.ma.reshape(a, (2, 3), order='F')
        a[0] += 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_reshape_external_ndarray(self, xp):
        a = xp.arange(6)
        return xp.ma.reshape(a, (2, 3))

    @testing.numpy_nlcpy_array_equal()
    def test_ma_filled(self, xp):
        data = testing.shaped_random([6], xp, dtype=numpy.uint32)
        mask = [0, 1, 0, 1, 0, 1]
        a = xp.ma.array(data, mask=mask, fill_value=10, hard_mask=True)
        b = a.filled()
        a[0] += 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_ma_filled_with_arg(self, xp):
        data = testing.shaped_random([6], xp, dtype=numpy.uint32)
        mask = [0, 1, 0, 1, 0, 1]
        a = xp.ma.array(data, mask=mask, hard_mask=True)
        b = a.filled(10)
        a[0] += 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_filled_ma(self, xp):
        data = testing.shaped_random([6], xp, dtype=numpy.uint32)
        mask = [0, 1, 0, 1, 0, 1]
        a = xp.ma.array(data, mask=mask, hard_mask=True)
        b = xp.ma.filled(a, 10)
        a[0] += 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_filled_ndarray(self, xp):
        a = testing.shaped_random([6], xp, dtype=numpy.uint32)
        b = xp.ma.filled(a, 10)
        a[0] += 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_filled_nomask(self, xp):
        a = xp.ma.array([1, 2, 3])
        b = a.filled()
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_filled_empty_mask(self, xp):
        a = xp.ma.array([1, 2, 3], mask=[0, 0, 0])
        b = a.filled()
        return b

    @testing.numpy_nlcpy_raises()
    def test_filled_uncastable_fill_value(self, xp):
        a = xp.ma.array([1, 2, 3], mask=[0, 1, 1])
        a.filled(5 + 6j)

    @testing.numpy_nlcpy_array_equal()
    def test_filled_etc(self, xp):
        a = [1, 3, 5]
        b = xp.ma.filled(a, 10)
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_mask_setter(self, xp):
        data = testing.shaped_random([6], xp, dtype=numpy.uint32)
        mask1 = xp.array([0, 1, 0, 1, 0, 1], dtype=numpy.bool_)
        mask2 = xp.array([1, 0, 1, 0, 1, 0], dtype=numpy.bool_)
        fill_value = testing.shaped_random([6], xp)
        a = xp.ma.array(data, mask=mask1, fill_value=fill_value, hard_mask=True)
        a.mask = mask2
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_harden_mask(self, xp):
        a = xp.ma.array(xp.arange(5), hard_mask=False)
        b = a.harden_mask()
        a[0] = 10
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_harden_mask_external(self, xp):
        a = xp.ma.array(xp.arange(5), hard_mask=False)
        b = xp.ma.harden_mask(a)
        a[0] = 10
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_harden_mask_external_ndarray(self, xp):
        a = xp.arange(5)
        return xp.ma.harden_mask(a)

    @testing.numpy_nlcpy_array_equal()
    def test_soften_mask(self, xp):
        a = xp.ma.array(xp.arange(5), hard_mask=True)
        b = a.soften_mask()
        a[0] += 10
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_soften_mask_external(self, xp):
        a = xp.ma.array(xp.arange(5), hard_mask=True)
        b = xp.ma.soften_mask(a)
        a[0] += 10
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_soften_mask_external_ndarray(self, xp):
        a = xp.arange(5)
        return xp.ma.soften_mask(a)

    @testing.numpy_nlcpy_array_equal()
    def test_unshare_mask(self, xp):
        data = testing.shaped_random([6], xp)
        mask = xp.array([0, 1, 0, 1, 0, 1])
        fill_value = testing.shaped_random([6], xp)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, hard_mask=True)
        b = a.unshare_mask()
        a[0] += 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_fill_value_setter(self, xp):
        data = testing.shaped_random([6], xp)
        mask = xp.array([0, 1, 0, 1, 0, 1])
        fill_value = xp.arange(6)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, hard_mask=True)
        a.fill_value = 10
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_fill_value_setter2(self, xp):
        data = testing.shaped_random([6], xp)
        mask = xp.array([0, 1, 0, 1, 0, 1])
        fill_value = xp.arange(6)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, hard_mask=True)
        a.fill_value = None
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_update_from_masked_array(self, xp):
        data = testing.shaped_random([6], xp)
        mask = xp.array([0, 1, 0, 1, 0, 1])
        fill_value = xp.arange(6)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, hard_mask=True)
        a.unshare_mask()
        b = xp.ma.array(1)
        b._update_from(a)
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_update_from_non_array(self, xp):
        data = testing.shaped_random([6], xp)
        mask = xp.array([0, 1, 0, 1, 0, 1])
        fill_value = xp.arange(6)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, hard_mask=True)
        a.unshare_mask()
        a._update_from(1)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_get_scalar_fill_value(self, xp):
        data = testing.shaped_random([6], xp)
        mask = xp.array([0, 1, 0, 1, 0, 1])
        a = xp.ma.array(data, mask=mask)
        a._fill_value = 10
        return a.fill_value

    @testing.numpy_nlcpy_array_equal()
    def test_set_array_fill_value(self, xp):
        data = testing.shaped_random([6], xp)
        mask = xp.array([0, 1, 0, 1, 0, 1])
        a = xp.ma.array(data, mask=mask)
        a.fill_value = xp.arange(6)
        return a

    def test_baseclass(self):
        a = nlcpy.ma.array([1, 2, 3])
        self.assertTrue(a.baseclass is nlcpy.ndarray)

    @testing.numpy_nlcpy_array_equal()
    def test_real(self, xp):
        data = testing.shaped_random([2, 3, 4], xp, dtype=numpy.complex64)
        mask = testing.shaped_random([2, 3, 4], xp, dtype=numpy.bool_)
        fill_value = testing.shaped_random([2, 3, 4], xp, dtype=numpy.complex64)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return a.real

    @testing.numpy_nlcpy_array_equal()
    def test_imag(self, xp):
        data = testing.shaped_random([2, 3, 4], xp, dtype=numpy.complex64)
        mask = testing.shaped_random([2, 3, 4], xp, dtype=numpy.bool_)
        fill_value = testing.shaped_random([2, 3, 4], xp, dtype=numpy.complex64)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return a.imag

    @testing.numpy_nlcpy_array_equal()
    def test_make_mask_none(self, xp):
        return xp.ma.make_mask_none([3])

    @testing.numpy_nlcpy_array_equal()
    def test_make_mask_none_with_dtype(self, xp):
        return xp.ma.make_mask_none([3, 4], dtype=numpy.int32)

    @testing.numpy_nlcpy_array_equal()
    def test_make_mask(self, xp):
        data = [[0, 1, 0], [1, 0, 1]]
        mask = [[1, 0, 0], [1, 0, 0]]
        m = xp.ma.array(data, mask=mask)
        return xp.ma.make_mask(m)

    @testing.numpy_nlcpy_array_equal()
    def test_make_mask_with_dtype(self, xp):
        m = [[0, 1, 0], [1, 0, 1]]
        return xp.ma.make_mask(m, dtype=numpy.uint32)

    @testing.numpy_nlcpy_array_equal()
    def test_make_mask_copy(self, xp):
        m = xp.array([[0, 1, 0], [1, 0, 1]], dtype=numpy.bool_)
        return xp.ma.make_mask(m, copy=True)

    @testing.numpy_nlcpy_array_equal()
    def test_make_mask_shrink(self, xp):
        m = xp.zeros([3, 4], dtype=numpy.bool_)
        return xp.ma.make_mask(m, shrink=True)

    @testing.numpy_nlcpy_array_equal()
    def test_make_mask_not_shrink(self, xp):
        m = xp.zeros([3, 4], dtype=numpy.bool_)
        return xp.ma.make_mask(m, shrink=False)

    @testing.numpy_nlcpy_array_equal()
    def test_make_mask_with_nomask(self, xp):
        return xp.ma.make_mask(xp.ma.nomask)

    @testing.numpy_nlcpy_array_equal()
    def test_mask_or(self, xp):
        m1 = xp.array([[0, 1, 0], [1, 0, 1]], dtype=numpy.bool_)
        m2 = xp.array([[1, 0, 0], [1, 0, 0]], dtype=numpy.bool_)
        return xp.ma.mask_or(m1, m2)

    @testing.numpy_nlcpy_array_equal()
    def test_mask_or_m1_nomask(self, xp):
        m1 = xp.ma.nomask
        m2 = xp.array([[1, 0, 0], [1, 0, 0]], dtype=numpy.bool_)
        return xp.ma.mask_or(m1, m2, copy=False)

    @testing.numpy_nlcpy_array_equal()
    def test_mask_or_m1_False(self, xp):
        m1 = False
        m2 = xp.array([[1, 0, 0], [1, 0, 0]], dtype=numpy.int32)
        return xp.ma.mask_or(m1, m2, copy=True)

    @testing.numpy_nlcpy_array_equal()
    def test_mask_or_m2_nomask(self, xp):
        m1 = xp.array([[0, 1, 0], [1, 0, 1]], dtype=numpy.bool_)
        m2 = xp.ma.nomask
        return xp.ma.mask_or(m1, m2, copy=False)

    @testing.numpy_nlcpy_array_equal()
    def test_mask_or_m2_False(self, xp):
        m1 = xp.array([[0, 1, 0], [1, 0, 1]], dtype=numpy.int32)
        m2 = False
        return xp.ma.mask_or(m1, m2, copy=True)

    @testing.numpy_nlcpy_array_equal()
    def test_mask_or_self(self, xp):
        m1 = xp.array([[0, 1, 0], [1, 0, 1]], dtype=numpy.bool_)
        m2 = m1
        m = xp.ma.mask_or(m1, m2)
        m1[0][0] = 1
        return m

    @testing.numpy_nlcpy_raises()
    def test_mask_or_incompatible_dtype(self, xp):
        m1 = xp.array([[0, 1, 0], [1, 0, 1]], dtype=numpy.bool_)
        m2 = xp.array([[1, 0, 0], [1, 0, 0]], dtype=numpy.int32)
        return xp.ma.mask_or(m1, m2)

    @testing.numpy_nlcpy_array_equal()
    def test_is_mask_not_bool_array(self, xp):
        m = xp.array([[0, 1, 0], [1, 0, 1]], dtype=xp.ma.MaskType)
        return xp.ma.is_mask(m)

    @testing.numpy_nlcpy_array_equal()
    def test_is_mask_not_array(self, xp):
        m = [[0, 1, 0], [1, 0, 1]]
        return xp.ma.is_mask(m)

    @testing.numpy_nlcpy_array_equal()
    def test_is_mask_not_bool_array2(self, xp):
        m = xp.array([[0, 1, 0], [1, 0, 1]])
        return xp.ma.is_mask(m)

    @testing.numpy_nlcpy_array_equal()
    def test_getmaskarray(self, xp):
        data = testing.shaped_random([10], xp)
        mask = testing.shaped_arange([10], xp)
        a = xp.ma.array(data, mask=mask)
        return xp.ma.getmaskarray(a)

    @testing.numpy_nlcpy_array_equal()
    def test_getmaskarray_with_nomask(self, xp):
        a = testing.shaped_random([10], xp)
        return xp.ma.getmaskarray(a)

    @testing.numpy_nlcpy_array_equal()
    def test_shrink_mask(self, xp):
        data = testing.shaped_random([10], xp)
        mask = testing.shaped_arange([10], xp)
        a = xp.ma.array(data, mask=mask)
        return a.shrink_mask()

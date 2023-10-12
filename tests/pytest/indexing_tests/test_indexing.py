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
import nlcpy
import warnings
import numpy

from nlcpy import testing


class TestIndexing(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_take_by_scalar(self, xp, dtype):
        a = testing.shaped_arange((2, 4, 3), xp, dtype)
        return a.take(2, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_external_take_by_scalar(self, xp, dtype):
        a = testing.shaped_arange((2, 4, 3), xp, dtype)
        return xp.take(a, 2, axis=1)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.numpy_nlcpy_array_equal()
    def test_take_by_array1(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange((2, 4, 3), xp, dtype_a)
        b = xp.array([[1, 0], [1, 0]], dtype=dtype_b)
        return a.take(b, axis=0)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.numpy_nlcpy_array_equal()
    def test_take_by_array2(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange((2, 4, 3), xp, dtype_a)
        b = xp.array([[1, 3], [2, 0]], dtype_b)
        return a.take(b, axis=1)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.numpy_nlcpy_array_equal()
    def test_take_by_array3(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange((2, 4, 3), xp, dtype_a)
        b = xp.array([[1, 2], [2, 0]], dtype_b)
        return a.take(b, axis=2)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.numpy_nlcpy_array_equal()
    def test_take_by_array_negative_axis1(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange((2, 4, 3), xp, dtype_a)
        b = xp.array([[1, 0], [1, 0]], dtype_b)
        return a.take(b, axis=-3)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.numpy_nlcpy_array_equal()
    def test_take_by_array_negative_axis2(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange((2, 4, 3), xp, dtype_a)
        b = xp.array([[1, 3], [2, 0]], dtype_b)
        return a.take(b, axis=-2)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.numpy_nlcpy_array_equal()
    def test_take_by_array_negative_axis3(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange((2, 4, 3), xp, dtype_a)
        b = xp.array([[1, 2], [2, 0]], dtype_b)
        return a.take(b, axis=-1)

    # NumPy does not support for dtype charcters 'lfdFD'.
    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('LfdFD', name='dtype_b')
    def test_take_ind_only_nlcpy(self, dtype_a, dtype_b):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            a = testing.shaped_arange((2, 4, 3), nlcpy, dtype_a)
            ind = [1, 0, 5]
            b = nlcpy.array(ind, dtype_b)
            testing.assert_array_equal(nlcpy.take(a, b), a.ravel()[ind])

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.for_all_dtypes(name='dtype_c')
    def test_take_with_out1(self, dtype_a, dtype_b, dtype_c):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            a = testing.shaped_arange((5, 4, 3), nlcpy, dtype_a)
            b = nlcpy.array([[1, 0], [1, 0]], dtype=dtype_b)
            c = nlcpy.empty((2, 2, 4, 3), dtype=dtype_c)
            res = nlcpy.take(a, b, out=c, axis=0)
            exp = a.get().take(b.get(), axis=0).astype(c.dtype)
            assert id(res) == id(c)
            testing.assert_array_equal(c, exp)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.for_all_dtypes(name='dtype_c')
    def test_take_with_out2(self, dtype_a, dtype_b, dtype_c):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            a = testing.shaped_arange((5, 4, 3), nlcpy, dtype_a)
            b = nlcpy.array([[1, 0], [1, 0]], dtype=dtype_b)
            c = nlcpy.empty((5, 2, 2, 3), dtype=dtype_c)
            res = nlcpy.take(a, b, out=c, axis=1)
            exp = a.get().take(b.get(), axis=1).astype(c.dtype)
            assert id(res) == id(c)
            testing.assert_array_equal(c, exp)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.for_all_dtypes(name='dtype_c')
    def test_take_with_out3(self, dtype_a, dtype_b, dtype_c):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            a = testing.shaped_arange((5, 4, 3), nlcpy, dtype_a)
            b = nlcpy.array([[1, 0], [1, 0]], dtype=dtype_b)
            c = nlcpy.empty((5, 4, 2, 2), dtype=dtype_c)
            res = nlcpy.take(a, b, out=c, axis=2)
            exp = a.get().take(b.get(), axis=2).astype(c.dtype)
            assert id(res) == id(c)
            testing.assert_array_equal(c, exp)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.for_all_dtypes(name='dtype_c')
    def test_take_with_out_no_axis(self, dtype_a, dtype_b, dtype_c):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            a = testing.shaped_arange((5, 4, 3), nlcpy, dtype_a)
            b = nlcpy.array([[1, 0], [1, 0]], dtype=dtype_b)
            c = nlcpy.empty(b.shape, dtype=dtype_c)
            res = nlcpy.take(a, b, out=c)
            exp = a.get().take(b.get()).astype(c.dtype)
            assert id(res) == id(c)
            testing.assert_array_equal(c, exp)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.numpy_nlcpy_raises()
    def test_take_with_out_shape_mismatch(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange((5, 4, 3), xp, dtype_a)
        b = xp.array([[1, 0], [1, 0]], dtype=dtype_b)
        c = xp.empty((3, 4), dtype=a.dtype)
        xp.take(a, b, out=c, axis=0)

    @testing.numpy_nlcpy_array_equal()
    def test_take_by_array_wrap_around1(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        b = xp.array([[1, 99999], [-2, 1]])
        return a.take(b, axis=0, mode='wrap')

    @testing.numpy_nlcpy_array_equal()
    def test_take_by_array_wrap_around2(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        b = xp.array([[1, 99999], [-2, 1]])
        return a.take(b, axis=1, mode='wrap')

    @testing.numpy_nlcpy_array_equal()
    def test_take_by_array_wrap_around3(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        b = xp.array([[1, 99999], [-2, 1]])
        return a.take(b, axis=2, mode='wrap')

    @testing.numpy_nlcpy_array_equal()
    def test_take_no_axis(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        b = xp.array([[10, 5], [3, 20]])
        return a.take(b)

    @testing.numpy_nlcpy_raises()
    def test_take_negative_invalid_axis(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        b = xp.array([[0, 1], [1, 0]])
        xp.take(a, b, axis=-4)

    @testing.numpy_nlcpy_raises()
    def test_take_invalid_axis(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        b = xp.array([[0, 1], [1, 0]])
        xp.take(a, b, axis=3)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilIL', name='dtype_b')
    @testing.numpy_nlcpy_array_equal()
    def test_take_0d_ind(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange((3, 4, 5), xp, dtype_a)
        b = xp.array(1, dtype=dtype_b)
        return a.take(b)

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.numpy_nlcpy_array_equal()
    def test_take_0d_ind_wrap_around(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange((3, 4, 5), xp, dtype_a)
        b = xp.array(1000, dtype=dtype_b)
        return xp.take(a, b, mode='wrap')

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_dtypes('?ilI', name='dtype_b')
    @testing.numpy_nlcpy_array_equal()
    def test_take_0d_ind_wrap_around_negative(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange((3, 4, 5), xp, dtype_a)
        b = xp.array(-1).astype(dtype_b)
        return xp.take(a, b, mode='wrap')

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_take_0d_array(self, xp, dtype):
        a = xp.array(3, dtype=dtype)
        return xp.take(a, 0)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_take_by_scalar_ind_not_int(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return xp.take(a, 2.2)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_diagonal(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_external_diagonal(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return xp.diagonal(a, 1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_diagonal_negative1(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(-1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_diagonal_negative2(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -1, -2)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_diagonal_negative3(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -1, 1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_diagonal_negative4(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -3, -1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_diagonal_negative5(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -1, -3)

    @testing.numpy_nlcpy_raises()
    def test_diagonal_invalid1(self, xp):
        a = testing.shaped_arange((3, 3, 3), xp)
        a.diagonal(0, 1, 3)

    @testing.numpy_nlcpy_raises()
    def test_diagonal_invalid2(self, xp):
        a = testing.shaped_arange((3, 3, 3), xp)
        a.diagonal(0, 2, -4)


@testing.parameterize(*(
    testing.product({
        'n': [-1, 0, 1, 2, 4, 7, 1.5],
        'ndim': [-1, 0, 1, 2, 3, 4]
    })
))
class TestDiagIndices(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_diag_indices(self, xp):
        return xp.diag_indices(self.n, self.ndim)


class TestSelect(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_select(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        condlist = [a > 3, a < 5]
        choicelist = [a, a + 100]
        return xp.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_select_default(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        condlist = [a < 3, a > 5]
        choicelist = [a, a + 100]
        default = -100
        return xp.select(condlist, choicelist, default)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_select_default_list(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        condlist = [a < 3, a > 5]
        choicelist = [a, a + 100]
        default = xp.arange(10) - 100
        return xp.select(condlist, choicelist, default)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_select_broadcast1(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        b = xp.arange(30, dtype=dtype).reshape(3, 10)
        condlist = [a < 3, b > 8]
        choicelist = [a, b]
        return xp.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_select_broadcast2(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        b = xp.arange(20, dtype=dtype).reshape(2, 10)
        condlist = [a < 4, b > 8]
        choicelist = [xp.repeat(a, 2).reshape(2, 10), b]
        return xp.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_select_broadcast3(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        condlist = [a < 4]
        choicelist = [a + 100]
        default = xp.arange(20).reshape(2, 10)
        return xp.select(condlist, choicelist, default)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_select_1D_choicelist(self, xp, dtype):
        a = xp.array(1)
        b = xp.array(3)
        condlist = [a < 3, b > 8]
        choicelist = [a, b]
        return xp.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_raises()
    def test_select_length_error(self, xp, dtype):
        a = xp.arange(10, dtype=dtype) + 1j
        condlist = [a > 3]
        choicelist = [a, a ** 2]
        xp.select(condlist, choicelist)

    @testing.numpy_nlcpy_raises()
    def test_select_type_error_choicelist(self, xp):
        a, b = list(range(10)), list(range(-10, 0))
        condlist = [0] * 10
        choicelist = [a, b]
        xp.select(condlist, choicelist)

    @testing.for_dtypes('fdFD')
    @testing.numpy_nlcpy_raises()
    def test_select_type_error_condlist(self, xp, dtype):
        condlist = xp.arange(10, dtype=dtype)
        choicelist = xp.arange(10)
        xp.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_raises()
    def test_select_not_broadcastable(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        b = xp.arange(20, dtype=dtype)
        condlist = [a < 3, b > 8]
        choicelist = [a, b]
        xp.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_raises()
    def test_select_default_shape_mismatch(self, xp, dtype):
        a = xp.arange(10)
        b = xp.arange(20)
        condlist = [a < 3, b > 8]
        choicelist = [a, b]
        xp.select(condlist, choicelist, [dtype(2)])

    @testing.numpy_nlcpy_raises()
    def test_select_integer_condlist(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        condlist = [[3, ] * 10, [2, ] * 10]
        choicelist = [a, a + 100]
        return xp.select(condlist, choicelist)

    @testing.numpy_nlcpy_raises()
    def test_select_empty_lists(self, xp):
        condlist = []
        choicelist = []
        return xp.select(condlist, choicelist)

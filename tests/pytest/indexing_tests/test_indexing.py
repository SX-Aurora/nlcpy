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

from nlcpy import testing


class TestIndexing(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_take_by_scalar(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        return a.take(2, axis=1)

    @testing.numpy_nlcpy_array_equal()
    def test_external_take_by_scalar(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        return xp.take(a, 2, axis=1)

    @testing.numpy_nlcpy_array_equal()
    def test_take_by_array(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        b = xp.array([[1, 3], [2, 0]])
        return a.take(b, axis=1)

    @testing.numpy_nlcpy_array_equal()
    def test_take_no_axis(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        b = xp.array([[10, 5], [3, 20]])
        return a.take(b)

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
        a = xp.arange(10, dtype=dtype)
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

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

from nlcpy import testing


class TestSplit(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_split_by_sections1(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_arange((3, 11), xp, dtype), order=order)
        return xp.split(a, (2, 4, 9), 1)

    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_split_by_sections2(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_arange((3, 11), xp, dtype), order=order)
        return xp.split(a, (2, 4, 9), -1)

    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_split_by_sections3(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_arange((3, 11), xp, dtype), order=order)
        return xp.split(a, (-9, 4, -2), 1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_split_by_sections4(self, xp, dtype):
        a = xp.arange(12).astype(dtype)
        return xp.split(a, (3, None))

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_split_by_sections5(self, xp, dtype):
        a = xp.arange(12).astype(dtype)
        return xp.split(a, (3, -15))

    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_split_out_of_bound1(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_arange((2, 3), xp, dtype), order=order)
        return xp.split(a, [3])

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_split_out_of_bound2(self, xp, dtype):
        a = testing.shaped_arange((0, ), xp, dtype)
        return xp.split(a, [1])

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_split_unordered_sections(self, xp, dtype):
        a = testing.shaped_arange((5, ), xp, dtype)
        return xp.split(a, [4, 2])

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_split_int_sections(self, xp, dtype):
        a = testing.shaped_arange((12, ), xp, dtype)
        return xp.split(a, 3)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_split_int_sections_large(self, xp, dtype):
        a = testing.shaped_arange((6000, ), xp, dtype)
        return xp.split(a, 2000)

    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_hsplit(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_arange((3, 12), xp, dtype), order=order)
        return xp.hsplit(a, 4)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_hsplit_vectors(self, xp, dtype):
        a = testing.shaped_arange((12,), xp, dtype)
        return xp.hsplit(a, 4)

    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_vsplit(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_arange((12, 3), xp, dtype), order=order)
        return xp.vsplit(a, 4)


class TestSplitFailure(unittest.TestCase):

    @testing.numpy_nlcpy_raises()
    def test_split_invalid_axis(self, xp):
        return xp.split(xp.empty([3, 4]), 1, 2)

    @testing.numpy_nlcpy_raises()
    def test_split_invalid_axis2(self, xp):
        return xp.split(xp.empty([3, 4]), 1, -3)

    @testing.numpy_nlcpy_raises()
    def test_split_zero_section(self, xp):
        return xp.split(xp.empty([3, 4]), 0)

    @testing.numpy_nlcpy_raises()
    def test_split_indivisible_section(self, xp):
        return xp.split(xp.arange(9), 2)

    @testing.numpy_nlcpy_raises()
    def test_split_negative_section(self, xp):
        return xp.split(xp.empty([3, 4]), -1)

    @testing.numpy_nlcpy_raises()
    def test_hsplit_0d(self, xp):
        return xp.hsplit(1, 1)

    @testing.numpy_nlcpy_raises()
    def test_vsplit_0d(self, xp):
        return xp.vsplit(1, 1)

    @testing.numpy_nlcpy_raises()
    def test_vsplit_1d(self, xp):
        return xp.vsplit(xp.arange(10), 5)

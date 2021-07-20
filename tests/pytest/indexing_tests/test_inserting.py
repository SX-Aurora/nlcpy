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


@testing.parameterize(*(
    testing.product({
        'shape_a': ((4, 4), (4, 4, 4), (10, 100), (100, 10)),
        'shape_val': ((2,), (10, ), (4, 4), (0, 4)),
        'wrap': (True, False),
    }))
)
class TestFillDiagonal(unittest.TestCase):

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_all_dtypes(name='dtype_val')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_val')
    @testing.numpy_nlcpy_array_equal()
    def test_fill_diagonal(self, xp, dtype_a, dtype_val, order_a, order_val):
        a = testing.shaped_arange(self.shape_a, xp, dtype=dtype_a, order=order_a)
        val = xp.asarray(
            testing.shaped_random(self.shape_val, xp, dtype_val), order=order_val)
        xp.fill_diagonal(a, val=val, wrap=self.wrap)
        return a

    @testing.for_all_dtypes(name='dtype_a')
    @testing.for_all_dtypes(name='dtype_val')
    @testing.numpy_nlcpy_array_equal()
    def test_fill_diagonal_strided(self, xp, dtype_a, dtype_val):
        a = testing.shaped_arange(self.shape_a, xp, dtype=dtype_a)
        a = a[[slice(0, None, 2) for _ in range(len(self.shape_a))]]
        val = testing.shaped_random(self.shape_val, xp, dtype_val)
        val = val[[slice(0, None, 2) for _ in range(len(self.shape_val))]]
        xp.fill_diagonal(a, val=val, wrap=self.wrap)
        return a


class TestFillDiagonalFailure(unittest.TestCase):

    @testing.numpy_nlcpy_raises()
    def test_fill_diagonal_1d(self, xp):
        xp.fill_diagonal(xp.arange(10), 1)

    @testing.numpy_nlcpy_raises()
    def test_fill_diagonal_shape_mismatch(self, xp):
        xp.fill_diagonal(xp.zeros([2, 3, 4]), 1)

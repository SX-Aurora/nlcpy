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
from nlcpy import testing

nan_dtypes = (
    numpy.float32,
    numpy.float64,
    numpy.complex64,
    numpy.complex128,
)

shapes = (
    (4,),
    (3, 4),
    (2, 3, 4),
)

shapes1 = (
    (3,),
    (3, 3),
    (3, 3, 3),
)


@testing.parameterize(*(
    testing.product({
        'shape': shapes,
        'shape1': shapes1,
    })
))
class TestArryEtc(unittest.TestCase):
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_case_01(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.conj()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_case_02(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.conjugate()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_case_03(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.cumsum()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_case_04(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.cumsum(axis=0)

    @testing.for_CF_orders()
    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_case_05(self, xp, dtype, order):
        a = testing.shaped_random(self.shape1, xp, dtype)
        b = testing.shaped_random(self.shape1, xp, dtype)
        return a.dot(b)

    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_case_06(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        return xp.prod(a)

    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_case_07_1(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        return xp.prod(a, axis=0)

    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_case_07_2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.prod()

    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_case_08(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.prod(axis=0)

    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_case_09(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.prod(axis=0, keepdims=True)

    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_case_10(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if a.ndim > 1:
            ret = a.prod(axis=1)
        else:
            ret = a.prod(axis=0)
        return ret

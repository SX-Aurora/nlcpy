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


@testing.parameterize(*(
    testing.product({
        'shape': [(2,), (2, 3), (2, 3, 4)],
    })
))
class TestFloating(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_array_equal()
    def test_signbit(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return xp.signbit(a)

    @testing.for_all_dtypes_combination(
        ('dtype_a', 'dtype_b'), no_complex=True)
    @testing.numpy_nlcpy_array_equal()
    def test_copysign_combination(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange(self.shape, xp, dtype_a)
        b = testing.shaped_reverse_arange(self.shape, xp, dtype_b)
        return xp.copysign(a, b)

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_copysign_float(self, xp, dtype):
        a = xp.array([-xp.inf, -3, -0.0, 0, 3, xp.inf], dtype=dtype)[:, None]
        b = xp.array([-xp.inf, -3, -0.0, 0, 3, xp.inf], dtype=dtype)[None, :]
        return xp.copysign(a, b)

    @testing.for_float_dtypes(name='ftype')
    @testing.for_dtypes(['i', 'l'], name='itype')
    @testing.numpy_nlcpy_array_equal()
    def test_ldexp(self, xp, ftype, itype):
        a = testing.shaped_arange(self.shape, xp, ftype)
        b = testing.shaped_reverse_arange(self.shape, xp, itype)
        return xp.ldexp(a, b)

    @testing.for_all_dtypes_combination(
        ('dtype_a', 'dtype_b'), no_complex=True)
    @testing.numpy_nlcpy_array_equal()
    def test_nextafter_combination(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange(self.shape, xp, dtype_a)
        b = testing.shaped_reverse_arange(self.shape, xp, dtype_b)
        return xp.nextafter(a, b)

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_nextafter_float(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype) - 3
        b = testing.shaped_reverse_arange(self.shape, xp, dtype) - 3
        b[0] = -xp.inf
        b[-1] = xp.inf
        a = a[:, None]
        b = b[None, :]
        return xp.nextafter(a, b)

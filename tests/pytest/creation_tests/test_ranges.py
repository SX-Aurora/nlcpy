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

import math
import sys
import unittest

from nlcpy import testing


class TestRanges(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_arange(self, xp, dtype):
        return xp.arange(10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_arange2(self, xp, dtype):
        return xp.arange(5, 10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_arange3(self, xp, dtype):
        return xp.arange(1, 11, 2, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_arange4(self, xp, dtype):
        return xp.arange(20, 2, -3, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_arange5(self, xp, dtype):
        return xp.arange(0, 100, None, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_arange6(self, xp, dtype):
        return xp.arange(0, 2, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_arange7(self, xp, dtype):
        return xp.arange(10, 11, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_arange8(self, xp, dtype):
        return xp.arange(10, 8, -1, dtype=dtype)

    @testing.numpy_nlcpy_raises()
    def test_arange9(self, xp):
        return xp.arange(10, dtype=xp.bool_)

    @testing.numpy_nlcpy_array_equal()
    def test_arange_no_dtype_int(self, xp):
        return xp.arange(1, 11, 2)

    @testing.numpy_nlcpy_array_equal()
    def test_arange_no_dtype_float(self, xp):
        return xp.arange(1.0, 11.0, 2.0)

    @testing.numpy_nlcpy_array_equal()
    def test_arange_negative_size(self, xp):
        return xp.arange(3, 1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace(self, xp, dtype):
        return xp.linspace(0, 10, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace2(self, xp, dtype):
        return xp.linspace(10, 0, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_zero_num(self, xp, dtype):
        return xp.linspace(0, 10, 0, dtype=dtype)

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_zero_num_no_endopoint_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 0, dtype=dtype, endpoint=False,
                              retstep=True)
        self.assertTrue(math.isnan(step))
        return x

    @testing.with_requires('numpy>=1.10', 'numpy<1.18')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_one_num_no_endopoint_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 1, dtype=dtype, endpoint=False,
                              retstep=True)
        self.assertTrue(math.isnan(step))
        return x

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_one_num(self, xp, dtype):
        return xp.linspace(0, 2, 1, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_no_endpoint(self, xp, dtype):
        return xp.linspace(0, 10, 5, dtype=dtype, endpoint=False)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 5, dtype=dtype, retstep=True)
        self.assertEqual(step, 2.5)
        return x

    @testing.numpy_nlcpy_allclose()
    def test_linspace_no_dtype_int(self, xp):
        return xp.linspace(0, 10)

    @testing.numpy_nlcpy_allclose()
    def test_linspace_no_dtype_float(self, xp):
        return xp.linspace(0.0, 10.0)

    @testing.numpy_nlcpy_allclose()
    def test_linspace_float_args_with_int_dtype(self, xp):
        return xp.linspace(0.1, 9.1, 11, dtype=int)

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_raises()
    def test_linspace_neg_num(self, xp):
        return xp.linspace(0, 10, -1)

    @testing.numpy_nlcpy_allclose()
    def test_linspace_float_overflow(self, xp):
        return xp.linspace(0., sys.float_info.max / 5, 10, dtype=float)

    # @testing.with_requires('numpy>=1.10')
    # @testing.numpy_nlcpy_array_equal()
    # def test_linspace_float_underflow(self, xp):
    #     # find minimum subnormal number
    #     x = sys.float_info.min
    #     while x / 2 > 0:
    #         x /= 2
    #     return xp.linspace(0., x, 10, dtype=float)

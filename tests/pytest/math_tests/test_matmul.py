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

import sys
import unittest

import numpy

from nlcpy import testing


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            # dot test
            ((3, 2), (2, 4)),
            ((3, 0), (0, 4)),
            ((0, 2), (2, 4)),
            ((3, 2), (2, 0)),
            ((2,), (2, 4)),
            ((0,), (0, 4)),
            ((3, 2), (2,)),
            ((3, 0), (0,)),
            ((2,), (2,)),
            ((0,), (0,)),
            ((10001,), (10001,)),
            # matmul test
            ((5, 3, 2), (5, 2, 4)),
            ((0, 3, 2), (0, 2, 4)),
            ((5, 3, 2), (2, 4)),
            ((0, 3, 2), (2, 4)),
            ((3, 2), (5, 2, 4)),
            ((3, 2), (0, 2, 4)),
            ((5, 3, 2), (1, 2, 4)),
            ((0, 3, 2), (1, 2, 4)),
            ((1, 3, 2), (5, 2, 4)),
            ((1, 3, 2), (0, 2, 4)),
            ((5, 3, 2), (2,)),
            ((5, 3, 0), (0,)),
            ((2,), (5, 2, 4)),
            ((0,), (5, 0, 4)),
            ((2, 2, 3, 2), (2, 2, 2, 4)),
            ((5, 0, 3, 2), (5, 0, 2, 4)),
            ((6, 5, 3, 2), (2, 4)),
            ((5, 0, 3, 2), (2, 4)),
            ((3, 2), (6, 5, 2, 4)),
            ((3, 2), (5, 0, 2, 4)),
            ((1, 5, 3, 2), (6, 1, 2, 4)),
            ((1, 0, 3, 2), (6, 1, 2, 4)),
            ((6, 1, 3, 2), (1, 5, 2, 4)),
            ((6, 1, 3, 2), (1, 0, 2, 4)),
            ((6, 5, 3, 2), (2,)),
            ((6, 5, 3, 0), (0,)),
            ((2,), (6, 5, 2, 4)),
            ((0,), (6, 5, 0, 4)),
            ((1, 3, 3), (10, 1, 3, 1)),
        ],
    }))
class TestMatmul(unittest.TestCase):

    @unittest.skipUnless(sys.version_info >= (3, 5),
                         'Only for Python3.5 or higher')
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-3)
    def test_operator_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_random(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_random(self.shape_pair[1], xp, dtype2)
        return x1 @ x2

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-3)
    def test_nlcpy_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_random(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_random(self.shape_pair[1], xp, dtype2)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            ((5, 3, 1), (3, 1, 4)),
            ((3, 2, 3), (3, 2, 4)),
            ((3, 2), ()),
            ((), (3, 2)),
            ((), ()),
            ((3, 2), (1,)),
            ((0, 2), (3, 0)),
            ((0, 1, 1), (2, 1, 1)),
        ],
    }))
class TestMatmulInvalidShape(unittest.TestCase):

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_raises(accept_error=ValueError)
    def test_invalid_shape(self, xp):
        shape1, shape2 = self.shape_pair
        x1 = testing.shaped_random(shape1, xp, numpy.float32)
        x2 = testing.shaped_random(shape2, xp, numpy.float32)
        xp.matmul(x1, x2)


class TestMatmulDtypeAndOrder(unittest.TestCase):

    @testing.for_orders('CFAK', name='order_x')
    @testing.for_orders('CFAK', name='order_y')
    @testing.for_orders('CFAK', name='order_out')
    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-3)
    def test_matmul_from_array(self, xp, dtype, order_x, order_y, order_out):
        x = testing.shaped_random((2, 2), xp, dtype)
        y = testing.shaped_random((2, 2), xp, dtype)
        x = xp.asarray(x, order=order_x)
        y = xp.asarray(y, order=order_y)
        return xp.matmul(x, y, order=order_out, dtype=dtype)

    @testing.for_orders('CFAK', name='order_x')
    @testing.for_orders('CFAK', name='order_y')
    @testing.for_orders('CFAK', name='order_out')
    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-3)
    def test_matmul_from_view(self, xp, dtype, order_x, order_y, order_out):
        x = testing.shaped_random((4, 4), xp, dtype)
        y = testing.shaped_random((4, 4), xp, dtype)
        x = xp.asarray(x, order=order_x)
        y = xp.asarray(y, order=order_y)
        x = x[::2, ::2]
        y = y[::2, ::2]
        return xp.matmul(x, y, order=order_out, dtype=dtype)

    @testing.for_orders('CF', name='order_x')
    @testing.for_orders('CF', name='order_y')
    @testing.for_orders('CF', name='order_out')
    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-3)
    def test_matmul_stride1(self, xp, dtype, order_x, order_y, order_out):
        x = testing.shaped_random((3, 1), xp, dtype)
        y = testing.shaped_random((1, 4), xp, dtype)
        return xp.matmul(x, y, order=order_out, dtype=dtype)

    @testing.for_orders('CF', name='order_x')
    @testing.for_orders('CF', name='order_y')
    @testing.for_orders('CF', name='order_out')
    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-3)
    def test_matmul_stride2(self, xp, dtype, order_x, order_y, order_out):
        x = testing.shaped_random((1, 3), xp, dtype)
        y = testing.shaped_random((3, 1), xp, dtype)
        return xp.matmul(x, y, order=order_out, dtype=dtype)

    @testing.for_orders('CF', name='order_x')
    @testing.for_orders('CF', name='order_y')
    @testing.for_orders('CF', name='order_out')
    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-3)
    def test_matmul_stride3(self, xp, dtype, order_x, order_y, order_out):
        x = testing.shaped_random((5, 6), xp, dtype)
        y = testing.shaped_random((7, 8), xp, dtype)
        x = xp.asarray(x, order=order_x)[:4, :4]
        y = xp.asarray(y, order=order_y)[:4, :4]
        return xp.matmul(x, y, order=order_out, dtype=dtype)

    # TODO
    # @testing.for_orders('CFAK', name='order_x')
    # @testing.for_orders('CFAK', name='order_y')
    # @testing.for_orders('CFAK', name='order_out')
    # @testing.for_all_dtypes(no_float16=True, no_bool=True)
    # @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-3)
    # def test_matmul_with_out(self, xp, dtype, order_x, order_y, order_out):
    #     x = testing.shaped_random((2,2), xp, dtype)
    #     y = testing.shaped_random((2,2), xp, dtype)
    #     x = xp.asarray(x, order=order_x)
    #     y = xp.asarray(y, order=order_y)
    #     out = xp.empty((2,2), dtype=dtype, order=order_out)
    #     xp.matmul(x, y, dtype=dtype, out=out)
    #     return out

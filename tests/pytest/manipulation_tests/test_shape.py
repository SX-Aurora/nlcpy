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

import nlcpy
from nlcpy import testing


class TestShape(unittest.TestCase):

    def test_reshape_strides(self):
        def func(xp):
            a = testing.shaped_arange((1, 1, 1, 2, 2), xp)
            return a.strides
        self.assertEqual(func(numpy), func(nlcpy))

    def test_reshape2(self):
        def func(xp):
            a = xp.zeros((8,), dtype=xp.float32)
            return a.reshape((1, 1, 1, 4, 1, 2)).strides
        self.assertEqual(func(numpy), func(nlcpy))

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_nocopy_reshape(self, xp, dtype, order):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = a.reshape(4, 3, 2, order=order)
        b[1] = 1
        return a

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_nocopy_reshape_with_order(self, xp, dtype, order):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = a.reshape(4, 3, 2, order=order)
        b[1] = 1
        return a

    @testing.for_orders('CFA')
    @testing.numpy_nlcpy_array_equal()
    def test_transposed_reshape2(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp).transpose(2, 0, 1)
        return a.reshape(2, 3, 4, order=order)

    @testing.for_orders('CFA')
    @testing.numpy_nlcpy_array_equal()
    def test_reshape_with_unknown_dimension(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.reshape(3, -1, order=order)

    @testing.numpy_nlcpy_raises()
    def test_reshape_with_multiple_unknown_dimensions(self):
        a = testing.shaped_arange((2, 3, 4))
        a.reshape(3, -1, -1)

    @testing.numpy_nlcpy_raises()
    def test_reshape_with_changed_arraysize(self):
        a = testing.shaped_arange((2, 3, 4))
        a.reshape(2, 4, 4)

    @testing.numpy_nlcpy_raises()
    def test_reshape_invalid_order(self):
        a = testing.shaped_arange((2, 3, 4))
        a.reshape(2, 4, 4, order='K')

    @testing.for_orders('CFA')
    @testing.numpy_nlcpy_array_equal()
    def test_external_reshape(self, xp, order):
        a = xp.zeros((8,), dtype=xp.float32)
        return xp.reshape(a, (1, 1, 1, 4, 1, 2), order=order)

    @testing.for_orders('CFA')
    @testing.numpy_nlcpy_array_equal()
    def test_ravel(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        a = a.transpose(2, 0, 1)
        return a.ravel(order)

    @testing.for_orders('CFA')
    @testing.numpy_nlcpy_array_equal()
    def test_ravel2(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.ravel(order)

    @testing.for_orders('CFA')
    @testing.numpy_nlcpy_array_equal()
    def test_ravel3(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        a = xp.array(a, order='f')
        return a.ravel(order)

    @testing.numpy_nlcpy_array_equal()
    def test_external_ravel(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        a = a.transpose(2, 0, 1)
        return xp.ravel(a)


@testing.parameterize(*testing.product({
    'order_init': ['C', 'F'],
    'order_reshape': ['C', 'F', 'A', 'c', 'f', 'a'],
    'shape_in_out': [((2, 3), (1, 6, 1)),  # (shape_init, shape_final)
                     ((6,), (2, 3)),
                     ((3, 3, 3), (9, 3))],
}))
class TestReshapeOrder(unittest.TestCase):

    @testing.with_requires('numpy>=1.12')
    def test_reshape_contiguity(self):
        shape_init, shape_final = self.shape_in_out

        a_nlcpy = testing.shaped_arange(shape_init, xp=nlcpy)
        a_nlcpy = nlcpy.asarray(a_nlcpy, order=self.order_init)
        b_nlcpy = a_nlcpy.reshape(shape_final, order=self.order_reshape)

        a_numpy = testing.shaped_arange(shape_init, xp=numpy)
        a_numpy = numpy.asarray(a_numpy, order=self.order_init)
        b_numpy = a_numpy.reshape(shape_final, order=self.order_reshape)

        assert b_nlcpy.flags.f_contiguous == b_numpy.flags.f_contiguous
        assert b_nlcpy.flags.c_contiguous == b_numpy.flags.c_contiguous

        testing.assert_array_equal(b_nlcpy.strides, b_numpy.strides)
        testing.assert_array_equal(b_nlcpy, b_numpy)

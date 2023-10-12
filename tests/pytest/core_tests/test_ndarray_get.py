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
from nlcpy import testing
import numpy
from numpy import testing as np_testing


class TestArrayGet(unittest.TestCase):

    def check_get(self, f, order='C'):
        a_ve = f(nlcpy)
        a_cpu = a_ve.get(order=order)
        b_cpu = f(numpy)
        np_testing.assert_array_equal(a_cpu, b_cpu)
        if order == 'F' or (order == 'A' and a_ve.flags.f_contiguous):
            assert a_cpu.flags.f_contiguous
        else:
            assert a_cpu.flags.c_contiguous

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_contiguous_array(self, dtype, order):
        def contiguous_array(xp):
            return testing.shaped_arange((3,), xp, dtype, order)
        self.check_get(contiguous_array, order)

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_non_contiguous_array(self, dtype, order):
        def non_contiguous_array(xp):
            return testing.shaped_arange((3, 3), xp, dtype, order)[0::2, 0::2]
        self.check_get(non_contiguous_array, order)

    @testing.multi_ve(2)
    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_get_multive(self, dtype, order):
        with nlcpy.venode.VE(1):
            src = testing.shaped_arange((2, 3), nlcpy, dtype, order)
            src = nlcpy.asarray(src, order='F')
        with nlcpy.venode.VE(0):
            dst = src.get()
        expected = testing.shaped_arange((2, 3), numpy, dtype, order)
        np_testing.assert_array_equal(dst, expected)

    def test_order_mismatch(self):
        arr = testing.shaped_arange((3,), nlcpy, dtype='f8')
        with self.assertRaises(ValueError):
            arr.get(order='K')


class TestArrayGetWithOut(unittest.TestCase):

    def setUp(self):
        self._prev_ve = nlcpy.venode.VE(0)

    def tearDown(self):
        self._prev_ve.apply()

    def check_get(self, f, out):
        a_ve = f(nlcpy)
        a_cpu = a_ve.get(out=out)
        b_cpu = f(numpy)
        assert a_cpu is out
        np_testing.assert_array_equal(a_cpu, b_cpu)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_contiguous_array(self, dtype, order):
        def contiguous_array(xp):
            return testing.shaped_arange((3,), xp, dtype, order)
        out = numpy.empty((3,), dtype, order)
        self.check_get(contiguous_array, out)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_contiguous_array_cross(self, dtype, order):
        def contiguous_array(xp):
            return testing.shaped_arange((3,), xp, dtype, order)
        out_order = 'C' if order == 'F' else 'F'
        out = numpy.empty((3,), dtype, out_order)
        self.check_get(contiguous_array, out)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_contiguous_array_with_error(self, dtype, order):
        out = numpy.empty((3, 3), dtype)[0:2, 0:2]
        with self.assertRaises(RuntimeError):
            a_ve = testing.shaped_arange((3, 3), nlcpy, dtype, order)[0:2, 0:2]
            a_ve.get(out=out)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_non_contiguous_array(self, dtype, order):
        def non_contiguous_array(xp):
            return testing.shaped_arange((3, 3), xp, dtype, order)[0::2, 0::2]
        out = numpy.empty((2, 2), dtype, order)
        self.check_get(non_contiguous_array, out)

    def test_out_not_numpy_array(self):
        arr = testing.shaped_arange((3,), nlcpy, dtype='f8')
        out = nlcpy.empty((3,), dtype='f8')
        with self.assertRaises(TypeError):
            arr.get(out=out)

    def test_out_dtype_mismatch(self):
        arr = testing.shaped_arange((3,), nlcpy, dtype='f8')
        out = numpy.empty((3,), dtype='f4')
        with self.assertRaises(TypeError):
            arr.get(out=out)

    def test_out_shape_mismatch(self):
        arr = testing.shaped_arange((3,), nlcpy, dtype='f8')
        out = numpy.empty((5,), dtype='f8')
        with self.assertRaises(ValueError):
            arr.get(out=out)

    @testing.multi_ve(2)
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_get_multive(self, dtype, order):
        with nlcpy.venode.VE(1):
            src = testing.shaped_arange((2, 3), nlcpy, dtype, order)
            src = nlcpy.asarray(src, order='F')
        with nlcpy.venode.VE(0):
            dst = numpy.empty((2, 3), dtype, order)
            src.get(out=dst)
        expected = testing.shaped_arange((2, 3), numpy, dtype, order)
        np_testing.assert_array_equal(dst, expected)

    @testing.multi_ve(2)
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_get_multive_with_use(self, dtype, order):
        with nlcpy.venode.VE(0):
            src = testing.shaped_arange((10,), nlcpy, dtype, order)
        nlcpy.venode.VE(1).use()
        dst = src[::2].get()
        expected = testing.shaped_arange((10,), numpy, dtype, order)[::2]
        assert nlcpy.venode.VE() == nlcpy.venode.VE(1)
        np_testing.assert_array_equal(dst, expected)

    @testing.multi_ve(2)
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_get_multive_with_out_with_use(self, dtype, order):
        with nlcpy.venode.VE(0):
            src = testing.shaped_arange((10,), nlcpy, dtype, order)
        dst = numpy.empty(10, dtype, order=('F' if order == 'C' else 'C'))
        nlcpy.venode.VE(1).use()
        src.get(out=dst)
        expected = testing.shaped_arange((10,), numpy, dtype, order)
        assert nlcpy.venode.VE() == nlcpy.venode.VE(1)
        np_testing.assert_array_equal(dst, expected)

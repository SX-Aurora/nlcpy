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

import unittest

import nlcpy
from nlcpy import testing
import numpy


class TestArraySet(unittest.TestCase):

    def check_set(self, f, a_ve):
        a_cpu = f(numpy)
        a_ve.set(a_cpu)
        b_ve = f(nlcpy)
        testing.assert_array_equal(a_ve, b_ve)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_contiguous_array(self, dtype, order):
        def contiguous_array(xp):
            return testing.shaped_arange((3,), xp, dtype, order)
        out = nlcpy.empty((3,), dtype, order)
        self.check_set(contiguous_array, out)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_contiguous_array_cross(self, dtype, order):
        def contiguous_array(xp):
            return testing.shaped_arange((3,), xp, dtype, order)
        out_order = 'C' if order == 'F' else 'F'
        out = nlcpy.empty((3,), dtype, out_order)
        self.check_set(contiguous_array, out)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_contiguous_array_with_error(self, dtype, order):
        a_cpu = numpy.empty((3, 3), dtype)[0:2, 0:2]
        with self.assertRaises(RuntimeError):
            a_ve = testing.shaped_arange((3, 3), nlcpy, dtype, order)[0:2, 0:2]
            a_ve.set(a_cpu)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_non_contiguous_array(self, dtype, order):
        def non_contiguous_array(xp):
            return testing.shaped_arange((3, 3), xp, dtype, order)[0::2, 0::2]
        out = nlcpy.empty((2, 2), dtype, order)
        self.check_set(non_contiguous_array, out)

    @testing.multi_ve(2)
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_set_multive(self, dtype, order):
        with nlcpy.venode.VE(1):
            dst = nlcpy.empty((2, 3), dtype, order)
        with nlcpy.venode.VE(0):
            src = testing.shaped_arange((2, 3), nlcpy, dtype, order).get()
        dst.set(src)
        expected = testing.shaped_arange((2, 3), nlcpy, dtype, order)
        testing.assert_array_equal(dst, expected)

    def test_not_numpy_array(self):
        dst = nlcpy.empty((3,), dtype='f8')
        src = testing.shaped_arange((3,), nlcpy, dtype='f8')
        with self.assertRaises(TypeError):
            dst.set(src)

    def test_dtype_mismatch(self):
        dst = nlcpy.empty((3,), dtype='f8')
        src = testing.shaped_arange((3,), numpy, dtype='f4')
        with self.assertRaises(TypeError):
            dst.set(src)

    def test_shape_mismatch(self):
        dst = nlcpy.empty((3,), dtype='f8')
        src = testing.shaped_arange((5,), numpy, dtype='f8')
        with self.assertRaises(ValueError):
            dst.set(src)

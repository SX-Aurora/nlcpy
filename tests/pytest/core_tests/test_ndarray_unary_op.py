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

import operator
import unittest

import numpy

import nlcpy
from nlcpy import testing


class TestArrayBoolOp(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_bool_empty(self, dtype):
        with testing.assert_warns(DeprecationWarning):
            self.assertFalse(bool(nlcpy.array((), dtype=dtype)))

    def test_bool_scalar_bool(self):
        self.assertTrue(bool(nlcpy.array(True, dtype=numpy.bool_)))
        self.assertFalse(bool(nlcpy.array(False, dtype=numpy.bool_)))

    @testing.for_all_dtypes()
    def test_bool_scalar(self, dtype):
        self.assertTrue(bool(nlcpy.array(1, dtype=dtype)))
        self.assertFalse(bool(nlcpy.array(0, dtype=dtype)))

    def test_bool_one_element_bool(self):
        self.assertTrue(bool(nlcpy.array([True], dtype=numpy.bool_)))
        self.assertFalse(bool(nlcpy.array([False], dtype=numpy.bool_)))

    @testing.for_all_dtypes()
    def test_bool_one_element(self, dtype):
        self.assertTrue(bool(nlcpy.array([1], dtype=dtype)))
        self.assertFalse(bool(nlcpy.array([0], dtype=dtype)))

    @testing.for_all_dtypes()
    def test_bool_two_elements(self, dtype):
        with self.assertRaises(ValueError):
            bool(nlcpy.array([1, 2], dtype=dtype))


class TestArrayUnaryOp(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_allclose()
    def check_array_op(self, op, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return op(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def check_array_op_full(self, op, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return op(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_allclose()
    def test_neg_array(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return operator.neg(a)

    def test_pos_array(self):
        self.check_array_op(operator.pos)

    @testing.with_requires('numpy<1.16')
    def test_pos_array_full(self):
        self.check_array_op_full(operator.pos)

    def test_abs_array(self):
        self.check_array_op_full(operator.abs)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_allclose()
    def check_zerodim_op(self, op, xp, dtype):
        a = xp.array(-2).astype(dtype)
        return op(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def check_zerodim_op_full(self, op, xp, dtype):
        a = xp.array(-2).astype(dtype)
        return op(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_allclose()
    def test_neg_zerodim(self, xp, dtype):
        a = xp.array(-2).astype(dtype)
        return operator.neg(a)

    def test_pos_zerodim(self):
        self.check_zerodim_op(operator.pos)

    def test_abs_zerodim(self):
        self.check_zerodim_op_full(operator.abs)

    @testing.with_requires('numpy<1.16')
    def test_abs_zerodim_full(self):
        self.check_zerodim_op_full(operator.abs)


class TestArrayIntUnaryOp(unittest.TestCase):

    @testing.for_int_dtypes()
    @testing.numpy_nlcpy_allclose()
    def check_array_op(self, op, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return op(a)

    def test_invert_array(self):
        self.check_array_op(operator.invert)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(accept_error=TypeError)
    def check_zerodim_op(self, op, xp, dtype):
        a = xp.array(-2).astype(dtype)
        return op(a)

    def test_invert_zerodim(self):
        self.check_zerodim_op(operator.invert)

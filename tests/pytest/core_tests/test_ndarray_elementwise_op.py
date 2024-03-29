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


class TestArrayElementwiseOp(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose(rtol=1e-6, accept_error=TypeError)
    def check_array_scalar_op(self, op, xp, x_type, y_type, swap=False,
                              no_bool=False, no_complex=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        if swap:
            return op(y_type(3), a)
        else:
            if y_type is numpy.bool_:
                return op(a, bool(3))
            else:
                return op(a, y_type(3))

    def test_add_scalar(self):
        self.check_array_scalar_op(operator.add)

    def test_radd_scalar(self):
        self.check_array_scalar_op(operator.add, swap=True)

    @testing.with_requires('numpy>=1.10')
    def test_iadd_scalar(self):
        self.check_array_scalar_op(operator.iadd)

    def test_sub_scalar(self):
        self.check_array_scalar_op(operator.sub, no_bool=True)

    def test_rsub_scalar(self):
        self.check_array_scalar_op(operator.sub, swap=True, no_bool=True)

    @testing.with_requires('numpy>=1.10')
    def test_isub_scalar(self):
        self.check_array_scalar_op(operator.isub, no_bool=True)

    def test_mul_scalar(self):
        self.check_array_scalar_op(operator.mul)

    def test_rmul_scalar(self):
        self.check_array_scalar_op(operator.mul, swap=True)

    @testing.with_requires('numpy>=1.10')
    def test_imul_scalar(self):
        self.check_array_scalar_op(operator.imul)

    def test_truediv_scalar(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_scalar_op(operator.truediv)

    def test_rtruediv_scalar(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_scalar_op(operator.truediv, swap=True)

    @testing.with_requires('numpy>=1.10')
    def test_itruediv_scalar(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_scalar_op(operator.itruediv)

    def test_floordiv_scalar(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_scalar_op(operator.floordiv, no_complex=True)

    def test_rfloordiv_scalar(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_scalar_op(operator.floordiv, swap=True,
                                       no_complex=True)

    @testing.with_requires('numpy>=1.10')
    def test_ifloordiv_scalar(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_scalar_op(operator.ifloordiv, no_complex=True)

    def test_pow_scalar(self):
        self.check_array_scalar_op(operator.pow)

    def test_rpow_scalar(self):
        self.check_array_scalar_op(operator.pow, swap=True)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose(atol=1.0, accept_error=TypeError)
    def check_ipow_scalar(self, xp, x_type, y_type):
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        if y_type is numpy.bool_:
            return operator.ipow(a, bool(3))
        else:
            return operator.ipow(a, y_type(3))

    @testing.with_requires('numpy>=1.10')
    def test_ipow_scalar(self):
        self.check_ipow_scalar()

    def test_lt_scalar(self):
        self.check_array_scalar_op(operator.lt, no_complex=False)

    def test_le_scalar(self):
        self.check_array_scalar_op(operator.le, no_complex=False)

    def test_gt_scalar(self):
        self.check_array_scalar_op(operator.gt, no_complex=False)

    def test_ge_scalar(self):
        self.check_array_scalar_op(operator.ge, no_complex=False)

    def test_eq_scalar(self):
        self.check_array_scalar_op(operator.eq)

    def test_ne_scalar(self):
        self.check_array_scalar_op(operator.ne)

    @testing.for_orders('CF', name='order_in')
    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose(accept_error=TypeError)
    def check_array_array_op(self, op, xp, x_type, y_type, order_in,
                             no_bool=False, no_complex=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type, order=order_in)
        b = xp.array([[6, 5, 4], [3, 2, 1]], y_type, order=order_in)
        return op(a, b)

    def test_add_array(self):
        self.check_array_array_op(operator.add)

    @testing.with_requires('numpy>=1.10')
    def test_iadd_array(self):
        self.check_array_array_op(operator.iadd)

    def test_sub_array(self):
        self.check_array_array_op(operator.sub, no_bool=True)

    @testing.with_requires('numpy>=1.10')
    def test_isub_array(self):
        self.check_array_array_op(operator.isub, no_bool=True)

    def test_mul_array(self):
        self.check_array_array_op(operator.mul)

    @testing.with_requires('numpy>=1.10')
    def test_imul_array(self):
        self.check_array_array_op(operator.imul)

    def test_truediv_array(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_array_op(operator.truediv)

    @testing.with_requires('numpy>=1.10')
    def test_itruediv_array(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_array_op(operator.itruediv)

    def test_floordiv_array(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_array_op(operator.floordiv, no_complex=True)

    @testing.with_requires('numpy>=1.10')
    def test_ifloordiv_array(self):
        if '1.16.1' <= numpy.lib.NumpyVersion(numpy.__version__) < '1.18.0':
            self.skipTest("NumPy Issue #12927")
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_array_op(operator.ifloordiv, no_complex=True)

    def test_pow_array(self):
        self.check_array_array_op(operator.pow)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose(atol=1.0, accept_error=TypeError)
    def check_ipow_array(self, xp, x_type, y_type):
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        b = xp.array([[6, 5, 4], [3, 2, 1]], y_type)
        return operator.ipow(a, b)

    @testing.with_requires('numpy>=1.10')
    def test_ipow_array(self):
        self.check_ipow_array()

    def test_lt_array(self):
        self.check_array_array_op(operator.lt, no_complex=True)

    def test_le_array(self):
        self.check_array_array_op(operator.le, no_complex=True)

    def test_gt_array(self):
        self.check_array_array_op(operator.gt, no_complex=True)

    def test_ge_array(self):
        self.check_array_array_op(operator.ge, no_complex=True)

    def test_eq_array(self):
        self.check_array_array_op(operator.eq)

    def test_ne_array(self):
        self.check_array_array_op(operator.ne)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose(accept_error=TypeError)
    def check_array_broadcasted_op(self, op, xp, x_type, y_type,
                                   no_bool=False, no_complex=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        b = xp.array([[1], [2]], y_type)
        return op(a, b)

    def test_broadcasted_add(self):
        self.check_array_broadcasted_op(operator.add)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_iadd(self):
        self.check_array_broadcasted_op(operator.iadd)

    def test_broadcasted_sub(self):
        self.check_array_broadcasted_op(operator.sub, no_bool=True)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_isub(self):
        self.check_array_broadcasted_op(operator.isub, no_bool=True)

    def test_broadcasted_mul(self):
        self.check_array_broadcasted_op(operator.mul)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_imul(self):
        self.check_array_broadcasted_op(operator.imul)

    def test_broadcasted_truediv(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_broadcasted_op(operator.truediv)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_itruediv(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_broadcasted_op(operator.itruediv)

    def test_broadcasted_floordiv(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_broadcasted_op(operator.floordiv, no_complex=True)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_ifloordiv(self):
        if '1.16.1' <= numpy.lib.NumpyVersion(numpy.__version__) < '1.18.0':
            self.skipTest("NumPy Issue #12927")
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_broadcasted_op(operator.ifloordiv,
                                            no_complex=True)

    def test_broadcasted_pow(self):
        self.check_array_broadcasted_op(operator.pow)

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose(atol=1.0, accept_error=TypeError)
    def check_broadcasted_ipow(self, xp, x_type, y_type):
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        b = xp.array([[1], [2]], y_type)
        return operator.ipow(a, b)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_ipow(self):
        self.check_broadcasted_ipow()

    def test_broadcasted_lt(self):
        self.check_array_broadcasted_op(operator.lt, no_complex=True)

    def test_broadcasted_le(self):
        self.check_array_broadcasted_op(operator.le, no_complex=True)

    def test_broadcasted_gt(self):
        self.check_array_broadcasted_op(operator.gt, no_complex=True)

    def test_broadcasted_ge(self):
        self.check_array_broadcasted_op(operator.ge, no_complex=True)

    def test_broadcasted_eq(self):
        self.check_array_broadcasted_op(operator.eq)

    def test_broadcasted_ne(self):
        self.check_array_broadcasted_op(operator.ne)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose()
    def check_array_doubly_broadcasted_op(self, op, xp, x_type, y_type,
                                          no_bool=False, no_complex=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        a = xp.array([[[1, 2, 3]], [[4, 5, 6]]], x_type)
        b = xp.array([[1], [2], [3]], y_type)
        return op(a, b)

    def test_doubly_broadcasted_add(self):
        self.check_array_doubly_broadcasted_op(operator.add)

    def test_doubly_broadcasted_sub(self):
        self.check_array_doubly_broadcasted_op(operator.sub, no_bool=True)

    def test_doubly_broadcasted_mul(self):
        self.check_array_doubly_broadcasted_op(operator.mul)

    def test_doubly_broadcasted_truediv(self):
        with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
            self.check_array_doubly_broadcasted_op(operator.truediv)

    def test_doubly_broadcasted_floordiv(self):
        with testing.numpy_nlcpy_errstate(divide='ignore'):
            self.check_array_doubly_broadcasted_op(operator.floordiv,
                                                   no_complex=True)

    def test_doubly_broadcasted_pow(self):
        self.check_array_doubly_broadcasted_op(operator.pow)

    def test_doubly_broadcasted_lt(self):
        self.check_array_doubly_broadcasted_op(operator.lt, no_complex=True)

    def test_doubly_broadcasted_le(self):
        self.check_array_doubly_broadcasted_op(operator.le, no_complex=True)

    def test_doubly_broadcasted_gt(self):
        self.check_array_doubly_broadcasted_op(operator.gt, no_complex=True)

    def test_doubly_broadcasted_ge(self):
        self.check_array_doubly_broadcasted_op(operator.ge, no_complex=True)

    def test_doubly_broadcasted_eq(self):
        self.check_array_doubly_broadcasted_op(operator.eq)

    def test_doubly_broadcasted_ne(self):
        self.check_array_doubly_broadcasted_op(operator.ne)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose()
    def check_array_reversed_op(self, op, xp, x_type, y_type, no_bool=False):
        if no_bool and x_type == numpy.bool_ and y_type == numpy.bool_:
            return xp.array(True)
        a = xp.array([1, 2, 3, 4, 5], x_type)
        b = xp.array([1, 2, 3, 4, 5], y_type)
        return op(a, b[::-1])

    def test_array_reversed_add(self):
        self.check_array_reversed_op(operator.add)

    def test_array_reversed_sub(self):
        self.check_array_reversed_op(operator.sub, no_bool=True)

    def test_array_reversed_mul(self):
        self.check_array_reversed_op(operator.mul)

    @testing.for_all_dtypes(no_bool=True)
    def check_typecast(self, val, dtype):
        operators = [
            operator.add, operator.sub, operator.mul, operator.truediv]

        for op in operators:
            with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
                a = op(val, (testing.shaped_arange((5,), numpy, dtype) - 2))
                b = op(val, (testing.shaped_arange((5,), nlcpy, dtype) - 2))
                b.get()
                self.assertEqual(a.dtype, b.dtype)

    def test_typecast_bool1(self):
        self.check_typecast(True)

    def test_typecast_bool2(self):
        self.check_typecast(False)

    def test_typecast_int1(self):
        self.check_typecast(0)

    def test_typecast_int2(self):
        self.check_typecast(-127)

    def test_typecast_int3(self):
        self.check_typecast(255)

    def test_typecast_int4(self):
        self.check_typecast(-32768)

    def test_typecast_int5(self):
        self.check_typecast(65535)

    def test_typecast_int6(self):
        self.check_typecast(-2147483648)

    def test_typecast_int7(self):
        self.check_typecast(4294967295)

    def test_typecast_float1(self):
        self.check_typecast(0.0)

    def test_typecast_float2(self):
        self.check_typecast(100000.0)


class TestArrayIntElementwiseOp(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose(accept_error=TypeError)
    def check_array_scalar_op(self, op, xp, x_type, y_type, swap=False):
        a = xp.array([[0, 1, 2], [1, 0, 2]], dtype=x_type)
        if swap:
            return op(y_type(2), a)
        else:
            return op(a, y_type(2))

    def test_lshift_scalar(self):
        self.check_array_scalar_op(operator.lshift)

    def test_rlshift_scalar(self):
        self.check_array_scalar_op(operator.lshift, swap=True)

    def test_rshift_scalar(self):
        self.check_array_scalar_op(operator.rshift)

    def test_rrshift_scalar(self):
        self.check_array_scalar_op(operator.rshift, swap=True)

    def test_and_scalar(self):
        self.check_array_scalar_op(operator.and_)

    def test_rand_scalar(self):
        self.check_array_scalar_op(operator.and_, swap=True)

    def test_or_scalar(self):
        self.check_array_scalar_op(operator.or_)

    def test_ror_scalar(self):
        self.check_array_scalar_op(operator.or_, swap=True)

    def test_xor_scalar(self):
        self.check_array_scalar_op(operator.xor)

    def test_rxor_scalar(self):
        self.check_array_scalar_op(operator.xor, swap=True)

    def test_mod_scalar(self):
        with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
            self.check_array_scalar_op(operator.mod)

    def test_rmod_scalar(self):
        with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
            self.check_array_scalar_op(operator.mod, swap=True)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose(accept_error=TypeError)
    def check_array_scalarzero_op(self, op, xp, x_type, y_type, swap=False):
        a = xp.array([[0, 1, 2], [1, 0, 2]], dtype=x_type)
        if swap:
            return op(y_type(0), a)
        else:
            return op(a, y_type(0))

    def test_lshift_scalarzero(self):
        self.check_array_scalarzero_op(operator.lshift)

    def test_rlshift_scalarzero(self):
        self.check_array_scalarzero_op(operator.lshift, swap=True)

    def test_rshift_scalarzero(self):
        self.check_array_scalarzero_op(operator.rshift)

    def test_rrshift_scalarzero(self):
        self.check_array_scalarzero_op(operator.rshift, swap=True)

    def test_and_scalarzero(self):
        self.check_array_scalarzero_op(operator.and_)

    def test_rand_scalarzero(self):
        self.check_array_scalarzero_op(operator.and_, swap=True)

    def test_or_scalarzero(self):
        self.check_array_scalarzero_op(operator.or_)

    def test_ror_scalarzero(self):
        self.check_array_scalarzero_op(operator.or_, swap=True)

    def test_xor_scalarzero(self):
        self.check_array_scalarzero_op(operator.xor)

    def test_rxor_scalarzero(self):
        self.check_array_scalarzero_op(operator.xor, swap=True)

    def test_mod_scalarzero(self):
        with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
            self.check_array_scalarzero_op(operator.mod)

    def test_rmod_scalarzero(self):
        with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
            self.check_array_scalarzero_op(operator.mod, swap=True)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose(accept_error=TypeError)
    def check_array_array_op(self, op, xp, x_type, y_type):
        a = xp.array([[0, 1, 2], [1, 0, 2]], dtype=x_type)
        b = xp.array([[0, 0, 1], [0, 1, 2]], dtype=y_type)
        return op(a, b)

    def test_lshift_array(self):
        self.check_array_array_op(operator.lshift)

    @testing.with_requires('numpy>=1.10')
    def test_ilshift_array(self):
        self.check_array_array_op(operator.ilshift)

    def test_rshift_array(self):
        self.check_array_array_op(operator.rshift)

    @testing.with_requires('numpy>=1.10')
    def test_irshift_array(self):
        self.check_array_array_op(operator.irshift)

    def test_and_array(self):
        self.check_array_array_op(operator.and_)

    @testing.with_requires('numpy>=1.10')
    def test_iand_array(self):
        self.check_array_array_op(operator.iand)

    def test_or_array(self):
        self.check_array_array_op(operator.or_)

    @testing.with_requires('numpy>=1.10')
    def test_ior_array(self):
        self.check_array_array_op(operator.ior)

    def test_xor_array(self):
        self.check_array_array_op(operator.xor)

    @testing.with_requires('numpy>=1.10')
    def test_ixor_array(self):
        self.check_array_array_op(operator.ixor)

    def test_mod_array(self):
        with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
            self.check_array_array_op(operator.mod)

    @testing.with_requires('numpy>=1.10')
    def test_imod_array(self):
        with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
            self.check_array_array_op(operator.imod)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose(accept_error=TypeError)
    def check_array_broadcasted_op(self, op, xp, x_type, y_type):
        a = xp.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=x_type)
        b = xp.array([[0, 0, 1]], dtype=y_type)
        return op(a, b)

    def test_broadcasted_lshift(self):
        self.check_array_broadcasted_op(operator.lshift)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_ilshift(self):
        self.check_array_broadcasted_op(operator.ilshift)

    def test_broadcasted_rshift(self):
        self.check_array_broadcasted_op(operator.rshift)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_irshift(self):
        self.check_array_broadcasted_op(operator.irshift)

    def test_broadcasted_and(self):
        self.check_array_broadcasted_op(operator.and_)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_iand(self):
        self.check_array_broadcasted_op(operator.iand)

    def test_broadcasted_or(self):
        self.check_array_broadcasted_op(operator.or_)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_ior(self):
        self.check_array_broadcasted_op(operator.ior)

    def test_broadcasted_xor(self):
        self.check_array_broadcasted_op(operator.xor)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_ixor(self):
        self.check_array_broadcasted_op(operator.ixor)

    def test_broadcasted_mod(self):
        with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
            self.check_array_broadcasted_op(operator.mod)

    @testing.with_requires('numpy>=1.10')
    def test_broadcasted_imod(self):
        with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
            self.check_array_broadcasted_op(operator.imod)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose(accept_error=TypeError)
    def check_array_doubly_broadcasted_op(self, op, xp, x_type, y_type):
        a = xp.array([[[0, 1, 2]], [[1, 0, 2]]], dtype=x_type)
        b = xp.array([[0], [0], [1]], dtype=y_type)
        return op(a, b)

    def test_doubly_broadcasted_lshift(self):
        self.check_array_doubly_broadcasted_op(operator.lshift)

    def test_doubly_broadcasted_rshift(self):
        self.check_array_doubly_broadcasted_op(operator.rshift)

    def test_doubly_broadcasted_and(self):
        self.check_array_doubly_broadcasted_op(operator.and_)

    def test_doubly_broadcasted_or(self):
        self.check_array_doubly_broadcasted_op(operator.or_)

    def test_doubly_broadcasted_xor(self):
        self.check_array_doubly_broadcasted_op(operator.xor)

    def test_doubly_broadcasted_mod(self):
        with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
            self.check_array_doubly_broadcasted_op(operator.mod)

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

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'], full=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-6, accept_error=TypeError)
    def check_array_scalar_op(self, op, xp, x_type, y_type, swap=False,
                              no_bool=False, no_complex=False, nomask=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        data = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        mask = xp.ma.nomask if nomask else xp.array([[0, 1, 0], [0, 0, 1]])
        a = xp.ma.array(data, mask=mask)
        if swap:
            ret = op(y_type(3), a)
        else:
            ret = op(a, y_type(3))
        return ret

    def test_add_scalar(self):
        self.check_array_scalar_op(operator.add)

    def test_sub_scalar(self):
        self.check_array_scalar_op(operator.sub, no_bool=True)

    def test_mul_scalar(self):
        self.check_array_scalar_op(operator.mul)

    def test_truediv_scalar(self):
        with testing.NumpyError(divide='ignore'):
            self.check_array_scalar_op(operator.truediv)

    def test_radd_scalar(self):
        self.check_array_scalar_op(operator.add, swap=True)

    def test_rsub_scalar(self):
        self.check_array_scalar_op(operator.sub, no_bool=True, swap=True)

    def test_rmul_scalar(self):
        self.check_array_scalar_op(operator.mul, swap=True)

    def test_rtruediv_scalar(self):
        with testing.NumpyError(divide='ignore'):
            self.check_array_scalar_op(operator.truediv, swap=True)

    def test_iadd_scalar(self):
        self.check_array_scalar_op(operator.iadd)

    def test_iadd_scalar_nomask(self):
        self.check_array_scalar_op(operator.iadd, nomask=True)

    def test_isub_scalar(self):
        self.check_array_scalar_op(operator.isub, no_bool=True)

    def test_isub_scalar_nomask(self):
        self.check_array_scalar_op(operator.isub, no_bool=True, nomask=True)

    def test_imul_scalar(self):
        self.check_array_scalar_op(operator.imul)

    def test_imul_scalar_nomask(self):
        self.check_array_scalar_op(operator.imul, nomask=True)

    def test_itruediv_scalar(self):
        with testing.NumpyError(divide='ignore'):
            self.check_array_scalar_op(operator.itruediv)

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
        mask = xp.array([[0, 1, 0], [1, 0, 1]])
        a = xp.ma.array(a, mask=mask)
        b = xp.array([[1], [2]], y_type)
        mask = xp.array([[0], [1]])
        b = xp.ma.array(b, mask=mask)
        res = op(a, b)
        return res

    def test_broadcasted_add(self):
        self.check_array_broadcasted_op(operator.add)

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
        a = xp.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], x_type)
        mask = xp.array([[0, 1, 0, 0], [1, 0, 1, 0]])
        a = xp.ma.array(a, mask=mask)
        b = xp.array([[1], [2], [3]], y_type)
        mask = xp.array([[0], [1], [0]])
        b = xp.ma.array(b, mask=mask)
        res = op(a, b)
        return res

    def test_doubly_broadcasted_add(self):
        self.check_array_doubly_broadcasted_op(operator.add)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_nlcpy_allclose()
    def check_array_reversed_op(self, op, xp, x_type, y_type, no_bool=False):
        if no_bool and x_type == numpy.bool_ and y_type == numpy.bool_:
            return xp.array(True)
        data = xp.array([1, 2, 3, 4, 5], dtype=x_type)
        mask = xp.array([0, 1, 0, 1, 0])
        a = xp.ma.array(data, mask=mask)

        data = xp.array([1, 2, 3, 4, 5], dtype=y_type)
        mask = xp.array([0, 0, 0, 1, 1])
        b = xp.ma.array(data, mask=mask)
        res = op(a, b[::-1])
        return res

    def test_array_reversed_add(self):
        self.check_array_reversed_op(operator.add)

    def test_array_reversed_sub(self):
        self.check_array_reversed_op(operator.sub, no_bool=True)

    def test_array_reversed_mul(self):
        self.check_array_reversed_op(operator.mul)

    def test_array_reversed_div(self):
        self.check_array_reversed_op(operator.truediv)

    @testing.for_all_dtypes(no_bool=True)
    def check_typecast(self, val, dtype):
        operators = [
            operator.add, operator.sub, operator.mul, operator.truediv]

        mask = [0, 1, 0, 1, 0]
        nval = testing.shaped_arange((5, ), numpy, dtype) - 2
        vval = testing.shaped_arange((5, ), nlcpy, dtype) - 2
        nval = numpy.ma.array(nval, mask=mask)
        vval = nlcpy.ma.array(vval, mask=mask)
        for op in operators:
            with testing.NumpyError(divide='ignore', invalid='ignore'):
                a = op(val, nval)
            b = op(val, vval)
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

    @testing.numpy_nlcpy_allclose()
    def test_divide_inf_mask(self, xp):
        a = xp.ma.array([3, 1, -1, 0])
        b = xp.ma.array([1, 0, 0, 0])
        return xp.ma.divide(a, b)

    @testing.numpy_nlcpy_allclose()
    def test_divide_domain_mask(self, xp):
        x = 3 / numpy.finfo(float).tiny
        a = xp.ma.array([3, 1, -1, 0, x])
        b = xp.ma.array([1, 0, 0, 0, 2])
        return xp.ma.divide(a, b)

    @testing.numpy_nlcpy_allclose()
    def test_true_divide_inf_mask(self, xp):
        a = xp.ma.array([3, 1, -1, 0])
        b = xp.ma.array([1, 0, 0, 0])
        return xp.ma.true_divide(a, b)

    @testing.numpy_nlcpy_allclose()
    def test_true_divide_domain_mask(self, xp):
        x = 3 / numpy.finfo(float).tiny
        a = xp.ma.array([3, 1, -1, 0, x])
        b = xp.ma.array([1, 0, 0, 0, 2])
        return xp.ma.true_divide(a, b)

    @testing.numpy_nlcpy_array_equal()
    def test_binary_op_both_nomask(self, xp):
        a = xp.ma.array(testing.shaped_random([2, 3, 4], xp))
        b = xp.ma.array(testing.shaped_random([2, 3, 4], xp))
        return a + b

    @testing.numpy_nlcpy_array_equal()
    def test_binary_op_both_0d(self, xp):
        a = xp.ma.array(1, xp)
        b = xp.ma.array(2, xp)
        return a + b

    @testing.numpy_nlcpy_allclose()
    def test_domained_binary_op_both_0d(self, xp):
        a = xp.ma.array(1, xp)
        b = xp.ma.array(2, xp)
        return a / b

    @testing.for_orders('CF', name='order_in')
    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'], full=True)
    @testing.numpy_nlcpy_allclose(accept_error=TypeError)
    def check_ma_ma_op(self, op_name, xp, x_type, y_type, order_in,
                       nomask=False, no_bool=False, no_complex=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        data1 = xp.array([[1, 2, 3], [4, 5, 6]], dtype=x_type, order=order_in)
        if nomask:
            a = xp.ma.array(data1, order=order_in)
        else:
            mask1 = xp.array([[1, 0, 0], [0, 1, 0]])
            a = xp.ma.array(data1, mask=mask1, order=order_in)
        data2 = xp.array([[6, 5, 4], [3, 2, 1]], dtype=y_type, order=order_in)
        mask2 = xp.array([[0, 1, 0], [0, 1, 0]])
        b = xp.ma.array(data2, mask=mask2, order=order_in)
        op = getattr(a, op_name)
        res = op(b)
        return res

    def test_add_ma(self):
        self.check_ma_ma_op('__add__')

    def test_sub_ma(self):
        self.check_ma_ma_op('__sub__')

    def test_mul_ma(self):
        self.check_ma_ma_op('__mul__')

    def test_div_ma(self):
        with testing.NumpyError(divide='ignore'):
            self.check_ma_ma_op('__div__')

    def test_truediv_ma(self):
        with testing.NumpyError(divide='ignore'):
            self.check_ma_ma_op('__truediv__')

    def test_iadd_ma(self):
        self.check_ma_ma_op('__iadd__')

    def test_iadd_ma_nomask(self):
        self.check_ma_ma_op('__iadd__', nomask=True)

    def test_isub_ma(self):
        self.check_ma_ma_op('__isub__')

    def test_isub_ma_nomask(self):
        self.check_ma_ma_op('__isub__', nomask=True)

    def test_imul_ma(self):
        self.check_ma_ma_op('__imul__')

    def test_imul_ma_nomask(self):
        self.check_ma_ma_op('__imul__', nomask=True)

    def test_itruediv_ma(self):
        with testing.NumpyError(divide='ignore'):
            self.check_ma_ma_op('__itruediv__')

    @testing.for_orders('CF', name='order_in')
    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'], full=True)
    @testing.numpy_nlcpy_allclose(accept_error=TypeError)
    def check_ma_ndarray_op(self, op_name, xp, x_type, y_type, order_in,
                            swap=False, no_bool=False, no_complex=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        data = xp.array([[1, 2, 3], [4, 5, 6]], dtype=x_type, order=order_in)
        mask = xp.array([[1, 0, 0], [0, 1, 0]])
        a = xp.ma.array(data, mask=mask, order=order_in)
        b = xp.array([[6, 5, 4], [3, 2, 1]], dtype=y_type, order=order_in)
        op = getattr(a, op_name)
        if swap:
            res = op(a)
        else:
            res = op(b)
        return res

    def test_add_array(self):
        self.check_ma_ndarray_op('__add__')

    def test_radd_array(self):
        self.check_ma_ndarray_op('__add__', swap=True)

    def test_sub_array(self):
        self.check_ma_ndarray_op('__sub__')

    def test_rsub_array(self):
        self.check_ma_ndarray_op('__sub__', swap=True)

    def test_mul_array(self):
        self.check_ma_ndarray_op('__mul__')

    def test_rmul_array(self):
        self.check_ma_ndarray_op('__mul__', swap=True)

    def test_div_array(self):
        with testing.NumpyError(divide='ignore'):
            self.check_ma_ndarray_op('__div__')

    def test_truediv_array(self):
        with testing.NumpyError(divide='ignore'):
            self.check_ma_ndarray_op('__truediv__')

    def test_rtruediv_array(self):
        with testing.NumpyError(divide='ignore'):
            self.check_ma_ndarray_op('__truediv__', swap=True)

    def test_iadd_array(self):
        self.check_ma_ndarray_op('__iadd__')

    def test_isub_array(self):
        self.check_ma_ndarray_op('__isub__')

    def test_imul_array(self):
        self.check_ma_ndarray_op('__imul__')

    @testing.numpy_nlcpy_raises()
    def test_idiv(self, xp):
        a = xp.ma.array([1, 2, 3])
        b = xp.ma.array([2, 3, 4])
        getattr(a, '__idiv__')(b)

    def test_itruediv_array(self):
        with testing.NumpyError(divide='ignore'):
            self.check_ma_ndarray_op('__itruediv__')

    @testing.numpy_nlcpy_raises()
    def test_itruediv_domain_mask(self, xp):
        a = xp.ma.array([1, 2, 3])
        b = xp.ma.array([0, 1, 2])
        getattr(a, '__itruediv__')(b)

    @testing.numpy_nlcpy_array_equal()
    def test_masked_binary_op_scalar_mask(self, xp):
        a = xp.ma.array(1)
        b = xp.ma.array(2)
        return a + b

    @testing.numpy_nlcpy_allclose()
    def test_domained_binary_op_scalar_mask(self, xp):
        a = xp.ma.array(1)
        b = xp.ma.array(2)
        return a / b

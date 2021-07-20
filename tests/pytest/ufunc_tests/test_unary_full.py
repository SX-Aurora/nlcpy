#
# * The source code in this file is developed independently by NEC Corporation.
#
# # NLCPy License #
#
#     Copyright (c) 2020-2021 NEC Corporation
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

import numpy
import unittest
import pytest

import nlcpy
from nlcpy import testing

float_types = [numpy.float32, numpy.float64]
complex_types = [numpy.complex64, numpy.complex128]
signed_int_types = [numpy.int32, numpy.int64]
unsigned_int_types = [numpy.uint32, numpy.uint64]
int_types = signed_int_types + unsigned_int_types
no_bool_types = float_types + int_types + complex_types
no_bool_no_uint_types = float_types + signed_int_types + complex_types
all_types = [numpy.bool] + float_types + int_types + complex_types
negative_types = (
    [numpy.bool] + float_types + signed_int_types + complex_types)
negative_no_complex_types = [numpy.bool] + float_types + signed_int_types
no_complex_types = [numpy.bool] + float_types + int_types
no_bool_no_complex_types = float_types + int_types

float16_op_set = [
    'sin',
    'cos',
    'tan',
    'arcsin',
    'arccos',
    'arctan',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    'arccosh',
    'arctanh',
    'deg2rad',
    'rad2deg',
    'radians',
    'degrees',
    'spacing',
]

minval = -100
maxval = 100

shapes = [(4, ),
          (4, 2),
          (4, 2, 3),
          (4, 2, 3, 5), ]


orders = ['C', 'F']
order_op = ['C', 'F', 'A', 'K', None]

ops = [
    'negative',
    'positive',
    'absolute',
    'fabs',
    'sign',
    'conj',
    'conjugate',
    'exp',
    'log',
    'sqrt',
    'square',
    'cbrt',
    'reciprocal',  # XXX: skip 0 input
    'sin',
    'cos',
    'tan',
    'arcsin',
    'arccos',
    'arctan',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    'arccosh',
    # 'arctanh', #XXX: skip test until compiler bug fixed
    'deg2rad',
    'rad2deg',
    'degrees',
    'radians',
    'invert',
    'logical_not',
    'isfinite',
    'isinf',
    'isnan',
    'signbit',
    'spacing',
    'floor',
    'ceil',
]


class TestUnaryCast(unittest.TestCase):

    ################################
    # *** testing parameter ***
    # input is array
    # out is False
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.for_orders(order_op, name='order_op')
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders,
        dtype_x=all_types,
        minval=minval, maxval=maxval,
        mode='array', is_out=False, is_where=False, is_dtype=False,
        seed=0
    )
    def test_unary_cast_array(self, xp, op, in1, order_op):
        in1 = xp.array(in1)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                func = getattr(xp, op)
                y = func(in1, order=order_op)
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar
    # out is False
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders,
        dtype_x=all_types,
        minval=minval, maxval=maxval,
        mode='scalar', is_out=False, is_where=False, is_dtype=False,
        seed=0
    )
    def test_unary_cast_scalar(self, xp, op, in1):
        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                func = getattr(xp, op)
                y = func(in1)
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array
    # out is False
    # where is False
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders,
        dtype_x=all_types, dtype_arg=all_types,
        minval=minval,
        maxval=maxval,
        mode='array', is_out=False, is_where=False, is_dtype=True,
        seed=0
    )
    def test_unary_cast_array_with_dtype(self, xp, op, in1, dtype):
        in1 = xp.array(in1)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                func = getattr(xp, op)
                y = func(in1, dtype=dtype)
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar
    # out is False
    # where is False
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders,
        dtype_x=all_types, dtype_arg=all_types,
        minval=minval, maxval=maxval,
        mode='scalar', is_out=False, is_where=False, is_dtype=True,
        seed=0
    )
    def test_unary_cast_scalar_with_dtype(self, xp, op, in1, dtype):
        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                func = getattr(xp, op)
                y = func(in1, dtype=dtype)
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array
    # out is True
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders, order_out=orders,
        dtype_x=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval,
        mode='array', is_out=True, is_where=False, is_dtype=False,
        seed=0
    )
    def test_unary_cast_array_with_out(self, xp, op, in1, out):
        in1 = xp.array(in1)
        out = xp.array(out)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                func = getattr(xp, op)
                y = func(in1, out=out)
                if xp is numpy and op in float16_op_set:
                    if in1.dtype == numpy.dtype('bool'):
                        y = func(in1, out=out, dtype='f4')
                        return y
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar
    # out is True
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders, order_out=orders,
        dtype_x=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval,
        mode='scalar', is_out=True, is_where=False, is_dtype=False,
        seed=0
    )
    def test_unary_cast_scalar_with_out(self, xp, op, in1, out):
        out = xp.array(out)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                func = getattr(xp, op)
                y = func(in1, out=out)
                if xp is numpy and op in float16_op_set:
                    if isinstance(in1, bool):
                        y = func(in1, out=out, dtype='f4')
                        return y
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array
    # out is True(for broadcast)
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders, order_out=orders,
        dtype_x=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval, mode='array',
        is_out=True, is_where=False, is_dtype=False, is_broadcast=True,
        seed=0
    )
    def test_unary_cast_array_with_out_broadcast(self, xp, op, in1, out):
        in1 = xp.array(in1)
        out = xp.array(out)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                func = getattr(xp, op)
                y = func(in1, out=out)
                if xp is numpy and op in float16_op_set:
                    if in1.dtype == numpy.dtype('bool'):
                        y = func(in1, out=out, dtype='f4')
                        return y
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array
    # out is True
    # where is False
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders, order_out=orders,
        dtype_x=all_types, dtype_out=all_types, dtype_arg=all_types,
        minval=minval, maxval=maxval,
        mode='array', is_out=True, is_where=False, is_dtype=True,
        seed=0
    )
    def test_unary_cast_array_with_out_with_dtype(self, xp, op, in1, out, dtype):
        in1 = xp.array(in1)
        out = xp.array(out)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                func = getattr(xp, op)
                y = func(in1, out=out, dtype=dtype)
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar
    # out is True
    # where is False
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders, order_out=orders,
        dtype_x=all_types, dtype_out=all_types, dtype_arg=all_types,
        minval=minval, maxval=maxval,
        mode='scalar', is_out=True, is_where=False, is_dtype=True,
        seed=0
    )
    def test_unary_cast_scalar_with_out_with_dtype(self, xp, op, in1, out, dtype):
        out = xp.array(out)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                func = getattr(xp, op)
                y = func(in1, out=out, dtype=dtype)
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array
    # out is True
    # where is True
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders, order_out=orders, order_where=orders,
        dtype_x=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval,
        mode='array', is_out=True, is_where=True, is_dtype=False,
        seed=0
    )
    def test_unary_cast_array_with_out_with_where(self, xp, op, in1, out, where):
        in1 = xp.array(in1)
        out = xp.array(out)
        where = xp.array(where)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                with numpy.warnings.catch_warnings():
                    numpy.warnings.simplefilter('ignore', numpy.ComplexWarning)
                    func = getattr(xp, op)
                    y = func(in1, out=out, where=where)
                    if xp is numpy and op in float16_op_set:
                        if in1.dtype == numpy.dtype('bool'):
                            y = func(in1, out=out, where=where, dtype='f4')
                            return y
                    if xp is nlcpy:
                        nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar
    # out is True
    # where is True
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders, order_out=orders,
        dtype_x=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval,
        mode='scalar', is_out=True, is_where=True, is_dtype=False,
        seed=0
    )
    def test_unary_cast_scalar_with_out_with_where(self, xp, op, in1, out, where):
        out = xp.array(out)
        where = xp.array(where)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                with numpy.warnings.catch_warnings():
                    numpy.warnings.simplefilter('ignore', numpy.ComplexWarning)
                    func = getattr(xp, op)
                    y = func(in1, out=out, where=where)
                    if xp is numpy and op in float16_op_set:
                        if isinstance(in1, bool):
                            y = func(in1, out=out, where=where, dtype='f4')
                            return y
                    if xp is nlcpy:
                        nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array
    # out is True(for broadcast)
    # where is True(for broadcast)
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders, order_out=orders,
        dtype_x=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval, mode='array',
        is_out=True, is_where=True, is_dtype=False, is_broadcast=True,
        seed=0
    )
    def test_unary_cast_array_with_out_with_where_broadcast(
            self, xp, op, in1, out, where):
        in1 = xp.array(in1)
        out = xp.array(out)
        where = xp.array(where)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                with numpy.warnings.catch_warnings():
                    numpy.warnings.simplefilter('ignore', numpy.ComplexWarning)
                    func = getattr(xp, op)
                    y = func(in1, out=out, where=where)
                    if xp is numpy and op in float16_op_set:
                        if in1.dtype == numpy.dtype('bool'):
                            y = func(in1, out=out, where=where, dtype='f4')
                            return y
                    if xp is nlcpy:
                        nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array
    # out is True
    # where is True
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders, order_out=orders, order_where=orders,
        dtype_x=all_types, dtype_out=all_types, dtype_arg=all_types,
        minval=minval, maxval=maxval,
        mode='array', is_out=True, is_where=True, is_dtype=True,
        seed=0
    )
    def test_unary_cast_array_with_out_with_where_with_dtype(
            self, xp, op, in1, out, where, dtype):
        in1 = xp.array(in1)
        out = xp.array(out)
        where = xp.array(where)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                with numpy.warnings.catch_warnings():
                    numpy.warnings.simplefilter('ignore', numpy.ComplexWarning)
                    func = getattr(xp, op)
                    y = func(in1, out=out, where=where, dtype=dtype)
                    if xp is nlcpy:
                        nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar
    # out is True
    # where is True
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders, order_out=orders, order_where=orders,
        dtype_x=all_types, dtype_out=all_types, dtype_arg=all_types,
        minval=minval, maxval=maxval,
        mode='scalar', is_out=True, is_where=True, is_dtype=True,
        seed=0
    )
    def test_unary_cast_scalar_with_out_with_where_with_dtype_broadcast(
            self, xp, op, in1, out, where, dtype):
        out = xp.array(out)
        where = xp.array(where)

        with testing.NumpyError(divide='ignore'):
            with testing.NlcpyError(divide='ignore', over='ignore', invalid='ignore'):
                with numpy.warnings.catch_warnings():
                    numpy.warnings.simplefilter('ignore', numpy.ComplexWarning)
                    func = getattr(xp, op)
                    y = func(in1, out=out, where=where, dtype=dtype)
                    if xp is nlcpy:
                        nlcpy.request.flush()
        return y

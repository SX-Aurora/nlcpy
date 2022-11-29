#
# * The source code in this file is developed independently by NEC Corporation.
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

import numpy
import unittest
import pytest
import gc
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

minval = -100
maxval = 100

shapes = [(4, ),
          (4, 2),
          (4, 2, 3), ]

orders = ['C', ]

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
    'reciprocal',
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


def adjust_input(xp, op, a):
    a = xp.array(a)
    if op in ('arcsin', 'arccos'):
        a /= maxval
    elif op == 'arccosh':
        a = a - minval + 1
    return a


class TestUnaryCast(unittest.TestCase):

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    ################################
    # *** testing parameter ***
    # input is array
    # out is False
    # where is False
    # dtype is False
    ################################
    @pytest.mark.fast_math
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders,
        dtype_x=all_types,
        minval=minval, maxval=maxval,
        mode='array', is_out=False, is_where=False, is_dtype=False,
    )
    def test_unary_cast_array(self, xp, op, in1):
        in1 = adjust_input(xp, op, in1)
        func = getattr(xp, op)
        y = func(in1)
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
    @pytest.mark.fast_math
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders,
        dtype_x=all_types,
        minval=minval, maxval=maxval,
        mode='scalar', is_out=False, is_where=False, is_dtype=False,
    )
    def test_unary_cast_scalar(self, xp, op, in1):
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
    @pytest.mark.fast_math
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders,
        dtype_x=all_types, dtype_arg=all_types,
        minval=minval,
        maxval=maxval,
        mode='array', is_out=False, is_where=False, is_dtype=True,
    )
    def test_unary_cast_array_with_dtype(self, xp, op, in1, dtype):
        in1 = adjust_input(xp, op, in1)
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
    @pytest.mark.fast_math
    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes,
        order_x=orders,
        dtype_x=all_types, dtype_arg=all_types,
        minval=minval, maxval=maxval,
        mode='scalar', is_out=False, is_where=False, is_dtype=True,
    )
    def test_unary_cast_scalar_with_dtype(self, xp, op, in1, dtype):
        func = getattr(xp, op)
        y = func(in1, dtype=dtype)
        if xp is nlcpy:
            nlcpy.request.flush()
        return y

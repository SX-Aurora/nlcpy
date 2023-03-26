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
from nlcpy.testing.types import all_types
from nlcpy.testing.types import float_types
from nlcpy.testing.types import f64_type


float16_op_set = [
    'logaddexp',
    'arctan2',
    'hypot',
    'nextafter',
]

minval = -100
maxval = 100

shapes = [((4, ), (4, )),
          ((4, 2), (4, 2)),
          ((4, 2, 3), (4, 2, 3)),
          ((4, 2, 3, 5), (4, 2, 3, 5))]

orders = ['C', 'F']
order_op = ['C', 'F', 'A', 'K', None]

ops = [
    'add',
    'subtract',
    'multiply',
    'divide',
    'logaddexp',
    'logaddexp2',
    'true_divide',
    'floor_divide',  # skip 0 divide case
    'power',
    'remainder',  # XXX skip 0 divide case(remove this if compiler bug fixed)
    'mod',  # XXX skip 0 divide case(remove this if compiler bug fixed)
    'fmod',
    'heaviside',
    'arctan2',
    'hypot',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'left_shift',
    'right_shift',
    'greater',
    'greater_equal',
    'less',
    'less_equal',
    'not_equal',
    'equal',
    'logical_and',
    'logical_or',
    'logical_xor',
    'minimum',
    'maximum',
    'fmax',
    'fmin',
    'copysign',
    'nextafter',
]

cnt = 0


@testing.with_requires('numpy>=1.19')
@testing.with_requires('numpy<1.20')
class TestBinaryCast(unittest.TestCase):

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    ################################
    # *** testing parameter ***
    # input is array_array
    # out is False
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.for_orders(order_op, name='order_op')
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders,
        dtype_x=all_types, dtype_y=all_types,
        minval=minval, maxval=maxval, mode='array_array',
        is_out=False, is_where=False, is_dtype=False, seed=0
    )
    def test_binary_cast_array_array(self, xp, op, in1, in2, order_op):
        in1 = xp.array(in1)
        in2 = xp.array(in2)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, order=order_op)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_scalar
    # out is False
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders,
        dtype_x=all_types, dtype_y=all_types,
        minval=minval, maxval=maxval, mode='array_scalar',
        is_out=False, is_where=False, is_dtype=False, seed=0
    )
    def test_binary_cast_array_scalar(self, xp, op, in1, in2):
        in1 = xp.array(in1)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar_scalar
    # out is False
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders,
        dtype_x=all_types, dtype_y=all_types,
        minval=minval, maxval=maxval, mode='scalar_scalar',
        is_out=False, is_where=False, is_dtype=False, seed=0
    )
    def test_binary_cast_scalar_scalar(self, xp, op, in1, in2):
        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_array
    # out is False
    # where is False
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_arg=all_types,
        minval=minval, maxval=maxval, mode='array_array',
        is_out=False, is_where=False, is_dtype=True, seed=0
    )
    def test_binary_cast_array_array_with_dtype(self, xp, op, in1, in2, dtype):
        in1 = xp.array(in1)
        in2 = xp.array(in2)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, dtype=dtype)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_scalar
    # out is False
    # where is False
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_arg=all_types,
        minval=minval, maxval=maxval, mode='array_scalar',
        is_out=False, is_where=False, is_dtype=True, seed=0
    )
    def test_binary_cast_array_scalar_with_dtype(
            self, xp, op, in1, in2, dtype):
        in1 = xp.array(in1)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, dtype=dtype)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar_scalar
    # out is False
    # where is False
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_arg=all_types,
        minval=minval, maxval=maxval, mode='scalar_scalar',
        is_out=False, is_where=False, is_dtype=True, seed=0
    )
    def test_binary_cast_scalar_scalar_with_dtype(
            self, xp, op, in1, in2, dtype):
        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, dtype=dtype)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_array
    # out is True
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval, mode='array_array',
        is_out=True, is_where=False, is_dtype=False, seed=0
    )
    def test_binary_cast_array_array_with_out(self, xp, op, in1, in2, out):
        in1 = xp.array(in1)
        in2 = xp.array(in2)
        out = xp.array(out)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, out=out)
            if xp is numpy and op in float16_op_set:
                if in1.dtype == numpy.dtype('bool') and \
                        in2.dtype == numpy.dtype('bool'):
                    y = func(in1, in2, out=out, dtype='f4')
                    return y
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_scalar
    # out is True
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval, mode='array_scalar',
        is_out=True, is_where=False, is_dtype=False, seed=0
    )
    def test_binary_cast_array_scalar_with_out(self, xp, op, in1, in2, out):
        in1 = xp.array(in1)
        out = xp.array(out)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, out=out)
            if xp is numpy and op in float16_op_set:
                if in1.dtype == numpy.dtype('bool') and \
                        isinstance(in2, bool):
                    y = func(in1, in2, out=out, dtype='f4')
                    return y
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar_scalar
    # out is True
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval, mode='scalar_scalar',
        is_out=True, is_where=False, is_dtype=False, seed=0
    )
    def test_binary_cast_scalar_scalar_with_out(self, xp, op, in1, in2, out):
        out = xp.array(out)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, out=out)
            if xp is numpy and op in float16_op_set:
                if isinstance(in1, bool) and \
                        isinstance(in2, bool):
                    y = func(in1, in2, out=out, dtype='f4')
                    return y
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_array
    # out is True(for broadcast)
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders,
        dtype_x=float_types, dtype_y=float_types, dtype_out=float_types,
        minval=minval, maxval=maxval, mode='array_array',
        is_out=True, is_where=False, is_dtype=False, is_broadcast=True,
        seed=0
    )
    def test_binary_cast_array_array_with_out_broadcast(
            self, xp, op, in1, in2, out):
        in1 = xp.array(in1)
        in2 = xp.array(in2)
        out = xp.array(out)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, out=out)
            if xp is numpy and op in float16_op_set:
                if in1.dtype == numpy.dtype('bool') and \
                        in2.dtype == numpy.dtype('bool'):
                    y = func(in1, in2, out=out, dtype='f4')
                    return y
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_array
    # out is True
    # where is False
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        dtype_arg=all_types,
        minval=minval, maxval=maxval, mode='array_array',
        is_out=True, is_where=False, is_dtype=True, seed=0
    )
    def test_binary_cast_array_array_with_out_with_dtype(
            self, xp, op, in1, in2, out, dtype):
        in1 = xp.array(in1)
        in2 = xp.array(in2)
        out = xp.array(out)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, out=out, dtype=dtype)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_scalar
    # out is True
    # where is False
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        dtype_arg=all_types,
        minval=minval, maxval=maxval, mode='array_scalar',
        is_out=True, is_where=False, is_dtype=True, seed=0
    )
    def test_binary_cast_array_scalar_with_out_with_dtype(
            self, xp, op, in1, in2, out, dtype):
        in1 = xp.array(in1)
        out = xp.array(out)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, out=out, dtype=dtype)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar_scalar
    # out is True
    # where is False
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        dtype_arg=all_types,
        minval=minval, maxval=maxval, mode='scalar_scalar',
        is_out=True, is_where=False, is_dtype=True, seed=0
    )
    def test_binary_cast_scalar_scalar_with_out_with_dtype(
            self, xp, op, in1, in2, out, dtype):
        out = xp.array(out)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, out=out, dtype=dtype)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_array
    # out is True
    # where is True
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders, order_where=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval, mode='array_array',
        is_out=True, is_where=True, is_dtype=False, seed=0
    )
    def test_binary_cast_array_array_with_out_with_where(
            self, xp, op, in1, in2, out, where):
        in1 = xp.array(in1)
        in2 = xp.array(in2)
        out = xp.array(out)
        where = xp.array(where)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            with numpy.warnings.catch_warnings():
                numpy.warnings.simplefilter(
                    'ignore', numpy.ComplexWarning)
                func = getattr(xp, op)
                y = func(in1, in2, out=out, where=where)
                if xp is numpy and op in float16_op_set:
                    if in1.dtype == numpy.dtype('bool') and \
                            in2.dtype == numpy.dtype('bool'):
                        y = func(in1, in2, out=out, where=where, dtype='f4')
                        return y
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_scalar
    # out is True
    # where is True
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders, order_where=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval, mode='array_scalar',
        is_out=True, is_where=True, is_dtype=False, seed=0
    )
    def test_binary_cast_array_scalar_with_out_with_where(
            self, xp, op, in1, in2, out, where):
        in1 = xp.array(in1)
        out = xp.array(out)
        where = xp.array(where)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            with numpy.warnings.catch_warnings():
                numpy.warnings.simplefilter(
                    'ignore', numpy.ComplexWarning)
                func = getattr(xp, op)
                y = func(in1, in2, out=out, where=where)
                if xp is numpy and op in float16_op_set:
                    if in1.dtype == numpy.dtype('bool') and \
                            isinstance(in2, bool):
                        y = func(in1, in2, out=out, where=where, dtype='f4')
                        return y
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar_scalar
    # out is True
    # where is True
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders, order_where=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        minval=minval, maxval=maxval, mode='scalar_scalar',
        is_out=True, is_where=True, is_dtype=False, seed=0
    )
    def test_binary_cast_scalar_scalar_with_out_with_where(
            self, xp, op, in1, in2, out, where):
        out = xp.array(out)
        where = xp.array(where)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            with numpy.warnings.catch_warnings():
                numpy.warnings.simplefilter(
                    'ignore', numpy.ComplexWarning)
                func = getattr(xp, op)
                y = func(in1, in2, out=out, where=where)
                if xp is numpy and op in float16_op_set:
                    if isinstance(in1, bool) and \
                            isinstance(in2, bool):
                        y = func(in1, in2, out=out, where=where, dtype='f4')
                        return y
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_array
    # out is True(for broadcast)
    # where is True(for broadcast)
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders, order_where=orders,
        dtype_x=float_types, dtype_y=float_types, dtype_out=float_types,
        minval=minval, maxval=maxval, mode='array_array',
        is_out=True, is_where=True, is_dtype=False, is_broadcast=True,
        seed=0
    )
    def test_binary_cast_array_array_with_out_with_where_broadcast(
            self, xp, op, in1, in2, out, where):
        in1 = xp.array(in1)
        in2 = xp.array(in2)
        out = xp.array(out)
        where = xp.array(where)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            with numpy.warnings.catch_warnings():
                numpy.warnings.simplefilter(
                    'ignore', numpy.ComplexWarning)
                func = getattr(xp, op)
                y = func(in1, in2, out=out, where=where)
                if xp is numpy and op in float16_op_set:
                    if in1.dtype == numpy.dtype('bool') and \
                            in2.dtype == numpy.dtype('bool'):
                        y = func(in1, in2, out=out, where=where, dtype='f4')
                        return y
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_array
    # out is True
    # where is True
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders, order_where=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        dtype_arg=all_types,
        minval=minval, maxval=maxval, mode='array_array',
        is_out=True, is_where=True, is_dtype=True, seed=0
    )
    def test_binary_cast_array_array_with_out_with_where_with_dtype(
            self, xp, op, in1, in2, out, where, dtype):
        in1 = xp.array(in1)
        in2 = xp.array(in2)
        out = xp.array(out)
        where = xp.array(where)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            with numpy.warnings.catch_warnings():
                numpy.warnings.simplefilter(
                    'ignore', numpy.ComplexWarning)
                func = getattr(xp, op)
                y = func(in1, in2, out=out, where=where, dtype=dtype)
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_scalar
    # out is True
    # where is True
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders, order_where=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        dtype_arg=all_types,
        minval=minval, maxval=maxval, mode='array_scalar',
        is_out=True, is_where=True, is_dtype=True, seed=0
    )
    def test_binary_cast_array_scalar_with_out_with_where_with_dtype(
            self, xp, op, in1, in2, out, where, dtype):
        in1 = xp.array(in1)
        out = xp.array(out)
        where = xp.array(where)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            with numpy.warnings.catch_warnings():
                numpy.warnings.simplefilter(
                    'ignore', numpy.ComplexWarning)
                func = getattr(xp, op)
                y = func(in1, in2, out=out, where=where, dtype=dtype)
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar_scalar
    # out is True
    # where is True
    # dtype is True
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes,
        order_x=orders, order_y=orders, order_out=orders, order_where=orders,
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        dtype_arg=all_types,
        minval=minval, maxval=maxval, mode='scalar_scalar',
        is_out=True, is_where=True, is_dtype=True, seed=0
    )
    def test_binary_cast_scalar_scalar_with_out_with_where_with_dtype(
            self, xp, op, in1, in2, out, where, dtype):
        out = xp.array(out)
        where = xp.array(where)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            with numpy.warnings.catch_warnings():
                numpy.warnings.simplefilter(
                    'ignore', numpy.ComplexWarning)
                func = getattr(xp, op)
                y = func(in1, in2, out=out, where=where, dtype=dtype)
                if xp is nlcpy:
                    nlcpy.request.flush()
        return y


###########################################################
#
# testing for broadcast
#
###########################################################
shapes_base = [(4, ),
               (4, 5),
               (4, 2, 3), ]

b_shapes = testing.shaped_rearrange_for_broadcast(shapes_base)


@testing.with_requires('numpy>=1.19')
@testing.with_requires('numpy<1.20')
class TestBinaryBroadcast(unittest.TestCase):

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    ################################
    # *** testing parameter ***
    # input is array_array
    # out is False
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, b_shapes,
        order_x=orders, order_y=orders,
        dtype_x=all_types, dtype_y=f64_type,
        minval=minval, maxval=maxval, mode='array_array',
        is_out=False, is_where=False, is_dtype=False, seed=0
    )
    def test_binary_broadcast_array_array(self, xp, op, in1, in2):
        in1 = xp.array(in1)
        in2 = xp.array(in2)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is scalar_array
    # out is False
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, b_shapes,
        order_x=orders, order_y=orders,
        dtype_x=all_types, dtype_y=f64_type,
        minval=minval, maxval=maxval, mode='array_scalar',
        is_out=False, is_where=False, is_dtype=False, seed=0
    )
    def test_binary_broadcast_array_scalar(self, xp, op, in1, in2):
        in1 = xp.array(in1)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_array
    # out is True
    # where is False
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, b_shapes,
        order_x=orders, order_y=orders, order_out=orders,
        dtype_x=all_types, dtype_y=f64_type, dtype_out=f64_type,
        minval=minval, maxval=maxval, mode='array_array',
        is_out=True, is_where=False, is_dtype=False, seed=0
    )
    def test_binary_broadcast_array_array_with_out(
            self, xp, op, in1, in2, out):
        in1 = xp.array(in1)
        in2 = xp.array(in2)
        out = xp.array(out)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, out=out)
            if xp is numpy and op in float16_op_set:
                if in1.dtype == numpy.dtype('bool') and \
                        in2.dtype == numpy.dtype('bool'):
                    y = func(in1, in2, out=out, dtype='f4')
                    return y
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

    ################################
    # *** testing parameter ***
    # input is array_array
    # out is True
    # where is True
    # dtype is False
    ################################
    @pytest.mark.full
    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, b_shapes,
        order_x=orders, order_y=orders, order_out=orders, order_where=orders,
        dtype_x=all_types, dtype_y=f64_type, dtype_out=f64_type,
        minval=minval, maxval=maxval, mode='array_array',
        is_out=True, is_where=True, is_dtype=False, seed=0
    )
    def test_binary_broadcast_array_array_with_where(
            self, xp, op, in1, in2, out, where):
        in1 = xp.array(in1)
        in2 = xp.array(in2)
        out = xp.array(out)
        where = xp.array(where)

        with testing.numpy_nlcpy_errstate(
                divide='ignore', over='ignore', invalid='ignore'):
            func = getattr(xp, op)
            y = func(in1, in2, out=out, where=where)
            if xp is nlcpy:
                nlcpy.request.flush()
        return y

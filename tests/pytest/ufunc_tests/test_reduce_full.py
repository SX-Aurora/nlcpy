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
all_types = [numpy.bool] + float_types + int_types + complex_types

ops = [
    'power',
    'multiply',
    'add',
    'subtract',
    'divide',
    'floor_divide',
    'true_divide',
    'mod',
    'remainder',
    'hypot',
    'fmod',
    'arctan2',
    'logaddexp',
    'logaddexp2',
    'right_shift',
    'left_shift',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'logical_and',
    'logical_or',
    'logical_xor',
    'less',
    'less_equal',
    'greater',
    'greater_equal',
    'equal',
    'not_equal',
    'maximum',
    'minimum',
    'fmax',
    'fmin',
    'heaviside',
    'copysign',
    'nextafter',
]


def convert_initial(initial, op, dtype_in, dtype=None, dtype_out=None):
    if initial in (None, numpy._NoValue):
        return initial
    if op == 'nextafter' and initial == 0:
        return 1
    dtype = numpy.dtype(dtype)
    dtype_out = numpy.dtype(dtype_out)
    if op == 'power' or dtype.char in 'IL' or dtype_out.char in 'IL':
        if numpy.real(initial) < 0:
            if type(initial) is complex:
                initial = -numpy.real(initial) + numpy.imag(initial) * 1j
            else:
                initial *= -1
    if op in ('floor_divide', 'fmod', 'mod'):
        initial *= 50
    return initial


def is_executable(op, initial=0, where=True, dtype_in=None, dtype=None, dtype_out=None):
    if op == 'logaddexp' and \
       dtype_out is not None and numpy.dtype(dtype_out).char == 'L':
        return False
    return True

    if where is not True:
        if initial is numpy._NoValue:
            initial = getattr(numpy, op).identity
        if initial is None:
            return False

    # NumPy returns incorrect value.
    if op == 'logaddexp' and \
       dtype_out is not None and numpy.dtype(dtype_out).char == 'L':
        return False

    if dtype is None:
        dtype = dtype_in if dtype_out is None else dtype_out
    if dtype is not None:
        dtype = numpy.dtype(dtype)
        if op in ('power', 'subtract', 'floor_divide'):
            return dtype.char in 'ilILfdFD'
        if op in ('divide', 'true_divide'):
            return dtype.char in 'fdFD'
        if op in ('mod', 'remainder', 'fmod'):
            return dtype.char in 'ilILfd'
        if op in ('bitwise_and', 'bitwise_or', 'bitwise_xor'):
            return dtype.char in '?ilIL'
        if op in ('left_shift', 'right_shift'):
            return dtype.char in 'ilIL'
        if op in ('arctan2', 'hypot', 'logaddexp', 'logaddexp2', 'heaviside', 'copysign',
                  'nextafter'):
            return dtype.char in 'fd'

    return True


def execute_ufunc(op, xp, in1, out=None, dtype=None,
                  axis=0, initial=numpy._NoValue, where=True, keepdims=False):
    dtype_out = None if out is None else out.dtype
    if not is_executable(op, initial, where, in1.dtype, dtype, dtype_out):
        return 0
    initial = convert_initial(initial, op, in1.dtype, dtype, dtype_out)
    return getattr(xp, op).reduce(
        in1, out=out, dtype=dtype, axis=axis,
        initial=initial, where=where, keepdims=keepdims)


@testing.parameterize(*testing.product({
    'initial': (numpy._NoValue, None, 0, 2, -2.63, -1.2 + 0.3j),
}))
@pytest.mark.full
class TestReduce(unittest.TestCase):

    shapes = ((1, 3, 2), (3, 2, 5), (5, 4, 2, 3))
    axes = (0, 1, -2, (1, 2), None)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[2:], shapes, dtype_x=all_types, ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce(self, xp, op, in1, axis):
        return execute_ufunc(
            op, xp, in1, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[2:], shapes, dtype_x=all_types, dtype_arg=all_types,
        is_dtype=True, ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce_with_dtype(self, xp, op, in1, dtype, axis):
        return execute_ufunc(
            op, xp, in1, dtype=dtype, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[2:], shapes, dtype_x=all_types, dtype_out=all_types,
        is_out=True, ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce_with_out(self, xp, op, in1, out, axis):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, out=out, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[2:], shapes, dtype_x=all_types, dtype_arg=all_types, dtype_out=all_types,
        is_out=True, is_dtype=True, ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce_with_dtype_and_out(self, xp, op, in1, dtype, out, axis):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, dtype=dtype, out=out, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[2:], shapes, dtype_x=all_types, dtype_out=all_types,
        ufunc_name='reduce', axes=axes, is_out=True, is_where=True, seed=0
    )
    def test_reduce_dtype_with_where(self, xp, op, in1, out, axis, where):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, out=out, axis=axis, initial=self.initial, where=where)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[2:], shapes, dtype_x=all_types, ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce_dtype_with_scalar_where(self, xp, op, in1, axis):
        return execute_ufunc(
            op, xp, in1, axis=axis, initial=self.initial, where=False)


@testing.parameterize(*testing.product({
    'initial': (numpy._NoValue, None, 2, -2.63, -1.2 + 0.3j),
}))
@pytest.mark.full
class TestReduce2(unittest.TestCase):

    shapes = ((1, 3, 2), (3, 2, 5))
    axes = (0, 1, -2, (1, 2))

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[:3], shapes, dtype_x=all_types, ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce2(self, xp, op, in1, axis):
        return execute_ufunc(
            op, xp, in1, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[:3], shapes, dtype_x=all_types, dtype_arg=all_types,
        is_dtype=True, ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce_power_with_dtype2(self, xp, op, in1, dtype, axis):
        return execute_ufunc(
            op, xp, in1, dtype=dtype, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[:3], shapes, dtype_x=all_types, dtype_out=all_types,
        is_out=True, ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce_with_out2(self, xp, op, in1, out, axis):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, out=out, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[:3], shapes, dtype_x=all_types, dtype_arg=all_types, dtype_out=all_types,
        is_out=True, is_dtype=True, ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce_with_dtype_and_out2(self, xp, op, in1, dtype, out, axis):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, dtype=dtype, out=out, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[:3], shapes, dtype_x=all_types, dtype_out=all_types,
        ufunc_name='reduce', axes=axes, is_out=True, is_where=True, seed=0
    )
    def test_reduce_dtype_with_where2(self, xp, op, in1, out, axis, where):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, out=out, axis=axis, initial=self.initial, where=where)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops[:3], shapes, dtype_x=all_types, ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce_dtype_with_scalar_where2(self, xp, op, in1, axis):
        return execute_ufunc(
            op, xp, in1, axis=axis, initial=self.initial, where=False)


@pytest.mark.full
class TestReduce3(unittest.TestCase):

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, (), dtype_x=all_types, dtype_out=all_types,
        is_out=True, mode='scalar', ufunc_name='reduce', axes=None
    )
    def test_reduce_scalar(self, xp, op, in1, out, axis):
        return execute_ufunc(op, xp, in1, axis=axis, out=out)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, (), dtype_x=all_types, dtype_out=all_types,
        is_out=True, mode='scalar', ufunc_name='reduce', axes=None
    )
    def test_reduce_scalar_with_initial(self, xp, op, in1, out, axis):
        return execute_ufunc(op, xp, in1, axis=axis, out=out, initial=2)

    @testing.numpy_nlcpy_raises
    def test_reduce_incorrect_axis(self, xp):
        return xp.add.reduce(xp.empty([2, 3, 4]), axis=3)

    @testing.numpy_nlcpy_raises
    def test_reduce_incorrect_axis2(self, xp):
        return xp.add.reduce(xp.empty([2, 3, 4]), axis=-4)

    @testing.numpy_nlcpy_raises
    def test_reduce_incorrect_axis3(self, xp):
        return xp.add.reduce(xp.empty([2, 3, 4]), axis=(1, 2, 3))

    @testing.numpy_nlcpy_raises
    def test_reduce_incorrect_out_shape(self, xp):
        return xp.add.reduce(xp.empty([2, 3, 4]), axis=(1, 2), out=xp.empty([4]))

    @testing.numpy_nlcpy_raises
    def test_reduce_power_with_negative_integer(self, xp):
        xp.power.reduce([1, -1])

    def test_reduce_too_large_left_shift(self):
        for dtype in 'iI':
            x = nlcpy.left_shift.reduce([1, 32], dtype=dtype)
            if x != 0:
                raise ValueError

        for dtype in 'lL':
            x = nlcpy.left_shift.reduce([1, 64], dtype=dtype)
            if x != 0:
                raise ValueError

    def test_reduce_with_nan(self):
        a = ((numpy.nan, 1), (1, numpy.nan))
        for op in ('maximum', 'minimum', 'fmax', 'fmin'):
            x = getattr(numpy, op).reduce(a, axis=0)
            y = getattr(nlcpy, op).reduce(a, axis=0)
            numpy.testing.assert_array_equal(x, y)

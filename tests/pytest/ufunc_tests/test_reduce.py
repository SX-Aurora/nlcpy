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

import warnings
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
all_types = [numpy.bool] + float_types + int_types + complex_types

ops = [
    'power',
    'add',
    'subtract',
    'multiply',
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


def execute_ufunc(op, xp, in1, out=None, dtype=None,
                  axis=0, initial=numpy._NoValue, where=True, keepdims=False):
    dtype_out = None if out is None else out.dtype
    if not is_executable(op, initial, where, in1.dtype, dtype, dtype_out):
        return 0
    initial = convert_initial(initial, op, in1.dtype, dtype, dtype_out)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', numpy.ComplexWarning)
        if op in ('power'):
            with xp.errstate(invalid='ignore', divide='ignore'):
                ret = getattr(xp, op).reduce(
                    in1, out=out, dtype=dtype, axis=axis,
                    initial=initial, where=where, keepdims=keepdims)
                nlcpy.request.flush()
    return ret


@testing.parameterize(*testing.product({
    'initial': (numpy._NoValue, None, 0, 2, -2.63, -3.2 + 0.3j),
}))
@pytest.mark.no_fast_math
class TestReduceDtype(unittest.TestCase):

    shape = ((3, 2), )

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shape, order_x='C', dtype_x=all_types,
        ufunc_name='reduce', axes=(0,), seed=0
    )
    def test_reduce(self, xp, op, in1, axis):
        return execute_ufunc(
            op, xp, in1, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shape, order_x='C', dtype_x=all_types, dtype_arg=all_types,
        is_dtype=True, ufunc_name='reduce', axes=(0,), seed=0
    )
    def test_reduce_with_dtype(self, xp, op, in1, dtype, axis):
        return execute_ufunc(
            op, xp, in1, dtype=dtype, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shape, order_x='C', order_out='C', dtype_x=all_types, dtype_out=all_types,
        is_out=True, ufunc_name='reduce', axes=(0,), seed=0
    )
    def test_reduce_with_out(self, xp, op, in1, out, axis):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, out=out, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shape, order_x='C', order_out='C',
        dtype_x=all_types, dtype_arg=all_types, dtype_out=all_types,
        is_out=True, is_dtype=True, ufunc_name='reduce', axes=(0,), seed=0
    )
    def test_reduce_with_dtype_and_out(self, xp, op, in1, dtype, out, axis):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, dtype=dtype, out=out, axis=axis, initial=self.initial)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shape, order_x='C', order_where='C', dtype_x=all_types,
        ufunc_name='reduce', axes=(0,), seed=0
    )
    def test_reduce_dtype_with_scalar_where(self, xp, op, in1, axis):
        return execute_ufunc(
            op, xp, in1, axis=axis, initial=self.initial, where=False)


@testing.parameterize(*testing.product({
    'initial': (numpy._NoValue, None, 1.1),
}))
@pytest.mark.no_fast_math
class TestReduceShape(unittest.TestCase):

    axes = (0, 1, -2, (0, 2), None)

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, ((2, 4, 3),), order_x='C', order_out='C',
        dtype_x=all_types, dtype_out=all_types, is_where=True, is_out=True,
        ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce_shape(self, xp, op, in1, out, where, axis):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, initial=self.initial, axis=axis, out=out, where=where)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, ((2, 4, 3),), order_x='C', order_out='C', dtype_x=all_types,
        dtype_out=all_types, is_where=True, is_out=True, ufunc_name='reduce',
        axes=axes, keepdims=True, seed=0
    )
    def test_reduce_shape_keepdims(self, xp, op, in1, out, where, axis):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, initial=self.initial, axis=axis,
            out=out, where=where, keepdims=True)


@pytest.mark.no_fast_math
class TestReduce(unittest.TestCase):

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, (), order_x='C', order_out='C', dtype_x=all_types, dtype_out=all_types,
        is_out=True, mode='scalar', ufunc_name='reduce', axes=None, seed=0
    )
    def test_reduce_scalar(self, xp, op, in1, out, axis):
        return execute_ufunc(op, xp, in1, axis=axis, out=out)

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


@pytest.mark.no_fast_math
class TestReduceKeepdims(unittest.TestCase):

    shape = ((3, 4), )
    axes = (0, 1, -1, (0, 1), None)

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shape, dtype_x=(numpy.float64,), dtype_arg=(numpy.float64,),
        ufunc_name='reduce', axes=axes, is_dtype=True, keepdims=True, seed=0
    )
    def test_reduce_keepdims_no_out_no_where(self, xp, op, in1, axis, dtype):
        return execute_ufunc(
            op, xp, in1, axis=axis, dtype=dtype, keepdims=True)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shape, dtype_x=(numpy.float64,), dtype_arg=(numpy.float64,),
        ufunc_name='reduce', axes=axes, is_out=True, is_dtype=True,
        keepdims=True, seed=0
    )
    def test_reduce_keepdims_with_out_no_where(self, xp, op, in1, axis, dtype, out):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, axis=axis, out=out,
            dtype=dtype, keepdims=True)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shape, dtype_x=(numpy.float64,), dtype_arg=(numpy.float64,),
        ufunc_name='reduce', axes=axes, is_out=True, is_where=True, is_dtype=True,
        keepdims=True, seed=0
    )
    def test_reduce_keepdims_with_out_with_where(self, xp, op, in1, axis, dtype,
                                                 out, where):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, axis=axis, out=out, dtype=dtype,
            where=where, keepdims=True)

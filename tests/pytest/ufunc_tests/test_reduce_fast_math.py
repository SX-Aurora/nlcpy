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
import nlcpy
import gc
from nlcpy import testing

float_types = [numpy.float32, numpy.float64]
complex_types = [numpy.complex64, numpy.complex128]
signed_int_types = [numpy.int32, numpy.int64]
unsigned_int_types = [numpy.uint32, numpy.uint64]
int_types = signed_int_types + unsigned_int_types
all_types = [numpy.bool] + float_types + int_types + complex_types

ops = [
    'power',
    'divide',
    'floor_divide',
    'true_divide',
    'mod',
    'remainder',
    'fmod',
    'arctan2',
    'logaddexp',
    'logaddexp2',
]


def convert_initial(initial, op, dtype_in, dtype=None):
    if initial in (None, numpy._NoValue):
        return initial
    dtype = numpy.dtype(dtype)
    if op == 'power' or dtype.char in 'IL':
        if numpy.real(initial) < 0:
            if type(initial) is complex:
                initial = -numpy.real(initial) + numpy.imag(initial) * 1j
            else:
                initial *= -1
    if op in ('floor_divide', 'fmod', 'mod'):
        initial *= 50
    return initial


def is_executable(op, initial=0, dtype_in=None, dtype=None):
    if dtype is None:
        dtype = dtype_in
    if dtype is not None:
        dtype = numpy.dtype(dtype)
        if op in ('power', 'floor_divide'):
            return dtype.char in 'ilILfdFD'
        if op in ('divide', 'true_divide'):
            return dtype.char in 'fdFD'
        if op in ('mod', 'remainder', 'fmod'):
            return dtype.char in 'ilILfd'
        if op in ('arctan2', 'logaddexp', 'logaddexp2'):
            return dtype.char in 'fd'
    return True


def execute_ufunc(op, xp, in1, dtype=None, axis=0, initial=numpy._NoValue):
    if not is_executable(op, initial, in1.dtype, dtype):
        return 0
    initial = convert_initial(initial, op, in1.dtype, dtype)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', numpy.ComplexWarning)
        if op in ('power'):
            with xp.errstate(invalid='ignore'):
                ret = getattr(xp, op).reduce(
                    in1, dtype=dtype, axis=axis, initial=initial)
                nlcpy.request.flush()
                return ret
        else:
            return getattr(xp, op).reduce(
                in1, dtype=dtype, axis=axis, initial=initial)


@testing.parameterize(*testing.product({
    'initial': (numpy._NoValue, None, 0, 2, -2.63, -3.2 + 0.3j),
}))
@pytest.mark.fast_math
class TestReduceDtype(unittest.TestCase):

    shape = ((4,), (3, 2), )

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


@testing.parameterize(*testing.product({
    'initial': (numpy._NoValue, None, 1.1),
}))
@pytest.mark.fast_math
class TestReduceAxis(unittest.TestCase):

    axes = (0, 1, -2, (0, 2), None)
    shapes = ((2, 4, 3),)

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes, order_x='C', dtype_x=all_types,
        ufunc_name='reduce', axes=axes, seed=0
    )
    def test_reduce_axis(self, xp, op, in1, axis):
        return execute_ufunc(
            op, xp, in1, initial=self.initial, axis=axis)


@pytest.mark.fast_math
class TestReduce(unittest.TestCase):

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, (), order_x='C', dtype_x=all_types,
        mode='scalar', ufunc_name='reduce', axes=None, seed=0
    )
    def test_reduce_scalar(self, xp, op, in1, axis):
        return execute_ufunc(op, xp, in1, axis=axis)

    @testing.numpy_nlcpy_raises
    def test_reduce_power_with_negative_integer(self, xp):
        xp.power.reduce([1, -1])

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
import warnings
import gc
import nlcpy
from nlcpy import testing
from nlcpy.testing.types import all_types


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


def adjust_dtype(xp, op, dtype_in, dtype, dtype_out):
    if xp is numpy:
        if dtype == numpy.bool_ or dtype is None and dtype_out == numpy.bool_:
            if op in ('divide', 'true_divide', 'logaddexp', 'logaddexp2',
                      'heaviside', 'arctan2', 'hypot', 'copysign', 'nextafter'):
                dtype = numpy.float32
            elif op in ('power', 'right_shift', 'left_shift', 'subtract'):
                dtype = numpy.int32
        elif dtype_in == numpy.bool_:
            if op == 'subtract':
                dtype = numpy.int32
    return dtype


def is_executable(op, dtype_in=None, dtype=None, dtype_out=None):
    if dtype_out is None:
        if op in (
            'divide', 'true_divide', 'arctan2', 'hypot', 'copysign',
            'logaddexp', 'logaddexp2', 'nextafter', 'heaviside',
            'power', 'floor_divide', 'mod', 'remainder', 'fmod',
            'right_shift', 'left_shift', 'subtract'
        ):
            return dtype != numpy.bool_ and not \
                (dtype is None and dtype_in == numpy.bool_)

    return True


def execute_ufunc(xp, op, in1, out=None, dtype=None, axis=0):
    dtype_out = None if out is None else out.dtype
    if not is_executable(op, in1.dtype, dtype, dtype_out):
        return 0
    dtype = adjust_dtype(xp, op, in1.dtype, dtype, dtype_out)
    ret = getattr(xp, op).accumulate(in1, out=out, dtype=dtype, axis=axis)
    return ret


@pytest.mark.no_fast_math
@testing.with_requires('numpy>=1.19')
@testing.with_requires('numpy<1.20')
class TestAccumulate(unittest.TestCase):

    shapes = ((2, 2),)
    axes = (0,)
    shapes2 = ((3, 2, 1, 4), (1, 3, 5, 2, 5))
    axes2 = (0, 1, -1)

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes, order_x='C', dtype_x=all_types,
        ufunc_name='accumulate', axes=axes, seed=0
    )
    def test_accumulate(self, xp, op, in1, axis):
        return execute_ufunc(xp, op, in1, axis=axis)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes, order_x='C', dtype_x=all_types, dtype_arg=all_types,
        is_dtype=True, ufunc_name='accumulate', axes=axes, seed=0
    )
    def test_accumulate_with_dtype(self, xp, op, in1, dtype, axis):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            return execute_ufunc(xp, op, in1, axis=axis, dtype=dtype)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes, order_x='C', order_out='C', dtype_x=all_types, dtype_out=all_types,
        is_out=True, ufunc_name='accumulate', axes=axes, seed=0
    )
    def test_accumulate_with_out(self, xp, op, in1, out, axis):
        out = xp.asarray(out)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            return execute_ufunc(xp, op, in1, out=out, axis=axis)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes, order_x='C', order_out='C', dtype_x=all_types, dtype_out=all_types,
        is_out=True, ufunc_name='accumulate', axes=axes, seed=0, is_broadcast=True
    )
    def test_accumulate_with_out_broadcast(self, xp, op, in1, out, axis):
        out = xp.asarray(out)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            return execute_ufunc(xp, op, in1, out=out, axis=axis)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes, order_x='C', order_out='C', dtype_x=all_types,
        dtype_arg=all_types, dtype_out=all_types, is_out=True, is_dtype=True,
        ufunc_name='accumulate', axes=axes, seed=0
    )
    def test_accumulate_with_dtype_and_out(self, xp, op, in1, dtype, out, axis):
        out = xp.asarray(out)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            return execute_ufunc(xp, op, in1, dtype=dtype, out=out, axis=axis)

    @testing.numpy_nlcpy_check_for_unary_ufunc(
        ops, shapes2, order_x='C', dtype_x=all_types, dtype_out=all_types,
        ufunc_name='accumulate', is_out=True, axes=axes2, seed=0
    )
    def test_accumulate_shape(self, xp, op, in1, out, axis):
        out = xp.asarray(out)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            return execute_ufunc(xp, op, in1, out=out, axis=axis)

    def test_accumulate_too_large_left_shift(self):
        for dtype in 'iI':
            x = nlcpy.left_shift.accumulate(nlcpy.array([1, 32, 32], dtype=dtype))
            if x[1] != 0 or x[2] != 0:
                raise ValueError

        for dtype in 'lL':
            x = nlcpy.left_shift.accumulate(nlcpy.array([1, 32, 32], dtype=dtype))
            if x[1] == 0 or x[2] != 0:
                raise ValueError

    def test_accumulate_with_nan(self):
        a = [[numpy.nan, 1], [1, numpy.nan]]
        for op in ('maximum', 'minimum', 'fmax', 'fmin'):
            x = getattr(numpy, op).accumulate(a)
            y = getattr(nlcpy, op).accumulate(a)
            numpy.testing.assert_array_equal(x, y)

    @testing.numpy_nlcpy_raises()
    def test_accumulate_accumulate_on_scalar(self, xp):
        xp.add.accumulate(1)

    @testing.numpy_nlcpy_raises()
    def test_accumulate_incorrect_axis(self, xp):
        xp.add.accumulate(xp.empty([2, 3, 4]), axis=3)

    @testing.numpy_nlcpy_raises()
    def test_accumulate_incorrect_axis2(self, xp):
        xp.add.accumulate(xp.empty([2, 3, 4]), axis=-4)

    @testing.numpy_nlcpy_raises()
    def test_accumulate_incorrect_out_shape(self, xp):
        xp.add.accumulate(xp.empty([2, 3, 4]), out=xp.empty([1, 3, 4]))

    @testing.for_dtypes("il")
    @testing.numpy_nlcpy_raises()
    def test_accumulate_power_with_negative_integer(self, xp, dtype):
        xp.power.accumulate(xp.array([1, -1], dtype=dtype))

    def test_accumulate_not_supported_dtype(self):
        for op in (
                'divide', 'true_divide', 'arctan2', 'hypot', 'logaddexp', 'logaddexp2',
                'nextafter', 'heaviside', 'floor_divide', 'mod', 'remainder', 'power'):
            try:
                getattr(nlcpy, op).accumulate([True])
                raise Exception
            except TypeError:
                pass

    @testing.numpy_nlcpy_array_equal()
    def test_accumulate_not_contiguous(self, xp):
        a = xp.moveaxis(xp.arange(24).reshape(2, 3, 4), 0, 1)
        return xp.add.accumulate(a, axis=2)

    @testing.numpy_nlcpy_array_equal()
    def test_accumulate_broadcast_not_contiguous(self, xp):
        a = xp.moveaxis(xp.arange(24).reshape(2, 3, 4), 0, 1)
        out = xp.zeros([3, 2, 4, 2])
        return xp.add.accumulate(a, axis=2, out=out)

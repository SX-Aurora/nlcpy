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


def adjust_dtype(xp, op, dtype, dtype_out):
    if xp is numpy:
        if dtype == numpy.bool_:
            if op in ('divide', 'true_divide', 'logaddexp', 'logaddexp2',
                      'heaviside', 'arctan2', 'hypot', 'copysign', 'nextafter'):
                dtype = numpy.float32
            elif op in ('power', 'right_shift', 'left_shift'):
                dtype = numpy.int32
    return dtype


def is_executable(op, dtype1=None, dtype2=None, dtype=None, dtype_out=None):
    if dtype_out is None:
        if op in (
            'divide', 'true_divide', 'arctan2', 'hypot', 'copysign',
            'logaddexp', 'logaddexp2', 'nextafter', 'heaviside',
            'power', 'floor_divide', 'mod', 'remainder', 'fmod',
            'right_shift', 'left_shift',
        ):
            return dtype != numpy.bool_ and \
                not (dtype is None and (dtype1 == numpy.bool_ or dtype2 == numpy.bool_))

    return True


def execute_ufunc(
    xp, op, in1, in2, out=None, dtype=None, order='K', casting='same_kind', where=True
):
    dtype_out = None if out is None else out.dtype
    if isinstance(in1, numpy.ndarray):
        dtype1 = in1.dtype
    else:
        dtype1 = numpy.dtype(type(in1))
    if isinstance(in2, numpy.ndarray):
        dtype2 = in2.dtype
    else:
        dtype2 = numpy.dtype(type(in2))
    if not is_executable(op, dtype1, dtype2, dtype, dtype_out):
        return 0
    dtype = adjust_dtype(xp, op, dtype, dtype_out)
    if op in ('fmod', 'power'):
        with xp.errstate(invalid='ignore', divide='ignore'):
            ret = getattr(xp, op).outer(
                in1, in2, out=out, dtype=dtype, order=order,
                casting=casting, where=where)
            nlcpy.request.flush()  # to capture warning
            return ret
    else:
        return getattr(xp, op).outer(
            in1, in2, out=out, dtype=dtype, order=order,
            casting=casting, where=where)


@pytest.mark.no_fast_math
@testing.with_requires('numpy>=1.19')
@testing.with_requires('numpy<1.20')
class TestOuter(unittest.TestCase):

    shapes = (((3, 4), (2, 3)),)
    castings = ('no', 'equiv', 'safe', 'same_kind')

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', order_arg='K', dtype_x=all_types,
        dtype_y=all_types, ufunc_name='outer', casting=castings, seed=0
    )
    def test_outer(self, xp, op, in1, in2, order, casting):
        return execute_ufunc(xp, op, in1, in2, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C',
        dtype_x=all_types, dtype_y=all_types, dtype_arg=all_types,
        is_dtype=True, ufunc_name='outer', casting=castings, seed=0
    )
    def test_outer_with_dtype(self, xp, op, in1, in2, dtype, casting, order):
        return execute_ufunc(
            op, xp, in1, in2, dtype=dtype, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', order_out='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        is_out=True, ufunc_name='outer', casting=castings, seed=0
    )
    def test_outer_with_out(self, xp, op, in1, in2, out, order, casting):
        out = xp.asarray(out)
        return execute_ufunc(op, xp, in1, in2, out=out, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', order_out='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types,
        is_out=True, ufunc_name='outer', casting=castings, seed=0,
        is_broadcast=True
    )
    def test_outer_with_out_broadcast(self, xp, op, in1, in2, out, order, casting):
        out = xp.asarray(out)
        return execute_ufunc(op, xp, in1, in2, out=out, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', order_out='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, dtype_arg=all_types, dtype_out=all_types,
        is_out=True, is_dtype=True, ufunc_name='outer', casting=castings, seed=0
    )
    def test_outer_with_dtype_and_out(
            self, xp, op, in1, in2, dtype, out, order, casting):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, in2, dtype=dtype, out=out, order=order, casting=casting)


@pytest.mark.no_fast_math
@testing.with_requires('numpy>=1.19')
@testing.with_requires('numpy<1.20')
class TestOuterArrayScalar(unittest.TestCase):

    shapes = ((3, 4),)
    castings = ('no', 'equiv', 'safe', 'same_kind')

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, mode='array_scalar',
        ufunc_name='outer', casting=castings, seed=0
    )
    def test_outer(self, xp, op, in1, in2, order, casting):
        return execute_ufunc(xp, op, in1, in2, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', dtype_x=all_types, dtype_y=all_types,
        dtype_arg=all_types, is_dtype=True, mode='array_scalar', ufunc_name='outer',
        casting=castings, seed=0
    )
    def test_outer_with_dtype(self, xp, op, in1, in2, dtype, casting, order):
        return execute_ufunc(op, xp, in2, in1, dtype=dtype, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', order_out='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types, is_out=True,
        mode='array_scalar', ufunc_name='outer', casting=castings, seed=0
    )
    def test_outer_with_out(self, xp, op, in1, in2, out, order, casting):
        out = xp.asarray(out)
        return execute_ufunc(op, xp, in1, in2, out=out, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', order_out='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types, is_out=True,
        mode='array_scalar', ufunc_name='outer', casting=castings, seed=0,
        is_broadcast=True
    )
    def test_outer_with_out_broadcast(self, xp, op, in1, in2, out, order, casting):
        out = xp.asarray(out)
        return execute_ufunc(op, xp, in2, in1, out=out, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', order_out='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, dtype_arg=all_types, dtype_out=all_types,
        is_out=True, is_dtype=True, mode='array_scalar', ufunc_name='outer',
        casting=castings, seed=0
    )
    def test_outer_with_dtype_and_out(
            self, xp, op, in1, in2, dtype, out, order, casting):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, in2, dtype=dtype, out=out, order=order, casting=casting)


@pytest.mark.no_fast_math
@testing.with_requires('numpy>=1.19')
@testing.with_requires('numpy<1.20')
class TestOuterScalar(unittest.TestCase):

    castings = ('no', 'equiv', 'safe', 'same_kind')

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, (), order_x='C', order_y='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, mode='scalar_scalar',
        ufunc_name='outer', casting=castings, seed=0
    )
    def test_outer(self, xp, op, in1, in2, order, casting):
        return execute_ufunc(xp, op, in1, in2, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, (), order_x='C', order_y='C', dtype_x=all_types, dtype_y=all_types,
        dtype_arg=all_types, is_dtype=True, mode='scalar_scalar', ufunc_name='outer',
        casting=castings, seed=0
    )
    def test_outer_with_dtype(self, xp, op, in1, in2, dtype, casting, order):
        return execute_ufunc(op, xp, in1, in2, dtype=dtype, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, (), order_x='C', order_y='C', order_out='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types, is_out=True,
        mode='scalar_scalar', ufunc_name='outer', casting=castings, seed=0
    )
    def test_outer_with_out(self, xp, op, in1, in2, out, order, casting):
        out = xp.asarray(out)
        return execute_ufunc(op, xp, in1, in2, out=out, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, (), order_x='C', order_y='C', order_out='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, dtype_out=all_types, is_out=True,
        mode='scalar_scalar', ufunc_name='outer', casting=castings, seed=0,
        is_broadcast=True
    )
    def test_outer_with_out_broadcast(self, xp, op, in1, in2, out, order, casting):
        out = xp.asarray(out)
        return execute_ufunc(op, xp, in1, in2, out=out, order=order, casting=casting)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, (), order_x='C', order_y='C', order_out='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, dtype_arg=all_types, dtype_out=all_types,
        is_out=True, is_dtype=True, mode='scalar_scalar', ufunc_name='outer',
        casting=castings, seed=0
    )
    def test_outer_with_dtype_and_out(
            self, xp, op, in1, in2, dtype, out, order, casting):
        out = xp.asarray(out)
        return execute_ufunc(
            op, xp, in1, in2, dtype=dtype, out=out, order=order, casting=casting)


@pytest.mark.no_fast_math
@testing.with_requires('numpy>=1.19')
@testing.with_requires('numpy<1.20')
class TestOuter2(unittest.TestCase):

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    def test_outer_too_large_left_shift(self):
        for dtype in 'iI':
            x = nlcpy.left_shift.outer(
                nlcpy.array([1], dtype=dtype), nlcpy.array([32], dtype=dtype))
            if x[0] != 0:
                raise ValueError

        for dtype in 'lL':
            x = nlcpy.left_shift.outer(
                nlcpy.array([1], dtype=dtype), nlcpy.array([64], dtype=dtype))
            if x[0] != 0:
                raise ValueError

    def test_outer_with_nan(self):
        a = [numpy.nan, 1]
        b = [1, numpy.nan]
        for op in ('maximum', 'minimum', 'fmax', 'fmin'):
            x = getattr(numpy, op).outer(a, b)
            y = getattr(nlcpy, op).outer(a, b)
            numpy.testing.assert_array_equal(x, y)

    @testing.for_dtypes("il")
    @testing.numpy_nlcpy_raises()
    def test_outer_power_with_negative_integer(self, xp, dtype):
        xp.power.outer(xp.array([1], dtype=dtype), xp.array([-1], dtype=dtype))

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


def adjust_dtype(xp, op, dtype):
    if xp is numpy:
        if dtype == numpy.bool:
            if op in ('divide', 'true_divide', 'logaddexp', 'logaddexp2', 'arctan2'):
                dtype = numpy.float32
            elif op == 'power':
                dtype = numpy.int32
    return dtype


def is_executable(op, dtype1=None, dtype2=None, dtype=None):
    if op in (
        'divide', 'true_divide', 'arctan2', 'logaddexp', 'logaddexp2',
        'power', 'floor_divide', 'mod', 'remainder', 'fmod',
    ):
        return dtype != numpy.bool and \
            not (dtype is None and (dtype1 == numpy.bool or dtype2 == numpy.bool))

    return True


def execute_ufunc(xp, op, in1, in2, dtype=None, order='K'):
    if isinstance(in1, numpy.ndarray):
        dtype1 = in1.dtype
        shape1 = in1.shape
    else:
        dtype1 = numpy.dtype(type(in1))
        shape1 = (1,)
    if isinstance(in2, numpy.ndarray):
        dtype2 = in2.dtype
        shape2 = in2.shape
    else:
        dtype2 = numpy.dtype(type(in2))
        shape2 = (1,)
    if not is_executable(op, dtype1, dtype2, dtype):
        return 0
    dtype = adjust_dtype(xp, op, dtype)
    if op in ('floor_divide', 'mod', 'fmod', 'remainder'):
        in1 = testing.shaped_arange(shape1, xp, dtype=dtype1) + 10
        in2 = testing.shaped_arange(shape2, xp, dtype=dtype2) + 1
    if op in ('power',):
        in1 = testing.shaped_arange(shape1, xp, dtype=dtype1) + 1
        in2 = testing.shaped_arange(shape2, xp, dtype=dtype2) + 1
    return getattr(xp, op).outer(in1, in2, dtype=dtype, order=order)


@pytest.mark.fast_math
class TestOuter(unittest.TestCase):

    shapes = (((3, 4), (2, 3)),)

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', order_arg='K', dtype_x=all_types,
        dtype_y=all_types, ufunc_name='outer', seed=0
    )
    def test_outer(self, xp, op, in1, in2, order, casting):
        return execute_ufunc(xp, op, in1, in2, order=order)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C',
        dtype_x=all_types, dtype_y=all_types, dtype_arg=all_types,
        is_dtype=True, ufunc_name='outer', seed=0
    )
    def test_outer_with_dtype(self, xp, op, in1, in2, dtype, order, casting):
        return execute_ufunc(
            op, xp, in1, in2, dtype=dtype, order=order)


@pytest.mark.fast_math
class TestOuterArrayScalar(unittest.TestCase):

    shapes = ((3, 4),)

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, mode='array_scalar',
        ufunc_name='outer', seed=0
    )
    def test_outer(self, xp, op, in1, in2, order, casting):
        return execute_ufunc(xp, op, in1, in2, order=order)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, shapes, order_x='C', order_y='C', dtype_x=all_types, dtype_y=all_types,
        dtype_arg=all_types, is_dtype=True, mode='array_scalar', ufunc_name='outer',
        seed=0
    )
    def test_outer_with_dtype(self, xp, op, in1, in2, dtype, order, casting):
        return execute_ufunc(op, xp, in2, in1, dtype=dtype, order=order)


@pytest.mark.fast_math
class TestOuterScalar(unittest.TestCase):

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, (), order_x='C', order_y='C', order_arg='K',
        dtype_x=all_types, dtype_y=all_types, mode='scalar_scalar',
        ufunc_name='outer', seed=0
    )
    def test_outer(self, xp, op, in1, in2, order, casting):
        return execute_ufunc(xp, op, in1, in2, order=order)

    @testing.numpy_nlcpy_check_for_binary_ufunc(
        ops, (), order_x='C', order_y='C', dtype_x=all_types, dtype_y=all_types,
        dtype_arg=all_types, is_dtype=True, mode='scalar_scalar', ufunc_name='outer',
        seed=0
    )
    def test_outer_with_dtype(self, xp, op, in1, in2, dtype, order, casting):
        return execute_ufunc(op, xp, in1, in2, dtype=dtype, order=order)


@pytest.mark.fast_math
class TestOuter2(unittest.TestCase):

    def tearDown(self):
        nlcpy.venode.synchronize_all_ve()
        gc.collect()

    @testing.for_dtypes("il")
    @testing.numpy_nlcpy_raises()
    def test_outer_power_with_negative_integer(self, xp, dtype):
        xp.power.outer(xp.array([1], dtype=dtype), xp.array([-1], dtype=dtype))

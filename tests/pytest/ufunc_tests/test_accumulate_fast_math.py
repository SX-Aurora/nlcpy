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
from nlcpy.testing.types import all_types


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


def adjust_dtype(xp, op, dtype_in, dtype):
    if xp is numpy:
        if dtype == numpy.bool_:
            if op in ('divide', 'true_divide', 'logaddexp', 'logaddexp2', 'arctan2'):
                dtype = numpy.float32
            elif op == 'power':
                dtype = numpy.int32
    return dtype


def is_executable(op, dtype_in=None, dtype=None):
    if op in (
        'divide', 'true_divide', 'arctan2', 'logaddexp', 'logaddexp2',
        'power', 'floor_divide', 'mod', 'remainder', 'fmod',
    ):
        return dtype != numpy.bool_ and not (dtype is None and dtype_in == numpy.bool_)

    return True


def execute_ufunc(xp, op, in1, dtype=None, axis=0):
    if not is_executable(op, in1.dtype, dtype):
        return 0
    dtype = adjust_dtype(xp, op, in1.dtype, dtype)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', numpy.ComplexWarning)
        if op in ('floor_divide'):
            with xp.errstate(over='ignore'):
                ret = getattr(xp, op).accumulate(in1, dtype=dtype, axis=axis)
                nlcpy.request.flush()
                return ret
        else:
            return getattr(xp, op).accumulate(in1, dtype=dtype, axis=axis)


@pytest.mark.fast_math
@testing.with_requires('numpy>=1.19')
@testing.with_requires('numpy<1.20')
class TestAccumulate(unittest.TestCase):

    shapes = ((4,), (2, 2),)
    axes = (0,)
    shapes2 = ((3, 2, 4),)
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
        return execute_ufunc(xp, op, in1, axis=axis, dtype=dtype)

    @testing.for_dtypes("il")
    @testing.numpy_nlcpy_raises()
    def test_accumulate_power_with_negative_integer(self, xp, dtype):
        xp.power.accumulate(xp.array([1, -1], dtype=dtype))

    def test_accumulate_not_supported_dtype(self):
        for op in (
                'divide', 'true_divide', 'arctan2', 'logaddexp', 'logaddexp2',
                'floor_divide', 'mod', 'remainder', 'power'):
            try:
                getattr(nlcpy, op).accumulate([True])
                raise Exception
            except TypeError:
                pass

    @testing.numpy_nlcpy_array_equal()
    def test_accumulate_not_contiguous(self, xp):
        a = xp.moveaxis(xp.arange(24).reshape(2, 3, 4), 0, 1)
        return xp.add.accumulate(a, axis=2)

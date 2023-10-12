#
# * The source code in this file is based on the soure code of CuPy.
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
# # CuPy License #
#
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

import itertools
import numpy
import unittest
import pytest
import warnings

import nlcpy
from nlcpy import testing
from nlcpy.testing.types import all_types
from nlcpy.testing.types import negative_types
from nlcpy.testing.types import float_types
from nlcpy.testing.types import no_bool_no_uint_types
from nlcpy.testing.types import no_complex_types
from nlcpy.testing.types import negative_no_complex_types
from nlcpy.testing.types import complex_types


shapes = ([3], [2, 3], [4, 2, 3])
values = (
    [-2, 2],
    [[-2, -1], [1, 2]],
    [[[-2, -1]], [[1, 2]]],
)


@testing.parameterize(*(
    testing.product({
        'arg1': ([testing.shaped_arange(s, numpy, dtype=d)
                  for s, d in itertools.product(shapes, all_types)
                  ] + [0, 0.0j, 0j, 2, 2.0, 2j, True, False]),
        'name': ['conj', 'angle', 'real', 'imag'],
    }) + testing.product({
        'arg1': ([numpy.array(v, dtype=d)
                  for v, d in itertools.product(values, negative_types)
                  ] + [0, 0.0j, 0j, 2, 2.0, 2j, -2, -2.0, -2j, True, False]),
        'name': ['angle'],
    }) + testing.product({
        'arg1': ([testing.shaped_arange(s, numpy, dtype=d) + 1
                  for s, d in itertools.product(shapes, all_types)
                  ] + [2, 2.0, 2j, True]),
        'name': ['reciprocal'],
    })
))
class TestArithmeticUnary(unittest.TestCase):

    @testing.numpy_nlcpy_allclose(atol=1e-5)
    def test_unary(self, xp):
        arg1 = self.arg1
        if isinstance(arg1, numpy.ndarray):
            arg1 = xp.asarray(arg1)
        y = getattr(xp, self.name)(arg1)

        is_over_1_13 = testing.numpy_satisfies('>=1.13.0')
        if is_over_1_13 and self.name in ('real', 'imag'):
            # From NumPy>=1.13, some functions return Python scalars for Python
            # scalar inputs.
            # We need to convert them to arrays to compare with nlcpy outputs.
            if xp is numpy and isinstance(arg1, (bool, int, float, complex)):
                y = xp.asarray(y)

            if xp is nlcpy and isinstance(arg1, bool):
                y = y.astype(int)

        return y


class TestComplex(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_equal()
    def test_real_ndarray(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        return x.real is x

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_equal()
    def test_real(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        return xp.real(x) is x

    @testing.for_complex_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_imag_ndarray(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        y = x.imag
        x += 1 + 1j
        return y

    @testing.for_complex_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_imag(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        y = xp.imag(x)
        x += 1 + 1j
        return y

    @testing.for_complex_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_imag_setter(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        x.imag = 10
        return x

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_raises()
    def test_imag_setter_no_complex(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        x.imag = 10

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_real_setter(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        x.real = 10
        return x

    @testing.for_complex_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_imag_setter_array(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        y = testing.shaped_arange((2, 3), xp, dtype=float)
        x.imag = y
        return x

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_raises()
    def test_imag_setter_array_no_complex(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        y = testing.shaped_arange((2, 3), xp, dtype=float)
        x.imag = y

    @testing.for_complex_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_imag_setter_complex_array(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        y = testing.shaped_arange((2, 3), xp, dtype='c16')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            x.imag = y
        return x

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_real_setter_array(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        y = testing.shaped_arange((2, 3), xp, dtype=float)
        x.real = y
        return x

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_real_setter_complex_array(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype=dtype)
        y = testing.shaped_arange((2, 3), xp, dtype='c16')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            x.real = y
        return x


@testing.parameterize(*(
    testing.product({
        'arg1': [testing.shaped_arange(s, numpy, dtype=d)
                 for s, d in itertools.product(shapes, no_bool_no_uint_types)
                 ] + [0, 0.0, 0j, 2, 2.0, 2j],
        'arg2': [testing.shaped_reverse_arange(s, numpy, dtype=d)
                 for s, d in itertools.product(shapes, no_bool_no_uint_types)
                 ] + [0, 0.0, 0j, 2, 2.0, 2j],
        'name': ['add', 'multiply', 'subtract'],
    })
    + testing.product({
        'arg1': [numpy.array(v, dtype=d)
                 for v, d in itertools.product(values, negative_types)
                 ] + [0, 0.0, 0j, 2, 2.0, 2j, -2, -2.0, -2j, True, False],
        'arg2': [numpy.array(v, dtype=d)
                 for v, d in itertools.product(values, negative_types)
                 ] + [0, 0.0, 0j, 2, 2.0, 2j, -2, -2.0, -2j, True, False],
        'name': ['divide', 'true_divide', 'subtract'],
    })
    + testing.product({
        'arg1': [numpy.array(v, dtype=d)
                 for v, d in itertools.product(values, float_types)
                 ] + [0.0, 2.0, -2.0],
        'arg2': [numpy.array(v, dtype=d)
                 for v, d in itertools.product(values, float_types)
                 ] + [0.0, 2.0, -2.0],
        'name': ['power', 'true_divide', 'subtract'],
    }) + testing.product({
        'arg1': [testing.shaped_arange(s, numpy, dtype=d)
                 for s, d in itertools.product(shapes, no_complex_types)
                 ] + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
        'arg2': [testing.shaped_reverse_arange(s, numpy, dtype=d)
                 for s, d in itertools.product(shapes, no_complex_types)
                 ] + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
        'name': ['floor_divide', 'fmod', 'remainder'],
    }) + testing.product({
        'arg1': [numpy.array(v, dtype=d)
                 for v, d in itertools.product(values, negative_no_complex_types)
                 ] + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
        'arg2': [numpy.array(v, dtype=d)
                 for v, d in itertools.product(values, negative_no_complex_types)
                 ] + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
        'name': ['floor_divide', 'fmod', 'remainder'],
    })
))
class TestArithmeticBinary(unittest.TestCase):

    @testing.numpy_nlcpy_allclose(atol=1e-4)
    @pytest.mark.no_fast_math
    def test_binary(self, xp):
        arg1 = self.arg1
        arg2 = self.arg2
        np1 = numpy.asarray(arg1)
        np2 = numpy.asarray(arg2)
        dtype1 = np1.dtype
        dtype2 = np2.dtype

        if self.name == 'power':
            # TODO: Fix this: xp.power(0j, 0)
            #     numpy => 1+0j
            #     nlcpy => nan + nanj
            c_arg1 = dtype1 in complex_types
            if c_arg1 and (np1 == 0j).any() and (np2 == 0).any():
                return xp.array(True)

        if isinstance(arg1, numpy.ndarray):
            arg1 = xp.asarray(arg1)
        if isinstance(arg2, numpy.ndarray):
            arg2 = xp.asarray(arg2)

        # NumPy>=1.13.0 does not support subtraction between booleans
        # TODO: Write a separate test to check both NumPy and nlcpy
        # raise TypeError.
        if testing.numpy_satisfies('>=1.13.0') and self.name == 'subtract':
            if dtype1 == numpy.bool_ and dtype2 == numpy.bool_:
                return xp.array(True)

        func = getattr(xp, self.name)
        with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
            y = func(arg1, arg2)
            nlcpy.request.flush()

        # NumPy returns different values (nan/inf) on division by zero
        # depending on the architecture.
        # As it is not possible for nlcpy to replicate this behavior, we ignore
        # the difference here.
        if self.name in ('floor_divide', 'remainder'):
            if y.dtype in (float_types + complex_types) and (np2 == 0).any():
                y = xp.asarray(y)
                with testing.numpy_nlcpy_errstate(invalid='ignore'):
                    y[y == numpy.inf] = numpy.nan
                    y[y == -numpy.inf] = numpy.nan
                    nlcpy.request.flush()

        return y


class TestArithmeticBinaryNanCheck(unittest.TestCase):

    @testing.for_complex_dtypes()
    @pytest.mark.no_fast_math
    def test_power_nan_check1(self, dtype):
        in1 = nlcpy.zeros(10, dtype=dtype)
        in2 = nlcpy.full(10, 0 + 0j, dtype=dtype)
        actual = nlcpy.power(in1, in2)
        desired = numpy.full(10, numpy.nan, dtype=dtype)
        desired.imag = numpy.nan
        assert numpy.array_equal(actual, desired, equal_nan=True)

    @testing.for_complex_dtypes()
    @pytest.mark.no_fast_math
    def test_power_nan_check2(self, dtype):
        in1 = nlcpy.zeros(10, dtype=dtype)
        in2 = nlcpy.full(10, 1 + 1j, dtype=dtype)
        actual = nlcpy.power(in1, in2)
        desired = numpy.full(10, numpy.nan, dtype=dtype)
        desired.imag = numpy.nan
        assert numpy.array_equal(actual, desired, equal_nan=True)

    @testing.for_complex_dtypes()
    def test_power_nan_check3(self, dtype):
        in1 = nlcpy.zeros(10, dtype=dtype)
        in2 = nlcpy.full(10, 0 + 1j, dtype=dtype)
        actual = nlcpy.power(in1, in2)
        desired = numpy.full(10, numpy.nan, dtype=dtype)
        desired.imag = numpy.nan
        assert numpy.array_equal(actual, desired, equal_nan=True)

    @testing.for_dtypes(float_types + complex_types)
    def test_floor_divide_nan_check_float_types(self, dtype):
        in1 = nlcpy.ones(10, dtype=dtype)
        in2 = nlcpy.zeros(10, dtype=dtype)
        with nlcpy.errstate(divide='ignore', invalid='ignore'):
            actual = nlcpy.floor_divide(in1, in2)
            desired = numpy.full(10, numpy.nan, dtype=dtype)
            assert numpy.array_equal(actual, desired, equal_nan=True)

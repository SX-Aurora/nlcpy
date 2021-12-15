#
# * The source code in this file is based on the soure code of CuPy.
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

import unittest
import numpy

from nlcpy import testing


class TestRavel(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_ravel(self, xp):
        data = testing.shaped_arange((2, 3, 4), xp)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return a.ravel()

    @testing.numpy_nlcpy_array_equal()
    def test_ravel_external(self, xp):
        data = testing.shaped_arange((2, 3, 4), xp)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return xp.ma.ravel(a)

    @testing.numpy_nlcpy_array_equal()
    def test_ravel_external_ndarray(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.ma.ravel(a)

    @testing.numpy_nlcpy_array_equal()
    def test_ravel_nomask(self, xp):
        data = testing.shaped_arange((2, 3, 4), xp)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, fill_value=fill_value)
        return a.ravel()

    @testing.numpy_nlcpy_array_equal()
    def test_ravel_copied(self, xp):
        data = testing.shaped_arange((4,), xp)
        mask = [0, 1, 0, 1]
        fill_value = testing.shaped_arange((4,), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, hard_mask=True)
        b = a.ravel()
        a[:] = 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_transposed_ravel(self, xp):
        data = testing.shaped_arange((2, 3, 4), xp)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        a = a.transpose(2, 0, 1)
        return a.ravel()


@testing.parameterize(
    {'repeats': 0, 'axis': None},
    {'repeats': 2, 'axis': None},
    {'repeats': 2, 'axis': 1},
    {'repeats': 2, 'axis': -1},
    {'repeats': [0, 0, 0], 'axis': 1},
    {'repeats': [1, 2, 3], 'axis': 1},
    {'repeats': [1, 2, 3], 'axis': -2},
)
class TestRepeat(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_array_repeat(self, xp):
        data = testing.shaped_arange((2, 3, 4), xp)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return a.repeat(self.repeats, self.axis)

    @testing.numpy_nlcpy_array_equal()
    def test_array_repeat_external(self, xp):
        data = testing.shaped_arange((2, 3, 4), xp)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return xp.ma.repeat(a, self.repeats, self.axis)

    @testing.numpy_nlcpy_array_equal()
    def test_array_repeat_external_ndarray(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.ma.repeat(a, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': 0},
    {'repeats': 2},
)
class TestScalarRepeat(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_scalar_repeat(self, xp, dtype):
        a = xp.ma.array(1, dtype=dtype)
        return a.repeat(self.repeats, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_masked_scalar_repeat(self, xp, dtype):
        a = xp.ma.array(1, mask=True, dtype=dtype)
        return a.repeat(self.repeats, axis=0)


@testing.parameterize(
    {'repeats': [2], 'axis': None},
    {'repeats': [2], 'axis': 1},
)
class TestRepeatListBroadcast(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_array_repeat(self, xp):
        data = testing.shaped_arange((2, 3, 4), xp)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return a.repeat(self.repeats, self.axis)


@testing.parameterize(
    {'repeats': 0, 'axis': None},
    {'repeats': 2, 'axis': None},
    {'repeats': 2, 'axis': 0},
    {'repeats': [1, 2, 3, 4], 'axis': None},
    {'repeats': [1, 2, 3, 4], 'axis': 0},
)
class TestRepeat1D(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_array_repeat(self, xp):
        data = testing.shaped_arange((4,), xp)
        mask = testing.shaped_random((4,), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((4,), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return a.repeat(self.repeats, self.axis)


@testing.parameterize(
    {'repeats': [2], 'axis': None},
    {'repeats': [2], 'axis': 0},
)
class TestRepeat1DListBroadcast(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_array_repeat(self, xp):
        data = testing.shaped_arange((4,), xp)
        mask = testing.shaped_random((4,), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((4,), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return a.repeat(self.repeats, self.axis)


@testing.parameterize(
    {'repeats': -3, 'axis': None},
    {'repeats': [-3, -3], 'axis': 0},
    {'repeats': [1, 2, 3], 'axis': None},
    {'repeats': [1, 2], 'axis': 1},
    {'repeats': 2, 'axis': -4},
    {'repeats': 2, 'axis': 3},
    {'repeats': 1 + 1j, 'axis': None},
)
class TestRepeatFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_repeat_failure(self, xp):
        a = xp.ma.array(testing.shaped_arange((2, 3, 4), xp))
        a.repeat(self.repeats, self.axis)


class TestRepeatFailure2(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_repeat_failure_ndarray_axis(self, xp):
        a = xp.ma.array(testing.shaped_arange((2, 3, 4), xp))
        a.repeat(2, xp.array([1]))


class TestResize(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_resize(self, xp):
        a = xp.ma.array(xp.arange(12))
        a.resize(3, 4)

    @testing.numpy_nlcpy_array_equal()
    def test_resize_external(self, xp):
        data = xp.arange(12)
        mask = testing.shaped_random((12), xp, dtype=numpy.bool_)
        a = xp.ma.array(data, mask=mask)
        return xp.ma.resize(a, (3, 4))

    @testing.numpy_nlcpy_array_equal()
    def test_resize_external_ndarray(self, xp):
        a = xp.arange(12)
        return xp.ma.resize(a, (3, 4))

    @testing.numpy_nlcpy_array_equal()
    def test_resize_external_scalar(self, xp):
        a = xp.ma.array(1, mask=True)
        return xp.ma.resize(a, 1)

    @testing.numpy_nlcpy_array_equal()
    def test_resize_external_scalar2(self, xp):
        a = xp.ma.array(1, mask=False)
        return xp.ma.resize(a, 1)

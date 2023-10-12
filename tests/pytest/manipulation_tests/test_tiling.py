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

import unittest
import numpy
import nlcpy
from nlcpy import testing


@testing.parameterize(
    {'reps': 0},
    {'reps': 1},
    {'reps': 2},
    {'reps': (0, 1)},
    {'reps': (2, 3)},
    {'reps': (2, 3, 4, 5)},
    {'reps': ()},
    {'reps': numpy.array(0)},
    {'reps': (numpy.array(0),)},
    {'reps': numpy.array([2, 3])},
)
class TestTile(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_array_tile(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.tile(x, self.reps)

    @testing.numpy_nlcpy_array_equal()
    def test_array_2d_tile(self, xp):
        if type(self.reps) in (list, tuple):
            if len(self.reps) > 2:
                return True
        elif isinstance(self.reps, numpy.ndarray):
            if self.reps.ndim > 2:
                return True
        x = testing.shaped_arange((2, 3), xp)
        return xp.tile(x, self.reps)

    @testing.numpy_nlcpy_array_equal()
    def test_tile_list(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp).tolist()
        return xp.tile(x, self.reps)


class TestTileFailure(unittest.TestCase):

    @testing.numpy_nlcpy_raises()
    def test_tile_failure(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        xp.tile(x, -3)

    def test_tile_reps_np_array_above_1d(self):
        x = nlcpy.empty((2, 3, 4))
        reps = numpy.empty((2, 2))
        with self.assertRaises(ValueError):
            nlcpy.tile(x, reps)

    def test_tile_reps_tuple_above_1d(self):
        x = nlcpy.empty((2, 3, 4))
        with self.assertRaises(ValueError):
            nlcpy.tile(x, ((0, 1), (2, 3)))
        with self.assertRaises(ValueError):
            nlcpy.tile(x, ((), ()))
        with self.assertRaises(TypeError):
            nlcpy.tile(x, ((0,), (1,)))
        with self.assertRaises(ValueError):
            nlcpy.tile(x, (((0,),), ((1,),)))
        with self.assertRaises(ValueError):
            nlcpy.tile(x, (((0, 1), (0, 1)), ((2, 3), (2, 3))))
        with self.assertRaises(ValueError):
            val = numpy.array(1)
            nlcpy.tile(x, (((0, 1), val),))
        with self.assertRaises(ValueError):
            val = numpy.array([])
            nlcpy.tile(x, (((0, 1), val),))
        with self.assertRaises(TypeError):
            val = numpy.array([0,])
            nlcpy.tile(x, (val,))
        with self.assertRaises(TypeError):
            nlcpy.tile(x, (0.1, 1.1))
        with self.assertRaises(TypeError):
            nlcpy.tile(x, (0.1j, 1.1j))


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
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_array_repeat(self, xp, order):
        x = testing.shaped_arange((2, 3, 4), xp, order=order)
        return xp.repeat(x, self.repeats, self.axis)

    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_array_repeat2(self, xp, order):
        x = testing.shaped_arange((2, 3, 4), xp, order=order)
        return x.repeat(self.repeats, self.axis)


@testing.parameterize(
    {'repeats': 0},
    {'repeats': 2},
)
class TestScalarRepeat(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_scalar_repeat(self, xp, dtype):
        a = xp.array(1, dtype=dtype)
        return xp.repeat(a, self.repeats, axis=0)


@testing.parameterize(
    {'repeats': [2], 'axis': None},
    {'repeats': [2], 'axis': 1},
)
class TestRepeatListBroadcast(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.repeat(x, self.repeats, self.axis)


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
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': [2], 'axis': None},
    {'repeats': [2], 'axis': 0},
)
class TestRepeat1DListBroadcast(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, self.repeats, self.axis)


class TestRepeatNotContiguous(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((3, 4, 5), xp)
        x = xp.moveaxis(x, 0, 1)
        return xp.repeat(x, [1, 2, 3], axis=1)


@testing.parameterize(
    {'repeats': -3, 'axis': None},
    {'repeats': [-3, -3], 'axis': 0},
    {'repeats': [1, 2, 3], 'axis': None},
    {'repeats': [1, 2], 'axis': 1},
    {'repeats': 2, 'axis': -4},
    {'repeats': 2, 'axis': 3},
    {'repeats': 1 + 1j, 'axis': None},
    {'repeats': [[0, 1], [2, 3]], 'axis': None},
)
class TestRepeatFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_repeat_failure(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        xp.repeat(x, self.repeats, self.axis)


class TestRepeatFailure2(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_repeat_failure_ndarray_axis(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        xp.repeat(x, 2, xp.array([1]))

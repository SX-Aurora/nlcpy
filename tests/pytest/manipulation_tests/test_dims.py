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
import pytest

import numpy
import nlcpy
from nlcpy import testing


class TestDims(unittest.TestCase):

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_broadcast_to(self, xp, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((3, 1, 4), xp, dtype)
        b = xp.broadcast_to(a, (2, 3, 3, 4))
        return b

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_raises()
    def test_broadcast_to_fail(self, xp, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((3, 1, 4), xp, dtype)
        xp.broadcast_to(a, (1, 3, 4))

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_raises()
    def test_broadcast_to_short_shape(self, xp, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((1, 3, 4), xp, dtype)
        xp.broadcast_to(a, (3, 4))

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_broadcast_to_numpy19(self, xp, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((3, 1, 4), xp, dtype)
        if xp is nlcpy:
            b = xp.broadcast_to(a, (2, 3, 3, 4))
        else:
            dummy = xp.empty((2, 3, 3, 4))
            b, _ = xp.broadcast_arrays(a, dummy)
        return b

    @testing.for_all_dtypes()
    def test_broadcast_to_fail_numpy19(self, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((3, 1, 4), nlcpy, dtype)
        with self.assertRaises(ValueError):
            nlcpy.broadcast_to(a, (1, 3, 4))

    @testing.for_all_dtypes()
    def test_broadcast_to_short_shape_numpy19(self, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((1, 3, 4), nlcpy, dtype)
        with self.assertRaises(ValueError):
            nlcpy.broadcast_to(a, (3, 4))

    @testing.numpy_nlcpy_array_equal()
    def test_expand_dims0(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, 0)

    @testing.numpy_nlcpy_array_equal()
    def test_expand_dims1(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, 1)

    @testing.numpy_nlcpy_array_equal()
    def test_expand_dims2(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, 2)

    @testing.numpy_nlcpy_array_equal()
    def test_expand_dims_negative1(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, -2)

    @testing.numpy_nlcpy_raises()
    def test_expand_dims_negative2(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, -4)

    @testing.numpy_nlcpy_array_equal()
    def test_expand_dims_tuple_axis(self, xp):
        a = testing.shaped_arange((2, 2, 2), xp)
        return [xp.expand_dims(a, axis) for axis in [
            (0, 1, 2),
            (0, -1, -2),
            (0, 3, 5),
            (0, -3, -5),
            (),
            (1,),
        ]]

    def test_expand_dims_out_of_range(self):
        for xp in (numpy, nlcpy):
            a = testing.shaped_arange((2, 2, 2), xp)
            for axis in [(1, -6), (1, 5)]:
                with pytest.raises(numpy.AxisError):
                    xp.expand_dims(a, axis)

    def test_expand_dims_repeated_axis(self):
        for xp in (numpy, nlcpy):
            a = testing.shaped_arange((2, 2, 2), xp)
            with pytest.raises(ValueError):
                xp.expand_dims(a, (1, 1))

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze()

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze2(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.squeeze(a)

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_int_axis1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=2)

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_int_axis2(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return xp.squeeze(a, axis=-3)

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_squeeze_int_axis_failure1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        xp.squeeze(a, axis=-9)

    def test_squeeze_int_axis_failure2(self):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), nlcpy)
        with self.assertRaises(nlcpy.core.error._AxisError):
            nlcpy.squeeze(a, axis=-9)

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_tuple_axis1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return xp.squeeze(a, axis=(2, 4))

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_tuple_axis2(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return xp.squeeze(a, axis=(-4, -3))

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_tuple_axis3(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return xp.squeeze(a, axis=(4, 2))

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_tuple_axis4(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return xp.squeeze(a, axis=())

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_squeeze_tuple_axis_failure1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        xp.squeeze(a, axis=(-9,))

    @testing.numpy_nlcpy_raises()
    def test_squeeze_tuple_axis_failure2(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        xp.squeeze(a, axis=(2, 2))

    def test_squeeze_tuple_axis_failure3(self):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), nlcpy)
        with self.assertRaises(nlcpy.core.error._AxisError):
            nlcpy.squeeze(a, axis=(-9,))

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_scalar1(self, xp):
        a = testing.shaped_arange((), xp)
        return xp.squeeze(a, axis=0)

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_scalar2(self, xp):
        a = testing.shaped_arange((), xp)
        return xp.squeeze(a, axis=-1)

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_squeeze_scalar_failure1(self, xp):
        a = testing.shaped_arange((), xp)
        xp.squeeze(a, axis=-2)

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_squeeze_scalar_failure2(self, xp):
        a = testing.shaped_arange((), xp)
        xp.squeeze(a, axis=1)

    def test_squeeze_scalar_failure3(self):
        a = testing.shaped_arange((), nlcpy)
        with self.assertRaises(nlcpy.core.error._AxisError):
            nlcpy.squeeze(a, axis=-2)

    def test_squeeze_scalar_failure4(self):
        a = testing.shaped_arange((), nlcpy)
        with self.assertRaises(nlcpy.core.error._AxisError):
            nlcpy.squeeze(a, axis=1)

    @testing.numpy_nlcpy_raises()
    def test_squeeze_failure(self, xp):
        a = testing.shaped_arange((2, 1, 3, 4), xp)
        xp.squeeze(a, axis=2)

    @testing.numpy_nlcpy_array_equal()
    def test_external_squeeze(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return xp.squeeze(a)


@testing.parameterize(
    {'shapes': [(), ()]},
    {'shapes': [(0,), (0,)]},
    {'shapes': [(1,), (1,)]},
    {'shapes': [(2,), (2,)]},
    {'shapes': [(0,), (1,)]},
    {'shapes': [(2, 3), (1, 3)]},
    {'shapes': [(2, 1, 3, 4), (3, 1, 4)]},
    {'shapes': [(4, 3, 2, 3), (2, 3)]},
    {'shapes': [(2, 0, 1, 1, 3), (2, 1, 0, 0, 3)]},
    {'shapes': [(0, 1, 1, 3), (2, 1, 0, 0, 3)]},
    {'shapes': [(0, 1, 1, 0, 3), (5, 2, 0, 1, 0, 0, 3), (2, 1, 0, 0, 0, 3)]},
)
class TestBroadcastArrays(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_broadcast_arrays(self, xp, dtype, order):
        arrays = [testing.shaped_arange(s, xp, dtype, order) for s in self.shapes]
        return xp.broadcast_arrays(*arrays)

    @testing.numpy_nlcpy_array_equal()
    def test_broadcast_arrays_with_list_input(self, xp):
        arrays = [testing.shaped_arange(s, xp).tolist() for s in self.shapes]
        return xp.broadcast_arrays(*arrays)


@testing.parameterize(
    {'shapes': [(3,), (2,)]},
    {'shapes': [(3, 2), (2, 3)]},
    {'shapes': [(3, 2), (3, 4)]},
    {'shapes': [(0, ), (2, )]},
)
class TestBroadcastArraysInvalidShape(unittest.TestCase):

    @testing.numpy_nlcpy_raises()
    def test_broadcast_arrays_invalid_shape(self, xp):
        arrays = [testing.shaped_arange(s, xp) for s in self.shapes]
        xp.broadcast_arrays(*arrays)


class TestBroadcastArraysFailure(unittest.TestCase):

    def test_broadcast_arrays_subok(self):
        try:
            nlcpy.broadcast_arrays(nlcpy.empty([1, 3]), nlcpy.empty([2, 1]), subok=True)
        except NotImplementedError:
            return
        raise Exception


class TestAtLeast(unittest.TestCase):

    def check_atleast(self, func, xp):
        a = testing.shaped_arange((), xp, 'i')
        b = testing.shaped_arange((2,), xp, 'f')
        c = testing.shaped_arange((3, 4), xp, 'd')
        d = testing.shaped_arange((4, 2, 3), xp, 'F', order='F')
        e = 1
        f = xp.float32(1)
        return func(a, b, c, d, e, f)

    @testing.numpy_nlcpy_array_equal()
    def test_atleast_1d(self, xp):
        return self.check_atleast(xp.atleast_1d, xp)

    @testing.numpy_nlcpy_array_equal()
    def test_atleast_1d2(self, xp):
        a = testing.shaped_arange((4, 2, 3), xp)
        return xp.atleast_1d(a)

    @testing.numpy_nlcpy_array_equal()
    def test_atleast_2d(self, xp):
        return self.check_atleast(xp.atleast_2d, xp)

    @testing.numpy_nlcpy_array_equal()
    def test_atleast_2d2(self, xp):
        a = testing.shaped_arange((4, 2, 3), xp)
        return xp.atleast_2d(a)

    @testing.numpy_nlcpy_array_equal()
    def test_atleast_3d(self, xp):
        return self.check_atleast(xp.atleast_3d, xp)

    @testing.numpy_nlcpy_array_equal()
    def test_atleast_3d2(self, xp):
        a = testing.shaped_arange((4, 2, 3), xp)
        return xp.atleast_3d(a)

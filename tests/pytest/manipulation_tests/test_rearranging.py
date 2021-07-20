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

from nlcpy import testing


@testing.parameterize(*(
    testing.product({
        'params': [
            ([10], None),
            ([4, 5], None),
            ([3, 5, 4, 2, 6], None),
            ([8], 0),
            ([5, 1, 2, 5, 4], 0),
            ([5, 1, 2, 5, 4], 1),
            ([5, 1, 2, 5, 4], 4),
            ([5, 1, 2, 5, 4], -2),
            ([3, 4, 5, 6, 7], (1, 3, 2)),
            ([4, 2, 1, 4, 5], (-1, 2, 3)),
            ([3, 4, 5, 6, 7], [1, 3, 2]),
            ([4, 2, 1, 4, 5], [-1, 2, 3]),
        ],
    })
))
class TestFlip(unittest.TestCase):
    @testing.for_all_dtypes(name='dtype_m')
    @testing.numpy_nlcpy_array_equal()
    def test_flip(self, xp, dtype_m):
        m = testing.shaped_arange(self.params[0], xp, dtype_m)
        axis = self.params[1]
        return xp.flip(m=m, axis=axis)


class TestFlipScalar(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_flip(self, xp):
        return xp.flip(1, None)


@testing.parameterize(*(
    testing.product({
        'params': [
            ([5, 1, 2, 5, 4], 4),
            ([5, 1, 2, 5, 4], -2),
            ([3, 4, 5, 6, 7], (1, 3, 2)),
            ([4, 2, 1, 4, 5], (-1, 2, 3)),
        ],
    })
))
class TestFlipArrayAxis(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_flip_array_axis(self, xp):
        m = testing.shaped_arange(self.params[0], xp)
        axis = xp.array(self.params[1], 'int64')
        return xp.flip(m, axis)


class TestFlipFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_flip_repeated_axis(self, xp):
        return xp.flip([1], (0, 0))

    @testing.numpy_nlcpy_raises()
    def test_flip_incompatible_axis1(self, xp):
        return xp.flip([1], 1)

    @testing.numpy_nlcpy_raises()
    def test_flip_incompatible_axis2(self, xp):
        return xp.flip([1], -2)

    @testing.numpy_nlcpy_raises()
    def test_flip_incompatible_axis3(self, xp):
        return xp.flip(xp.arange(24).reshape(1, 2, 3, 4), xp.arange(4).reshape(2, 2))


@testing.parameterize(*(
    testing.product({
        'shape': [
            [10, ],
            [4, 5],
            [1, 7, 4],
            [4, 4, 4, 4],
            [3, 4, 5, 6, 7],
            [3, 5, 4, 2, 6],
            [5, 1, 2, 5, 4],
        ],
    })
))
class TestFlipud(unittest.TestCase):
    @testing.for_all_dtypes(name='dtype_m')
    @testing.numpy_nlcpy_array_equal()
    def test_flipud(self, xp, dtype_m):
        m = testing.shaped_arange(self.shape, xp, dtype_m)
        return xp.flipud(m)


class TestFlipudFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_flipud_0d(self, xp):
        return xp.flipud(1)


@testing.parameterize(*(
    testing.product({
        'shape': [
            [4, 5],
            [1, 7, 4],
            [4, 4, 4, 4],
            [3, 4, 5, 6, 7],
            [3, 5, 4, 2, 6],
            [5, 1, 2, 5, 4],
        ],
    })
))
class TestFliplr(unittest.TestCase):
    @testing.for_all_dtypes(name='dtype_m')
    @testing.numpy_nlcpy_array_equal()
    def test_fliplr(self, xp, dtype_m):
        m = testing.shaped_arange(self.shape, xp, dtype_m)
        return xp.fliplr(m)


class TestFliplrFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_fliplr_0d(self, xp):
        return xp.fliplr(1)

    @testing.numpy_nlcpy_raises()
    def test_fliplr_1d(self, xp):
        return xp.fliplr([1])


class TestRoll(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(accept_error=TypeError)
    def test_roll(self, xp, dtype):
        x = xp.arange(10, dtype)
        return xp.roll(x, 2)

    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_roll2(self, xp, dtype, order):
        x = testing.shaped_arange((5, 2), xp, dtype, order)
        return xp.roll(x, 1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_negative(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, -2)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_with_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 1, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_with_negative_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 1, axis=-1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_double_shift(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)
        return xp.roll(x, 35)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_double_shift_with_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 11, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_zero_array(self, xp, dtype):
        x = testing.shaped_arange((), xp, dtype)
        return xp.roll(x, 5)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_scalar_shift_multi_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 1, axis=(0, 1))

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_scalar_shift_duplicate_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 1, axis=(0, 0))

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_large_shift(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 50, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_multi_shift_multi_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, (2, 1), axis=(0, 1))

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_multi_shift_multi_axis_with_negative_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, (2, 1), axis=(0, -1))

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_multi_shift_multi_axis_with_same_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, (2, 1), axis=(1, -1))

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_multi_shift_scalar_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, (2, 1, 3), axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_roll_multi_shift_axis_none(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, (2, 1, 3), axis=None)


class TestRollFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_roll_invalid_shift(self, xp):
        x = testing.shaped_arange((5, 2), xp)
        xp.roll(x, 'str', axis=0)

    @testing.numpy_nlcpy_raises()
    def test_roll_shape_mismatch(self, xp):
        x = testing.shaped_arange((5, 2, 3), xp)
        xp.roll(x, (2, 2, 2), axis=(0, 1))

    @testing.numpy_nlcpy_raises()
    def test_roll_invalid_axis1(self, xp):
        x = testing.shaped_arange((5, 2), xp)
        xp.roll(x, 1, axis=2)

    @testing.numpy_nlcpy_raises()
    def test_roll_invalid_axis2(self, xp):
        x = testing.shaped_arange((5, 2), xp)
        xp.roll(x, 1, axis=-3)

    @testing.numpy_nlcpy_raises()
    def test_roll_invalid_axis_length(self, xp):
        x = testing.shaped_arange((5, 2, 2), xp)
        xp.roll(x, shift=(1, 0), axis=(0, 1, 2))

    @testing.numpy_nlcpy_raises()
    def test_roll_invalid_axis_type(self, xp):
        x = testing.shaped_arange((5, 2), xp)
        xp.roll(x, 2, axis='0')

    @testing.numpy_nlcpy_raises()
    def test_roll_invalid_negative_axis1(self, xp):
        x = testing.shaped_arange((5, 2), xp)
        xp.roll(x, 1, axis=-3)

    @testing.numpy_nlcpy_raises()
    def test_roll_invalid_negative_axis2(self, xp):
        x = testing.shaped_arange((5, 2), xp)
        xp.roll(x, 1, axis=(1, -3))

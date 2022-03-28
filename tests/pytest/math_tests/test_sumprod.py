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


@testing.parameterize(*(
    testing.product({
        'shape': [(2,), (2, 3), (2, 3, 4)],
    })
))
class TestSumprod1(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_sum_all(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return xp.sum(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_external_sum_all(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return xp.sum(a)


class TestSumprod2(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_sum_all2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype)
        return xp.sum(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_sum_all_transposed(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(2, 0, 1)
        return xp.sum(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_sum_all_transposed2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype).transpose(2, 0, 1)
        return xp.sum(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_sum_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.sum(a, axis=1)

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_allclose(rtol=1e-5)
    def test_sum_axis_huge(self, xp):
        a = testing.shaped_random((2048, 1, 1024), xp, 'f4')
        a = xp.broadcast_to(a, (2048, 1024, 1024))
        return xp.sum(a, axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_external_sum_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.sum(a, axis=1)

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_nlcpy_allclose()
    def test_sum_axis2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype)
        return xp.sum(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(contiguous_check=False)
    def test_sum_axis_transposed(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(2, 0, 1)
        return xp.sum(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(contiguous_check=False)
    def test_sum_axis_transposed2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype).transpose(2, 0, 1)
        return xp.sum(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_sum_axes(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return xp.sum(a, axis=(1, 3))

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_sum_axes2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40, 50), xp, dtype)
        return xp.sum(a, axis=(1, 3))

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_sum_axes3(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return xp.sum(a, axis=(0, 2, 3))

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_sum_axes4(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40, 50), xp, dtype)
        return xp.sum(a, axis=(0, 2, 3))

    @testing.for_all_dtypes_combination(names=['src_dtype', 'dst_dtype'])
    @testing.numpy_nlcpy_allclose()
    def test_sum_dtype(self, xp, src_dtype, dst_dtype):
        if not numpy.can_cast(src_dtype, dst_dtype):
            return xp.array([])  # skip
        a = testing.shaped_arange((2, 3, 4), xp, src_dtype)
        return xp.sum(a, dtype=dst_dtype)

    @testing.numpy_nlcpy_allclose()
    def test_sum_keepdims(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.sum(a, axis=1, keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_sum_out(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty((2, 4), dtype=dtype)
        xp.sum(a, axis=1, out=b)
        return b

    def test_sum_out_wrong_shape(self):
        a = testing.shaped_arange((2, 3, 4))
        b = nlcpy.empty((2, 3))
        with self.assertRaises(NotImplementedError):
            nlcpy.sum(a, axis=1, out=b)


axes = [0, 1, 2]


@testing.parameterize(*testing.product({'axis': axes}))
class TestCumsum(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_cumsum(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.cumsum(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_cumsum_2dim(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.cumsum(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(contiguous_check=False)
    def test_cumsum_axis(self, xp, dtype):
        n = len(axes)
        a = testing.shaped_arange(tuple(range(4, 4 + n)), xp, dtype)
        return xp.cumsum(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(accept_error=nlcpy.core.error._AxisError)
    def test_cumsum_axis_empty(self, xp, dtype):
        n = len(axes)
        a = testing.shaped_arange(tuple(range(0, n)), xp, dtype)
        return xp.cumsum(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_invalid_axis_lower1(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.cumsum(a, axis=-a.ndim - 1)

    @testing.for_all_dtypes()
    def test_invalid_axis_lower2(self, dtype):
        a = testing.shaped_arange((4, 5), nlcpy, dtype)
        with self.assertRaises(nlcpy.core.error._AxisError):
            return nlcpy.cumsum(a, axis=-a.ndim - 1)

    @testing.for_all_dtypes()
    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_invalid_axis_upper1(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.cumsum(a, axis=a.ndim + 1)

    @testing.for_all_dtypes()
    def test_invalid_axis_upper2(self, dtype):
        a = testing.shaped_arange((4, 5), nlcpy, dtype)
        with self.assertRaises(nlcpy.core.error._AxisError):
            return nlcpy.cumsum(a, axis=a.ndim + 1)

    @testing.numpy_nlcpy_allclose()
    def test_cumsum_arraylike(self, xp):
        return xp.cumsum((1, 2, 3))

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_cumsum_numpy_array(self, xp, dtype):
        a_numpy = numpy.arange(8, dtype=dtype)
        return xp.cumsum(a_numpy)


@testing.with_requires('numpy>=1.14')  # NumPy issue #9251
class TestDiff(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_diff_1dim(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.diff(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_diff_1dim_with_n(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.diff(a, n=3)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_diff_2dim_without_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.diff(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_diff_2dim_with_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.diff(a, axis=-2)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_diff_2dim_with_n_and_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.diff(a, 2, 1)

    @testing.with_requires('numpy>=1.16')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_diff_2dim_with_prepend(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        b = testing.shaped_arange((4, 1), xp, dtype)
        return xp.diff(a, axis=-1, prepend=b)

    @testing.with_requires('numpy>=1.16')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_diff_2dim_with_append(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        b = testing.shaped_arange((1, 5), xp, dtype)
        return xp.diff(a, axis=0, append=b, n=2)

    @testing.with_requires('numpy>=1.16')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_diff_2dim_with_scalar_append(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.diff(a, prepend=1, append=0)

    @testing.with_requires('numpy>=1.16')
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_allclose()
    def test_diff_4dim(self, xp, order):
        a = xp.asarray(testing.shaped_arange((2, 3, 4, 5), xp), order=order)
        return xp.diff(a)

    @testing.with_requires('numpy>=1.16')
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_allclose()
    def test_diff_4dim_axis1(self, xp, order):
        a = xp.asarray(testing.shaped_arange((2, 3, 4, 5), xp), order=order)
        return xp.diff(a, axis=1)

    @testing.with_requires('numpy>=1.16')
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_allclose()
    def test_diff_4dim_axis2(self, xp, order):
        a = xp.asarray(testing.shaped_arange((2, 3, 4, 5), xp), order=order)
        return xp.diff(a, axis=3)

    @testing.with_requires('numpy>=1.16')
    @testing.numpy_nlcpy_allclose()
    def test_diff_4dim_not_contiguous(self, xp):
        a = xp.moveaxis(testing.shaped_arange((2, 3, 4, 5), xp), 0, 1)
        return xp.diff(a, axis=1)

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

import nlcpy
from nlcpy import testing


class TestSort(unittest.TestCase):

    # Test ranks

    @testing.numpy_nlcpy_raises()
    def test_sort_zero_dim(self):
        a = testing.shaped_random((), nlcpy)
        with self.assertRaises(TypeError):
            a.sort()

    @testing.numpy_nlcpy_raises()
    def test_external_sort_zero_dim(self):
        a = testing.shaped_random((), nlcpy)
        with self.assertRaises(TypeError):
            return nlcpy.sort(a)

    @testing.numpy_nlcpy_array_equal()
    def test_sort_two_dim(self, xp):
        a = testing.shaped_random((2, 3), xp)
        a.sort()
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_external_sort_two_dim(self, xp):
        a = testing.shaped_random((2, 3), xp)
        return xp.sort(a)

    @testing.numpy_nlcpy_array_equal()
    def test_sort_three_dim(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort()
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_external_three_dim(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a)

    # Test dtypes

    @testing.for_all_dtypes(no_float16=True, no_bool=False, no_complex=False)
    @testing.numpy_nlcpy_allclose()
    def test_sort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        a.sort()
        return a

    @testing.for_all_dtypes(no_float16=True, no_bool=False, no_complex=False)
    @testing.numpy_nlcpy_allclose()
    def test_external_sort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.sort(a)

    @testing.for_dtypes([numpy.float16])
    def test_external_sort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), numpy, dtype)
        with self.assertRaises(TypeError):
            return nlcpy.sort(a)

    # Test contiguous arrays

    @testing.numpy_nlcpy_allclose()
    def test_sort_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)  # C contiguous view
        a.sort()
        return a

    @testing.numpy_nlcpy_allclose()
    def test_external_sort_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)  # C contiguous view
        return xp.sort(a)

    @testing.numpy_nlcpy_allclose()
    def test_external_sort_non_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)[::2]  # Non contiguous view
        return xp.sort(a)

    # Test axis

    @testing.numpy_nlcpy_array_equal()
    def test_sort_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=0)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_external_sort_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a, axis=0)

    @testing.numpy_nlcpy_array_equal()
    def test_sort_negative_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=-2)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_external_sort_negative_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a, axis=-2)

    @testing.numpy_nlcpy_array_equal()
    def test_external_sort_none_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a, axis=None)

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_sort_invalid_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=3)

    def test_sort_invalid_axis2(self):
        a = testing.shaped_random((2, 3, 3), nlcpy)
        with self.assertRaises(nlcpy.core.error._AxisError):
            a.sort(axis=3)

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_external_sort_invalid_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        xp.sort(a, axis=3)

    def test_external_sort_invalid_axis2(self):
        a = testing.shaped_random((2, 3, 3), nlcpy)
        with self.assertRaises(nlcpy.core.error._AxisError):
            nlcpy.sort(a, axis=3)

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_sort_invalid_negative_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=-4)

    def test_sort_invalid_negative_axis2(self):
        a = testing.shaped_random((2, 3, 3), nlcpy)
        with self.assertRaises(nlcpy.core.error._AxisError):
            a.sort(axis=-4)

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_external_sort_invalid_negative_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        xp.sort(a, axis=-4)

    def test_external_sort_invalid_negative_axis2(self):
        a = testing.shaped_random((2, 3, 3), nlcpy)
        with self.assertRaises(nlcpy.core.error._AxisError):
            nlcpy.sort(a, axis=-4)


@testing.parameterize(*testing.product({
    'external': [False, True],
}))
class TestArgsort(unittest.TestCase):

    def argsort(self, a, axis=-1, xp=numpy):
        if self.external:
            return xp.argsort(a, axis=axis)
        else:
            return a.argsort(axis=axis)

    # Test base cases

    @testing.for_all_dtypes(no_float16=True, no_bool=False, no_complex=False)
    @testing.numpy_nlcpy_array_equal()
    def test_argsort_zero_dim(self, xp, dtype):
        a = xp.array([])
        return self.argsort(a, xp=xp)

    @testing.for_all_dtypes(no_float16=True, no_bool=False, no_complex=False)
    @testing.numpy_nlcpy_array_equal()
    def test_argsort_one_dim(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return self.argsort(a, xp=xp)

    @testing.for_all_dtypes(no_float16=True, no_bool=False, no_complex=False)
    @testing.numpy_nlcpy_array_equal()
    def test_argsort_multi_dim(self, xp, dtype):
        a = testing.shaped_random((2, 3, 3), xp, dtype)
        return self.argsort(a, xp=xp)

    @testing.numpy_nlcpy_array_equal()
    def test_argsort_non_contiguous(self, xp):
        a = xp.array([1, 0, 2, 3])[::2]
        return self.argsort(a, xp=xp)

    # Test unsupported dtype

    @testing.for_dtypes([numpy.float16])
    def test_argsort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), numpy, dtype)
        with self.assertRaises(TypeError):
            return nlcpy.argsort(a, xp=nlcpy)

    # Test axis

    @testing.numpy_nlcpy_array_equal()
    def test_argsort_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return self.argsort(a, axis=0, xp=xp)

    @testing.numpy_nlcpy_array_equal()
    def test_argsort_negative_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return self.argsort(a, axis=2, xp=xp)

    @testing.numpy_nlcpy_array_equal()
    def test_argsort_none_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return self.argsort(a, axis=None, xp=xp)

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_argsort_invalid_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return self.argsort(a, axis=3, xp=xp)

    def test_argsort_invalid_axis2(self):
        a = testing.shaped_random((2, 3, 3), nlcpy)
        with self.assertRaises(nlcpy.core.error._AxisError):
            return self.argsort(a, axis=3, xp=nlcpy)

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_argsort_invalid_negative_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return self.argsort(a, axis=-4)

    def test_argsort_invalid_negative_axis2(self):
        a = testing.shaped_random((2, 3, 3), nlcpy)
        with self.assertRaises(nlcpy.core.error._AxisError):
            return self.argsort(a, axis=-4, xp=nlcpy)

    # Misc tests

    def test_argsort_original_array_not_modified_one_dim(self):
        a = testing.shaped_random((10,), nlcpy)
        b = nlcpy.array(a)
        self.argsort(a, xp=nlcpy)
        testing.assert_allclose(a, b)

    def test_argsort_original_array_not_modified_multi_dim(self):
        a = testing.shaped_random((2, 3, 3), nlcpy)
        b = nlcpy.array(a)
        self.argsort(a, xp=nlcpy)
        testing.assert_allclose(a, b)

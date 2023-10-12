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

import nlcpy
import numpy
from nlcpy import testing
from nlcpy.testing import condition


@testing.parameterize(*(
    testing.product({
        'pat': [
            ((10,), None),
            ((3, 3), 1),
            ((3, 4, 4), 0),
            ((5, 6, 6), -1),
            ((3, 6, 4, 5, 5), 2),
            ((4, 2, 6, 7, 5), 4),
        ],
        'ord': [None, numpy.inf, -numpy.inf, 0, 1, 2, -1, -2, 5, -4],
        'keepdims': [True, False],
    })
))
class TestNormVector(unittest.TestCase):
    @testing.for_dtypes("?ilILdD")
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_norm_vector(self, xp, dtype, order):
        shape, axis = self.pat
        x = xp.asarray(testing.shaped_random(shape, xp, dtype), order=order)
        args = dict()
        args["x"] = x
        if self.ord is not None:
            args["ord"] = self.ord
        if axis is not None:
            args["axis"] = axis
        if self.keepdims:
            args["keepdims"] = self.keepdims
        if self.ord in (-1, -2, -4):
            with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
                ret = xp.linalg.norm(**args)
                nlcpy.request.flush()
                return ret
        else:
            return xp.linalg.norm(**args)

    @testing.for_dtypes("fF")
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-6, rtol=1e-6)
    def test_norm_vector_single(self, xp, dtype, order):
        shape, axis = self.pat
        x = xp.asarray(testing.shaped_random(shape, xp, dtype), order=order)
        if self.ord in (-1, -2, -4):
            with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
                ret = xp.linalg.norm(x, self.ord, axis, self.keepdims)
                nlcpy.request.flush()
                return ret
        else:
            return xp.linalg.norm(x, self.ord, axis, self.keepdims)


@testing.parameterize(*(
    testing.product({
        'pat': [
            ((0, 3, 3), 1),
            ((3, 0, 3), 1),
        ],
        'ord': [None, 0, 1, 2, -1, -2, 5, -4],
        'keepdims': [True, False],
    })
))
class TestNormVectorZeroSizeArray(unittest.TestCase):
    @testing.for_dtypes("ilILdD")
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_norm_vector_zero_size_array(self, xp, dtype, order):
        shape, axis = self.pat
        x = xp.asarray(testing.shaped_random(shape, xp, dtype), order=order)
        if self.ord in (-1, -2, -4):
            with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
                ret = xp.linalg.norm(x, self.ord, axis, self.keepdims)
                nlcpy.request.flush()
                return ret
        else:
            return xp.linalg.norm(x, self.ord, axis, self.keepdims)

    @testing.for_dtypes("fF")
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-6, rtol=1e-6)
    def test_norm_vector_zero_size_array_single(self, xp, dtype, order):
        shape, axis = self.pat
        x = xp.asarray(testing.shaped_random(shape, xp, dtype), order=order)
        if self.ord in (-1, -2, -4):
            with testing.numpy_nlcpy_errstate(divide='ignore', invalid='ignore'):
                ret = xp.linalg.norm(x, self.ord, axis, self.keepdims)
                nlcpy.request.flush()
                return ret
        else:
            return xp.linalg.norm(x, self.ord, axis, self.keepdims)


@testing.parameterize(*(
    testing.product({
        'pat': [
            ((3, 3), None),
            ((2, 3, 3), (0, 1)),
            ((3, 4, 4), (-1, 0)),
            ((4, 5, 5), (1, 0)),
            ((4, 5, 3, 3), (1, -2)),
            ((3, 6, 4, 5, 5), (2, 4)),
            ((7, 3, 5, 5, 5), (3, 1)),
        ],
        'ord': [None, 'fro', 'nuc', numpy.inf, -numpy.inf, 1, 2, -1, -2],
        'keepdims': [True, False],
    })
))
class TestNormMatrix(unittest.TestCase):
    @testing.for_dtypes("?ilILdD")
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_norm_matrix(self, xp, dtype, order):
        shape, axis = self.pat
        x = xp.asarray(testing.shaped_random(shape, xp, dtype), order=order)
        return xp.linalg.norm(x, self.ord, axis, self.keepdims)

    @testing.for_dtypes("fF")
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-6, rtol=1e-6)
    def test_norm_matrix_single(self, xp, dtype, order):
        shape, axis = self.pat
        x = xp.asarray(testing.shaped_random(shape, xp, dtype), order=order)
        return xp.linalg.norm(x, self.ord, axis, self.keepdims)


class TestNormMatrixNotContiguous(unittest.TestCase):
    @condition.repeat(times=1000)
    @testing.numpy_nlcpy_allclose(atol=1e-4, rtol=1e-4)
    def test_norm_matrix_not_contiguous(self, xp):
        x = xp.arange(3 * 4 * 5).reshape(3, 4, 5).astype('f')
        x = xp.moveaxis(x, 0, 1)
        return xp.linalg.norm(x, 'fro', (1, 0))

    @condition.repeat(times=1000)
    @testing.numpy_nlcpy_allclose(atol=1e-4, rtol=1e-4)
    def test_norm_matrix_not_contiguous2(self, xp):
        x = xp.arange(10 * 10 * 10).reshape(10, 10, 10).astype('f')
        x = xp.moveaxis(x, 0, 1)
        return xp.linalg.norm(x, 'fro', (1, 0))

    @testing.numpy_nlcpy_allclose(atol=1e-4, rtol=1e-4)
    def test_norm_matrix_not_contiguous3(self, xp):
        x = xp.arange(4 * 5 * 5).reshape(4, 5, 5).astype('f')
        x = xp.moveaxis(x, 1, 2)
        return xp.linalg.norm(x, 'fro', (1, 0))

    @testing.numpy_nlcpy_allclose(atol=1e-4, rtol=1e-4)
    def test_norm_matrix_not_contiguous4(self, xp):
        x = xp.arange(3 * 4 * 5 * 5).reshape(3, 4, 5, 5).astype('f')
        x = xp.moveaxis(x, 2, 3)
        return xp.linalg.norm(x, 'fro', (0, 2))

    @testing.numpy_nlcpy_allclose(atol=1e-4, rtol=1e-4)
    def test_norm_matrix_not_contiguous_2d(self, xp):
        x = xp.arange(4 * 5).reshape(4, 5).astype('f')
        x = xp.moveaxis(x, 0, 1)
        return xp.linalg.norm(x, 'fro', (0, 1))


@testing.parameterize(*(
    testing.product({
        'pat': [
            ((0, 3, 3), (0, 1)),
            ((3, 5, 7, 9, 0), (0, 1)),
        ],
        'ord': [None, 'fro', 'nuc', 1, -1],
        'keepdims': [True, False],
    })
))
class TestNormMatrixZeroSizeArray(unittest.TestCase):
    @testing.for_dtypes("ilILdD")
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_norm_matrix_zero_size_array(self, xp, dtype, order):
        shape, axis = self.pat
        x = xp.asarray(testing.shaped_random(shape, xp, dtype), order=order)
        return xp.linalg.norm(x, self.ord, axis, self.keepdims)

    @testing.for_dtypes("fF")
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-6, rtol=1e-6)
    def test_norm_matrix_zero_size_array_single(self, xp, dtype, order):
        shape, axis = self.pat
        x = xp.asarray(testing.shaped_random(shape, xp, dtype), order=order)
        return xp.linalg.norm(x, self.ord, axis, self.keepdims)


class TestNormFailureAxis(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_norm_too_large_dim(self, xp):
        return xp.linalg.norm(xp.empty([2, 3, 4, 5]), xp.inf)

    @testing.numpy_nlcpy_raises()
    def test_norm_too_many_axis(self, xp):
        return xp.linalg.norm(xp.empty([2, 3, 4, 5]), axis=(0, 1, 2))

    @testing.numpy_nlcpy_raises()
    def test_norm_duplicated_axis(self, xp):
        return xp.linalg.norm(xp.empty([2, 3, 4, 5]), axis=(0, 0))

    @testing.numpy_nlcpy_raises()
    def test_norm_non_integer_axis(self, xp):
        return xp.linalg.norm(xp.empty([2, 3, 4, 5]), axis='A')

    @testing.numpy_nlcpy_raises()
    def test_norm_zero_size_reduction_1(self, xp):
        return xp.linalg.norm(xp.empty([3, 0, 2]), xp.inf, axis=(1, 0))

    @testing.numpy_nlcpy_raises()
    def test_norm_zero_size_reduction_2(self, xp):
        return xp.linalg.norm(xp.empty([3, 0, 2]), -xp.inf, axis=(1, 0))

    @testing.numpy_nlcpy_raises()
    def test_norm_zero_size_reduction_3(self, xp):
        return xp.linalg.norm(xp.empty([3, 0, 2]), 1, axis=(0, 1))


@testing.parameterize(*(
    testing.product({
        'ord': ['fro', 'nuc'],
    })
))
class TestNormFailureInvalidOrdForVector(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_norm_invalid_ord_for_vector(self, xp):
        return xp.linalg.norm(xp.empty([2]), self.ord)


@testing.parameterize(*(
    testing.product({
        'ord': [0, 3],
    })
))
class TestNormFailureInvalidOrdForMatrix(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_norm_invalid_ord_for_matrix(self, xp):
        return xp.linalg.norm(xp.empty([2, 2]), self.ord)

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


@testing.parameterize(*(
    testing.product({
        'shape': [
            (3, 5),
            (3, 7, 4),
            (4, 3, 6, 5),
            (5, 3, 4, 2, 6),
            (4, 3, 6, 4, 5),
            (0, 3, 3)
        ],
        'full_matrices': [True, False],
    })
))
class TestSvd(unittest.TestCase):
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_svd_general_compute_uv(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        args = dict()
        args["a"] = a
        if not self.full_matrices:
            args["full_matrices"] = self.full_matrices
        u, s, vh = xp.linalg.svd(**args)
        ret = [
            (i.shape,
             i.dtype,
             i.flags.c_contiguous,
             i.flags.f_contiguous
             ) for i in (u, s, vh)
        ]
        if a.size == 0:
            return ret
        if self.full_matrices:
            min_mn = min(self.shape[-1], self.shape[-2])
            u = u[..., :self.shape[-2], :min_mn]
            vh = vh[..., :min_mn, :self.shape[-1]]
        if type(u) is nlcpy.ndarray:
            u = u.get()
            s = s.get()
            vh = vh.get()
        x = numpy.matmul((u * s[..., None, :]), vh)
        tol = 1e-5 if a.dtype.char in 'fF' else 1e-12
        numpy.testing.assert_allclose(a, x, atol=tol, rtol=tol)
        return ret

    @testing.for_orders('CF')
    @testing.for_dtypes('fF')
    @testing.numpy_nlcpy_allclose(atol=1e-5, rtol=1e-5)
    def test_svd_general_single(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        return xp.linalg.svd(a, self.full_matrices, False)

    @testing.for_orders('CF')
    @testing.for_dtypes('?ilILdD')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_svd_general_double(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        return xp.linalg.svd(a, self.full_matrices, False)


@testing.parameterize(*(
    testing.product({
        'shape': [
            (3, 3),
            (3, 7, 7),
            (4, 3, 6, 6),
            (6, 3, 4, 5, 5),
            (0, 3, 3)
        ],
        'full_matrices': [True, False],
    })
))
class TestSvdHermitian(unittest.TestCase):
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_svd_hermitian_compute_uv(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        m = a.shape[-1]
        for i in range(1, m):
            for j in range(i + 1):
                a[..., i, j] = xp.conj(a[..., j, i])
        u, s, vh = xp.linalg.svd(a, self.full_matrices, True, True)
        ret = [
            (i.shape,
             i.dtype,
             i.flags.c_contiguous,
             i.flags.f_contiguous,
             ) for i in (u, s, vh)
        ]
        if self.full_matrices:
            min_mn = min(self.shape[-1], self.shape[-2])
            u = u[..., :self.shape[-1], :min_mn]
            vh = vh[..., :min_mn, :self.shape[-2]]
        if type(u) is nlcpy.ndarray:
            u = u.get()
            s = s.get()
            vh = vh.get()
            a = a.get()
        for i in range(m):
            a[..., i, i] = numpy.real(a[..., i, i])
        x = numpy.matmul((u * s[..., None, :]), vh)
        tol = 1e-5 if a.dtype.char in 'fF' else 1e-12
        numpy.testing.assert_allclose(a, x, atol=tol, rtol=tol)
        return ret

    @testing.for_orders('CF')
    @testing.for_dtypes('fF')
    @testing.numpy_nlcpy_allclose(atol=1e-5, rtol=1e-5)
    def test_svd_hermitian_single(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        return xp.linalg.svd(a, self.full_matrices, False, True)

    @testing.for_orders('CF')
    @testing.for_dtypes('?ilILdD')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_svd_hermitian_double(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        return xp.linalg.svd(a, self.full_matrices, False, True)


class TestSvdFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_svd_0d(self, xp):
        return xp.linalg.svd(1)

    @testing.numpy_nlcpy_raises()
    def test_svd_1d(self, xp):
        return xp.linalg.svd(xp.arange(3))

    @testing.numpy_nlcpy_raises()
    def test_svd_not_square(self, xp):
        return xp.linalg.svd(xp.empty([3, 4]), hermitian=True)


@testing.parameterize(*(
    testing.product({
        'shape': [
            (3, 3),
            (3, 7, 7),
            (4, 3, 6, 6),
            (6, 3, 4, 5, 5),
            (2, 3, 3, 4, 5, 5),
            (0, 3, 3)
        ],
    })
))
class TestCholesky(unittest.TestCase):
    @testing.for_orders('CF')
    @testing.for_dtypes('?ilILdD')
    @testing.numpy_nlcpy_allclose(atol=1e-9, rtol=1e-9)
    def test_cholesky_double(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        m = a.shape[-1]
        a = a.reshape(-1, m, m)
        if dtype == numpy.bool_:
            for i in range(a.shape[0]):
                a[i] = xp.eye(m)
        else:
            for i in range(a.shape[0]):
                a[i] = numpy.dot(a[i], numpy.conjugate(a[i].T))
                a[i] += numpy.eye(a[i].shape[0], a[i].shape[1], dtype=a.dtype) * 100
        a = xp.asarray(a.reshape(self.shape), order=order)
        return xp.linalg.cholesky(a)

    @testing.for_orders('CF')
    @testing.for_dtypes('fF')
    @testing.numpy_nlcpy_allclose(atol=1e-6, rtol=1e-6)
    def test_cholesky_single(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        m = a.shape[-1]
        a = a.reshape(-1, m, m)
        for i in range(a.shape[0]):
            a[i] = numpy.dot(a[i], xp.conjugate(a[i].T))
            a[i] += numpy.eye(a[i].shape[0], a[i].shape[1], dtype=a.dtype) * 100
        a = xp.asarray(a.reshape(self.shape), order=order)
        return xp.linalg.cholesky(a)


class TestCholeskyFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_cholesky_0d(self, xp):
        return xp.linalg.cholesky(1)

    @testing.numpy_nlcpy_raises()
    def test_cholesky_1d(self, xp):
        return xp.linalg.cholesky(xp.arange(3))

    @testing.numpy_nlcpy_raises()
    def test_cholesky_not_square(self, xp):
        return xp.linalg.cholesky(xp.empty([3, 4]))

    @testing.numpy_nlcpy_raises()
    def test_cholesky_not_positive_definite(self, xp):
        return xp.linalg.cholesky([[1, 2], [3, 4]])


@testing.parameterize(*(
    testing.product({
        'shape': [
            (3, 5),
            (5, 4),
            (4, 4),
            (1, 7),
            (6, 1),
            (0, 3),
            (4, 0),
            (0, 0),
        ],
        'mode': [None, 'reduced', 'complete', 'r', 'raw', 'full', 'e', 'economic']
    })
))
class TestQr(unittest.TestCase):
    @testing.for_orders('CF')
    @testing.for_dtypes('fF')
    @testing.numpy_nlcpy_allclose(atol=1e-6, rtol=1e-6)
    def test_qr_single(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        args = dict()
        args["a"] = a
        if self.mode is not None:
            args["mode"] = self.mode
        if self.mode == 'complete':
            q, r = xp.linalg.qr(**args)
            m, n = a.shape
            k = min(m, n)
            q[:, k:m] = 0
            r[k:m] = 0
            return q, r
        else:
            return xp.linalg.qr(**args)

    @testing.for_orders('CF')
    @testing.for_dtypes('?ilILdD')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_qr_double(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        if self.mode == 'complete':
            q, r = xp.linalg.qr(a, self.mode)
            m, n = a.shape
            k = min(m, n)
            q[:, k:m] = 0
            r[k:m] = 0
            return q, r
        elif self.mode is not None:
            return xp.linalg.qr(a, mode=self.mode)
        else:
            return xp.linalg.qr(a)


class TestQrFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_qr_0d(self, xp):
        return xp.linalg.qr(1)

    @testing.numpy_nlcpy_raises()
    def test_qr_1d(self, xp):
        return xp.linalg.qr(xp.arange(3))

    @testing.numpy_nlcpy_raises()
    def test_qr_incompatible_mode(self, xp):
        return xp.linalg.qr(xp.ones([2, 2]), mode='hoge')

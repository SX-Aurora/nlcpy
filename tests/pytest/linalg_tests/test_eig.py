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
        'shape': [
            (3, 3),
            (4, 5, 5),
            (0, 4, 4),
            (3, 4, 6, 6),
            (7, 3, 5, 5),
            (5, 0, 4, 6, 6),
            (2, 5, 3, 4, 4),
            (6, 3, 4, 3, 3),
        ],
    })
))
class TestEig(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_eig(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        args = dict()
        args["a"] = a
        w, v = xp.linalg.eig(**args)
        ret = [w.shape, v.shape, v.dtype, v.flags.c_contiguous, v.flags.f_contiguous]
        m = a.shape[-1]
        a = a.reshape([-1, m, m])
        w = w.reshape([-1, m])
        v = v.reshape([-1, m, m])
        if type(a) is nlcpy.ndarray:
            a = a.get()
            w = w.get()
            v = v.get()
        x = numpy.array([numpy.dot(a[i], v[i]) for i in range(a.shape[0])])
        y = numpy.array([w[i] * v[i] for i in range(a.shape[0])])
        tol = 1e-5 if a.dtype.char in 'fF' else 1e-12
        numpy.testing.assert_allclose(x, y, atol=tol, rtol=tol)
        return ret


class TestEigFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_eig_0d(self, xp):
        return xp.linalg.eig(1)

    @testing.numpy_nlcpy_raises()
    def test_eig_1d(self, xp):
        a = xp.empty([3])
        return xp.linalg.eig(a)

    @testing.numpy_nlcpy_raises()
    def test_eig_not_square(self, xp):
        a = xp.empty([3, 4])
        return xp.linalg.eig(a)


@testing.parameterize(*(
    testing.product({
        'shape': [
            (3, 3),
            (4, 5, 5),
            (0, 4, 4),
            (4, 3, 6, 6),
            (2, 5, 3, 4, 4),
            (5, 4, 1, 3, 3)
        ],
        'uplo': ['U', 'L', 'u', 'l'],
    })
))
class TestEigh(unittest.TestCase):
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_eigh(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        m = a.shape[-1]
        for i in range(1, m):
            for j in range(i + 1):
                a[..., i, j] = xp.conj(a[..., j, i])
        args = dict()
        args["a"] = a
        if self.uplo != 'L':
            args["UPLO"] = self.uplo
        w, v = xp.linalg.eigh(**args)
        ret = [
            (i.shape,
             i.dtype,
             i.flags.c_contiguous,
             i.flags.f_contiguous) for i in (w, v)]

        a = a.reshape([-1, m, m])
        w = w.reshape([-1, m])
        v = v.reshape([-1, m, m])
        if type(a) is nlcpy.ndarray:
            a = a.get()
            w = w.get()
            v = v.get()
        for i in range(m):
            a[..., i, i] = numpy.real(a[..., i, i])
        x = numpy.array([numpy.dot(a[i], v[i]) for i in range(a.shape[0])])
        y = numpy.array([w[i] * v[i] for i in range(a.shape[0])])
        tol = 1e-5 if a.dtype.char in 'fF' else 1e-12
        numpy.testing.assert_allclose(x, y, atol=tol, rtol=tol)
        return numpy.array(ret, dtype=object)


class TestEighFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_eigh_0d(self, xp):
        return xp.linalg.eigh(1)

    @testing.numpy_nlcpy_raises()
    def test_eigh_1d(self, xp):
        a = xp.empty([3])
        return xp.linalg.eigh(a)

    @testing.numpy_nlcpy_raises()
    def test_eigh_not_square(self, xp):
        a = xp.empty([3, 4])
        return xp.linalg.eigh(a)

    @testing.numpy_nlcpy_raises()
    def test_eigh_uplo_is_not_lu(self, xp):
        a = xp.empty([3, 3])
        return xp.linalg.eigh(a, UPLO='Z')


@testing.parameterize(*(
    testing.product({
        'shape': [
            (3, 3),
            (4, 5, 5),
            (6, 4, 4),
            (0, 3, 3),
            (3, 5, 4, 4),
            (2, 5, 3, 4, 4),
            (3, 7, 1, 3, 3)
        ],
    })
))
class TestEigvals(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.for_orders('CF')
    def test_eigvals(self, dtype, order):
        a = nlcpy.asarray(testing.shaped_random(self.shape, nlcpy, dtype), order=order)
        args = dict()
        args["a"] = a
        w1 = nlcpy.linalg.eigvals(**args)
        w2, v = nlcpy.linalg.eig(**args)
        numpy.testing.assert_allclose(w1, w2, atol=1e-12, rtol=1e-12)


class TestEigvalsFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_eigvals_0d(self, xp):
        return xp.linalg.eigvals(1)

    @testing.numpy_nlcpy_raises()
    def test_eigvals_1d(self, xp):
        a = xp.empty([3])
        return xp.linalg.eigvals(a)

    @testing.numpy_nlcpy_raises()
    def test_eigvals_not_square(self, xp):
        a = xp.empty([3, 4])
        return xp.linalg.eigvals(a)


@testing.parameterize(*(
    testing.product({
        'shape': [
            (3, 3),
            (4, 5, 5),
            (6, 4, 4),
            (0, 3, 3),
            (2, 5, 3, 4, 4),
            (3, 7, 1, 3, 3)
        ],
        'uplo': ['U', 'L', 'u', 'l'],
    })
))
class TestEigvalsh(unittest.TestCase):
    @testing.for_dtypes('fF')
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_allclose(atol=1e-5, rtol=1e-5)
    def test_eigvalsh_single(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        args = dict()
        args["a"] = a
        if self.uplo != 'L':
            args["UPLO"] = self.uplo
        return xp.linalg.eigvalsh(**args)

    @testing.for_dtypes('?ilILdD')
    @testing.for_orders('CF')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_eigvalsh_double(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        return xp.linalg.eigvalsh(a, self.uplo)


class TestEigvalshFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_eigvalsh_0d(self, xp):
        return xp.linalg.eigvalsh(1)

    @testing.numpy_nlcpy_raises()
    def test_eigvalsh_1d(self, xp):
        a = xp.empty([3])
        return xp.linalg.eigvalsh(a)

    @testing.numpy_nlcpy_raises()
    def test_eigvalsh_not_square(self, xp):
        a = xp.empty([3, 4])
        return xp.linalg.eigvalsh(a)

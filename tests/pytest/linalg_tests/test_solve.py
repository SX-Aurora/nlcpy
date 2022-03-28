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
from nlcpy.testing import condition


@testing.parameterize(*(
    testing.product({
        'shape': [
            ((3, 3), (3,)),
            ((3, 3), (3, 4)),
            ((3, 4, 4), (3, 4)),
            ((3, 4, 4), (3, 4, 5)),
            ((7, 6, 5, 5), (7, 6, 5)),
            ((7, 6, 5, 5), (7, 6, 5, 4)),
            ((4, 2, 3, 5, 5), (4, 2, 3, 5)),
            ((4, 2, 3, 5, 5), (4, 2, 3, 5, 6)),
            ((6, 6), (7, 4, 3, 6, 5)),
            ((2, 3, 5, 5), (4, 2, 3, 5, 6)),
            ((7, 5, 2, 4, 4), (4, 6)),
            ((4, 2, 3, 5, 5), (3, 5, 5)),
            ((0, 3, 3), (0, 3)),
            ((3, 6, 6), (1, 6, 6)),
            ((1, 6, 6), (3, 6, 7)),
        ],
    })
))
class TestSolve(unittest.TestCase):
    @condition.retry(10)
    @testing.for_dtypes("?ilILdD", name='dt_a')
    @testing.for_dtypes("?ilILdD", name='dt_b')
    @testing.for_orders("CF", name='order_a')
    @testing.for_orders("CF", name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_solve(self, xp, dt_a, dt_b, order_a, order_b):
        if dt_a != numpy.bool_:
            a = xp.asarray(
                testing.shaped_random(self.shape[0], xp, dt_a), order=order_a)
        else:
            n = self.shape[0][-1]
            a = xp.empty(self.shape[0])
            a[..., :, :] = xp.eye(n)
        b = xp.asarray(
            testing.shaped_random(self.shape[1], xp, dt_b), order=order_b)
        args = dict()
        args["a"] = a
        args["b"] = b
        return xp.linalg.solve(**args)

    @condition.retry(10)
    @testing.for_dtypes("fF", name='dt_a')
    @testing.for_dtypes("fF", name='dt_b')
    @testing.for_orders("CF", name='order_a')
    @testing.for_orders("CF", name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-6, rtol=1e-6)
    def test_solve_single(self, xp, dt_a, dt_b, order_a, order_b):
        a = xp.asarray(
            testing.shaped_random(self.shape[0], xp, dt_a), order=order_a)
        b = xp.asarray(
            testing.shaped_random(self.shape[1], xp, dt_b), order=order_b)
        return xp.linalg.solve(a, b)

    @condition.retry(10)
    @testing.for_dtypes("fF", name='dt_a')
    @testing.for_dtypes("?ilILdD", name='dt_b')
    @testing.for_orders("CF", name='order_a')
    @testing.for_orders("CF", name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_solve_single_a(self, xp, dt_a, dt_b, order_a, order_b):
        a = xp.asarray(
            testing.shaped_random(self.shape[0], xp, dt_a), order=order_a)
        b = xp.asarray(
            testing.shaped_random(self.shape[1], xp, dt_b), order=order_b)
        return xp.linalg.solve(a, b)

    @condition.retry(10)
    @testing.for_dtypes("?ilILdD", name='dt_a')
    @testing.for_dtypes("fF", name='dt_b')
    @testing.for_orders("CF", name='order_a')
    @testing.for_orders("CF", name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_solve_single_b(self, xp, dt_a, dt_b, order_a, order_b):
        if dt_a != numpy.bool_:
            a = xp.asarray(
                testing.shaped_random(self.shape[0], xp, dt_a), order=order_a)
        else:
            n = self.shape[0][-1]
            a = xp.empty(self.shape[0])
            a[..., :, :] = xp.eye(n)
        b = xp.asarray(
            testing.shaped_random(self.shape[1], xp, dt_b), order=order_b)
        return xp.linalg.solve(a, b)


class TestSolveFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_solve_0d(self, xp):
        return xp.linalg.solve(1, 1)

    @testing.numpy_nlcpy_raises()
    def test_solve_1d(self, xp):
        return xp.linalg.solve([1], [1])

    @testing.numpy_nlcpy_raises()
    def test_solve_not_square(self, xp):
        a = testing.shaped_random([3, 4])
        b = testing.shaped_random([4])
        return xp.linalg.solve(a, b)

    @testing.numpy_nlcpy_raises()
    def test_solve_shape_mismatch_1(self, xp):
        a = testing.shaped_random([3, 3])
        b = testing.shaped_random([4])
        return xp.linalg.solve(a, b)

    @testing.numpy_nlcpy_raises()
    def test_solve_shape_mismatch_2(self, xp):
        a = testing.shaped_random([4, 3, 3])
        b = testing.shaped_random([3])
        return xp.linalg.solve(a, b)

    @testing.numpy_nlcpy_raises()
    def test_solve_shape_mismatch_3(self, xp):
        a = testing.shaped_random([3, 3])
        b = testing.shaped_random([4, 5])
        return xp.linalg.solve(a, b)

    @testing.numpy_nlcpy_raises()
    def test_solve_shape_mismatch_4(self, xp):
        a = testing.shaped_random([5, 6, 3, 3])
        b = testing.shaped_random([2, 3, 3])
        return xp.linalg.solve(a, b)

    @testing.numpy_nlcpy_raises()
    def test_solve_singular(self, xp):
        return xp.linalg.solve(xp.zeros([3, 3]), xp.zeros([3]))


@testing.parameterize(*(
    testing.product({
        'shape': [
            ((3, 3), (3,)),
            ((3, 3), (3, 4)),
            ((4, 5), (4,)),
            ((4, 5), (4, 6)),
            ((5, 3), (5,)),
            ((5, 3), (5, 4)),
            ((0, 3), (0, 4)),
            ((3, 0), (3,)),
            ((3, 4), (3, 0)),
        ],
        'rcond': ['warn', None, -1, 0.5, 1],
    })
))
class TestLstsq(unittest.TestCase):
    @testing.for_dtypes("?ilILdD", name='dt_a')
    @testing.for_dtypes("?ilILdD", name='dt_b')
    @testing.for_orders("CF", name='order_a')
    @testing.for_orders("CF", name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_lstsq(self, xp, dt_a, dt_b, order_a, order_b):
        a = xp.asarray(
            testing.shaped_random(self.shape[0], xp, dt_a), order=order_a)
        b = xp.asarray(
            testing.shaped_random(self.shape[1], xp, dt_b), order=order_b)
        args = dict()
        args["a"] = a
        args["b"] = b
        if self.rcond != 'warn':
            args["rcond"] = self.rcond
        return xp.linalg.lstsq(**args)

    @testing.for_dtypes("fF", name='dt_a')
    @testing.for_dtypes("fF", name='dt_b')
    @testing.for_orders("CF", name='order_a')
    @testing.for_orders("CF", name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-5, rtol=1e-5)
    def test_lstsq_single(self, xp, dt_a, dt_b, order_a, order_b):
        a = xp.asarray(
            testing.shaped_random(self.shape[0], xp, dt_a), order=order_a)
        b = xp.asarray(
            testing.shaped_random(self.shape[1], xp, dt_b), order=order_b)
        if self.rcond != 'warn':
            return xp.linalg.lstsq(a, b, self.rcond)
        else:
            return xp.linalg.lstsq(a, b)

    @testing.for_dtypes("fF", name='dt_a')
    @testing.for_dtypes("?ilILdD", name='dt_b')
    @testing.for_orders("CF", name='order_a')
    @testing.for_orders("CF", name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_lstsq_single_a(self, xp, dt_a, dt_b, order_a, order_b):
        a = xp.asarray(
            testing.shaped_random(self.shape[0], xp, dt_a), order=order_a)
        b = xp.asarray(
            testing.shaped_random(self.shape[1], xp, dt_b), order=order_b)
        return xp.linalg.lstsq(a, b, self.rcond)

    @testing.for_dtypes("?ilILdD", name='dt_a')
    @testing.for_dtypes("fF", name='dt_b')
    @testing.for_orders("CF", name='order_a')
    @testing.for_orders("CF", name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_lstsq_single_b(self, xp, dt_a, dt_b, order_a, order_b):
        a = xp.asarray(
            testing.shaped_random(self.shape[0], xp, dt_a), order=order_a)
        b = xp.asarray(
            testing.shaped_random(self.shape[1], xp, dt_b), order=order_b)
        return xp.linalg.lstsq(a, b, rcond=self.rcond)


class TestLstsqFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_lstsq_0d(self, xp):
        return xp.linalg.lstsq(1, 1)

    @testing.numpy_nlcpy_raises()
    def test_lstsq_1d(self, xp):
        return xp.linalg.lstsq([1], [1])

    @testing.numpy_nlcpy_raises()
    def test_lstsq_incompatible_dim(self, xp):
        a = testing.shaped_random([3, 4])
        b = testing.shaped_random([4])
        return xp.linalg.lstsq(a, b)

    def test_lstsq_not_converge(self):
        a = nlcpy.ones([200, 200])
        a[0, 0] = nlcpy.nan
        with self.assertRaises(nlcpy.linalg.LinAlgError):
            nlcpy.linalg.lstsq(a, nlcpy.zeros([200, 1]))


@testing.parameterize(*(
    testing.product({
        'shape': [
            (10, 10),
            (8, 7, 7),
            (3, 4, 5, 5),
            (3, 6, 4, 5, 5),
            (0, 3, 3),
            (3, 0, 2, 6, 6),
        ],
    })
))
class TestInv(unittest.TestCase):
    @condition.retry(10)
    @testing.for_dtypes("?ilILdD")
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_inv(self, xp, dtype, order):
        if dtype != numpy.bool_:
            a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        else:
            n = self.shape[-1]
            a = xp.empty(self.shape)
            a[..., :, :] = xp.eye(n)
        args = dict()
        args["a"] = a
        return xp.linalg.inv(**args)

    @condition.retry(10)
    @testing.for_dtypes("fF")
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-6, rtol=1e-6)
    def test_inv_single(self, xp, dtype, order):
        a = xp.asarray(testing.shaped_random(self.shape, xp, dtype), order=order)
        return xp.linalg.inv(a)


class TestInvFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_inv_0d(self, xp):
        return xp.linalg.inv(1)

    @testing.numpy_nlcpy_raises()
    def test_inv_1d(self, xp):
        return xp.linalg.inv([1])

    @testing.numpy_nlcpy_raises()
    def test_inv_not_square(self, xp):
        return xp.linalg.inv(testing.shaped_random([3, 4]))

    @testing.numpy_nlcpy_raises()
    def test_inv_singular(self, xp):
        return xp.linalg.inv(xp.zeros([3, 3]))

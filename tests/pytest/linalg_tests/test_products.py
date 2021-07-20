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


class TestInner_0d(unittest.TestCase):
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_inner_0d(self, xp):
        a = testing.shaped_random((1,), xp)[0]
        b = testing.shaped_random((1,), xp)[0]
        return xp.inner(a, b)


@testing.parameterize(*(
    testing.product({
        'shape': [(0,), (1000,)],
    })
))
class TestInner_1d(unittest.TestCase):
    @testing.for_dtypes("ilILdD", name='dt_a')
    @testing.for_dtypes("ilILdD", name='dt_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_inner_1d(self, xp, dt_a, dt_b):
        a = testing.shaped_random(self.shape, xp, dt_a)
        b = testing.shaped_random(self.shape, xp, dt_b)
        return xp.inner(a, b)

    @testing.for_dtypes("fF", name='dt_a')
    @testing.for_dtypes("fF", name='dt_b')
    @testing.numpy_nlcpy_allclose(atol=1e-6, rtol=1e-6)
    def test_inner_1d_single(self, xp, dt_a, dt_b):
        a = testing.shaped_random(self.shape, xp, dt_a)
        b = testing.shaped_random(self.shape, xp, dt_b)
        return xp.inner(a, b)

    @testing.for_dtypes("fF", name='dt_a')
    @testing.for_dtypes("ilILdD", name='dt_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_inner_1d_single_a(self, xp, dt_a, dt_b):
        a = testing.shaped_random(self.shape, xp, dt_a)
        b = testing.shaped_random(self.shape, xp, dt_b)
        return xp.inner(a, b)

    @testing.for_dtypes("ilILdD", name='dt_a')
    @testing.for_dtypes("fF", name='dt_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_inner_1d_single_b(self, xp, dt_a, dt_b):
        a = testing.shaped_random(self.shape, xp, dt_a)
        b = testing.shaped_random(self.shape, xp, dt_b)
        return xp.inner(a, b)


class TestInner_broadcast(unittest.TestCase):
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_inner_broadcast1(self, xp, order):
        a = testing.shaped_random((1,), xp)[0]
        b = xp.asarray(testing.shaped_random((1000,), xp), order=order)
        return xp.inner(a, b)

    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_inner_broadcast2(self, xp, order):
        a = xp.asarray(testing.shaped_random((1000,), xp), order=order)
        b = testing.shaped_random((1,), xp)[0]
        return xp.inner(a, b)


@testing.parameterize(*(
    testing.product({
        'shape_a': [(10, ), (5, 4), (3, 4, 5), (2, 4, 3), (3, 0, 4)],
        'shape_b': [(10, ), (3, 6), (4, 2, 1), (3, 5), (0, 5, 9)],
        'param_out': [None, (0, 'C'), (0, 'F'), (3, 'C'), (3, 'F')],
    })
))
class TestOuter(unittest.TestCase):
    @testing.for_dtypes('fF', name='dt_a')
    @testing.for_dtypes('fF', name='dt_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-5, rtol=1e-5)
    def test_outer_single_both(self, xp, dt_a, dt_b, order_a, order_b):
        a = xp.asarray(testing.shaped_random(self.shape_a, xp, dt_a), order=order_a)
        b = xp.asarray(testing.shaped_random(self.shape_b, xp, dt_b), order=order_b)
        args = dict()
        args["a"] = a
        args["b"] = b
        if self.param_out is not None:
            if self.param_out[0] != 0:
                shape_out = [self.param_out[0], a.size, b.size]
            else:
                shape_out = [a.size, b.size]
            dt_out = numpy.result_type(dt_a, dt_b)
            out = xp.empty(shape_out, dtype=dt_out, order=self.param_out[1])
            args["out"] = out
        return xp.outer(**args)

    @testing.for_dtypes('fF', name='dt_a')
    @testing.for_dtypes('?ilILdD', name='dt_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_outer_single_a(self, xp, dt_a, dt_b, order_a, order_b):
        a = xp.asarray(testing.shaped_random(self.shape_a, xp, dt_a), order=order_a)
        b = xp.asarray(testing.shaped_random(self.shape_b, xp, dt_b), order=order_b)
        if self.param_out is not None:
            if self.param_out[0] != 0:
                shape_out = [self.param_out[0], a.size, b.size]
            else:
                shape_out = [a.size, b.size]
            dt_out = numpy.result_type(dt_a, dt_b)
            out = xp.empty(shape_out, dtype=dt_out, order=self.param_out[1])
            return xp.outer(a, b, out)
        else:
            return xp.outer(a, b)

    @testing.for_dtypes('?ilILdD', name='dt_a')
    @testing.for_dtypes('fF', name='dt_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_outer_single_b(self, xp, dt_a, dt_b, order_a, order_b):
        a = xp.asarray(testing.shaped_random(self.shape_a, xp, dt_a), order=order_a)
        b = xp.asarray(testing.shaped_random(self.shape_b, xp, dt_b), order=order_b)
        if self.param_out is not None:
            if self.param_out[0] != 0:
                shape_out = [self.param_out[0], a.size, b.size]
            else:
                shape_out = [a.size, b.size]
            dt_out = numpy.result_type(dt_a, dt_b)
            out = xp.empty(shape_out, dtype=dt_out, order=self.param_out[1])
            return xp.outer(a, b, out)
        else:
            return xp.outer(a, b)

    @testing.for_dtypes('?ilILdD', name='dt_a')
    @testing.for_dtypes('?ilILdD', name='dt_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_outer_double(self, xp, dt_a, dt_b, order_a, order_b):
        a = xp.asarray(testing.shaped_random(self.shape_a, xp, dt_a), order=order_a)
        b = xp.asarray(testing.shaped_random(self.shape_b, xp, dt_b), order=order_b)
        if self.param_out is not None:
            if self.param_out[0] != 0:
                shape_out = [self.param_out[0], a.size, b.size]
            else:
                shape_out = [a.size, b.size]
            dt_out = numpy.result_type(dt_a, dt_b)
            out = xp.empty(shape_out, dtype=dt_out, order=self.param_out[1])
            return xp.outer(a, b, out)
        else:
            return xp.outer(a, b)


class TestOuterFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_outer_incompatible_out(self, xp):
        return xp.outer(1, 2, xp.empty(1))


@testing.parameterize(*(
    testing.product({
        'shape': [
            ((2, 3, 4), (3, 4, 2)),
            ((1, 1), (1, 1)),
            ((1, 1), (1, 2)),
            ((1, 2), (2, 1)),
            ((2, 1), (1, 1)),
            ((1, 2), (2, 3)),
            ((2, 1), (1, 3)),
            ((2, 3), (3, 1)),
            ((2, 3), (3, 4)),
            ((0, 3), (3, 4)),
            ((2, 3), (3, 0)),
            ((0, 3), (3, 0)),
            ((3, 0), (0, 4)),
            ((2, 3, 0), (3, 0, 2)),
            ((0, 0), (0, 0)),
            ((3,), (3,)),
            ((2,), (2, 4)),
            ((4, 2), (2,)),
            ((), ()),
            ((), (2, 4)),
            ((4, 2), ()),
        ],
    })
))
class TestDot(unittest.TestCase):

    @testing.for_dtypes('?fF', name='dt_a')
    @testing.for_dtypes('?fF', name='dt_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-5, rtol=1e-5, contiguous_check=False)
    def test_dot_single(self, xp, dt_a, dt_b, order_a, order_b):
        dt_a = xp.dtype(dt_a)
        dt_b = xp.dtype(dt_b)
        if numpy.result_type(dt_a, dt_b).char not in 'fF':
            return 0
        shape_a, shape_b = self.shape
        a = xp.asarray(
            xp.arange(xp.prod(shape_a)).reshape(shape_a), dtype=dt_a, order=order_a)
        b = xp.asarray(
            xp.arange(xp.prod(shape_b)).reshape(shape_b), dtype=dt_b, order=order_b)
        return xp.dot(a, b)

    @testing.for_all_dtypes(name='dt_a')
    @testing.for_all_dtypes(name='dt_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12, contiguous_check=False)
    def test_dot_double(self, xp, dt_a, dt_b, order_a, order_b):
        dt_a = xp.dtype(dt_a)
        dt_b = xp.dtype(dt_b)
        if numpy.result_type(dt_a, dt_b).char not in 'dD':
            return 0
        shape_a, shape_b = self.shape
        a = xp.asarray(
            xp.arange(xp.prod(shape_a)).reshape(shape_a), dtype=dt_a, order=order_a)
        b = xp.asarray(
            xp.arange(xp.prod(shape_b)).reshape(shape_b), dtype=dt_b, order=order_b)
        return xp.dot(a, b)

    @testing.for_dtypes('?ilIL', name='dt_a')
    @testing.for_dtypes('?ilIL', name='dt_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_array_equal()
    def test_dot_int(self, xp, dt_a, dt_b, order_a, order_b):
        dt_a = xp.dtype(dt_a)
        dt_b = xp.dtype(dt_b)
        shape_a, shape_b = self.shape
        a = xp.asarray(
            xp.arange(xp.prod(shape_a)).reshape(shape_a), dtype=dt_a, order=order_a)
        b = xp.asarray(
            xp.arange(xp.prod(shape_b)).reshape(shape_b), dtype=dt_b, order=order_b)
        return xp.dot(a, b)

    @testing.for_dtypes('?fF', name='dt_a')
    @testing.for_dtypes('?fF', name='dt_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-5, rtol=1e-5, contiguous_check=False)
    def test_dot_single_with_out(self, xp, dt_a, dt_b, order_a, order_b):
        dt_a = xp.dtype(dt_a)
        dt_b = xp.dtype(dt_b)
        dt_out = numpy.result_type(dt_a, dt_b)
        if dt_out.char not in 'fF':
            return 0
        shape_a, shape_b = self.shape
        a = xp.asarray(
            xp.arange(xp.prod(shape_a)).reshape(shape_a), dtype=dt_a, order=order_a)
        b = xp.asarray(
            xp.arange(xp.prod(shape_b)).reshape(shape_b), dtype=dt_b, order=order_b)
        if a.ndim == 0:
            shape_out = b.shape
        elif b.ndim == 0:
            shape_out = a.shape
        elif a.ndim == 1 and b.ndim == 1:
            shape_out = ()
        elif b.ndim == 1:
            shape_out = shape_a[:-1]
        elif a.ndim == 1:
            shape_out = shape_b[:-2] + shape_b[-1:]
        else:
            shape_out = shape_a[:-1] + shape_b[:-2] + shape_b[-1:]
        out = xp.empty(shape_out, dt_out)
        return xp.dot(a, b, out)

    @testing.for_all_dtypes(name='dt_a')
    @testing.for_all_dtypes(name='dt_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12, contiguous_check=False)
    def test_dot_double_with_out(self, xp, dt_a, dt_b, order_a, order_b):
        dt_a = xp.dtype(dt_a)
        dt_b = xp.dtype(dt_b)
        dt_out = numpy.result_type(dt_a, dt_b)
        if dt_out.char not in ('dD'):
            return 0
        shape_a, shape_b = self.shape
        a = xp.asarray(
            xp.arange(xp.prod(shape_a)).reshape(shape_a), dtype=dt_a, order=order_a)
        b = xp.asarray(
            xp.arange(xp.prod(shape_b)).reshape(shape_b), dtype=dt_b, order=order_b)
        if a.ndim == 0:
            shape_out = b.shape
        elif b.ndim == 0:
            shape_out = a.shape
        elif a.ndim == 1 and b.ndim == 1:
            shape_out = ()
        elif b.ndim == 1:
            shape_out = shape_a[:-1]
        elif a.ndim == 1:
            shape_out = shape_b[:-2] + shape_b[-1:]
        else:
            shape_out = shape_a[:-1] + shape_b[:-2] + shape_b[-1:]
        out = xp.empty(shape_out, dt_out)
        return xp.dot(a, b, out)

    @testing.for_dtypes('?ilIL', name='dt_a')
    @testing.for_dtypes('?ilIL', name='dt_b')
    @testing.for_orders('CF', name='order_a')
    @testing.for_orders('CF', name='order_b')
    @testing.numpy_nlcpy_array_equal()
    def test_dot_int_with_out(self, xp, dt_a, dt_b, order_a, order_b):
        dt_a = xp.dtype(dt_a)
        dt_b = xp.dtype(dt_b)
        dt_out = numpy.result_type(dt_a, dt_b)
        shape_a, shape_b = self.shape
        a = xp.asarray(
            xp.arange(xp.prod(shape_a)).reshape(shape_a), dtype=dt_a, order=order_a)
        b = xp.asarray(
            xp.arange(xp.prod(shape_b)).reshape(shape_b), dtype=dt_b, order=order_b)
        if a.ndim == 0:
            shape_out = b.shape
        elif b.ndim == 0:
            shape_out = a.shape
        elif a.ndim == 1 and b.ndim == 1:
            shape_out = ()
        elif b.ndim == 1:
            shape_out = shape_a[:-1]
        elif a.ndim == 1:
            shape_out = shape_b[:-2] + shape_b[-1:]
        else:
            shape_out = shape_a[:-1] + shape_b[:-2] + shape_b[-1:]
        out = xp.empty(shape_out, dt_out)
        return xp.dot(a, b, out=out)


class TestDotFailure(unittest.TestCase):

    @testing.numpy_nlcpy_raises()
    def test_dot_with_out_f_contiguous(self, xp):
        a = xp.ones([2, 3, 4])
        b = xp.ones([3, 4, 2])
        out = xp.empty([2, 3, 3, 2], order='F')
        xp.dot(a, b, out)

    @testing.for_all_dtypes(name='dt_a')
    @testing.for_all_dtypes(name='dt_b')
    @testing.for_all_dtypes(name='dt_out')
    @testing.numpy_nlcpy_raises()
    def test_dot_incompatible_out_dtype(self, xp, dt_a, dt_b, dt_out):
        if dt_out == numpy.result_type(dt_a, dt_b):
            raise Exception
        a = xp.ones([2, 3, 4], dtype=dt_a)
        b = xp.ones([3, 4, 2], dtype=dt_b)
        out = xp.empty([2, 3, 3, 2], dtype=dt_out)
        xp.dot(a, b, out=out)

    @testing.numpy_nlcpy_raises()
    def test_dot_incompatible_out_shape(self, xp):
        a = xp.ones([2, 3, 4])
        b = xp.ones([3, 4, 2])
        out = xp.empty([2, 2, 3, 3])
        xp.dot(a, b, out=out)

    @testing.numpy_nlcpy_raises()
    def test_dot_incompatible_input_shape(self, xp):
        a = xp.ones([2, 3, 4])
        b = xp.ones([2, 3, 4])
        xp.dot(a, b)

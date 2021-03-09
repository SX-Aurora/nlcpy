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


def astype_without_warning(x, dtype, *args, **kwargs):
    dtype = numpy.dtype(dtype)
    # nlcpy interpret bool as int32
    if dtype == numpy.dtype(bool):
        dtype = numpy.dtype('int32')
    return x.astype(dtype, *args, **kwargs)


class TestArrayCopyAndView(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_view(self, xp):
        a = testing.shaped_arange((4,), xp, dtype=numpy.float32)
        b = a.view(dtype=numpy.int32)
        b[:] = 0
        return a

    @testing.for_dtypes([numpy.int32, numpy.int64])
    @testing.numpy_nlcpy_array_equal()
    def test_view_itemsize(self, xp, dtype):
        a = testing.shaped_arange((4,), xp, dtype=numpy.int32)
        b = a.view(dtype=dtype)
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_view_0d(self, xp):
        a = xp.array(1.5, dtype=numpy.float32)
        return a.view(dtype=numpy.int32)

    """failure
    @testing.for_dtypes([numpy.int32, numpy.int64])
    @testing.numpy_nlcpy_raises()
    def test_view_non_contiguous_raise(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2), xp, dtype=numpy.int32).transpose(
            0, 2, 1)
        a.view(dtype=dtype)
    """

    @testing.numpy_nlcpy_array_equal()
    def test_flatten(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.flatten()

    @testing.numpy_nlcpy_array_equal()
    def test_flatten_copied(self, xp):
        a = testing.shaped_arange((4,), xp)
        b = a.flatten()
        a[:] = 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_transposed_flatten(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp).transpose(2, 0, 1)
        return a.flatten()

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_fill(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a.fill(1)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_fill_with_numpy_scalar_ndarray(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a.fill(numpy.ones((), dtype=dtype))
        return a

    @testing.for_all_dtypes()
    def test_fill_with_numpy_nonscalar_ndarray(self, dtype):
        a = testing.shaped_arange((2, 3, 4), nlcpy, dtype)
        with self.assertRaises(ValueError):
            a.fill(numpy.ones((1,), dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_transposed_fill(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = a.transpose(2, 0, 1)
        b.fill(1)
        return b

    @testing.for_orders(['C', 'F', 'A', 'K', None])
    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_nlcpy_array_equal()
    def test_astype(self, xp, src_dtype, dst_dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, src_dtype)
        return astype_without_warning(a, dst_dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    def test_astype_type(self, src_dtype, dst_dtype, order):
        a = testing.shaped_arange((2, 3, 4), nlcpy, src_dtype)
        b = astype_without_warning(a, dst_dtype, order=order)
        a_cpu = testing.shaped_arange((2, 3, 4), numpy, src_dtype)
        b_cpu = astype_without_warning(a_cpu, dst_dtype, order=order)
        self.assertEqual(b.dtype.type, b_cpu.dtype.type)

    @testing.for_orders('CAK')
    @testing.for_all_dtypes()
    def test_astype_type_c_contiguous_no_copy(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), nlcpy, dtype)
        b = a.astype(dtype, order=order, copy=False)
        self.assertTrue(b is a)

    @testing.for_orders('FAK')
    @testing.for_all_dtypes()
    def test_astype_type_f_contiguous_no_copy(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), nlcpy, dtype)
        a = nlcpy.asarray(a, order='F')
        b = a.astype(dtype, order=order, copy=False)
        self.assertTrue(b is a)

    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_nlcpy_array_equal()
    def test_astype_strides(self, xp, src_dtype, dst_dtype):
        src = xp.empty((1, 2, 3), dtype=src_dtype)
        return numpy.array(
            astype_without_warning(src, dst_dtype, order='K').strides)

    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_nlcpy_array_equal()
    def test_astype_strides_negative(self, xp, src_dtype, dst_dtype):
        src = xp.empty((2, 3), dtype=src_dtype)[::-1, :]
        return numpy.array(
            astype_without_warning(src, dst_dtype, order='K').strides)

    """TODO
    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_nlcpy_array_equal()
    def test_astype_strides_swapped(self, xp, src_dtype, dst_dtype):
        src = xp.swapaxes(xp.empty((2, 3, 4), dtype=src_dtype), 1, 0)
        return numpy.array(
            astype_without_warning(src, dst_dtype, order='K').strides)
    """

    """failure
    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_nlcpy_array_equal()
    def test_astype_strides_broadcast(self, xp, src_dtype, dst_dtype):
        src, _ = xp.broadcast_arrays(xp.empty((2,), dtype=src_dtype),
                                     xp.empty((2, 3, 2), dtype=src_dtype))
        return numpy.array(
            astype_without_warning(src, dst_dtype, order='K').strides)
    """

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_diagonal1(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_diagonal2(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(-1, 2, 0)

    @testing.for_orders('CF')
    @testing.for_dtypes([numpy.int32, numpy.int64,
                         numpy.float32, numpy.float64])
    @testing.numpy_nlcpy_array_equal()
    def test_isinstance_numpy_copy(self, xp, dtype, order):
        a = numpy.arange(100, dtype=dtype).reshape(10, 10, order=order)
        b = xp.empty(a.shape, dtype=dtype, order=order)
        b[:] = a
        return b


@testing.parameterize(
    {'src_order': 'C'},
    {'src_order': 'F'},
)
class TestNumPyArrayCopyView(unittest.TestCase):
    @testing.for_orders('CF')
    @testing.for_dtypes([numpy.int32, numpy.int64,
                         numpy.float32, numpy.float64])
    @testing.numpy_nlcpy_array_equal()
    def test_isinstance_numpy_view_copy_f(self, xp, dtype, order):
        a = numpy.arange(100, dtype=dtype).reshape(
            10, 10, order=self.src_order)
        a = a[2:5, 1:8]
        b = xp.empty(a.shape, dtype=dtype, order=order)
        b[:] = a
        return b

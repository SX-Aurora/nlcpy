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
import warnings

import numpy
import nlcpy
from nlcpy import testing


def astype_without_warning(x, dtype, *args, **kwargs):
    with testing.numpy_nlcpy_errstate(invalid='ignore'):
        dtype = numpy.dtype(dtype)
        # nlcpy interpret bool as int32
        if dtype == numpy.dtype(bool):
            dtype = numpy.dtype('int32')
        if x.dtype.kind == 'c' and dtype.kind not in 'c':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', numpy.ComplexWarning)
                ret = x.astype(dtype, *args, **kwargs)
        else:
            ret = x.astype(dtype, *args, **kwargs)
        nlcpy.request.flush()
    return ret


class TestArrayCopyAndView(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_view(self, xp):
        data = testing.shaped_random([4], xp, dtype=numpy.float32)
        mask = [1, 1, 0, 1]
        fill_value = xp.arange(4)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        b = a.view(dtype=numpy.int32)
        b.data[0] = 100
        b.mask[0] = 1
        b.fill_value = 10
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_view_same_itemsize(self, xp):
        data = testing.shaped_arange((4,), xp, dtype=numpy.int32)
        mask = [0, 1, 0, 1]
        fill_value = 5
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        b = a.view(dtype=numpy.float32)
        b.fill_value = 10
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_view_ndarray(self, xp):
        data = testing.shaped_arange((4,), xp, dtype=numpy.int32)
        mask = [0, 1, 0, 1]
        fill_value = 5
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        b = a.view(dtype=numpy.float64, type=xp.ndarray)
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_view_different_itemsize1(self, xp):
        data = testing.shaped_arange((4,), xp, dtype=numpy.int64)
        a = xp.ma.array(data)
        b = a.view(dtype=numpy.float32)
        return b

    @testing.numpy_nlcpy_raises()
    def test_view_different_itemsize2(self, xp):
        data = testing.shaped_arange((4,), xp, dtype=numpy.int64)
        mask = [0, 1, 0, 1]
        fill_value = xp.arange(4)
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        a.view(dtype=numpy.float32)

    @testing.numpy_nlcpy_array_equal()
    def test_view_0d(self, xp):
        a = xp.ma.array(1.5, dtype=numpy.float32)
        return a.view(dtype=numpy.int32)

    @testing.numpy_nlcpy_array_equal()
    def test_view_same_dtype(self, xp):
        data = xp.arange(3)
        fill_value = xp.arange(1, 4)
        a = xp.ma.array(data, fill_value=fill_value)
        b = a.view()
        fill_value[0] = 10
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_view_fill_value(self, xp):
        data = xp.arange(3)
        fill_value = xp.arange(1, 4)
        a = xp.ma.array(data, fill_value=fill_value)
        b = a.view(fill_value=5)
        return b

    @testing.numpy_nlcpy_raises()
    def test_view_specify_type_twice(self, xp):
        xp.ma.array(1).view(xp.ndarray, xp.ndarray)

    @testing.numpy_nlcpy_array_equal()
    def test_flatten(self, xp):
        data = testing.shaped_arange((2, 3, 4), xp)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return a.flatten()

    @testing.numpy_nlcpy_array_equal()
    def test_flatten_copied(self, xp):
        data = testing.shaped_arange((4,), xp)
        mask = [0, 1, 0, 1]
        fill_value = testing.shaped_arange((4,), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, hard_mask=True)
        b = a.flatten()
        a[:] = 1
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_transposed_flatten(self, xp):
        data = testing.shaped_arange((2, 3, 4), xp)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        a = a.transpose(2, 0, 1)
        return a.flatten()

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_fill(self, xp, dtype):
        data = testing.shaped_arange((2, 3, 4), xp, dtype=dtype)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        a.fill(1)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_fill_with_numpy_scalar_ndarray(self, xp, dtype):
        data = testing.shaped_arange((2, 3, 4), xp, dtype=dtype)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        a.fill(numpy.ones((), dtype=dtype))
        return a

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_fill_with_numpy_nonscalar_ndarray(self, xp, dtype):
        data = testing.shaped_arange((2, 3, 4), xp, dtype=dtype)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        if xp is nlcpy:
            a.fill(numpy.ones((1,), dtype=dtype))
            return a
        else:
            a.fill(1)
            return a

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_transposed_fill(self, xp, dtype):
        data = testing.shaped_arange((2, 3, 4), xp, dtype=dtype)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        b = xp.ma.transpose(a, (2, 0, 1))
        b.fill(1)
        return b

    @testing.for_orders(['C', 'F', 'A', 'K', None])
    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_nlcpy_array_equal()
    def test_astype(self, xp, src_dtype, dst_dtype, order):
        data = testing.shaped_arange((2, 3, 4), xp)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, dtype=src_dtype)
        return astype_without_warning(a, dst_dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    def test_astype_type(self, src_dtype, dst_dtype, order):
        data = testing.shaped_arange((2, 3, 4), nlcpy)
        mask = testing.shaped_arange((2, 3, 4), nlcpy) % 2
        fill_value = testing.shaped_arange((2, 3, 4), nlcpy) + 1
        a = nlcpy.ma.array(data, mask=mask, fill_value=fill_value, dtype=src_dtype)
        b = astype_without_warning(a, dst_dtype, order=order)

        data = testing.shaped_arange((2, 3, 4), numpy)
        mask = testing.shaped_arange((2, 3, 4), numpy) % 2
        fill_value = testing.shaped_arange((2, 3, 4), numpy) + 1
        a_cpu = numpy.ma.array(data, mask=mask, fill_value=fill_value, dtype=src_dtype)
        b_cpu = astype_without_warning(a_cpu, dst_dtype, order=order)

        self.assertEqual(b.dtype.type, b_cpu.dtype.type)
        self.assertEqual(b._data.dtype.type, b_cpu._data.dtype.type)

    @testing.for_orders('CAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_astype_type_c_contiguous_no_copy(self, xp, dtype, order):
        data = testing.shaped_arange((2, 3, 4), xp)
        mask = testing.shaped_random((2, 3, 4), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, dtype=dtype)
        b = a.astype(dtype, order=order, copy=False)
        return b is a

    @testing.for_orders('FAK')
    @testing.for_all_dtypes()
    def test_astype_type_f_contiguous_no_copy(self, dtype, order):
        data = testing.shaped_arange((2, 3, 4), nlcpy)
        mask = testing.shaped_random((2, 3, 4), nlcpy, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3, 4), nlcpy) + 1
        a = nlcpy.ma.array(
            data, mask=mask, fill_value=fill_value, dtype=dtype, order='F')
        b = a.astype(dtype, order=order, copy=False)
        self.assertTrue(b is a)

    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_nlcpy_array_equal()
    def test_astype_strides(self, xp, src_dtype, dst_dtype):
        data = xp.empty((1, 2, 3), dtype=src_dtype)
        mask = testing.shaped_random((1, 2, 3), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((1, 2, 3), nlcpy) + 1
        src = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return numpy.array(
            astype_without_warning(src, dst_dtype, order='K').strides)

    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_nlcpy_array_equal()
    def test_astype_strides_negative(self, xp, src_dtype, dst_dtype):
        data = xp.empty((2, 3), dtype=src_dtype)
        mask = testing.shaped_random((2, 3), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((2, 3), nlcpy) + 1
        src = xp.ma.array(data, mask=mask, fill_value=fill_value)[::-1, :]
        return numpy.array(
            astype_without_warning(src, dst_dtype, order='K').strides)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_diagonal1(self, xp, dtype):
        data = testing.shaped_arange((3, 4, 5), xp, dtype=dtype)
        mask = testing.shaped_random((3, 4, 5), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((3, 4, 5), nlcpy) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value, hard_mask=True)
        b = a.diagonal(1, 2, 0)
        if dtype != numpy.bool_:
            a[0] = 1
        return b

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_diagonal2(self, xp, dtype):
        data = testing.shaped_arange((3, 4, 5), xp, dtype=dtype)
        mask = testing.shaped_random((3, 4, 5), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((3, 4, 5), nlcpy) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return a.diagonal(-1, 2, 0)

    @testing.numpy_nlcpy_array_equal()
    def test_diagonal_external(self, xp):
        data = testing.shaped_arange((3, 4, 5), xp)
        mask = testing.shaped_random((3, 4, 5), xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange((3, 4, 5), nlcpy) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return xp.ma.diagonal(a, -1, 2, 0)

    @testing.numpy_nlcpy_array_equal()
    def test_diagonal_external_ndarray(self, xp):
        a = testing.shaped_arange((3, 4, 5), xp)
        return xp.ma.diagonal(a, -1, 2, 0)

    @testing.for_orders('CF')
    @testing.for_dtypes([numpy.int32, numpy.int64,
                         numpy.float32, numpy.float64])
    @testing.numpy_nlcpy_array_equal()
    def test_isinstance_numpy_copy(self, xp, dtype, order):
        with xp.errstate(invalid='ignore'):
            data = numpy.arange(100, dtype=dtype)
            mask = testing.shaped_random((100,), numpy, dtype=numpy.bool_)
            fill_value = numpy.arange(1, 101)
            a = numpy.ma.array(data, mask=mask, fill_value=fill_value)
            a = a.reshape(10, 10, order=order)
            b = xp.ma.array(xp.empty(a.shape), dtype=dtype, order=order)
            b[:] = a
            nlcpy.request.flush()
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
        with xp.errstate(invalid='ignore'):
            data = numpy.arange(100, dtype=dtype).reshape(
                10, 10, order=self.src_order)
            mask = testing.shaped_random((10, 10), numpy, dtype=numpy.bool_)
            fill_value = numpy.arange(1, 101, dtype=dtype).reshape(
                10, 10, order=self.src_order)
            a = numpy.ma.array(data, mask=mask, fill_value=fill_value,
                               order=self.src_order)
            a = a[2:5, 1:8]
            b = xp.ma.array(xp.empty(a.shape), dtype=dtype, order=order)
            b[:] = a
            nlcpy.request.flush()
        return b

    @testing.for_orders('CF')
    @testing.for_dtypes([numpy.int32, numpy.int64,
                         numpy.float32, numpy.float64])
    @testing.numpy_nlcpy_array_equal()
    def test_copy(self, xp, dtype, order):
        data = numpy.arange(12, dtype=dtype).reshape(
            3, 4, order=self.src_order)
        mask = testing.shaped_random((3, 4), numpy, dtype=numpy.bool_)
        fill_value = numpy.arange(1, 13, dtype=dtype).reshape(
            3, 4, order=self.src_order)
        a = xp.ma.array(
            data, mask=mask, fill_value=fill_value, order=self.src_order,
            hard_mask=True)
        b = a.copy(order=order)
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_copy_external(self, xp):
        data = numpy.arange(12).reshape(3, 4)
        mask = testing.shaped_random((3, 4), numpy, dtype=numpy.bool_)
        fill_value = numpy.arange(1, 13).reshape(3, 4)
        a = xp.ma.array(
            data, mask=mask, fill_value=fill_value, hard_mask=True)
        b = xp.ma.copy(a)
        return b

    @testing.numpy_nlcpy_array_equal()
    def test_copy_external_ndarray(self, xp):
        a = xp.arange(12).reshape(3, 4)
        return xp.ma.copy(a)


@testing.parameterize(*(
    testing.product({
        'shapes': ((2, 3, 4), (3, 1, 2), (4, 2, 3, 5,)),
        'axes': ((0, 2), (2, 0), (0, -1), (1, 1), (-3, 1), (2, -3)),
    }))
)
class TestSwapAxes(unittest.TestCase):

    @testing.for_orders('CF')
    @testing.numpy_nlcpy_array_equal()
    def test_swapaxes(self, xp, order):
        data = testing.shaped_arange(self.shapes, xp, order=order)
        mask = testing.shaped_random(self.shapes, xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange(self.shapes, xp, order=order) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return a.swapaxes(self.axes[0], self.axes[1])

    @testing.numpy_nlcpy_array_equal()
    def test_swapaxes_external(self, xp):
        data = testing.shaped_arange(self.shapes, xp)
        mask = testing.shaped_random(self.shapes, xp, dtype=numpy.bool_)
        fill_value = testing.shaped_arange(self.shapes, xp) + 1
        a = xp.ma.array(data, mask=mask, fill_value=fill_value)
        return xp.ma.swapaxes(a, self.axes[0], self.axes[1])

    @testing.numpy_nlcpy_array_equal()
    def test_swapaxes_external_ndarray(self, xp):
        a = testing.shaped_arange(self.shapes, xp)
        return xp.ma.swapaxes(a, self.axes[0], self.axes[1])


@testing.parameterize(*(
    testing.product({
        'axes': ((3, 0), (0, 3), (-4, 1), (1, -4)),
    }))
)
class TestSwapAxesFailure(unittest.TestCase):

    @testing.numpy_nlcpy_raises()
    def test_swapaxes_failure(self, xp):
        data = testing.shaped_arange((2, 3, 4), xp)
        a = xp.ma.array(data)
        a.swapaxes(self.axes[0], self.axes[1])


def createMaskedArray(shape, xp):
    data = testing.shaped_arange(shape, xp)
    mask = testing.shaped_random(shape, xp, dtype=numpy.bool_)
    fill_value = testing.shaped_arange(shape, xp) + 1
    return xp.ma.array(data, mask=mask, fill_value=fill_value)


class TestSqueeze(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze1(self, xp):
        a = createMaskedArray((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze()

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze2(self, xp):
        a = createMaskedArray((2, 3, 4), xp)
        return a.squeeze()

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_int_axis1(self, xp):
        a = createMaskedArray((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=2)

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_int_axis2(self, xp):
        a = createMaskedArray((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=-3)

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_raises()
    def test_squeeze_int_axis_failure1(self, xp):
        a = createMaskedArray((1, 2, 1, 3, 1, 1, 4, 1), xp)
        a.squeeze(axis=-9)

    def test_squeeze_int_axis_failure2(self):
        a = createMaskedArray((1, 2, 1, 3, 1, 1, 4, 1), nlcpy)
        with self.assertRaises(nlcpy.core.error._AxisError):
            a.squeeze(axis=-9)

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_tuple_axis1(self, xp):
        a = createMaskedArray((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=(2, 4))

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_tuple_axis2(self, xp):
        a = createMaskedArray((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=(-4, -3))

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_tuple_axis3(self, xp):
        a = createMaskedArray((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=(4, 2))

    @testing.numpy_nlcpy_array_equal()
    def test_squeeze_tuple_axis4(self, xp):
        a = createMaskedArray((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=())

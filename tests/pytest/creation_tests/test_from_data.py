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
from nlcpy import testing
import tempfile
import numpy
from io import StringIO


class TestFromData(unittest.TestCase):

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_array(self, xp, dtype, order):
        return xp.array([[1, 2, 3], [2, 3, 4]], dtype=dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_array_from_empty_list(self, xp, dtype, order):
        return xp.array([], dtype=dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_array_from_nested_empty_list(self, xp, dtype, order):
        return xp.array([[], []], dtype=dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_array_from_numpy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_array_from_numpy_scalar(self, xp, dtype, order):
        a = numpy.array(2, dtype=dtype)
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_array_equal()
    def test_array_from_numpy_broad_cast(self, xp, dtype, order):
        a = testing.shaped_arange((2, 1, 4), numpy, dtype)
        a = numpy.broadcast_to(a, (2, 3, 4))
        return xp.array(a, order=order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_from_list_of_numpy(self, xp, dtype, src_order, dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of numpy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), numpy, dtype, src_order) + (12 * i)
            for i in range(2)]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_from_list_of_numpy_view(self, xp, dtype, src_order,
                                           dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of numpy.ndarray>)

        # create a list of view of ndarrays
        a = [
            (testing.shaped_arange((3, 8), numpy,
                                   dtype, src_order) + (24 * i))[:, ::2]
            for i in range(2)]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_from_list_of_numpy_scalar(self, xp, dtype, order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of numpy.ndarray>)
        a = [numpy.array(i, dtype=dtype) for i in range(2)]
        return xp.array(a, order=order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_from_nested_list_of_numpy(self, xp, dtype, src_order,
                                             dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of numpy.ndarray>)
        a = [
            [testing.shaped_arange(
                (3, 4), numpy, dtype, src_order) + (12 * i)]
            for i in range(2)]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_from_list_of_nlcpy(self, xp, dtype, src_order, dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of nlcpy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), xp, dtype, src_order) + (12 * i)
            for i in range(2)]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_from_list_of_nlcpy_view(self, xp, dtype, src_order,
                                           dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of nlcpy.ndarray>)

        # create a list of view of ndarrays
        a = [
            (testing.shaped_arange((3, 8), xp,
                                   dtype, src_order) + (24 * i))[:, ::2]
            for i in range(2)]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_from_nested_list_of_nlcpy(self, xp, dtype, src_order,
                                             dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of nlcpy.ndarray>)
        a = [
            [testing.shaped_arange((3, 4), xp, dtype, src_order) + (12 * i)]
            for i in range(2)]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_from_list_of_nlcpy_scalar(self, xp, dtype, order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of nlcpy.ndarray>)
        a = [xp.array(i, dtype=dtype) for i in range(2)]
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_from_nested_list_of_nlcpy_scalar(self, xp, dtype, order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of nlcpy.ndarray>)
        a = [[xp.array(i, dtype=dtype) for i in range(2)],
             [xp.array(i, dtype=dtype) for i in range(2)]]
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_array_copy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_array_copy_is_copied(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, order=order)
        a.fill(0)
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_nlcpy_array_equal()
    def test_array_copy_with_dtype(self, xp, dtype1, dtype2, order):
        # complex to real makes no sense
        a = testing.shaped_arange((2, 3, 4), xp, dtype1)
        return xp.array(a, dtype=dtype2, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_nlcpy_array_equal()
    def test_array_copy_with_dtype_char(self, xp, dtype1, dtype2, order):
        # complex to real makes no sense
        a = testing.shaped_arange((2, 3, 4), xp, dtype1)
        return xp.array(a, dtype=numpy.dtype(dtype2).char, order=order)

    @testing.for_orders('CFAK')
    @testing.numpy_nlcpy_array_equal()
    def test_array_copy_with_dtype_being_none(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.array(a, dtype=None, order=order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_copy_list_of_numpy_with_dtype(self, xp, dtype1, dtype2,
                                                 src_order, dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of numpy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), numpy, dtype1, src_order) + (12 * i)
            for i in range(2)]
        return xp.array(a, dtype=dtype2, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_copy_list_of_numpy_with_dtype_char(self, xp, dtype1,
                                                      dtype2, src_order,
                                                      dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of numpy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), numpy, dtype1, src_order) + (12 * i)
            for i in range(2)]
        return xp.array(a, dtype=numpy.dtype(dtype2).char, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_copy_list_of_nlcpy_with_dtype(self, xp, dtype1, dtype2,
                                                 src_order, dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of nlcpy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), xp, dtype1, src_order) + (12 * i)
            for i in range(2)]
        return xp.array(a, dtype=dtype2, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_nlcpy_array_equal(strides_check=True)
    def test_array_copy_list_of_nlcpy_with_dtype_char(self, xp, dtype1, dtype2,
                                                      src_order, dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # nlcpy.array(<list of nlcpy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), xp, dtype1, src_order) + (12 * i)
            for i in range(2)]
        return xp.array(a, dtype=numpy.dtype(dtype2).char, order=dst_order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_array_no_copy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, copy=False, order=order)
        a.fill(0)
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_array_f_contiguous_input(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype, order='F')
        b = xp.array(a, copy=False, order=order)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_array_f_contiguous_output(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, copy=False, order='F')
        assert b.flags.f_contiguous
        return b

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_array_no_copy_ndmin(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, copy=False, ndmin=5)
        assert a.shape == (2, 3, 4)
        a.fill(0)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_asarray(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.asarray(a)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_asarray_is_not_copied(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asarray(a)
        a.fill(0)
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_asarray_with_order(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asarray(a, order=order)
        if order in ['F', 'f']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_asarray_preserves_numpy_array_order(self, xp, dtype, order):
        a_numpy = testing.shaped_arange((2, 3, 4), numpy, dtype, order)
        b = xp.asarray(a_numpy)
        assert b.flags.f_contiguous == a_numpy.flags.f_contiguous
        assert b.flags.c_contiguous == a_numpy.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_asanyarray_with_order(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asanyarray(a, order=order)
        if order in ['F', 'f']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_asarray_from_numpy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        b = xp.asarray(a, order=order)
        if order in ['F', 'f']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_asarray_with_order_copy_behavior(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asarray(a, order=order)
        a.fill(0)
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_copy(self, xp, dtype, order):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = xp.copy(a, order=order)
        a[1] = 1
        return b

    @testing.for_CF_orders()
    @testing.numpy_nlcpy_equal()
    def test_copy_order(self, xp, order):
        a = xp.zeros((2, 3, 4), order=order)
        b = xp.copy(a)
        return (b.flags.c_contiguous, b.flags.f_contiguous)


@testing.parameterize(
    *testing.product({
        'shape': [(4, ), (4, 2), (4, 2, 3), (5, 4, 2, 3), (5, 4, 2, 3, 2)],
        'ndmin': [0, 1, 2, 3, 4, 5, 6],
        'copy': [True, False],
        'xp': [numpy, nlcpy]
    })
)
class TestArrayPreservationOfShape(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_nlcpy_array(self, dtype):
        a = testing.shaped_arange(self.shape, self.xp, dtype)
        nlcpy.array(a, copy=self.copy, ndmin=self.ndmin)
        # Check if nlcpy.ndarray does not alter
        # the shape of the original array.
        assert a.shape == self.shape


@testing.parameterize(
    *testing.product({
        'shape': [(4, ), (4, 2), (4, 2, 3), (5, 4, 2, 3), (5, 4, 2, 3, 2)],
        'ndmin': [0, 1, 2, 3, 4, 5, 6],
        'copy': [True, False],
        'xp': [numpy, nlcpy]
    })
)
class TestArrayCopy(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_nlcpy_array(self, dtype):
        a = testing.shaped_arange(self.shape, self.xp, dtype)
        actual = nlcpy.array(a, copy=self.copy, ndmin=self.ndmin)
        should_copy = (self.xp is numpy) or self.copy
        # TODO: Better determination of copy.
        is_copied = not ((actual is a) or (actual.base is a) or
                         (actual.base is a.base and a.base is not None))
        assert should_copy == is_copied


class TestArrayInvalidObject(unittest.TestCase):

    def test_invalid_type(self):
        a = numpy.array([1, 2, 3], dtype=object)
        with self.assertRaises(NotImplementedError):
            nlcpy.array(a)


@testing.parameterize(
    *testing.product({
        'count': [-2, -1, 0, 1, 2],
        'sep': ['', ','],
        'offset': [0, 8, 16]
    })
)
class TestFromFile(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal
    @testing.for_all_dtypes()
    def test_fromfile(self, xp, dtype):
        a = testing.shaped_random([10], numpy, dtype)
        with tempfile.TemporaryFile() as fh:
            a.tofile(fh, sep=self.sep)
            return xp.fromfile(
                fh, dtype=dtype, count=self.count, sep=self.sep, offset=self.offset)


class TestLoadtxt(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal
    @testing.for_all_dtypes()
    def test_loadtxt_dtype(self, xp, dtype):
        a = testing.shaped_random([2, 3, 4], xp, dtype)
        txt = StringIO(str(a).translate(str.maketrans({'[': None, ']': None})))
        return xp.loadtxt(txt, dtype=dtype)

    @testing.numpy_nlcpy_array_equal
    def test_loadtxt_comments(self, xp):
        a = testing.shaped_random([2, 3, 4], xp)
        txt = StringIO(str(a).translate(str.maketrans({'[': '@comment\n', ']': None})))
        return xp.loadtxt(txt, comments='@')

    @testing.numpy_nlcpy_array_equal
    def test_loadtxt_delimiter(self, xp):
        a = testing.shaped_random([2, 3, 4], xp)
        trans = str.maketrans({'[': None, ']': None, ' ': '@'})
        txt = StringIO(str(a).translate(trans).replace('\n@', ''))
        return xp.loadtxt(txt, delimiter='@')

    @testing.numpy_nlcpy_array_equal
    def test_loadtxt_converters(self, xp):
        a = testing.shaped_random([2, 3, 4], xp)
        txt = StringIO(str(a).translate(str.maketrans({'[': None, ']': None})))
        conv = {1: lambda s: float(s + s)}
        return xp.loadtxt(txt, converters=conv)

    @testing.numpy_nlcpy_array_equal
    def test_loadtxt_skiprows(self, xp):
        a = testing.shaped_random([2, 3, 4], xp)
        txt = StringIO(str(a).translate(str.maketrans({'[': None, ']': None})))
        return xp.loadtxt(txt, skiprows=1)

    @testing.numpy_nlcpy_array_equal
    def test_loadtxt_usecols(self, xp):
        a = testing.shaped_random([2, 3, 4], xp)
        txt = StringIO(str(a).translate(str.maketrans({'[': None, ']': None})))
        return xp.loadtxt(txt, usecols=1)

    @testing.numpy_nlcpy_array_equal
    def test_loadtxt_unpack(self, xp):
        a = testing.shaped_random([2, 3, 4], xp)
        txt = StringIO(str(a).translate(str.maketrans({'[': None, ']': None})))
        return xp.loadtxt(txt, unpack=True)

    @testing.numpy_nlcpy_array_equal
    def test_loadtxt_ndmin(self, xp):
        a = testing.shaped_random([10], xp)
        txt = StringIO(str(a).translate(str.maketrans({'[': None, ']': None})))
        return xp.loadtxt(txt, ndmin=2)

    @testing.numpy_nlcpy_raises
    def test_loadtxt_encoding(self, xp):
        return xp.loadtxt("", encoding="dummy")

    @testing.numpy_nlcpy_array_equal
    def test_loadtxt_maxrows(self, xp):
        a = testing.shaped_random([2, 3, 4], xp)
        txt = StringIO(str(a).translate(str.maketrans({'[': None, ']': None})))
        return xp.loadtxt(txt, max_rows=2)

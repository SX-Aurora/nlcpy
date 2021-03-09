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
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#     THE SOFTWARE.
#


import unittest

import numpy
import nlcpy
from nlcpy import testing


class DummyError(Exception):
    pass


@testing.parameterize(*(
    testing.product({
        'val': [0, 3, -5.2, complex(1.2, -3.4)],
        'casting': ['no', 'equiv', 'safe', 'same_kind', 'unsafe'],
    })
))
class TestCopyScalar(unittest.TestCase):
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_copyto_scalar(self, xp, dst_dtype):
        if numpy.can_cast(self.val, dst_dtype, casting=self.casting):
            dst = xp.asanyarray(-999, dtype=dst_dtype)  # make some 0-dim array
            src = self.val
            xp.copyto(dst, src, casting=self.casting)
            return dst
        else:
            return -1

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_copyto_scalar_masked(self, xp, dst_dtype):
        if numpy.can_cast(self.val, dst_dtype, casting=self.casting):
            dst = xp.asanyarray(-999, dtype=dst_dtype)  # make some 0-dim array
            src = self.val
            where = xp.asanyarray(1, dtype='bool')
            xp.copyto(dst, src, where=where, casting=self.casting)
            return dst
        else:
            return -1

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.numpy_nlcpy_raises()
    def test_copyto_scalar_fail_casting(self, xp, dst_dtype):
        if not numpy.can_cast(self.val, dst_dtype, casting=self.casting):
            dst = xp.asanyarray(-999, dtype=dst_dtype)  # make some 0-dim array
            src = self.val
            xp.copyto(dst, src, casting=self.casting)
        else:
            raise DummyError()


@testing.parameterize(*(
    testing.product({
        'shape': [
            (5,),
            (3, 3),
            (4, 5, 5),
            (0, 4, 4),
            (4, 3, 6, 6),
        ],
        'casting': ['no', 'equiv', 'safe', 'same_kind', 'unsafe']
    })
))
class TestCopyNdarray(unittest.TestCase):
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.for_all_dtypes(name='src_dtype')
    @testing.for_orders('CF', name='dst_order')
    @testing.for_orders('CF', name='src_order')
    @testing.numpy_nlcpy_array_equal()
    def test_copyto_ndarray(self, xp, dst_dtype, src_dtype, dst_order, src_order):
        if numpy.can_cast(src_dtype, dst_dtype, casting=self.casting):
            dst = xp.empty(self.shape, dtype=dst_dtype, order=dst_order)
            src = xp.asarray(
                testing.shaped_random(self.shape, xp, src_dtype), order=src_order)
            xp.copyto(dst, src, casting=self.casting)
            return dst
        else:
            return -1

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.for_all_dtypes(name='src_dtype')
    @testing.for_orders('CF', name='dst_order')
    @testing.for_orders('CF', name='src_order')
    @testing.for_orders('CF', name='where_order')
    @testing.numpy_nlcpy_array_equal()
    def test_copyto_ndarray_masked(self, xp, dst_dtype, src_dtype,
                                   dst_order, src_order, where_order):
        if numpy.can_cast(src_dtype, dst_dtype, casting=self.casting):
            dst = xp.zeros(self.shape, dtype=dst_dtype, order=dst_order)
            src = xp.asarray(
                testing.shaped_random(self.shape, xp, src_dtype), order=src_order)
            where = xp.asarray(
                testing.shaped_random(self.shape, xp, 'bool'), order=where_order)
            xp.copyto(dst, src, where=where, casting=self.casting)
            return dst
        else:
            return -1

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.for_all_dtypes(name='src_dtype')
    @testing.numpy_nlcpy_raises()
    def test_copyto_scalar_fail_casting(self, xp, dst_dtype, src_dtype):
        if not numpy.can_cast(src_dtype, dst_dtype, casting=self.casting):
            dst = xp.empty(self.shape, dtype=dst_dtype)
            src = testing.shaped_random(self.shape, xp, src_dtype)
            xp.copyto(dst, src, casting=self.casting)
        else:
            raise DummyError()


@testing.parameterize(*(
    testing.product({
        'pat_shapes': [
            # (src_shape, dst_shape)
            ((1,), (3,)),
            ((3, 1), (3, 4)),
            ((), (5, 6)),
            ((2, 4), (3, 2, 4)),
            ((1,), (3, 6, 9)),
            ((2, 1, 4), (5, 2, 4, 4)),
            ((4, 1, 1, 6), (4, 3, 7, 6)),
        ],
    })
))
class TestCopyNdarrayBroadcast(unittest.TestCase):
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.for_all_dtypes(name='src_dtype')
    @testing.for_orders('CF', name='dst_order')
    @testing.for_orders('CF', name='src_order')
    @testing.numpy_nlcpy_array_equal()
    def test_copyto_ndarray_broadcast(self, xp, dst_dtype, src_dtype,
                                      dst_order, src_order):
        if numpy.can_cast(src_dtype, dst_dtype):
            src_shape, dst_shape = self.pat_shapes
            dst = xp.empty(dst_shape, dtype=dst_dtype, order=dst_order)
            src = xp.asarray(
                testing.shaped_random(src_shape, xp, src_dtype), order=src_order)
            xp.copyto(dst, src)
            return dst
        else:
            return -1


@testing.parameterize(*(
    testing.product({
        'pat_shapes': [
            # (src_shape, where_shape, dst_shape)
            # (src_shape, where_shape, dst_shape)
            ((), (1,), (3,)),
            ((1,), (1,), (3,)),
            ((3, 1), (3, 4), (3, 4)),
            ((2, 4), (1, 4), (2, 4)),
            ((), (3, 4), (2, 3, 4)),
            ((2, 4, 4), (4, 4), (2, 4, 4)),
            ((5, 2, 3, 4), (2, 1, 4), (5, 2, 3, 4)),
        ],
    })
))
class TestCopyNdarrayBroadcastMasked(unittest.TestCase):
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.for_all_dtypes(name='src_dtype')
    @testing.for_orders('CF', name='dst_order')
    @testing.for_orders('CF', name='src_order')
    @testing.for_orders('CF', name='where_order')
    @testing.numpy_nlcpy_array_equal()
    def test_copyto_ndarray_broadcast_masked(self, xp, dst_dtype, src_dtype,
                                             dst_order, src_order, where_order):
        if numpy.can_cast(src_dtype, dst_dtype):
            src_shape, where_shape, dst_shape = self.pat_shapes
            dst = xp.zeros(dst_shape, dtype=dst_dtype, order=dst_order)
            src = xp.asarray(
                testing.shaped_random(src_shape, xp, src_dtype), order=src_order)
            where = xp.asarray(
                testing.shaped_random(where_shape, xp, 'bool'), order=where_order)
            xp.copyto(dst, src, where=where)
            return dst
        else:
            return -1


class TestCopyOtherDst(unittest.TestCase):
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.for_all_dtypes(name='src_dtype')
    @testing.for_orders('CF', name='dst_order')
    @testing.for_orders('CF', name='src_order')
    @testing.numpy_nlcpy_array_equal()
    def test_copyto_other_dst1(self, xp, src_dtype, dst_dtype, dst_order, src_order):
        if numpy.can_cast(src_dtype, dst_dtype):
            dst = xp.empty(
                (2, 4, 3), dtype=dst_dtype, order=dst_order).transpose(0, 2, 1)
            src = xp.asarray(
                testing.shaped_random((2, 3, 4), xp, src_dtype), order=src_order)
            xp.copyto(dst, src)
            return dst
        else:
            return -1

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.for_all_dtypes(name='src_dtype')
    @testing.for_orders('CF', name='dst_order')
    @testing.for_orders('CF', name='src_order')
    @testing.numpy_nlcpy_array_equal()
    def test_copyto_other_dst2(self, xp, src_dtype, dst_dtype, dst_order, src_order):
        if numpy.can_cast(src_dtype, dst_dtype):
            dst = xp.empty((2, 3, 5, 4), dtype=dst_dtype, order=dst_order)[:, :, 3, :]
            src = xp.asarray(
                testing.shaped_random((2, 3, 4), xp, src_dtype), order=src_order)
            xp.copyto(dst, src)
            return dst
        else:
            return -1


@testing.parameterize(*(
    testing.product({
        'dst': [
            None,
            [[1, 2, 3], [4, 5, 6]],
            ((0.0, 0.1, 0.2), (1.0, 1.1, 1.2)),
        ]
    })
))
class TestCopyIllegaldst1(unittest.TestCase):
    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_raises()
    def test_copyto_illegal_dst1(self, xp):
        dst = self.dst
        src = xp.ones((2, 3))
        xp.copyto(dst, src)


class TestCopyIllegaldst2(unittest.TestCase):
    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_raises()
    def test_copyto_illegal_dst2(self, xp):
        # make opposite ndarray
        if xp == numpy:
            dst = nlcpy.empty(2, 3)
        else:  # xp == "nlcpy"
            dst = numpy.empty(2, 3)
        src = xp.ones((2, 3))
        xp.copyto(dst, src)


@testing.parameterize(*(
    testing.product({
        'src': [
            # some array-like whose shape is (2, 3)
            ((0.0, 0.1, 0.2), (1.0, 1.1, 1.2)),  # tuple (float)
            [[0, 1, 2], [11, 12, 13]],           # list (int)
            [(0, 1j, 2j), (1, 1 + 1j, 1 + 2j)],  # tuple in list (complex)
            numpy.arange(6).reshape(2, 3),       # numpy.ndarray (float64)
        ],
        'dst_shape': [
            (2, 3),
            (2, 2, 3)  # broadcast
        ],
        'casting': ['no', 'equiv', 'safe', 'same_kind', 'unsafe']
    })
))
class TestCopyOtherSrc(unittest.TestCase):
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_copyto_other_src(self, xp, dst_dtype):
        src_dtype = numpy.asanyarray(self.src).dtype
        if numpy.can_cast(src_dtype, dst_dtype, casting=self.casting):
            dst = xp.empty(self.dst_shape, dtype=dst_dtype)
            xp.copyto(dst, self.src, casting=self.casting)
            return dst
        else:
            return -1

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.numpy_nlcpy_raises()
    def test_copyto_other_src_fail_casting(self, xp, dst_dtype):
        src_dtype = numpy.asanyarray(self.src).dtype
        if not numpy.can_cast(src_dtype, dst_dtype, casting=self.casting):
            dst = xp.empty(self.dst_shape, dtype=dst_dtype)
            xp.copyto(dst, self.src, casting=self.casting)
        else:
            raise DummyError()


class TestCopyIllegalSrc(unittest.TestCase):
    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_raises()
    def test_copyto_illegal_src1(self, xp):
        dst = xp.empty((2, 3))
        src = None
        xp.coptyto(dst, src, where=self.where)

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_raises()
    def test_copyto_illegal_src2(self, xp):
        dst = xp.empty((2, 3))
        src = xp.ones((2, 4))  # cann not broadcast
        xp.coptyto(dst, src, where=self.where)


@testing.parameterize(*(
    testing.product({
        'where': [
            None,
            True,
            [False, True, False],
            ((1,), (0,),),
            numpy.ones((2, 3), dtype='bool'),  # numpy.ndarray
        ]
    })
))
class TestCopyOtherWhere(unittest.TestCase):
    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_array_equal()
    def test_copyto_other_where(self, xp):
        dst = xp.zeros((2, 3))
        src = xp.ones((2, 3))
        where = self.where
        xp.copyto(dst, src, where=where)
        return dst


class TestCopyIllegalWhere(unittest.TestCase):
    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_raises()
    def test_copyto_illegal_where1(self, xp):
        dst = xp.empty((2, 3))
        src = xp.ones((2, 3))
        where = xp.ones((2, 4), dtype="bool")  # cann not braodcast
        xp.copyto(dst, src, where=where)

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_raises()
    def test_copyto_illegal_where2(self, xp):
        dst = xp.empty((2, 3))
        src = xp.ones((2, 3))
        where = xp.ones((2, 3), dtype="int64")  # dtype is not "bool"
        xp.copyto(dst, src, where=where)

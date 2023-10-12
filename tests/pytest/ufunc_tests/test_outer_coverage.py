#
# * The source code in this file is developed independently by NEC Corporation.
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

import unittest
import numpy
import nlcpy
from nlcpy import testing


class TestOuterCoverage(unittest.TestCase):

    @testing.numpy_nlcpy_allclose()
    def test_outer_out_tuple(self, xp):
        a = xp.array([1, 2, 3], dtype='f8')
        o = (xp.empty((3, 3)),)
        xp.add.outer(a, a, out=o)
        return o[0]

    def test_outer_out_tuple_2elems(self):
        a = nlcpy.array([1, 2, 3], dtype='f8')
        with self.assertRaises(ValueError):
            nlcpy.add.outer(a, a, out=(1, 2))

    def test_outer_out_not_array(self):
        a = nlcpy.array([1, 2, 3], dtype='f8')
        with self.assertRaises(TypeError):
            nlcpy.add.outer(a, a, out=1)

    def test_outer_ldexp_b_not_u8(self):
        a = nlcpy.array([1, 2, 3], dtype='f8')
        b = nlcpy.array([1, 2, 3], dtype='f8')
        with self.assertRaises(TypeError):
            nlcpy.ldexp.outer(a, b)
        b = nlcpy.array([1, 2, 3], dtype='c8')
        with self.assertRaises(TypeError):
            nlcpy.ldexp.outer(a, b)

    def test_outer_ldexp_0(self):
        a = nlcpy.array([1, 0, 1], dtype='bool')
        b = nlcpy.array([1, 2, 3], dtype='f8')
        with self.assertRaises(TypeError):
            nlcpy.ldexp.outer(a, b)

    def test_outer_ldexp_1(self):
        a = nlcpy.array([1, 0, 1], dtype='i8')
        b = nlcpy.array([1, 2, 3], dtype='u8')
        with self.assertRaises(TypeError):
            nlcpy.ldexp.outer(a, b)

    def test_outer_no_attribute(self):
        a = nlcpy.array([1, 0, 1], dtype='i8')
        b = nlcpy.array([1, 2, 3], dtype='u8')
        with self.assertRaises(AttributeError):
            nlcpy.sin.outer(a, b)

    def test_outer_calc_dtype_failure_0(self):
        a = nlcpy.array([1, 2, 3], dtype='f8')
        o = nlcpy.empty((3, 3), dtype='bool')
        with self.assertRaises(TypeError):
            nlcpy.add.outer(a, a, out=o, dtype='i2')

    def test_outer_calc_dtype_failure_1(self):
        a = nlcpy.array([1, 2, 3], dtype='f8')
        o = nlcpy.empty((3, 3), dtype='bool')
        with self.assertRaises(TypeError):
            nlcpy.divide.outer(a, a, out=o, dtype='bool')

    @testing.numpy_nlcpy_allclose()
    def test_outer_calc_dtype_promote_0(self, xp):
        a = xp.array([1, 2, 3], dtype='i4')
        o = xp.empty((3, 3), dtype='i4')
        xp.add.outer(a, a, out=o, dtype='i2')
        return o

    @testing.numpy_nlcpy_allclose()
    def test_outer_calc_dtype_promote_1(self, xp):
        a = xp.array([1, 2, 3], dtype='f4')
        o = xp.empty((3, 3), dtype='f4')
        xp.add.outer(a, a, out=o, dtype='f2')
        return o

    def test_outer_dtype_cast_failure_0(self):
        a = nlcpy.array([1, 2, 3], dtype='i4')
        b = nlcpy.array([1, 2, 3], dtype='f8')
        o = nlcpy.empty((3, 3), dtype='i4')
        with self.assertRaises(TypeError):
            nlcpy.add.outer(a, b, out=o, dtype='i2')

    def test_outer_dtype_cast_failure_1(self):
        a = nlcpy.array([1, 2, 3], dtype='f4')
        b = nlcpy.array([1, 2, 3], dtype='f4')
        o = nlcpy.empty((3, 3), dtype='i4')
        with self.assertRaises(TypeError):
            nlcpy.add.outer(a, b, out=o, dtype='f4')

    @testing.numpy_nlcpy_allclose()
    def test_outer_floor_divide_0(self, xp):
        a = xp.array([1, 2, 3], dtype='i4')
        return xp.floor_divide.outer(a, a)

    def test_outer_floor_divide_1(self):
        a = nlcpy.array([1, 2, 3], dtype='i4')
        b = nlcpy.array([1, 2, 3], dtype='u8')
        o = nlcpy.empty((3, 3), dtype='i8')
        nlcpy.floor_divide.outer(a, b, out=o)
        desired = [[1, 0, 0], [2, 1, 0], [3, 1, 1]]
        testing.assert_array_equal(o, desired)

    @testing.numpy_nlcpy_allclose()
    def test_outer_out_bcast(self, xp):
        a = xp.ones((2, 1), dtype='i4')
        o = xp.empty((2, 2, 2, 1), dtype='i4')
        xp.add.outer(a, a, out=o)
        return o

    def test_outer_out_bcast_failure(self):
        a = nlcpy.ones((2, 1), dtype='i4')
        o = nlcpy.empty((2, 2, 3), dtype='i4')
        with self.assertRaises(ValueError):
            nlcpy.add.outer(a, a, out=o)

    def test_outer_unknown_order(self):
        with self.assertRaises(ValueError):
            nlcpy.add.outer([1, 2], [1, 2], order='Z')

    def test_outer_where_cast_failure(self):
        a = nlcpy.ones([1, 2, 3])
        w = nlcpy.array([1, 2, 3], dtype='f4')
        with self.assertRaises(TypeError):
            nlcpy.add.outer(a, a, where=w)

    @testing.numpy_nlcpy_allclose()
    def test_outer_where_all_false(self, xp):
        a = xp.ones(3)
        o = xp.ones((3, 3))
        w = xp.array([False, False, False], dtype='bool')
        return xp.add.outer(a, a, out=o, where=w)

    @testing.numpy_nlcpy_allclose()
    def test_outer_where(self, xp):
        a = xp.array([1, 2, 3])
        o = xp.zeros((3, 3))
        w = xp.array([True, False, True], dtype='bool')
        return xp.add.outer(a, a, out=o, where=w)

    def test_outer_where_bcast_failure(self):
        a = nlcpy.array([1, 2, 3])
        o = nlcpy.zeros((3, 3))
        w = nlcpy.array([True, False], dtype='bool')
        with self.assertRaises(ValueError):
            nlcpy.add.outer(a, a, out=o, where=w)

    def test_outer_size_0(self):
        a = nlcpy.array([1, 2, 3])
        b = nlcpy.array([])
        assert nlcpy.add.outer(a, b).size == 0
        assert nlcpy.add.outer(b, a).size == 0

    @testing.numpy_nlcpy_allclose()
    def test_outer_not_contiguous(self, xp):
        a = xp.ones((3, 3), dtype='i4')
        b = xp.arange(25, dtype='i4').reshape(5, 5)
        return xp.add.outer(a[:, ::2], b[:, ::2])

    def test_outer_bool_bool_failure(self):
        a = nlcpy.array([True, False, True], dtype='bool')
        b = nlcpy.array([False, False, True], dtype='bool')
        with self.assertRaises(TypeError):
            nlcpy.floor_divide.outer(a, b)
        with self.assertRaises(TypeError):
            nlcpy.arctan2.outer(a, b)
        with self.assertRaises(AttributeError):
            nlcpy.sin.outer(a, b)

    @testing.numpy_nlcpy_allclose()
    def test_outer_bool_bool_divide(self, xp):
        a = xp.array([True, False, True], dtype='bool')
        b = xp.array([True, True, True], dtype='bool')
        return xp.divide.outer(a, b)

    @testing.numpy_nlcpy_allclose()
    def test_outer_np_ndarray_scalar(self, xp):
        a = numpy.array(1)
        return xp.add.outer(a, a)

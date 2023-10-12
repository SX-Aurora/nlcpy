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
import warnings
import numpy
import nlcpy
from nlcpy import testing


class TestReduceCoverage(unittest.TestCase):

    @testing.numpy_nlcpy_allclose()
    def test_reduce_axis_ndarray(self, xp):
        a = xp.array([1, 2, 3], dtype='i8')
        return xp.add.reduce(a, axis=xp.array(0))

    @testing.numpy_nlcpy_allclose()
    def test_reduce_copyto(self, xp):
        a = xp.array([[1, 2], [3, 4]], dtype='i8')
        o = xp.array(0, dtype='f8')
        xp.add.reduce(a, out=o, dtype='f4', axis=None)
        return o

    def test_reduce_flexible_dtype(self):
        with self.assertRaises(TypeError):
            nlcpy.add.reduce([1, 2], dtype=numpy.dtype("i4, f4"))

    def test_reduce_subtract_failure_0(self):
        a = nlcpy.array([[1, 2], [3, 4]], dtype='i8')
        with self.assertRaises(TypeError):
            nlcpy.subtract.reduce(a, dtype='bool')

    def test_reduce_subtract_failure_1(self):
        a = nlcpy.array([[1, 2], [3, 4]], dtype='i8')
        o = nlcpy.empty(2, dtype='bool')
        with self.assertRaises(TypeError):
            nlcpy.subtract.reduce(a, out=o)

    def test_reduce_subtract_failure_2(self):
        a = nlcpy.array([[1, 0], [0, 1]], dtype='bool')
        with self.assertRaises(TypeError):
            nlcpy.subtract.reduce(a)

    @testing.with_requires('numpy<1.20')
    @testing.numpy_nlcpy_allclose()
    def test_reduce_where_tuple(self, xp):
        a = xp.array([1, 2, 3])
        w = (xp.array([True, False, True]),)
        return xp.add.reduce(a, where=w)

    def test_reduce_keepdims_not_integer(self):
        a = nlcpy.array([1, 2, 3])
        with self.assertRaises(TypeError):
            nlcpy.add.reduce(a, keepdims=1.2)

    def test_reduce_keepdims_not_size_1(self):
        a = nlcpy.ones((2, 3, 2))
        with self.assertRaises(TypeError):
            nlcpy.add.reduce(a, keepdims=(0, 1))

    def test_reduce_axis_size_zero(self):
        a = nlcpy.ones((2, 0, 2))
        with self.assertRaises(ValueError):
            nlcpy.add.reduce(a, axis=1, initial=None)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array(self, xp):
        a = xp.array([])
        return xp.add.reduce(a)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_out(self, xp):
        a = xp.array([])
        o = xp.array(0)
        return xp.add.reduce(a, out=o)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_boolean_op(self, xp):
        a = xp.array([])
        return xp.logical_and.reduce(a)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_dtype(self, xp):
        a = xp.array([], dtype='i4')
        return xp.add.reduce(a, dtype='f4')

    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_add_bool(self, xp):
        a = xp.array([], dtype='bool')
        return xp.add.reduce(a)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_add_u32(self, xp):
        a = xp.array([], dtype='u4')
        return xp.add.reduce(a)

    @testing.with_requires('numpy<1.20')
    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_initial_complex(self, xp):
        a = xp.array([])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            return xp.add.reduce(a, initial=1 + 1j, axis=0, dtype='f8')

    @testing.with_requires('numpy<1.20')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_initial_inf(self, xp, dtype):
        a = xp.array([])
        return xp.add.reduce(a, initial=xp.inf, dtype=dtype)

    @testing.with_requires('numpy<1.20')
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_initial_neg_inf(self, xp, dtype):
        a = xp.array([])
        return xp.add.reduce(a, initial=-xp.inf, dtype=dtype)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_initial_novalue_0(self, xp):
        a = xp.array([])
        return xp.add.reduce(a, initial=xp._NoValue)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_initial_novalue_1(self, xp):
        a = xp.empty((2, 3, 0))
        return xp.minimum.reduce(a, axis=0, initial=xp._NoValue)

    @testing.with_requires('numpy<1.20')
    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_initial_np_inf(self, xp):
        a = xp.array([1, 2, 3])
        return xp.add.reduce(a, initial=numpy.inf)

    @testing.with_requires('numpy<1.20')
    @testing.numpy_nlcpy_allclose()
    def test_reduce_zero_size_array_initial_np_neg_inf(self, xp):
        a = xp.array([1, 2, 3])
        return xp.add.reduce(a, initial=-numpy.inf)

    def test_reduce_initial_not_scalar(self):
        a = nlcpy.array([1, 2, 3])
        with self.assertRaises(ValueError):
            nlcpy.add.reduce(a, initial=(1, 2))

    @testing.numpy_nlcpy_allclose()
    def test_reduce_power_scalar(self, xp):
        a = xp.array(1, dtype='i4')
        return xp.power.reduce(a, initial=1, dtype='i4')

    @testing.numpy_nlcpy_allclose()
    def test_reduce_power_scalar_where_false(self, xp):
        a = xp.array(1, dtype='i4')
        return xp.power.reduce(a, initial=1, dtype='i4', where=False)

    def test_reduce_power_integer_to_negative(self):
        a = nlcpy.array(-11, dtype='i4')
        with self.assertRaises(ValueError):
            return nlcpy.power.reduce(
                a, initial=1, dtype='i4', where=True)

    # @testing.numpy_nlcpy_allclose()
    # def test_reduce_hypot_where_axis_0(self, xp):
    #     a = xp.ones((2, 2, 2), dtype='f4')
    #     w = xp.ones((2, 2, 2), dtype='bool')
    #     return xp.hypot.reduce(a, where=w, axis=(0, 1))

    @testing.numpy_nlcpy_allclose()
    def test_reduce_hypot_where_axis_1(self, xp):
        a = xp.ones((2, 2, 2), dtype='f4')
        w = xp.zeros((2, 2, 2), dtype='bool')
        return xp.hypot.reduce(a, where=w, axis=(0, 1, 2))

    @testing.numpy_nlcpy_allclose()
    def test_reduce_divide_0(self, xp):
        a = xp.array([1, 2], dtype='f4')
        return xp.divide.reduce(a, dtype='f8')

    @testing.with_requires('numpy<1.20')
    @testing.numpy_nlcpy_allclose()
    def test_reduce_logical_and_0(self, xp):
        a = xp.array([1, 2], dtype='i4')
        return xp.logical_and.reduce(a, dtype='f8')

    @testing.numpy_nlcpy_allclose()
    def test_reduce_add_0(self, xp):
        a = xp.array([1, 2], dtype='i4')
        return xp.add.reduce(a)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_add_1(self, xp):
        a = xp.array([1, 2], dtype='u4')
        return xp.add.reduce(a)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_add_2(self, xp):
        a = xp.array([1, 2], dtype='i4')
        return xp.add.reduce(a, initial=xp._NoValue)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_first_loop_0(self, xp):
        a = xp.ones((1, 2, 3), dtype='i4')
        return xp.maximum.reduce(a, axis=0, initial=xp._NoValue)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_first_loop_1(self, xp):
        a = xp.ones((1, 2, 3), dtype='i4')
        o = xp.zeros((2, 3))
        return xp.maximum.reduce(a, axis=0, initial=xp._NoValue, out=o)

    @testing.numpy_nlcpy_allclose()
    def test_reduce_first_loop_2(self, xp):
        a = xp.ones((1, 2, 3), dtype='i4')
        o = xp.zeros(3)
        return xp.maximum.reduce(a, axis=(0, 1), initial=xp._NoValue, out=o)

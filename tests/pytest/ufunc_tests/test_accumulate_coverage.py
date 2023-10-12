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
import nlcpy
from nlcpy import testing


class TestAccumulateCoverage(unittest.TestCase):

    def test_accumulate_not_binary_op(self):
        with self.assertRaises(ValueError):
            nlcpy.invert.accumulate(1)
        with self.assertRaises(ValueError):
            nlcpy.logical_not.accumulate(1)

    def test_accumulate_out_not_array(self):
        x = nlcpy.array([1, 2])
        with self.assertRaises(TypeError):
            nlcpy.add.accumulate(x, out=1)
        with self.assertRaises(ValueError):
            nlcpy.add.accumulate(x, out=(0, 1))
        with self.assertRaises(TypeError):
            nlcpy.add.accumulate(x, out=(0,))

    def test_accumulate_out_tuple(self):
        x = nlcpy.array([1, 2, 3], dtype='i8')
        out = (nlcpy.empty(3, dtype='i8'),)
        actual = nlcpy.add.accumulate(x, out=out)
        desired = nlcpy.array([1, 3, 6], dtype='i8')
        testing.assert_array_equal(actual, desired)
        testing.assert_array_equal(out[0], desired)

    def test_accumulate_out_shape_mismatch(self):
        x = nlcpy.array([1, 2], dtype='i8')
        out = nlcpy.empty(3, dtype='i8')
        with self.assertRaises(ValueError):
            nlcpy.divide.accumulate(x, out=out)
        with self.assertRaises(ValueError):
            nlcpy.add.accumulate(x, out=out)

    @testing.numpy_nlcpy_allclose()
    def test_accumulate_divide_with_out(self, xp):
        x = xp.array([1, 2, 3], dtype='f8')
        out = xp.empty(3, dtype='f8')
        xp.divide.accumulate(x, out=out)
        return out

    def test_accumulate_in_above_out_ndim(self):
        x = nlcpy.empty((2, 3, 4))
        out = nlcpy.empty((2, 3))
        with self.assertRaises(ValueError):
            nlcpy.add.accumulate(x, out=out)

    def test_accumulate_dtype_not_supported(self):
        with self.assertRaises(TypeError):
            nlcpy.add.accumulate([1, 2], dtype='i1')
        with self.assertRaises(TypeError):
            nlcpy.add.accumulate([1, 2], dtype='i2')
        with self.assertRaises(TypeError):
            nlcpy.add.accumulate([1, 2], dtype='f2')

    def test_accumulate_fp16_i8_not_supported_op(self):
        with self.assertRaises(TypeError):
            nlcpy.divide.accumulate([1, 2], dtype='bool')
        with self.assertRaises(TypeError):
            nlcpy.floor_divide.accumulate([1, 2], dtype='bool')

    @testing.with_requires('numpy<1.20')
    @testing.numpy_nlcpy_allclose()
    def test_accumulate_work_array_0(self, xp):
        x = xp.array([1, 2, 3], dtype='f8')
        o = xp.empty(3, dtype='bool')
        xp.divide.accumulate(x, out=o, dtype='bool')
        return o

    @testing.with_requires('numpy<1.20')
    @testing.numpy_nlcpy_allclose()
    def test_accumulate_work_array_1(self, xp):
        x = xp.array([1, 2, 3], dtype='f8')
        o = xp.empty(3, dtype='i4')
        xp.divide.accumulate(x, out=o, dtype='i4')
        return o

    @testing.numpy_nlcpy_allclose()
    def test_accumulate_work_array_2(self, xp):
        x = xp.array([1, 2, 3], dtype='f8')
        o = xp.empty(3, dtype='bool')
        xp.greater.accumulate(x, out=o, dtype='bool')
        return o

    def test_accumulate_axis_list(self):
        x = nlcpy.empty((2, 3, 4))
        with self.assertRaises(TypeError):
            nlcpy.add.accumulate(x, axis=[0, 1])

    @testing.numpy_nlcpy_allclose()
    def test_accumulate_axis_array_0d(self, xp):
        x = xp.array([[2, 3], [1, 2]])
        axis = xp.array(0)
        return xp.add.accumulate(x, axis=axis)

    def test_accumulate_axis_array_above_0d(self):
        x = nlcpy.empty((2, 3, 4))
        axis = nlcpy.array([0, 1])
        with self.assertRaises(TypeError):
            nlcpy.add.accumulate(x, axis=axis)

    def test_accumulate_axis_tuple_failure(self):
        x = nlcpy.array([[2, 3], [1, 2]])
        with self.assertRaises(TypeError):
            nlcpy.add.accumulate(x, axis=(nlcpy.array(1.),))
        with self.assertRaises(TypeError):
            nlcpy.add.accumulate(x, axis=(nlcpy.array([[1], [2]]),))
        with self.assertRaises(nlcpy.AxisError):
            nlcpy.add.accumulate(x, axis=(0, 2))
        with self.assertRaises(ValueError):
            nlcpy.add.accumulate(x, axis=(0, 1))

    def test_accumulate_axis_none_above_1d_array(self):
        x = nlcpy.empty((2, 3, 4))
        with self.assertRaises(ValueError):
            nlcpy.add.accumulate(x, axis=None)

    @testing.numpy_nlcpy_allclose()
    def test_accumulate_axis_none(self, xp):
        x = xp.array([1, 2, 3])
        return xp.add.accumulate(x, axis=None)

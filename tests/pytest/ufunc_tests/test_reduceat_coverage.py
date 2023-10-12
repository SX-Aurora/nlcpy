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


class TestReduceatCoverage(unittest.TestCase):

    def test_reduceat_a_none(self):
        with self.assertRaises(TypeError):
            nlcpy.add.reduceat(None, 1)

    def test_reduceat_indices_none(self):
        with self.assertRaises(ValueError):
            nlcpy.add.reduceat([1, 2], None)

    def test_reduceat_indices_too_deep(self):
        with self.assertRaises(ValueError):
            nlcpy.add.reduceat([1, 2], [[1, 0], [0, 1]])

    def test_reduceat_indices_too_small_depth(self):
        with self.assertRaises(ValueError):
            nlcpy.add.reduceat([1, 2], nlcpy.array(0))

    def test_reduceat_a_size_0_indices_not_0(self):
        with self.assertRaises(IndexError):
            nlcpy.add.reduceat(nlcpy.array([]), nlcpy.array([0, 1]))

    def test_reduceat_axis_list(self):
        with self.assertRaises(TypeError):
            nlcpy.add.reduceat([1, 2], [0, 1], axis=[0,])

    def test_reduceat_axis_not_integer_array(self):
        with self.assertRaises(TypeError):
            nlcpy.add.reduceat([1, 2], [0, 1], axis=nlcpy.array(1.1))

    @testing.numpy_nlcpy_allclose()
    def test_reduceat_axis_scalar_array(self, xp):
        return xp.add.reduceat([1, 2], [0, 1], axis=xp.array(0))

    def test_reduceat_axis_tuple_not_integer_array(self):
        with self.assertRaises(TypeError):
            nlcpy.add.reduceat([1, 2], [0, 1], axis=(nlcpy.array(1.1),))

    def test_reduceat_axis_tuple_out_of_bounds(self):
        with self.assertRaises(nlcpy.AxisError):
            nlcpy.add.reduceat([1, 2], [0, 1], axis=(-1,))

    def test_reduceat_axis_tuple_failure(self):
        with self.assertRaises(ValueError):
            nlcpy.add.reduceat([1, 2], [0, 1], axis=(0, 0))

    def test_reduceat_axis_none_a_ndim(self):
        with self.assertRaises(ValueError):
            nlcpy.add.reduceat([1, 2], [0, 1], axis=None)

    def test_reduceat_out_not_array(self):
        with self.assertRaises(TypeError):
            nlcpy.add.reduceat([1, 2], [0, 1], out=1)

    def test_reduceat_out_tuple_multi_entry(self):
        with self.assertRaises(ValueError):
            nlcpy.add.reduceat([1, 2], [0, 1], out=(1, 2))

    def test_reduceat_out_tuple_one_entry_not_array(self):
        with self.assertRaises(TypeError):
            nlcpy.add.reduceat([1, 2], [0, 1], out=(1,))

    @testing.numpy_nlcpy_allclose()
    def test_reduceat_out_tuple_one_entry_array(self, xp):
        o = xp.empty(2)
        xp.add.reduceat([1, 2], [0, 1], out=(o,))
        return o

    def test_reduceat_out_ndim_too_small(self):
        with self.assertRaises(ValueError):
            nlcpy.add.reduceat([1, 2], [0, 1], out=(nlcpy.array(0),))

    @testing.numpy_nlcpy_allclose()
    def test_reduceat_f_order(self, xp):
        a = xp.ones((3, 3), order='f')
        return xp.add.reduceat(a, [0, 1, 2]).tolist()

    @testing.with_requires('numpy<1.20')
    @testing.numpy_nlcpy_allclose()
    def test_reduceat_fmod_bool(self, xp):
        a = xp.array([1, 2])
        o = xp.empty(2, dtype='i4')
        return xp.floor_divide.reduceat(a, [0, 1], out=o, dtype='bool')

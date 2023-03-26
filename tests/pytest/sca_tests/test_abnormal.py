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

import numpy
import unittest

import nlcpy
from nlcpy import testing
from nlcpy.testing.types import float_types
from nlcpy.testing.types import no_float_types


class TestOptimizeLeading(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_convert_optimized_0d(self, dtype):
        xin = nlcpy.array(1, dtype=dtype)
        with self.assertRaises(ValueError):
            _ = nlcpy.sca.convert_optimized_array(xin)

    @testing.for_all_dtypes()
    def test_convert_optimized_5d(self, dtype):
        xin = nlcpy.random.rand(2, 3, 2, 3, 2).astype(dtype)
        with self.assertRaises(ValueError):
            _ = nlcpy.sca.convert_optimized_array(xin)

    @testing.for_all_dtypes()
    def test_create_optimized_size0(self, dtype):
        with self.assertRaises(ValueError):
            _ = nlcpy.sca.create_optimized_array(0)

    @testing.for_all_dtypes()
    def test_create_optimized_5d(self, dtype):
        with self.assertRaises(ValueError):
            _ = nlcpy.sca.create_optimized_array((2, 3, 2, 3, 2))

    @testing.for_dtypes(float_types)
    def test_convert_valid_dtype(self, dtype):
        xin = nlcpy.random.rand(2, 3)
        xopt = nlcpy.sca.convert_optimized_array(xin, dtype=dtype)
        assert xopt.dtype == dtype
        assert xopt.shape == xin.shape

    @testing.for_dtypes(no_float_types)
    def test_convert_invalid_dtype(self, dtype):
        xin = nlcpy.random.rand(2, 3)
        with self.assertRaises(TypeError):
            _ = nlcpy.sca.convert_optimized_array(xin, dtype=dtype)

    @testing.for_dtypes(float_types)
    def test_create_valid_dtype(self, dtype):
        shape_in = (2, 3)
        xopt = nlcpy.sca.create_optimized_array(shape_in, dtype=dtype)
        assert xopt.dtype == dtype
        assert xopt.shape == shape_in

    @testing.for_dtypes(no_float_types)
    def test_create_invalid_dtype(self, dtype):
        shape_in = (2, 3)
        with self.assertRaises(TypeError):
            _ = nlcpy.sca.create_optimized_array(shape_in, dtype=dtype)

    @testing.for_all_dtypes(name='dt1')
    @testing.for_dtypes(float_types, name='dt2')
    def test_convert_not_contiguous_1(self, dt1, dt2):
        xbase = nlcpy.random.rand(5, 6, 5, 6).astype(dt1)
        xin = xbase[::2, ::3, ::2, ::3]
        xopt = nlcpy.sca.convert_optimized_array(xin, dtype=dt2)
        testing.assert_allclose(xin, xopt)
        assert xopt.strides != xin.strides
        assert xopt.dtype == dt2

    @testing.for_all_dtypes(name='dt1')
    @testing.for_dtypes(float_types, name='dt2')
    def test_convert_not_contiguous_2(self, dt1, dt2):
        xbase = nlcpy.random.rand(3, 4, 3, 4).astype(dt1)
        xin = nlcpy.moveaxis(xbase, 0, 2)
        xopt = nlcpy.sca.convert_optimized_array(xin, dtype=dt2)
        testing.assert_allclose(xin, xopt)
        assert xopt.strides != xin.strides
        assert xopt.dtype == dt2

    @testing.for_dtypes(float_types)
    def test_convert_no_dtype(self, dtype):
        xin = nlcpy.random.rand(2, 3).astype(dtype=dtype)
        xopt = nlcpy.sca.convert_optimized_array(xin)
        assert xopt.dtype == dtype
        assert xopt.shape == xin.shape

    @testing.for_dtypes(float_types)
    def test_convert_from_numpy(self, dtype):
        xin = numpy.random.rand(2, 3).astype(dtype=dtype)
        xopt = nlcpy.sca.convert_optimized_array(xin)
        assert xopt.dtype == dtype
        assert xopt.shape == xin.shape
        assert type(xin) != type(xopt)


class TestCreateDescriptor(unittest.TestCase):

    @testing.for_dtypes(float_types)
    def test_create_descriptor_0d(self, dtype):
        xin = nlcpy.array(1, dtype=dtype)
        with self.assertRaises(ValueError):
            _ = nlcpy.sca.create_descriptor(xin)

    @testing.for_dtypes(float_types)
    def test_create_descriptor_5d(self, dtype):
        xin = nlcpy.random.rand(2, 3, 2, 3, 2).astype(dtype)
        with self.assertRaises(ValueError):
            _ = nlcpy.sca.create_descriptor(xin)

    @testing.for_dtypes(float_types)
    def test_create_descriptor_not_monotonously_increasing(self, dtype):
        xin = nlcpy.random.rand(2, 3, 2, 3).astype(dtype).T
        with self.assertRaises(ValueError):
            _ = nlcpy.sca.create_descriptor(xin)

    @testing.for_dtypes(float_types)
    def test_create_descriptor_indivisible(self, dtype):
        xin = nlcpy.random.rand(4, 5).astype(dtype)[:, ::2]
        with self.assertRaises(ValueError):
            _ = nlcpy.sca.create_descriptor(xin)

    @testing.for_dtypes(float_types)
    def test_create_descriptor_divisible_1(self, dtype):
        xin = nlcpy.random.rand(5).astype(dtype)[::2]
        _ = nlcpy.sca.create_descriptor(xin)

    @testing.for_dtypes(float_types)
    def test_create_descriptor_divisible_2(self, dtype):
        xin = nlcpy.random.rand(2, 4).astype(dtype)[::2]
        _ = nlcpy.sca.create_descriptor(xin)

    @testing.for_dtypes(float_types)
    def test_create_descriptor_diff_ndim(self, dtype):
        xin1 = nlcpy.random.rand(2, 3).astype(dtype=dtype)
        xin2 = nlcpy.random.rand(2, 3, 2).astype(dtype=dtype)
        with self.assertRaises(ValueError):
            _, _ = nlcpy.sca.create_descriptor((xin1, xin2))

    @testing.for_dtypes(float_types)
    def test_create_descriptor_from_numpy(self, dtype):
        xin1 = numpy.random.rand(2, 3).astype(dtype=dtype)
        with self.assertRaises(TypeError):
            _ = nlcpy.sca.create_descriptor((xin1))


class TestCreateDescription(unittest.TestCase):

    @testing.for_dtypes(float_types)
    def test_out_of_index_axis_xn(self, dtype):
        xin = nlcpy.random.rand(2, 3, 2, 3).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        with self.assertRaises(IndexError):
            dx[..., -3]

    @testing.for_dtypes(float_types)
    def test_out_of_index_axis_xp(self, dtype):
        xin = nlcpy.random.rand(2, 3, 2, 3).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        with self.assertRaises(IndexError):
            dx[..., 3]

    @testing.for_dtypes(float_types)
    def test_out_of_index_axis_yn(self, dtype):
        xin = nlcpy.random.rand(2, 3, 2, 3).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        with self.assertRaises(IndexError):
            dx[..., -2, :]

    @testing.for_dtypes(float_types)
    def test_out_of_index_axis_yp(self, dtype):
        xin = nlcpy.random.rand(2, 3, 2, 3).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        with self.assertRaises(IndexError):
            dx[..., 2, :]

    @testing.for_dtypes(float_types)
    def test_out_of_index_axis_zn(self, dtype):
        xin = nlcpy.random.rand(2, 3, 2, 3).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        with self.assertRaises(IndexError):
            dx[:, -3, ...]

    @testing.for_dtypes(float_types)
    def test_out_of_index_axis_zp(self, dtype):
        xin = nlcpy.random.rand(2, 3, 2, 3).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        with self.assertRaises(IndexError):
            dx[:, 3, ...]

    @testing.for_dtypes(float_types)
    def test_out_of_index_axis_wn(self, dtype):
        xin = nlcpy.random.rand(2, 3, 2, 3).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        with self.assertRaises(IndexError):
            dx[-2, ...]

    @testing.for_dtypes(float_types)
    def test_out_of_index_axis_wp(self, dtype):
        xin = nlcpy.random.rand(2, 3, 2, 3).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        with self.assertRaises(IndexError):
            dx[2, ...]

    @testing.for_dtypes(float_types)
    def test_diff_dtypes(self, dtype):
        xin1 = nlcpy.random.rand(2, 3).astype(dtype=dtype)
        if dtype == nlcpy.dtype('f4'):
            xin2 = nlcpy.random.rand(2, 3).astype(dtype='f8')
        elif dtype == nlcpy.dtype('f8'):
            xin2 = nlcpy.random.rand(2, 3).astype(dtype='f4')
        else:
            raise TypeError
        dxin1, dxin2 = nlcpy.sca.create_descriptor((xin1, xin2))
        with self.assertRaises(TypeError):
            dxin1[...] + dxin2[...]

    @testing.for_dtypes(float_types)
    def test_invalid_divide_1(self, dtype):
        xin = nlcpy.random.rand(2, 3).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        with self.assertRaises(TypeError):
            1 / dx[...]

    @testing.for_dtypes(float_types)
    def test_invalid_divide_2(self, dtype):
        xin = nlcpy.random.rand(2, 3).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        coef = nlcpy.array(1, dtype=dtype)
        with self.assertRaises(TypeError):
            coef / dx[...]

    @testing.for_dtypes(float_types, name='dtype1')
    @testing.for_dtypes(no_float_types, name='dtype2')
    def test_invalid_dtype_for_coef(self, dtype1, dtype2):
        xin = nlcpy.random.rand(2, 3).astype(dtype1)
        dx = nlcpy.sca.create_descriptor(xin)
        coef = nlcpy.array(1, dtype=dtype2)
        with self.assertRaises(TypeError):
            dx[...] * coef

    @testing.for_dtypes(float_types)
    def test_assign_multiple_coef_for_single_description(self, dtype):
        xin = nlcpy.random.rand(2, 3).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        coef1 = nlcpy.array(1, dtype=dtype)
        coef2 = nlcpy.array(2, dtype=dtype)
        with self.assertRaises(TypeError):
            dx[...] * coef1 * coef2

    @testing.for_dtypes(float_types)
    def test_assign_multiple_coef_for_multiple_description(self, dtype):
        xin = nlcpy.arange(10).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        coef1 = nlcpy.array(-1, dtype=dtype)
        coef2 = nlcpy.array(2, dtype=dtype)
        coef3 = nlcpy.array(3, dtype=dtype)
        desc = dx[0] * coef1 + dx[0] * coef2 + dx[0] * coef3
        res_sca = nlcpy.sca.create_kernel(desc).execute()
        res_naive = xin * coef1 + xin * coef2 + xin * coef3
        testing.assert_allclose(res_sca, res_naive)

    @testing.for_dtypes(float_types)
    def test_assign_numpy_factor(self, dtype):
        xin = nlcpy.arange(10).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        coef = numpy.array(-1, dtype=dtype)
        desc = dx[0] * coef
        res_sca = nlcpy.sca.create_kernel(desc).execute()
        res_naive = xin * coef
        testing.assert_allclose(res_sca, res_naive)

    @testing.for_dtypes(float_types)
    def test_coef_array_invalid_shape_1(self, dtype):
        xin = nlcpy.random.rand(5, 6).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        coef_array = nlcpy.random.rand(3, 6).astype(dtype)
        desc = (dx[-1, 0] + dx[1, 0] + dx[0, -1] + dx[0, 1]) * coef_array
        with self.assertRaises(ValueError):
            nlcpy.sca.create_kernel(desc)

    @testing.for_dtypes(float_types)
    def test_coef_array_invalid_shape_2(self, dtype):
        xin = nlcpy.random.rand(5, 6).astype(dtype)
        dx = nlcpy.sca.create_descriptor(xin)
        coef_array = nlcpy.random.rand(5, 4).astype(dtype)
        desc = (dx[-1, 0] + dx[1, 0] + dx[0, -1] + dx[0, 1]) * coef_array
        with self.assertRaises(ValueError):
            nlcpy.sca.create_kernel(desc)

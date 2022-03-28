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


float_types = [numpy.float32, numpy.float64]


TOL_SINGLE = 1e-6
TOL_DOUBLE = 1e-12


class TestAutgenOut(unittest.TestCase):

    @testing.for_dtypes(float_types)
    def test_autogen_1d(self, dtype):
        xin = nlcpy.random.rand(10).astype(dtype=dtype)
        dxin = nlcpy.sca.create_descriptor(xin)
        res = nlcpy.sca.create_kernel(dxin[...]).execute()
        assert id(xin) != id(res)
        testing.assert_allclose(xin, res)

    @testing.for_dtypes(float_types)
    def test_autogen_2d(self, dtype):
        xin = nlcpy.random.rand(5, 6).astype(dtype=dtype)
        dxin = nlcpy.sca.create_descriptor(xin)
        res = nlcpy.sca.create_kernel(dxin[...]).execute()
        assert id(xin) != id(res)
        testing.assert_allclose(xin, res)

    @testing.for_dtypes(float_types)
    def test_autogen_3d(self, dtype):
        xin = nlcpy.random.rand(3, 4, 5).astype(dtype=dtype)
        dxin = nlcpy.sca.create_descriptor(xin)
        res = nlcpy.sca.create_kernel(dxin[...]).execute()
        assert id(xin) != id(res)
        testing.assert_allclose(xin, res)

    @testing.for_dtypes(float_types)
    def test_autogen_4d(self, dtype):
        xin = nlcpy.random.rand(3, 4, 5, 6).astype(dtype=dtype)
        dxin = nlcpy.sca.create_descriptor(xin)
        res = nlcpy.sca.create_kernel(dxin[...]).execute()
        assert id(xin) != id(res)
        testing.assert_allclose(xin, res)

    @testing.for_dtypes(float_types)
    def test_autogen_multi_ndarray_1d_same_shape(self, dtype):
        xin = nlcpy.random.rand(10).astype(dtype=dtype)
        yin = nlcpy.random.rand(10).astype(dtype=dtype)
        dxin, dyin = nlcpy.sca.create_descriptor((xin, yin))
        res_sca = nlcpy.sca.create_kernel(dxin[...] + dyin[...]).execute()
        assert id(xin) != id(res_sca)
        assert id(yin) != id(res_sca)
        res_naive = xin + yin
        testing.assert_allclose(res_sca, res_naive)

    @testing.for_dtypes(float_types)
    def test_autogen_multi_ndarray_1d_diff_shape(self, dtype):
        xin = nlcpy.random.rand(5).astype(dtype=dtype)
        yin = nlcpy.random.rand(10).astype(dtype=dtype)
        dxin, dyin = nlcpy.sca.create_descriptor((xin, yin))
        res_sca = nlcpy.sca.create_kernel(dxin[...] + dyin[...]).execute()
        assert id(xin) != id(res_sca)
        assert id(yin) != id(res_sca)
        res_naive = xin + yin[:5]
        testing.assert_allclose(res_sca, res_naive)

    @testing.for_dtypes(float_types)
    def test_autogen_multi_ndarray_2d_same_shape(self, dtype):
        xin = nlcpy.random.rand(5, 5).astype(dtype=dtype)
        yin = nlcpy.random.rand(5, 5).astype(dtype=dtype)
        dxin, dyin = nlcpy.sca.create_descriptor((xin, yin))
        res_sca = nlcpy.sca.create_kernel(dxin[...] + dyin[...]).execute()
        assert id(xin) != id(res_sca)
        assert id(yin) != id(res_sca)
        res_naive = xin + yin
        testing.assert_allclose(res_sca, res_naive)

    @testing.for_dtypes(float_types)
    def test_autogen_multi_ndarray_2d_diff_shape(self, dtype):
        xin = nlcpy.random.rand(5, 5).astype(dtype=dtype)
        yin = nlcpy.random.rand(4, 6).astype(dtype=dtype)
        dxin, dyin = nlcpy.sca.create_descriptor((xin, yin))
        res_sca = nlcpy.sca.create_kernel(dxin[...] + dyin[...]).execute()
        assert id(xin) != id(res_sca)
        assert id(yin) != id(res_sca)
        res_naive = xin[:4, :] + yin[:, :5]
        testing.assert_allclose(res_sca, res_naive)

    @testing.for_dtypes(float_types)
    def test_autogen_multi_ndarray_3d_same_shape(self, dtype):
        xin = nlcpy.random.rand(5, 5, 5).astype(dtype=dtype)
        yin = nlcpy.random.rand(5, 5, 5).astype(dtype=dtype)
        dxin, dyin = nlcpy.sca.create_descriptor((xin, yin))
        res_sca = nlcpy.sca.create_kernel(dxin[...] + dyin[...]).execute()
        assert id(xin) != id(res_sca)
        assert id(yin) != id(res_sca)
        res_naive = xin + yin
        testing.assert_allclose(res_sca, res_naive)

    @testing.for_dtypes(float_types)
    def test_autogen_multi_ndarray_3d_diff_shape(self, dtype):
        xin = nlcpy.random.rand(5, 4, 6).astype(dtype=dtype)
        yin = nlcpy.random.rand(4, 6, 7).astype(dtype=dtype)
        dxin, dyin = nlcpy.sca.create_descriptor((xin, yin))
        res_sca = nlcpy.sca.create_kernel(dxin[...] + dyin[...]).execute()
        assert id(xin) != id(res_sca)
        assert id(yin) != id(res_sca)
        res_naive = xin[:4, :, :] + yin[:, :4, :6]
        testing.assert_allclose(res_sca, res_naive)

    @testing.for_dtypes(float_types)
    def test_autogen_multi_ndarray_4d_same_shape(self, dtype):
        xin = nlcpy.random.rand(5, 5, 5, 5).astype(dtype=dtype)
        yin = nlcpy.random.rand(5, 5, 5, 5).astype(dtype=dtype)
        dxin, dyin = nlcpy.sca.create_descriptor((xin, yin))
        res_sca = nlcpy.sca.create_kernel(dxin[...] + dyin[...]).execute()
        assert id(xin) != id(res_sca)
        assert id(yin) != id(res_sca)
        res_naive = xin + yin
        testing.assert_allclose(res_sca, res_naive)

    @testing.for_dtypes(float_types)
    def test_autogen_multi_ndarray_4d_diff_shape(self, dtype):
        xin = nlcpy.random.rand(5, 4, 6, 7).astype(dtype=dtype)
        yin = nlcpy.random.rand(4, 6, 7, 3).astype(dtype=dtype)
        dxin, dyin = nlcpy.sca.create_descriptor((xin, yin))
        res_sca = nlcpy.sca.create_kernel(dxin[...] + dyin[...]).execute()
        assert id(xin) != id(res_sca)
        assert id(yin) != id(res_sca)
        res_naive = xin[:4, :, :, :3] + yin[:, :4, :6, :]
        testing.assert_allclose(res_sca, res_naive)

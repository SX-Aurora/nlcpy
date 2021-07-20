#
# * The source code in this file is developed independently by NEC Corporation.
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

import numpy
import unittest

import nlcpy
from nlcpy import testing


float_types = [numpy.float32, numpy.float64]


TOL_SINGLE = 1e-6
TOL_DOUBLE = 1e-12


class TestMultiNdarray(unittest.TestCase):

    @testing.for_dtypes(float_types)
    def test_multi_ndarray_1(self, dtype):
        xin = nlcpy.random.rand(5, 5).astype(dtype=dtype)
        yin = nlcpy.random.rand(4, 6).astype(dtype=dtype)
        # compute with sca
        dxin, dyin = nlcpy.sca.create_descriptor((xin, yin))
        desc = dxin[-1, 1] + dxin[1, -1] + dyin[-1, -1] + dyin[1, 1]
        res_sca = nlcpy.sca.create_kernel(desc).execute()
        # compute with naive
        x_tmp = xin[:4, :]
        y_tmp = yin[:, :5]
        res_naive = nlcpy.zeros((4, 5), dtype=dtype)
        res_naive[1:-1, 1:-1] = (x_tmp[:-2, 2:] + x_tmp[2:, :-2] +
                                 y_tmp[:-2, :-2] + y_tmp[2:, 2:])

        rtol = TOL_SINGLE if dtype == numpy.float32 else TOL_DOUBLE
        testing.assert_allclose(res_sca, res_naive, rtol=rtol)

    @testing.for_dtypes(float_types)
    def test_multi_ndarray_2(self, dtype):
        xin = nlcpy.random.rand(5, 5).astype(dtype=dtype)
        yin = nlcpy.random.rand(7, 7).astype(dtype=dtype)
        # compute with sca
        dxin, dyin = nlcpy.sca.create_descriptor((xin, yin))
        desc = dxin[-1, 1] + dxin[1, -1] + dyin[-1, -1] + dyin[1, 1]
        res_sca = nlcpy.sca.create_kernel(desc).execute()
        # compute with naive
        x_tmp = xin[:, :]
        y_tmp = yin[:5, :5]
        res_naive = nlcpy.zeros((5, 5), dtype=dtype)
        res_naive[1:-1, 1:-1] = (x_tmp[:-2, 2:] + x_tmp[2:, :-2] +
                                 y_tmp[:-2, :-2] + y_tmp[2:, 2:])

        rtol = TOL_SINGLE if dtype == numpy.float32 else TOL_DOUBLE
        testing.assert_allclose(res_sca, res_naive, rtol=rtol)

    @testing.for_dtypes(float_types)
    def test_multi_ndarray_3(self, dtype):
        xin = nlcpy.random.rand(7, 7).astype(dtype=dtype)
        yin = nlcpy.random.rand(6, 8).astype(dtype=dtype)
        # compute with sca
        dxin, dyin = nlcpy.sca.create_descriptor((xin, yin))
        desc = dxin[-2, 2] + dxin[1, -1] + dyin[0, 0]
        res_sca = nlcpy.sca.create_kernel(desc).execute()
        # compute with naive
        x_tmp = xin[:6, :]
        y_tmp = yin[:, :7]
        res_naive = nlcpy.zeros((6, 7), dtype=dtype)
        res_naive[2:-1, 1:-2] = x_tmp[:-3, 3:] + x_tmp[3:, :-3] + y_tmp[2:-1, 1:-2]

        rtol = TOL_SINGLE if dtype == numpy.float32 else TOL_DOUBLE
        testing.assert_allclose(res_sca, res_naive, rtol=rtol)

    @testing.for_dtypes(float_types)
    def test_multi_ndarray_4(self, dtype):
        xin = nlcpy.random.rand(5, 5).astype(dtype=dtype)
        yin = nlcpy.random.rand(5, 5).astype(dtype=dtype)
        zin = nlcpy.random.rand(5, 5).astype(dtype=dtype)
        # compute with sca
        dxin, dyin, dzin = nlcpy.sca.create_descriptor((xin, yin, zin))
        desc = (
            dxin[-1, 0] +
            dxin[0, 1] +
            dxin[1, 0] +
            dxin[0, -1] +
            dyin[0, 0] +
            dzin[0, 0])
        res_sca = nlcpy.sca.create_kernel(desc).execute()
        # compute with naive
        res_naive = nlcpy.zeros((5, 5), dtype=dtype)
        res_naive[1:-1, 1:-1] = (
            xin[:-2, 1:-1] +
            xin[1:-1, 2:] +
            xin[2:, 1:-1] +
            xin[1:-1, :-2] +
            yin[1:-1, 1:-1] +
            zin[1:-1, 1:-1])

        rtol = TOL_SINGLE if dtype == numpy.float32 else TOL_DOUBLE
        testing.assert_allclose(res_sca, res_naive, rtol=rtol)

    @testing.for_dtypes(float_types)
    def test_multi_ndarray_5(self, dtype):
        xin = nlcpy.random.rand(5, 5).astype(dtype=dtype)
        yin = nlcpy.random.rand(5, 5).astype(dtype=dtype)
        zin = nlcpy.random.rand(5, 5).astype(dtype=dtype)
        # compute with sca
        dxin, dyin, dzin = nlcpy.sca.create_descriptor((xin, yin, zin))
        desc = (
            dxin[-1, 0] +
            dxin[0, 1] +
            dxin[1, 0] +
            dxin[0, -1] +
            dyin[-1, -1] +
            dzin[1, 1])
        res_sca = nlcpy.sca.create_kernel(desc).execute()
        # compute with naive
        res_naive = nlcpy.zeros((5, 5), dtype=dtype)
        res_naive[1:-1, 1:-1] = (
            xin[:-2, 1:-1] +
            xin[1:-1, 2:] +
            xin[2:, 1:-1] +
            xin[1:-1, :-2] +
            yin[:-2, :-2] +
            zin[2:, 2:])

        rtol = TOL_SINGLE if dtype == numpy.float32 else TOL_DOUBLE
        testing.assert_allclose(res_sca, res_naive, rtol=rtol)

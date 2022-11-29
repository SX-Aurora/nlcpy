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
from nlcpy import venode
from nlcpy import testing


nve = nlcpy.venode.get_num_available_venodes()


@testing.multi_ve(nve)
@testing.parameterize(*testing.product({
    'veid': [i for i in range(nve)],
}))
class TestCreationVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_empty(self):
        res = nlcpy.empty(10)
        assert res.size == 10
        assert res.venode == venode.VE(self.veid)

    def test_empty_like(self):
        x = nlcpy.empty(10)
        res = nlcpy.empty_like(x)
        assert res.size == 10
        assert res.venode == venode.VE(self.veid)

    def test_eye(self):
        res_vp = nlcpy.eye(2, dtype=int)
        res_np = numpy.eye(2, dtype=int)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_identity(self):
        res_vp = nlcpy.identity(3)
        res_np = numpy.identity(3)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ones(self):
        res_vp = nlcpy.ones(3)
        res_np = numpy.ones(3)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ones_like(self):
        x = nlcpy.empty(3)
        y = numpy.empty(3)
        res_vp = nlcpy.ones_like(x)
        res_np = numpy.ones_like(y)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_zeros(self):
        res_vp = nlcpy.zeros(3)
        res_np = numpy.zeros(3)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_zeros_like(self):
        x = nlcpy.empty(3)
        y = numpy.empty(3)
        res_vp = nlcpy.zeros_like(x)
        res_np = numpy.zeros_like(y)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_full(self):
        res_vp = nlcpy.full((2, 2), 10)
        res_np = numpy.full((2, 2), 10)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_full_like(self):
        x = nlcpy.empty((2, 2))
        y = numpy.empty((2, 2))
        res_vp = nlcpy.full_like(x, 10)
        res_np = numpy.full_like(y, 10)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_arange(self):
        res_vp = nlcpy.arange(10)
        res_np = numpy.arange(10)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_linspace(self):
        res_vp = nlcpy.linspace(2.0, 3.0, num=5)
        res_np = numpy.linspace(2.0, 3.0, num=5)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_logspace(self):
        res_vp = nlcpy.logspace(2.0, 3.0, num=5)
        res_np = numpy.logspace(2.0, 3.0, num=5)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_meshgrid(self):
        nx, ny = (3, 2)
        x_vp = nlcpy.linspace(0, 1, nx)
        y_vp = nlcpy.linspace(0, 1, ny)
        xv_vp, yv_vp = nlcpy.meshgrid(x_vp, y_vp)
        x_np = numpy.linspace(0, 1, nx)
        y_np = numpy.linspace(0, 1, ny)
        xv_np, yv_np = nlcpy.meshgrid(x_np, y_np)
        testing.assert_allclose(xv_vp, xv_np, atol=1e-6, rtol=1e-6)
        testing.assert_allclose(yv_vp, yv_np, atol=1e-6, rtol=1e-6)
        assert xv_vp.venode == venode.VE(self.veid)
        assert yv_vp.venode == venode.VE(self.veid)

    def test_diag(self):
        x = nlcpy.arange(9).reshape((3, 3))
        res_vp = nlcpy.diag(x)
        y = numpy.arange(9).reshape((3, 3))
        res_np = numpy.diag(y)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_diagflat(self):
        res_vp = nlcpy.diagflat([[1, 2], [3, 4]])
        res_np = numpy.diagflat([[1, 2], [3, 4]])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_tri(self):
        res_vp = nlcpy.tri(3, 5, 2, dtype=int)
        res_np = numpy.tri(3, 5, 2, dtype=int)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_tril(self):
        res_vp = nlcpy.tril([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], -1)
        res_np = numpy.tril([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], -1)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_triu(self):
        res_vp = nlcpy.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], -1)
        res_np = numpy.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], -1)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

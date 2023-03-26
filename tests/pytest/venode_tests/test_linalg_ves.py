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
class TestLinalgVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_dot(self):
        res_vp = nlcpy.dot(3, 4)
        res_np = numpy.dot(3, 4)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_inner(self):
        res_vp = nlcpy.inner([1, 2, 3], [0, 1, 2])
        res_np = numpy.inner([1, 2, 3], [0, 1, 2])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_outer(self):
        res_vp = nlcpy.outer(nlcpy.ones((5,)), nlcpy.linspace(-2, 2, 5))
        res_np = numpy.outer(numpy.ones((5,)), numpy.linspace(-2, 2, 5))
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_matmul(self):
        res_vp = nlcpy.matmul([[1, 0], [0, 1]], [[4, 1], [2, 2]])
        res_np = numpy.matmul([[1, 0], [0, 1]], [[4, 1], [2, 2]])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_svd(self):
        numpy.random.seed(0)
        a = numpy.random.randn(9, 6) + 1j * numpy.random.randn(9, 6)
        u_vp, s_vp, vh_vp = nlcpy.linalg.svd(a, full_matrices=True)
        testing.assert_allclose(a, nlcpy.dot(u_vp[:, :6] * s_vp, vh_vp))
        assert u_vp.venode == venode.VE(self.veid)
        assert s_vp.venode == venode.VE(self.veid)
        assert vh_vp.venode == venode.VE(self.veid)

    def test_cholesky(self):
        res_vp = nlcpy.linalg.cholesky([[1, -2j], [2j, 5]])
        res_np = numpy.linalg.cholesky([[1, -2j], [2j, 5]])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_qr(self):
        numpy.random.seed(0)
        a = numpy.random.randn(9, 6)
        q_vp, r_vp = nlcpy.linalg.qr(a)
        q_np, r_np = numpy.linalg.qr(a)
        testing.assert_allclose(q_vp, q_np, atol=1e-6, rtol=1e-6)
        testing.assert_allclose(r_vp, r_np, atol=1e-6, rtol=1e-6)
        assert q_vp.venode == venode.VE(self.veid)
        assert r_vp.venode == venode.VE(self.veid)

    def test_eig(self):
        w_vp, v_vp = nlcpy.linalg.eig(nlcpy.diag((1, 2, 3)))
        w_np, v_np = numpy.linalg.eig(numpy.diag((1, 2, 3)))
        testing.assert_allclose(w_vp, w_np, atol=1e-6, rtol=1e-6)
        testing.assert_allclose(v_vp, v_np, atol=1e-6, rtol=1e-6)
        assert w_vp.venode == venode.VE(self.veid)
        assert v_vp.venode == venode.VE(self.veid)

    def test_eigh(self):
        w_vp, v_vp = nlcpy.linalg.eigh([[1, -2j], [2j, 5]])
        w_np, v_np = numpy.linalg.eigh([[1, -2j], [2j, 5]])
        testing.assert_allclose(w_vp, w_np, atol=1e-6, rtol=1e-6)
        testing.assert_allclose(v_vp, v_np, atol=1e-6, rtol=1e-6)
        assert w_vp.venode == venode.VE(self.veid)
        assert v_vp.venode == venode.VE(self.veid)

    def test_eigvals(self):
        res_vp = nlcpy.linalg.eigvals(nlcpy.diag((-1, 1)))
        res_np = numpy.linalg.eigvals(numpy.diag((-1, 1)))
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_eigvalsh(self):
        res_vp = nlcpy.linalg.eigvalsh([[1, -2j], [2j, 5]])
        res_np = numpy.linalg.eigvalsh([[1, -2j], [2j, 5]])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_norm(self):
        a_vp = nlcpy.arange(9) - 4
        res_vp = nlcpy.linalg.norm(a_vp)
        a_np = numpy.arange(9) - 4
        res_np = numpy.linalg.norm(a_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_solve(self):
        res_vp = nlcpy.linalg.solve([[3, 1], [1, 2]], [9, 8])
        res_np = numpy.linalg.solve([[3, 1], [1, 2]], [9, 8])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_lstsq(self):
        A_vp = nlcpy.array(([0, 1, 2, 3], nlcpy.ones(4))).T
        y_vp = nlcpy.array([-1, 0.2, 0.9, 2.1])
        m_vp, c_vp = nlcpy.linalg.lstsq(A_vp, y_vp, rcond=None)[0]
        A_np = numpy.array(([0, 1, 2, 3], numpy.ones(4))).T
        y_np = numpy.array([-1, 0.2, 0.9, 2.1])
        m_np, c_np = numpy.linalg.lstsq(A_np, y_np, rcond=None)[0]
        testing.assert_allclose(m_vp, m_np, atol=1e-6, rtol=1e-6)
        testing.assert_allclose(c_vp, c_np, atol=1e-6, rtol=1e-6)
        assert m_vp.venode == venode.VE(self.veid)
        assert c_vp.venode == venode.VE(self.veid)

    def test_inv(self):
        res_vp = nlcpy.linalg.inv([[1., 2.], [3., 4.]])
        res_np = numpy.linalg.inv([[1., 2.], [3., 4.]])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

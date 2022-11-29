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
class TestFftVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_fft(self):
        t_vp = nlcpy.arange(256)
        res_vp = nlcpy.fft.fft(nlcpy.sin(t_vp))
        t_np = numpy.arange(256)
        res_np = numpy.fft.fft(numpy.sin(t_np))
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ifft(self):
        res_vp = nlcpy.fft.ifft([0, 4, 0, 0])
        res_np = numpy.fft.ifft([0, 4, 0, 0])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_fft2(self):
        a = numpy.mgrid[:5, :5][0]
        res_vp = nlcpy.fft.fft2(a)
        res_np = numpy.fft.fft2(a)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ifft2(self):
        a_vp = 4 * nlcpy.eye(4)
        res_vp = nlcpy.fft.ifft2(a_vp)
        a_np = 4 * numpy.eye(4)
        res_np = numpy.fft.ifft2(a_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_fftn(self):
        a = numpy.mgrid[:3, :3, :3][0]
        res_vp = nlcpy.fft.fftn(a, axes=(1, 2))
        res_np = numpy.fft.fftn(a, axes=(1, 2))
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ifftn(self):
        a_vp = nlcpy.eye(4)
        res_vp = nlcpy.fft.ifftn(nlcpy.fft.fftn(a_vp, axes=(0,)), axes=(1,))
        a_np = numpy.eye(4)
        res_np = numpy.fft.ifftn(numpy.fft.fftn(a_np, axes=(0,)), axes=(1,))
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_rfft(self):
        res_vp = nlcpy.fft.rfft([0, 1, 0, 0])
        res_np = numpy.fft.rfft([0, 1, 0, 0])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_irfft(self):
        res_vp = nlcpy.fft.irfft([1, -1j, -1])
        res_np = numpy.fft.irfft([1, -1j, -1])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_rfft2(self):
        a_vp = nlcpy.ones((2, 2, 2))
        res_vp = nlcpy.fft.rfft2(a_vp)
        a_np = numpy.ones((2, 2, 2))
        res_np = numpy.fft.rfft2(a_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_irfft2(self):
        a_vp = nlcpy.zeros((3, 2, 2))
        a_vp[0, 0, 0] = 3 * 2 * 2
        res_vp = nlcpy.fft.irfft2(a_vp)
        a_np = numpy.zeros((3, 2, 2))
        a_np[0, 0, 0] = 3 * 2 * 2
        res_np = numpy.fft.irfft2(a_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_rfftn(self):
        a_vp = nlcpy.ones((2, 2, 2))
        res_vp = nlcpy.fft.rfftn(a_vp)
        a_np = numpy.ones((2, 2, 2))
        res_np = numpy.fft.rfftn(a_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_irfftn(self):
        a_vp = nlcpy.zeros((3, 2, 2))
        a_vp[0, 0, 0] = 3 * 2 * 2
        res_vp = nlcpy.fft.irfftn(a_vp)
        a_np = numpy.zeros((3, 2, 2))
        a_np[0, 0, 0] = 3 * 2 * 2
        res_np = numpy.fft.irfftn(a_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_hfft(self):
        s_vp = nlcpy.array([1, 2, 3, 4, 3, 2])
        res_vp = nlcpy.fft.hfft(s_vp[:4])
        s_np = numpy.array([1, 2, 3, 4, 3, 2])
        res_np = numpy.fft.hfft(s_np[:4])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ihfft(self):
        s_vp = nlcpy.array([15, -4, 0, -1, 0, -4])
        res_vp = nlcpy.fft.ihfft(s_vp)
        s_np = numpy.array([15, -4, 0, -1, 0, -4])
        res_np = numpy.fft.ihfft(s_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_fftfreq(self):
        res_vp = nlcpy.fft.fftfreq(8, d=.1)
        res_np = numpy.fft.fftfreq(8, d=.1)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_rfftfreq(self):
        res_vp = nlcpy.fft.rfftfreq(8, d=.1)
        res_np = numpy.fft.rfftfreq(8, d=.1)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_fftshift(self):
        f_vp = nlcpy.fft.fftfreq(10, 0.1)
        res_vp = nlcpy.fft.fftshift(f_vp)
        f_np = numpy.fft.fftfreq(10, 0.1)
        res_np = numpy.fft.fftshift(f_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ifftshift(self):
        f_vp = nlcpy.fft.fftfreq(9, d=1. / 9).reshape(3, 3)
        res_vp = nlcpy.fft.ifftshift(nlcpy.fft.fftshift(f_vp))
        f_np = numpy.fft.fftfreq(9, d=1. / 9).reshape(3, 3)
        res_np = numpy.fft.ifftshift(numpy.fft.fftshift(f_np))
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)


@testing.multi_ve(nve)
@testing.parameterize(*testing.product({
    'veid': [i for i in range(nve)],
}))
class TestFftReuseVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_fft(self):
        t_vp = nlcpy.arange(256)
        _ = nlcpy.fft.fft(nlcpy.sin(t_vp))
        res_vp = nlcpy.fft.fft(nlcpy.sin(t_vp))
        t_np = numpy.arange(256)
        res_np = numpy.fft.fft(numpy.sin(t_np))
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ifft(self):
        _ = nlcpy.fft.ifft([0, 4, 0, 0])
        res_vp = nlcpy.fft.ifft([0, 4, 0, 0])
        res_np = numpy.fft.ifft([0, 4, 0, 0])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_fft2(self):
        a = numpy.mgrid[:5, :5][0]
        _ = nlcpy.fft.fft2(a)
        res_vp = nlcpy.fft.fft2(a)
        res_np = numpy.fft.fft2(a)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ifft2(self):
        a_vp = 4 * nlcpy.eye(4)
        _ = nlcpy.fft.ifft2(a_vp)
        res_vp = nlcpy.fft.ifft2(a_vp)
        a_np = 4 * numpy.eye(4)
        res_np = numpy.fft.ifft2(a_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_fftn(self):
        a = numpy.mgrid[:3, :3, :3][0]
        _ = nlcpy.fft.fftn(a, axes=(1, 2))
        res_vp = nlcpy.fft.fftn(a, axes=(1, 2))
        res_np = numpy.fft.fftn(a, axes=(1, 2))
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ifftn(self):
        a_vp = nlcpy.eye(4)
        _ = nlcpy.fft.ifftn(nlcpy.fft.fftn(a_vp, axes=(0,)), axes=(1,))
        res_vp = nlcpy.fft.ifftn(nlcpy.fft.fftn(a_vp, axes=(0,)), axes=(1,))
        a_np = numpy.eye(4)
        res_np = numpy.fft.ifftn(numpy.fft.fftn(a_np, axes=(0,)), axes=(1,))
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_rfft(self):
        _ = nlcpy.fft.rfft([0, 1, 0, 0])
        res_vp = nlcpy.fft.rfft([0, 1, 0, 0])
        res_np = numpy.fft.rfft([0, 1, 0, 0])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_irfft(self):
        _ = nlcpy.fft.irfft([1, -1j, -1])
        res_vp = nlcpy.fft.irfft([1, -1j, -1])
        res_np = numpy.fft.irfft([1, -1j, -1])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_rfft2(self):
        a_vp = nlcpy.ones((2, 2, 2))
        _ = nlcpy.fft.rfft2(a_vp)
        res_vp = nlcpy.fft.rfft2(a_vp)
        a_np = numpy.ones((2, 2, 2))
        res_np = numpy.fft.rfft2(a_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_irfft2(self):
        a_vp = nlcpy.zeros((3, 2, 2))
        a_vp[0, 0, 0] = 3 * 2 * 2
        _ = nlcpy.fft.irfft2(a_vp)
        res_vp = nlcpy.fft.irfft2(a_vp)
        a_np = numpy.zeros((3, 2, 2))
        a_np[0, 0, 0] = 3 * 2 * 2
        res_np = numpy.fft.irfft2(a_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_rfftn(self):
        a_vp = nlcpy.ones((2, 2, 2))
        _ = nlcpy.fft.rfftn(a_vp)
        res_vp = nlcpy.fft.rfftn(a_vp)
        a_np = numpy.ones((2, 2, 2))
        res_np = numpy.fft.rfftn(a_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_irfftn(self):
        a_vp = nlcpy.zeros((3, 2, 2))
        a_vp[0, 0, 0] = 3 * 2 * 2
        _ = nlcpy.fft.irfftn(a_vp)
        res_vp = nlcpy.fft.irfftn(a_vp)
        a_np = numpy.zeros((3, 2, 2))
        a_np[0, 0, 0] = 3 * 2 * 2
        res_np = numpy.fft.irfftn(a_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_hfft(self):
        s_vp = nlcpy.array([1, 2, 3, 4, 3, 2])
        _ = nlcpy.fft.hfft(s_vp[:4])
        res_vp = nlcpy.fft.hfft(s_vp[:4])
        s_np = numpy.array([1, 2, 3, 4, 3, 2])
        res_np = numpy.fft.hfft(s_np[:4])
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ihfft(self):
        s_vp = nlcpy.array([15, -4, 0, -1, 0, -4])
        _ = nlcpy.fft.ihfft(s_vp)
        res_vp = nlcpy.fft.ihfft(s_vp)
        s_np = numpy.array([15, -4, 0, -1, 0, -4])
        res_np = numpy.fft.ihfft(s_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_fftfreq(self):
        _ = nlcpy.fft.fftfreq(8, d=.1)
        res_vp = nlcpy.fft.fftfreq(8, d=.1)
        res_np = numpy.fft.fftfreq(8, d=.1)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_rfftfreq(self):
        _ = nlcpy.fft.rfftfreq(8, d=.1)
        res_vp = nlcpy.fft.rfftfreq(8, d=.1)
        res_np = numpy.fft.rfftfreq(8, d=.1)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_fftshift(self):
        f_vp = nlcpy.fft.fftfreq(10, 0.1)
        _ = nlcpy.fft.fftshift(f_vp)
        res_vp = nlcpy.fft.fftshift(f_vp)
        f_np = numpy.fft.fftfreq(10, 0.1)
        res_np = numpy.fft.fftshift(f_np)
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ifftshift(self):
        f_vp = nlcpy.fft.fftfreq(9, d=1. / 9).reshape(3, 3)
        _ = nlcpy.fft.ifftshift(nlcpy.fft.fftshift(f_vp))
        res_vp = nlcpy.fft.ifftshift(nlcpy.fft.fftshift(f_vp))
        f_np = numpy.fft.fftfreq(9, d=1. / 9).reshape(3, 3)
        res_np = numpy.fft.ifftshift(numpy.fft.fftshift(f_np))
        testing.assert_allclose(res_vp, res_np, atol=1e-6, rtol=1e-6)
        assert res_vp.venode == venode.VE(self.veid)

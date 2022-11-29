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
import pytest

import nlcpy
from nlcpy import venode
from nlcpy import testing


nve = nlcpy.venode.get_num_available_venodes()


@testing.multi_ve(nve)
@testing.parameterize(*testing.product({
    'veid': [i for i in range(nve)],
}))
class TestSCAVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_sca_simple(self):
        base = nlcpy.arange(25, dtype='f4').reshape(5, 5)
        db = nlcpy.sca.create_descriptor(base)
        desc_i = db[-1, 0] + db[1, 0] + db[0, -1] + db[0, 1]
        kern = nlcpy.sca.create_kernel(desc_i)
        res_sca = kern.execute()
        res_naive = nlcpy.zeros_like(base)
        res_naive[1:-1, 1:-1] = (
            base[:-2, 1:-1] +
            base[2:, 1:-1] +
            base[1:-1, :-2] +
            base[1:-1, 2:])
        testing.assert_allclose(res_sca, res_naive, atol=1e-6, rtol=1e-6)
        assert res_sca.venode == venode.VE(self.veid)
        nlcpy.sca.destroy_kernel(kern)

    def test_sca_empty_description(self):
        base = nlcpy.arange(25, dtype='f4').reshape(5, 5)
        db = nlcpy.sca.create_descriptor(base)
        desc_i = nlcpy.sca.empty_description()
        desc_i += db[-1, 0] + db[1, 0] + db[0, -1] + db[0, 1]
        kern = nlcpy.sca.create_kernel(desc_i)
        res_sca = kern.execute()
        res_naive = nlcpy.zeros_like(base)
        res_naive[1:-1, 1:-1] = (
            base[:-2, 1:-1] +
            base[2:, 1:-1] +
            base[1:-1, :-2] +
            base[1:-1, 2:])
        testing.assert_allclose(res_sca, res_naive, atol=1e-6, rtol=1e-6)
        assert res_sca.venode == venode.VE(self.veid)
        nlcpy.sca.destroy_kernel(kern)

    def test_sca_convert_optimized_array(self):
        base = nlcpy.arange(25, dtype='f4').reshape(5, 5)
        opt = nlcpy.sca.convert_optimized_array(base)
        testing.assert_allclose(base, opt)
        assert opt.venode == venode.VE(self.veid)

    def test_sca_create_optimized_array(self):
        opt = nlcpy.sca.create_optimized_array((3, 5, 7))
        testing.assert_allclose(nlcpy.zeros((3, 5, 7)), opt)
        assert opt.venode == venode.VE(self.veid)


@testing.multi_ve(2)
class TestSCAVEsErrorChk(unittest.TestCase):

    def test_sca_create_kernel_err0(self):
        with venode.VE(1):
            base = nlcpy.arange(25, dtype='f4').reshape(5, 5)
        db = nlcpy.sca.create_descriptor(base)
        desc_i = nlcpy.sca.empty_description()
        desc_i += db[-1, 0] + db[1, 0] + db[0, -1] + db[0, 1]
        with venode.VE(0):
            with pytest.raises(ValueError):
                kern = nlcpy.sca.create_kernel(desc_i)
        with venode.VE(1):
            kern = nlcpy.sca.create_kernel(desc_i)
        with venode.VE(0):
            _ = kern.execute()  # not raise
        with venode.VE(1):
            res_sca = kern.execute()
        with venode.VE(1):
            res_naive = nlcpy.zeros_like(base)
            res_naive[1:-1, 1:-1] = (
                base[:-2, 1:-1] +
                base[2:, 1:-1] +
                base[1:-1, :-2] +
                base[1:-1, 2:])
        testing.assert_allclose(res_sca, res_naive, atol=1e-6, rtol=1e-6)
        assert res_sca.venode == venode.VE(1)
        nlcpy.sca.destroy_kernel(kern)

    def test_sca_create_kernel_err1(self):
        with venode.VE(1):
            base = nlcpy.arange(25, dtype='f4').reshape(5, 5)
        with venode.VE(0):
            out0 = nlcpy.zeros_like(base)
        with venode.VE(1):
            out1 = nlcpy.zeros_like(base)
        db, do0, do1 = nlcpy.sca.create_descriptor((base, out0, out1))
        desc_i = nlcpy.sca.empty_description()
        desc_i += db[-1, 0] + db[1, 0] + db[0, -1] + db[0, 1]
        desc_o0 = do0[0, 0]
        desc_o1 = do1[0, 0]
        with venode.VE(0):
            with pytest.raises(ValueError):
                _ = nlcpy.sca.create_kernel(desc_i, desc_o=desc_o0)
        with venode.VE(1):
            with pytest.raises(ValueError):
                _ = nlcpy.sca.create_kernel(desc_i, desc_o=desc_o0)
        with venode.VE(1):
            kern = nlcpy.sca.create_kernel(desc_i, desc_o=desc_o1)
        _ = kern.execute()
        with venode.VE(1):
            res_naive = nlcpy.zeros_like(base)
            res_naive[1:-1, 1:-1] = (
                base[:-2, 1:-1] +
                base[2:, 1:-1] +
                base[1:-1, :-2] +
                base[1:-1, 2:])
        testing.assert_allclose(out1, res_naive, atol=1e-6, rtol=1e-6)
        assert out1.venode == venode.VE(1)
        nlcpy.sca.destroy_kernel(kern)

    def test_sca_create_kernel_err2(self):
        with venode.VE(1):
            base = nlcpy.arange(25, dtype='f4').reshape(5, 5)
        with venode.VE(0):
            coef0 = nlcpy.arange(9, dtype='f4').reshape(3, 3)
        with venode.VE(1):
            coef1 = nlcpy.arange(9, dtype='f4').reshape(3, 3)
        db = nlcpy.sca.create_descriptor(base)
        desc_i = (db[-1, 0] + db[1, 0] + db[0, -1] + db[0, 1]) * coef0
        with venode.VE(0):
            with pytest.raises(ValueError):
                _ = nlcpy.sca.create_kernel(desc_i)
        with venode.VE(1):
            with pytest.raises(ValueError):
                _ = nlcpy.sca.create_kernel(desc_i)
        desc_i = (db[-1, 0] + db[1, 0] + db[0, -1] + db[0, 1]) * coef1
        with venode.VE(0):
            with pytest.raises(ValueError):
                _ = nlcpy.sca.create_kernel(desc_i)
        with venode.VE(1):
            kern = nlcpy.sca.create_kernel(desc_i)
        res_sca = kern.execute()
        with venode.VE(1):
            res_naive = nlcpy.zeros_like(base)
            res_naive[1:-1, 1:-1] = (
                base[:-2, 1:-1] +
                base[2:, 1:-1] +
                base[1:-1, :-2] +
                base[1:-1, 2:]) * coef1
        testing.assert_allclose(res_sca, res_naive, atol=1e-6, rtol=1e-6)
        assert res_sca.venode == venode.VE(1)
        nlcpy.sca.destroy_kernel(kern)

    def test_sca_create_kernel_err3(self):
        with venode.VE(1):
            base = nlcpy.arange(25, dtype='f4').reshape(5, 5)
        with venode.VE(0):
            coef0 = nlcpy.arange(4, dtype='f4')
        with venode.VE(1):
            coef1 = nlcpy.arange(4, dtype='f4')
        db = nlcpy.sca.create_descriptor(base)
        desc_i = (
            db[-1, 0] * coef0[0] +
            db[1, 0] * coef0[1] +
            db[0, -1] * coef0[2] +
            db[0, 1] * coef0[3])
        with venode.VE(0):
            with pytest.raises(ValueError):
                _ = nlcpy.sca.create_kernel(desc_i)
        with venode.VE(1):
            with pytest.raises(ValueError):
                _ = nlcpy.sca.create_kernel(desc_i)
        desc_i = (
            db[-1, 0] * coef1[0] +
            db[1, 0] * coef1[1] +
            db[0, -1] * coef1[2] +
            db[0, 1] * coef1[3])
        with venode.VE(0):
            with pytest.raises(ValueError):
                _ = nlcpy.sca.create_kernel(desc_i)
        with venode.VE(1):
            kern = nlcpy.sca.create_kernel(desc_i)
        res_sca = kern.execute()
        with venode.VE(1):
            res_naive = nlcpy.zeros_like(base)
            res_naive[1:-1, 1:-1] = (
                base[:-2, 1:-1] * coef1[0] +
                base[2:, 1:-1] * coef1[1] +
                base[1:-1, :-2] * coef1[2] +
                base[1:-1, 2:] * coef1[3])
        testing.assert_allclose(res_sca, res_naive, atol=1e-6, rtol=1e-6)
        assert res_sca.venode == venode.VE(1)
        nlcpy.sca.destroy_kernel(kern)

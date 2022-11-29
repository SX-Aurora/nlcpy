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

import pytest
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
class TestNdarrayVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_getitem_int(self):
        x_vp = nlcpy.arange(5 * 5 * 5).reshape(5, 5, 5)
        res_vp = x_vp[2:4, 1:3, 2:]
        x_np = numpy.arange(5 * 5 * 5).reshape(5, 5, 5)
        res_np = x_np[2:4, 1:3, 2:]
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_getitem_boolean(self):
        nlcpy.random.seed(10)
        mask = nlcpy.random.randint(0, 2, (5, 5, 5)).astype('bool')
        x_vp = nlcpy.arange(5 * 5 * 5).reshape(5, 5, 5)
        res_vp = x_vp[mask]
        x_np = numpy.arange(5 * 5 * 5).reshape(5, 5, 5)
        res_np = x_np[mask.get()]
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_getitem_advanced(self):
        nlcpy.random.seed(10)
        indices = nlcpy.random.randint(0, 5, (5, 5))
        x_vp = nlcpy.arange(5 * 5 * 5 * 5).reshape(5, 5, 5, 5)
        res_vp = x_vp[indices, 2:]
        x_np = numpy.arange(5 * 5 * 5 * 5).reshape(5, 5, 5, 5)
        res_np = x_np[indices.get(), 2:]
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_setitem_int(self):
        x_vp = nlcpy.arange(5 * 5 * 5).reshape(5, 5, 5)
        x_vp[2:4, 1:3, 2:] = -123
        x_np = numpy.arange(5 * 5 * 5).reshape(5, 5, 5)
        x_np[2:4, 1:3, 2:] = -123
        testing.assert_array_equal(x_vp, x_np)
        assert x_vp.venode == venode.VE(self.veid)

    def test_setitem_boolean(self):
        nlcpy.random.seed(10)
        mask = nlcpy.random.randint(0, 2, (5, 5, 5)).astype('bool')
        x_vp = nlcpy.arange(5 * 5 * 5).reshape(5, 5, 5)
        x_vp[mask] = -123
        x_np = numpy.arange(5 * 5 * 5).reshape(5, 5, 5)
        x_np[mask.get()] = -123
        testing.assert_array_equal(x_vp, x_np)
        assert x_vp.venode == venode.VE(self.veid)

    def test_setitem_advanced(self):
        nlcpy.random.seed(10)
        indices = nlcpy.random.randint(0, 5, (5, 5))
        x_vp = nlcpy.arange(5 * 5 * 5 * 5).reshape(5, 5, 5, 5)
        x_vp[indices, 2:] = -123
        x_np = numpy.arange(5 * 5 * 5 * 5).reshape(5, 5, 5, 5)
        x_np[indices.get(), 2:] = -123
        testing.assert_array_equal(x_vp, x_np)
        assert x_vp.venode == venode.VE(self.veid)

    def test_resize(self):
        a_vp = nlcpy.array([[0, 1], [2, 3]], order='C')
        a_vp.resize((2, 1), refcheck=False)
        a_np = numpy.array([[0, 1], [2, 3]], order='C')
        a_np.resize((2, 1), refcheck=False)
        testing.assert_array_equal(a_vp, a_np)
        assert a_vp.venode == venode.VE(self.veid)


@testing.multi_ve(2)
class TestNdarrayVEsErr(unittest.TestCase):

    def test_getitem_boolean_err(self):
        with venode.VE(1):
            nlcpy.random.seed(10)
            mask = nlcpy.random.randint(0, 2, (5, 5, 5)).astype('bool')
        with venode.VE(0):
            x_vp = nlcpy.arange(5 * 5 * 5).reshape(5, 5, 5)
            with pytest.raises(ValueError):
                x_vp[mask]

    def test_getitem_advanced_err(self):
        with venode.VE(1):
            nlcpy.random.seed(10)
            indices = nlcpy.random.randint(0, 5, (5, 5))
        with venode.VE(0):
            x_vp = nlcpy.arange(5 * 5 * 5 * 5).reshape(5, 5, 5, 5)
            with pytest.raises(ValueError):
                x_vp[indices, 2:]

    def test_setitem_boolean_err(self):
        with venode.VE(1):
            nlcpy.random.seed(10)
            mask = nlcpy.random.randint(0, 2, (5, 5, 5)).astype('bool')
        with venode.VE(0):
            x_vp = nlcpy.arange(5 * 5 * 5).reshape(5, 5, 5)
            with pytest.raises(ValueError):
                x_vp[mask] = -123
        with venode.VE(0):
            val = nlcpy.array(-123)
        with venode.VE(1):
            x_vp = nlcpy.arange(5 * 5 * 5).reshape(5, 5, 5)
            with pytest.raises(ValueError):
                x_vp[mask] = val

    def test_setitem_advanced_err(self):
        with venode.VE(1):
            nlcpy.random.seed(10)
            indices = nlcpy.random.randint(0, 5, (5, 5))
        with venode.VE(0):
            x_vp = nlcpy.arange(5 * 5 * 5 * 5).reshape(5, 5, 5, 5)
            with pytest.raises(ValueError):
                x_vp[indices, 2:] = -123
        with venode.VE(0):
            val = nlcpy.array(-123)
        with venode.VE(1):
            x_vp = nlcpy.arange(5 * 5 * 5 * 5).reshape(5, 5, 5, 5)
            with pytest.raises(ValueError):
                x_vp[indices] = val

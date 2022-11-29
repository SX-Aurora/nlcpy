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
class TestMaskedArrayVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_masked_array(self):
        mask = [[False, True, False], [False, False, True]]
        data_vp = nlcpy.arange(6).reshape((2, 3))
        res_vp = nlcpy.ma.MaskedArray(data_vp, mask=mask)
        data_np = numpy.arange(6).reshape((2, 3))
        res_np = numpy.ma.MaskedArray(data_np, mask=mask)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)
        assert res_vp.mask.venode == venode.VE(self.veid)

    def test_masked_copy(self):
        mask = [[False, True, False], [False, False, True]]
        data_vp = nlcpy.arange(6).reshape((2, 3))
        res_vp = nlcpy.ma.MaskedArray(data_vp, mask=mask).copy()
        data_np = numpy.arange(6).reshape((2, 3))
        res_np = numpy.ma.MaskedArray(data_np, mask=mask).copy()
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)
        assert res_vp.mask.venode == venode.VE(self.veid)

    def test_masked_add(self):
        mask = [[False, True, False], [False, False, True]]
        data_vp = nlcpy.arange(6).reshape((2, 3))
        ma_vp = nlcpy.ma.MaskedArray(data_vp, mask=mask).copy()
        res_vp = ma_vp + 1
        data_np = numpy.arange(6).reshape((2, 3))
        ma_np = numpy.ma.MaskedArray(data_np, mask=mask).copy()
        res_np = ma_np + 1
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)
        assert res_vp.mask.venode == venode.VE(self.veid)


@testing.multi_ve(2)
class TestMaskedArrayVEsErr(unittest.TestCase):

    def test_masked_array_err(self):
        mask = [[False, True, False], [False, False, True]]
        with venode.VE(0):
            mask_vp0 = nlcpy.ma.make_mask(mask)
            data_vp0 = nlcpy.arange(6).reshape((2, 3))
        with venode.VE(1):
            mask_vp1 = nlcpy.ma.make_mask(mask)
            data_vp1 = nlcpy.arange(6).reshape((2, 3))
        with venode.VE(0):
            with pytest.raises(ValueError):
                _ = nlcpy.ma.MaskedArray(data_vp0, mask=mask_vp1)
        with venode.VE(0):
            with pytest.raises(ValueError):
                _ = nlcpy.ma.MaskedArray(data_vp1, mask=mask_vp0)
        with venode.VE(0):
            with pytest.raises(ValueError):
                _ = nlcpy.ma.MaskedArray(data_vp1, mask=mask)
        with venode.VE(0):
            res_vp0 = nlcpy.ma.MaskedArray(data_vp0, mask=mask)

        with venode.VE(1):
            with pytest.raises(ValueError):
                _ = nlcpy.ma.MaskedArray(data_vp0, mask=mask_vp1)
        with venode.VE(1):
            with pytest.raises(ValueError):
                _ = nlcpy.ma.MaskedArray(data_vp1, mask=mask_vp0)
        with venode.VE(1):
            with pytest.raises(ValueError):
                _ = nlcpy.ma.MaskedArray(data_vp0, mask=mask)
        with venode.VE(1):
            res_vp1 = nlcpy.ma.MaskedArray(data_vp1, mask=mask)

        testing.assert_array_equal(res_vp0, res_vp1)
        assert res_vp0.venode == venode.VE(0)
        assert res_vp1.venode == venode.VE(1)

    def test_masked_add_err(self):
        mask = [[False, True, False], [False, False, True]]
        with venode.VE(0):
            mask_vp0 = nlcpy.ma.make_mask(mask)
            data_vp0 = nlcpy.arange(6).reshape((2, 3))
        with venode.VE(1):
            mask_vp1 = nlcpy.ma.make_mask(mask)
            data_vp1 = nlcpy.arange(6).reshape((2, 3))
        with venode.VE(0):
            ma_vp0 = nlcpy.ma.MaskedArray(data_vp0, mask=mask_vp0)
        with venode.VE(1):
            ma_vp1 = nlcpy.ma.MaskedArray(data_vp1, mask=mask_vp1)

        with venode.VE(0):
            with pytest.raises(ValueError):
                _ = ma_vp0 + ma_vp1
        with venode.VE(1):
            with pytest.raises(ValueError):
                _ = ma_vp0 + ma_vp1
        with venode.VE(0):
            res_vp0 = ma_vp0 + ma_vp0
        with venode.VE(1):
            res_vp1 = ma_vp1 + ma_vp1

        testing.assert_array_equal(res_vp0, res_vp1)
        assert res_vp0.venode == venode.VE(0)
        assert res_vp1.venode == venode.VE(1)

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
class TestSortingVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_argsort(self):
        res_vp = nlcpy.argsort([3, 1, 2])
        res_np = numpy.argsort([3, 1, 2])
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_sort(self):
        res_vp = nlcpy.sort([[1, 4], [3, 1]])
        res_np = numpy.sort([[1, 4], [3, 1]])
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_argmax(self):
        a_vp = nlcpy.arange(6).reshape(2, 3) + 10
        res_vp = nlcpy.argmax(a_vp)
        a_np = numpy.arange(6).reshape(2, 3) + 10
        res_np = numpy.argmax(a_np)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_argmin(self):
        a_vp = nlcpy.arange(6).reshape(2, 3) + 10
        res_vp = nlcpy.argmin(a_vp)
        a_np = numpy.arange(6).reshape(2, 3) + 10
        res_np = numpy.argmin(a_np)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_nanargmax(self):
        res_vp = nlcpy.nanargmax([[nlcpy.nan, 4], [2, 3]])
        res_np = numpy.nanargmax([[numpy.nan, 4], [2, 3]])
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_nanargmin(self):
        res_vp = nlcpy.nanargmin([[nlcpy.nan, 4], [2, 3]])
        res_np = numpy.nanargmin([[numpy.nan, 4], [2, 3]])
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_argwhere(self):
        x_vp = nlcpy.arange(6).reshape(2, 3)
        res_vp = nlcpy.argwhere(x_vp > 1)
        x_np = numpy.arange(6).reshape(2, 3)
        res_np = numpy.argwhere(x_np > 1)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_nonzero(self):
        res_vp = nlcpy.nonzero([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
        res_np = nlcpy.nonzero([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
        for i in range(2):
            testing.assert_array_equal(res_vp[i], res_np[i])
            assert res_vp[i].venode == venode.VE(self.veid)

    def test_count_nonzero(self):
        res_vp = nlcpy.count_nonzero(nlcpy.eye(4))
        res_np = numpy.count_nonzero(numpy.eye(4))
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

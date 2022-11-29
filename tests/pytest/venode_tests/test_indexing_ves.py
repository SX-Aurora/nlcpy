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
class TestIndexingVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_diag_indices(self):
        res_vp = nlcpy.diag_indices(4)
        res_np = numpy.diag_indices(4)
        testing.assert_array_equal(res_vp[0], res_np[0])
        testing.assert_array_equal(res_vp[1], res_np[1])
        assert res_vp[0].venode == venode.VE(self.veid)
        assert res_vp[1].venode == venode.VE(self.veid)

    def test_nonzero(self):
        x_vp = nlcpy.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
        res_vp = nlcpy.nonzero(x_vp)
        x_np = numpy.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
        res_np = numpy.nonzero(x_np)
        testing.assert_array_equal(res_vp[0], res_np[0])
        testing.assert_array_equal(res_vp[1], res_np[1])
        assert res_vp[0].venode == venode.VE(self.veid)
        assert res_vp[1].venode == venode.VE(self.veid)

    def test_where(self):
        a_vp = nlcpy.arange(10)
        res_vp = nlcpy.where(a_vp < 5, a_vp, a_vp * 10)
        a_np = numpy.arange(10)
        res_np = numpy.where(a_np < 5, a_np, a_np * 10)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_diag(self):
        x = nlcpy.arange(9).reshape((3, 3))
        res_vp = nlcpy.diag(x)
        y = numpy.arange(9).reshape((3, 3))
        res_np = numpy.diag(y)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_diagonal(self):
        a_vp = nlcpy.arange(4).reshape(2, 2)
        res_vp = nlcpy.diagonal(a_vp)
        a_np = numpy.arange(4).reshape(2, 2)
        res_np = numpy.diagonal(a_np)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_select(self):
        x_vp = nlcpy.arange(10)
        cond_vp = [x_vp < 3, x_vp > 5]
        choice_vp = [x_vp, x_vp ** 2]
        res_vp = nlcpy.select(cond_vp, choice_vp)
        x_np = numpy.arange(10)
        cond_np = [x_np < 3, x_np > 5]
        choice_np = [x_np, x_np ** 2]
        res_np = nlcpy.select(cond_np, choice_np)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_take(self):
        indices = [0, 1, 4]
        a_vp = nlcpy.array([4, 3, 5, 7, 6, 8])
        res_vp = nlcpy.take(a_vp, indices)
        a_np = numpy.array([4, 3, 5, 7, 6, 8])
        res_np = numpy.take(a_np, indices)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_fill_diagonal(self):
        a_vp = nlcpy.zeros((3, 3), int)
        nlcpy.fill_diagonal(a_vp, 5)
        a_np = numpy.zeros((3, 3), int)
        numpy.fill_diagonal(a_np, 5)
        testing.assert_array_equal(a_vp, a_np)
        assert a_vp.venode == venode.VE(self.veid)

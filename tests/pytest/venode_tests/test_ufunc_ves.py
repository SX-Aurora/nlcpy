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
class TestUfuncVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_power(self):
        x_vp = nlcpy.arange(6).reshape(2, 3)
        res_vp = nlcpy.power(x_vp, 2)
        x_np = numpy.arange(6).reshape(2, 3)
        res_np = nlcpy.power(x_np, 2)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_add_reduce(self):
        x_vp = nlcpy.arange(9).reshape(3, 3)
        res_vp = nlcpy.add.reduce(x_vp, axis=0)
        x_np = numpy.arange(9).reshape(3, 3)
        res_np = numpy.add.reduce(x_np, axis=0)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_add_accumulate(self):
        x_vp = nlcpy.arange(9).reshape(3, 3)
        res_vp = nlcpy.add.accumulate(x_vp, axis=0)
        x_np = numpy.arange(9).reshape(3, 3)
        res_np = numpy.add.accumulate(x_np, axis=0)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_add_reduceat(self):
        x_vp = nlcpy.linspace(0, 15, 16).reshape(4, 4)
        res_vp = nlcpy.add.reduceat(x_vp, [0, 3, 1, 2, 0])
        x_np = numpy.linspace(0, 15, 16).reshape(4, 4)
        res_np = numpy.add.reduceat(x_np, [0, 3, 1, 2, 0])
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_add_outer(self):
        res_vp = nlcpy.add.outer([1, 2, 3], [4, 5, 6])
        res_np = numpy.add.outer([1, 2, 3], [4, 5, 6])
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

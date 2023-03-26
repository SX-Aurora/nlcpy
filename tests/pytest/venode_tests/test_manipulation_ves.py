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
class TestManipulationVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_copyto(self):
        src_vp = nlcpy.arange(10)
        dst_vp = nlcpy.empty(10)
        nlcpy.copyto(dst_vp, src_vp)
        src_np = numpy.arange(10)
        dst_np = numpy.empty(10)
        numpy.copyto(dst_np, src_np)
        testing.assert_array_equal(dst_vp, dst_np)
        assert src_vp.venode == venode.VE(self.veid)
        assert dst_vp.venode == venode.VE(self.veid)

    def test_reshape(self):
        x_vp = nlcpy.arange(6).reshape(2, 3)
        res_vp = nlcpy.reshape(x_vp, (3, -1))
        y_np = numpy.arange(6).reshape(2, 3)
        res_np = nlcpy.reshape(y_np, (3, -1))
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ravel(self):
        x_vp = nlcpy.arange(6).reshape(2, 3)
        res_vp = nlcpy.ravel(x_vp)
        y_np = numpy.arange(6).reshape(2, 3)
        res_np = numpy.ravel(y_np)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_moveaxis(self):
        x = nlcpy.zeros((3, 4, 5))
        res_vp = nlcpy.moveaxis(x, 0, -1)
        y = numpy.zeros((3, 4, 5))
        res_np = numpy.moveaxis(y, 0, -1)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_rollaxis(self):
        x = nlcpy.ones((3, 4, 5, 6))
        res_vp = nlcpy.rollaxis(x, 3, 1)
        y = numpy.ones((3, 4, 5, 6))
        res_np = numpy.rollaxis(y, 3, 1)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_swapaxes(self):
        x = nlcpy.array([[1, 2, 3]])
        res_vp = nlcpy.swapaxes(x, 0, 1)
        y = numpy.array([[1, 2, 3]])
        res_np = numpy.swapaxes(y, 0, 1)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_transpose(self):
        x = nlcpy.arange(4).reshape((2, 2))
        res_vp = nlcpy.transpose(x)
        y = numpy.arange(4).reshape((2, 2))
        res_np = numpy.transpose(y)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_atleast_1d(self):
        res_vp = nlcpy.atleast_1d(1.0)
        res_np = numpy.atleast_1d(1.0)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_atleast_2d(self):
        res_vp = nlcpy.atleast_2d(1.0)
        res_np = numpy.atleast_2d(1.0)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_atleast_3d(self):
        res_vp = nlcpy.atleast_3d(1.0)
        res_np = numpy.atleast_3d(1.0)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_broadcast_arrays(self):
        x_vp = nlcpy.array([[1, 2, 3]])
        y_vp = nlcpy.array([[4], [5]])
        res_vp = nlcpy.broadcast_arrays(x_vp, y_vp)
        x_np = numpy.array([[1, 2, 3]])
        y_np = numpy.array([[4], [5]])
        res_np = numpy.broadcast_arrays(x_np, y_np)
        testing.assert_array_equal(res_vp[0], res_np[0])
        testing.assert_array_equal(res_vp[1], res_np[1])
        assert res_vp[0].venode == venode.VE(self.veid)
        assert res_vp[1].venode == venode.VE(self.veid)

    def test_broadcast_to(self):
        x = nlcpy.array([1, 2, 3])
        res_vp = nlcpy.broadcast_to(x, (3, 3))
        y = numpy.array([1, 2, 3])
        res_np = numpy.broadcast_to(y, (3, 3))
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_expand_dims(self):
        x = nlcpy.array([1, 2])
        res_vp = nlcpy.expand_dims(x, axis=0)
        y = numpy.array([1, 2])
        res_np = numpy.expand_dims(y, axis=0)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_squeeze(self):
        x = nlcpy.array([[[0], [1], [2]]])
        res_vp = nlcpy.squeeze(x)
        y = numpy.array([[[0], [1], [2]]])
        res_np = numpy.squeeze(y)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_concatenate(self):
        x_vp = nlcpy.array([[1, 2], [3, 4]])
        y_vp = nlcpy.array([[5, 6]])
        res_vp = nlcpy.concatenate((x_vp, y_vp), axis=0)
        x_np = numpy.array([[1, 2], [3, 4]])
        y_np = numpy.array([[5, 6]])
        res_np = numpy.concatenate((x_np, y_np), axis=0)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_hstack(self):
        x_vp = nlcpy.array([1, 2, 3])
        y_vp = nlcpy.array([2, 3, 4])
        res_vp = nlcpy.hstack((x_vp, y_vp))
        x_np = numpy.array([1, 2, 3])
        y_np = numpy.array([2, 3, 4])
        res_np = numpy.hstack((x_np, y_np))
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_stack(self):
        arrays = [numpy.arange(12).reshape(3, 4) for _ in range(10)]
        res_vp = nlcpy.stack(arrays, axis=0)
        res_np = numpy.stack(arrays, axis=0)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_vstack(self):
        x_vp = nlcpy.array([1, 2, 3])
        y_vp = nlcpy.array([2, 3, 4])
        res_vp = nlcpy.vstack((x_vp, y_vp))
        x_np = numpy.array([1, 2, 3])
        y_np = numpy.array([2, 3, 4])
        res_np = numpy.vstack((x_np, y_np))
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_block(self):
        A_vp = nlcpy.eye(2) * 2
        B_vp = nlcpy.eye(3) * 3
        res_vp = nlcpy.block(
            [[A_vp, nlcpy.zeros((2, 3))],
             [nlcpy.ones((3, 2)), B_vp]])
        A_np = numpy.eye(2) * 2
        B_np = numpy.eye(3) * 3
        res_np = numpy.block(
            [[A_np, numpy.zeros((2, 3))],
             [numpy.ones((3, 2)), B_np]])
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_split(self):
        x_vp = nlcpy.arange(9.0)
        res_vp = nlcpy.split(x_vp, 3)
        x_np = numpy.arange(9.0)
        res_np = numpy.split(x_np, 3)
        for i in range(3):
            testing.assert_array_equal(res_vp[i], res_np[i])
            assert res_vp[i].venode == venode.VE(self.veid)

    def test_hsplit(self):
        x_vp = nlcpy.arange(16.0).reshape(4, 4)
        res_vp = nlcpy.hsplit(x_vp, 2)
        x_np = numpy.arange(16.0).reshape(4, 4)
        res_np = numpy.hsplit(x_np, 2)
        for i in range(2):
            testing.assert_array_equal(res_vp[i], res_np[i])
            assert res_vp[i].venode == venode.VE(self.veid)

    def test_vsplit(self):
        x_vp = nlcpy.arange(16.0).reshape(4, 4)
        res_vp = nlcpy.vsplit(x_vp, 2)
        x_np = numpy.arange(16.0).reshape(4, 4)
        res_np = numpy.vsplit(x_np, 2)
        for i in range(2):
            testing.assert_array_equal(res_vp[i], res_np[i])
            assert res_vp[i].venode == venode.VE(self.veid)

    def test_tile(self):
        x_vp = nlcpy.array([0, 1, 2])
        res_vp = nlcpy.tile(x_vp, 2)
        x_np = numpy.array([0, 1, 2])
        res_np = numpy.tile(x_np, 2)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_repeat(self):
        res_vp = nlcpy.repeat(3, 4)
        res_np = numpy.repeat(3, 4)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_append(self):
        res_vp = nlcpy.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
        res_np = numpy.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_delete(self):
        x_vp = nlcpy.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
        res_vp = nlcpy.delete(x_vp, 1, 0)
        x_np = numpy.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
        res_np = numpy.delete(x_np, 1, 0)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_insert(self):
        x_vp = nlcpy.array([[1, 1], [2, 2], [3, 3]])
        res_vp = nlcpy.insert(x_vp, 1, 5)
        x_np = numpy.array([[1, 1], [2, 2], [3, 3]])
        res_np = numpy.insert(x_np, 1, 5)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_resize(self):
        x_vp = nlcpy.array([[0, 1], [2, 3]])
        res_vp = nlcpy.resize(x_vp, (2, 3))
        x_np = numpy.array([[0, 1], [2, 3]])
        res_np = numpy.resize(x_np, (2, 3))
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_unique(self):
        res_vp = nlcpy.unique([1, 1, 2, 2, 3, 3])
        res_np = numpy.unique([1, 1, 2, 2, 3, 3])
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_flip(self):
        x_vp = nlcpy.arange(8).reshape((2, 2, 2))
        res_vp = nlcpy.flip(x_vp, 0)
        x_np = numpy.arange(8).reshape((2, 2, 2))
        res_np = numpy.flip(x_np, 0)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_fliplr(self):
        x_vp = nlcpy.diag([1., 2., 3.])
        res_vp = nlcpy.fliplr(x_vp)
        x_np = numpy.diag([1., 2., 3.])
        res_np = numpy.fliplr(x_np)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_flipud(self):
        x_vp = nlcpy.diag([1., 2., 3.])
        res_vp = nlcpy.flipud(x_vp)
        x_np = numpy.diag([1., 2., 3.])
        res_np = numpy.flipud(x_np)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_roll(self):
        x_vp = nlcpy.arange(10)
        res_vp = nlcpy.roll(x_vp, 2)
        x_np = numpy.arange(10)
        res_np = numpy.roll(x_np, 2)
        testing.assert_array_equal(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

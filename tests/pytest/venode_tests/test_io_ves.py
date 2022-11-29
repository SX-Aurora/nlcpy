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
from tempfile import TemporaryFile
from tempfile import mkstemp


nve = nlcpy.venode.get_num_available_venodes()


@testing.multi_ve(nve)
@testing.parameterize(*testing.product({
    'veid': [i for i in range(nve)],
}))
class TestIOVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_save_load(self):
        base = nlcpy.array([[1, 2, 3], [4, 5, 6]])
        nlcpy.save('/tmp/123', base)
        res_vp = nlcpy.load('/tmp/123.npy')
        testing.assert_array_equal(res_vp, base)
        assert res_vp.venode == venode.VE(self.veid)

    def test_savez(self):
        outfile = TemporaryFile()
        x = nlcpy.arange(10)
        y = nlcpy.sin(x)
        nlcpy.savez(outfile, x=x, y=y)
        _ = outfile.seek(0)
        npzfile = nlcpy.load(outfile)
        testing.assert_array_equal(npzfile['x'], x)
        testing.assert_array_equal(npzfile['y'], y)
        assert npzfile['x'].venode == venode.VE(self.veid)
        assert npzfile['y'].venode == venode.VE(self.veid)

    def test_savez_compressed(self):
        test_array = nlcpy.random.rand(3, 2)
        test_vector = nlcpy.random.rand(4)
        nlcpy.savez_compressed('/tmp/123', a=test_array, b=test_vector)
        loaded = nlcpy.load('/tmp/123.npz')
        testing.assert_array_equal(test_array, loaded['a'])
        testing.assert_array_equal(test_vector, loaded['b'])
        assert loaded['a'].venode == venode.VE(self.veid)
        assert loaded['b'].venode == venode.VE(self.veid)

    def test_loadtxt_savetxt(self):
        c = nlcpy.array([[0., 1.], [2., 3.]])
        nlcpy.savetxt('/tmp/test.out', c, delimiter=',')
        res_vp = nlcpy.loadtxt('/tmp/test.out', delimiter=',')
        testing.assert_array_equal(res_vp, c)
        assert res_vp.venode == venode.VE(self.veid)

    def test_fromfile(self):
        x = numpy.random.uniform(0, 1, 5)
        fname = mkstemp()[1]
        x.tofile(fname)
        res_vp = nlcpy.fromfile(fname)
        testing.assert_array_equal(res_vp, x)
        assert res_vp.venode == venode.VE(self.veid)

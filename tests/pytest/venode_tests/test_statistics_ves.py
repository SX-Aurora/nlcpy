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
class TestStatisticsVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_amax(self):
        a_vp = nlcpy.arange(4).reshape((2, 2))
        res_vp = nlcpy.amax(a_vp)
        a_np = numpy.arange(4).reshape((2, 2))
        res_np = numpy.amax(a_np)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_amin(self):
        a_vp = nlcpy.arange(4).reshape((2, 2))
        res_vp = nlcpy.amin(a_vp)
        a_np = numpy.arange(4).reshape((2, 2))
        res_np = numpy.amin(a_np)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_nanmax(self):
        a_vp = nlcpy.array([[1, 2], [3, nlcpy.nan]])
        res_vp = nlcpy.nanmax(a_vp)
        a_np = numpy.array([[1, 2], [3, numpy.nan]])
        res_np = numpy.nanmax(a_np)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_nanmin(self):
        a_vp = nlcpy.array([[1, 2], [3, nlcpy.nan]])
        res_vp = nlcpy.nanmin(a_vp)
        a_np = numpy.array([[1, 2], [3, numpy.nan]])
        res_np = numpy.nanmin(a_np)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_ptp(self):
        x_vp = nlcpy.array([[4, 9, 2, 10], [6, 9, 7, 12]])
        res_vp = nlcpy.ptp(x_vp, axis=1)
        x_np = numpy.array([[4, 9, 2, 10], [6, 9, 7, 12]])
        res_np = numpy.ptp(x_np, axis=1)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_percentile(self):
        a_vp = nlcpy.array([[10, 7, 4], [3, 2, 1]])
        res_vp = nlcpy.percentile(a_vp, 50)
        a_np = numpy.array([[10, 7, 4], [3, 2, 1]])
        res_np = numpy.percentile(a_np, 50)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_nanpercentile(self):
        a_vp = nlcpy.array([[10., 7., 4.], [3., 2., 1.]])
        a_vp[0][1] = nlcpy.nan
        res_vp = nlcpy.nanpercentile(a_vp, 50)
        a_np = numpy.array([[10., 7., 4.], [3., 2., 1.]])
        a_np[0][1] = numpy.nan
        res_np = numpy.nanpercentile(a_np, 50)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_quantile(self):
        a_vp = nlcpy.array([[10, 7, 4], [3, 2, 1]])
        res_vp = nlcpy.quantile(a_vp, .5)
        a_np = numpy.array([[10, 7, 4], [3, 2, 1]])
        res_np = numpy.quantile(a_np, .5)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_nanquantile(self):
        a_vp = nlcpy.array([[10., 7., 4.], [3., 2., 1.]])
        a_vp[0][1] = nlcpy.nan
        res_vp = nlcpy.nanquantile(a_vp, .5)
        a_np = numpy.array([[10., 7., 4.], [3., 2., 1.]])
        a_np[0][1] = numpy.nan
        res_np = numpy.nanquantile(a_np, .5)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_average(self):
        res_vp = nlcpy.average([1, 2, 3, 4])
        res_np = numpy.average([1, 2, 3, 4])
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_mean(self):
        res_vp = nlcpy.mean([1, 2, 3, 4])
        res_np = numpy.mean([1, 2, 3, 4])
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_median(self):
        res_vp = nlcpy.median([1, 2, 3, 4])
        res_np = numpy.median([1, 2, 3, 4])
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_nanmedian(self):
        res_vp = nlcpy.nanmedian([1, nlcpy.nan, 3, 4])
        res_np = numpy.nanmedian([1, numpy.nan, 3, 4])
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_nanmean(self):
        res_vp = nlcpy.nanmean([1, nlcpy.nan, 3, 4])
        res_np = numpy.nanmean([1, numpy.nan, 3, 4])
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_nanstd(self):
        res_vp = nlcpy.nanstd([1, nlcpy.nan, 3, 4])
        res_np = numpy.nanstd([1, numpy.nan, 3, 4])
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_nanvar(self):
        res_vp = nlcpy.nanvar([1, nlcpy.nan, 3, 4])
        res_np = numpy.nanvar([1, numpy.nan, 3, 4])
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_std(self):
        res_vp = nlcpy.std([1, 2, 3, 4])
        res_np = numpy.std([1, 2, 3, 4])
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_var(self):
        res_vp = nlcpy.var([1, 2, 3, 4])
        res_np = numpy.var([1, 2, 3, 4])
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_correlate(self):
        res_vp = nlcpy.correlate([1, 2, 3], [0, 1, 0.5])
        res_np = numpy.correlate([1, 2, 3], [0, 1, 0.5])
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_correcoef(self):
        x_vp = nlcpy.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                            [2, 1, 8, 3, 7, 5, 10, 7, 2]])
        res_vp = nlcpy.corrcoef(x_vp)
        x_np = numpy.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                            [2, 1, 8, 3, 7, 5, 10, 7, 2]])
        res_np = numpy.corrcoef(x_np)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_cov(self):
        x_vp = nlcpy.array([[0, 2], [1, 1], [2, 0]]).T
        res_vp = nlcpy.cov(x_vp)
        x_np = numpy.array([[0, 2], [1, 1], [2, 0]]).T
        res_np = numpy.cov(x_np)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_histogram(self):
        res_vp = nlcpy.histogram([1, 2, 1], bins=[0, 1, 2, 3])
        res_np = numpy.histogram([1, 2, 1], bins=[0, 1, 2, 3])
        testing.assert_allclose(res_vp[0], res_np[0])
        testing.assert_allclose(res_vp[1], res_np[1])
        assert res_vp[0].venode == venode.VE(self.veid)
        assert res_vp[1].venode == venode.VE(self.veid)

    def test_histogram2d(self):
        numpy.random.seed(42)
        z = numpy.random.randn(2, 50)
        H_vp, xedges_vp, yedges_vp = nlcpy.histogram2d(z[0], z[1], bins=5)
        H_np, xedges_np, yedges_np = numpy.histogram2d(z[0], z[1], bins=5)
        testing.assert_allclose(H_vp, H_np)
        testing.assert_allclose(xedges_vp, xedges_np)
        testing.assert_allclose(yedges_vp, yedges_np)
        assert H_vp.venode == venode.VE(self.veid)
        assert xedges_vp.venode == venode.VE(self.veid)
        assert yedges_vp.venode == venode.VE(self.veid)

    def test_histogramdd(self):
        numpy.random.seed(12)
        r = numpy.random.randn(100, 3)
        H_vp, edges_vp = nlcpy.histogramdd(r, bins=(5, 8, 4))
        H_np, edges_np = numpy.histogramdd(r, bins=(5, 8, 4))
        testing.assert_allclose(H_vp, H_np)
        testing.assert_allclose(edges_vp[0], edges_np[0])
        testing.assert_allclose(edges_vp[1], edges_np[1])
        testing.assert_allclose(edges_vp[2], edges_np[2])
        assert H_vp.venode == venode.VE(self.veid)

    def test_bincount(self):
        res_vp = nlcpy.bincount(nlcpy.arange(5))
        res_np = numpy.bincount(numpy.arange(5))
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_histogram_bin_edges(self):
        arr_vp = nlcpy.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
        res_vp = nlcpy.histogram_bin_edges(arr_vp, bins='auto', range=(0, 1))
        arr_np = numpy.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
        res_np = numpy.histogram_bin_edges(arr_np, bins='auto', range=(0, 1))
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

    def test_digitize(self):
        x_vp = nlcpy.array([0.2, 6.4, 3.0, 1.6])
        bins_vp = nlcpy.array([0.0, 1.0, 2.5, 4.0, 10.0])
        res_vp = nlcpy.digitize(x_vp, bins_vp)
        x_np = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins_np = numpy.array([0.0, 1.0, 2.5, 4.0, 10.0])
        res_np = numpy.digitize(x_np, bins_np)
        testing.assert_allclose(res_vp, res_np)
        assert res_vp.venode == venode.VE(self.veid)

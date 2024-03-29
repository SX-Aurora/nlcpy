#
# * The source code in this file is based on the soure code of NumPy.
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
# # NumPy License #
#
#     Copyright (c) 2005-2020, NumPy Developers.
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

import sys
import unittest
import pytest

import nlcpy as vp
import numpy
from numpy.linalg import LinAlgError
from numpy.testing import (
    assert_, assert_raises, assert_equal, assert_allclose,
    assert_warns, assert_no_warnings, assert_array_equal,
    assert_array_almost_equal, suppress_warnings)

from nlcpy.random import Generator, MT19937, SeedSequence
from nlcpy import testing

random = Generator(MT19937())
multi_ve_node_max = vp.venode.get_num_available_venodes()


@pytest.fixture(scope='module', params=[True, False])
def endpoint(request):
    return request.param


class TestSeed(object):
    @testing.multi_ve(multi_ve_node_max)
    def test_scalar(self):
        s1 = {}
        s2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                s = Generator(MT19937(0))
                s1[ve] = s.integers(1000)
                assert_equal(s1[ve], vp.array([694]))
                s = Generator(MT19937(4294967295))
                s2[ve] = s.integers(1000)
                assert_equal(s2[ve], vp.array([318]))
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(s1[ve].get(), s1[ve + 1].get())
            assert_equal(s2[ve].get(), s2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_array(self):
        import numpy
        s1 = {}
        s2 = {}
        s3 = {}
        s4 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                s = Generator(MT19937(range(10)))
                s1[ve] = s.integers(1000)
                assert_equal(s1[ve], vp.array([762]))
                s = Generator(MT19937(numpy.arange(10)))
                s2[ve] = s.integers(1000)
                assert_equal(s2[ve], vp.array([762]))
                s = Generator(MT19937([0]))
                s3[ve] = s.integers(1000)
                assert_equal(s3[ve], vp.array([694]))
                s = Generator(MT19937([4294967295]))
                s4[ve] = s.integers(1000)
                assert_equal(s4[ve], vp.array([318]))
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(s1[ve].get(), s1[ve + 1].get())
            assert_equal(s2[ve].get(), s2[ve + 1].get())
            assert_equal(s3[ve].get(), s3[ve + 1].get())
            assert_equal(s4[ve].get(), s4[ve + 1].get())

    def __exclude_test_seedsequence(self):
        s = MT19937(SeedSequence(0))
        assert_equal(s.random_raw(1), 2058676884)

    @testing.multi_ve(multi_ve_node_max)
    def test_invalid_scalar(self):
        # seed must be an unsigned 32 bit integer
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_raises(TypeError, MT19937, -0.5)
                assert_raises(ValueError, MT19937, -1)

    @testing.multi_ve(multi_ve_node_max)
    def test_invalid_array(self):
        # seed must be an unsigned integer
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_raises(TypeError, MT19937, [-0.5])
                assert_raises(ValueError, MT19937, [-1])
                assert_raises(ValueError, MT19937, [1, -2, 4294967296])

    @testing.multi_ve(multi_ve_node_max)
    def test_noninstantized_bitgen(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_raises(TypeError, Generator, MT19937)


class TestBinomial(object):
    @testing.multi_ve(multi_ve_node_max)
    def test_n_zero(self):
        # Tests the corner case of n == 0 for the binomial distribution.
        # binomial(0, p) should be zero for any p in [0, 1].
        # This test addresses issue #3480.
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                zeros = vp.zeros(2, dtype='int')
                for p in [0, .5, 1]:
                    assert_(random.binomial(0, p) == 0)
                    # assert_array_equal(random.binomial(zeros, p), zeros)
                    assert_array_equal(random.binomial(0, p, 2), zeros)

    @testing.multi_ve(multi_ve_node_max)
    def test_p_is_nan(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_raises(ValueError, random.binomial, 1, vp.nan)


class TestMultinomial(object):
    def __exclude_test_basic(self):
        random.multinomial(100, [0.2, 0.8])

    def __exclude_test_zero_probability(self):
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def __exclude_test_int_negative_interval(self):
        assert_(-5 <= random.integers(-5, -1) < -1)
        x = random.integers(-5, -1, 5)
        assert_(vp.all(-5 <= x))
        assert_(vp.all(x < -1))

    def __exclude_test_size(self):
        p = [0.5, 0.5]
        assert_equal(random.multinomial(1, p, vp.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, vp.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, vp.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, vp.array((2, 2))).shape,
                     (2, 2, 2))

        assert_raises(TypeError, random.multinomial, 1, p,
                      float(1))

    def __exclude_test_invalid_prob(self):
        assert_raises(ValueError, random.multinomial, 100, [1.1, 0.2])
        assert_raises(ValueError, random.multinomial, 100, [-.1, 0.9])

    def __exclude_test_invalid_n(self):
        assert_raises(ValueError, random.multinomial, -1, [0.8, 0.2])
        assert_raises(ValueError, random.multinomial, [-1] * 10, [0.8, 0.2])

    def __exclude_test_p_non_contiguous(self):
        p = vp.arange(15.)
        p /= vp.sum(p[1::3])
        pvals = p[1::3]
        random = Generator(MT19937(1432985819))
        non_contig = random.multinomial(100, pvals=pvals)
        random = Generator(MT19937(1432985819))
        contig = random.multinomial(100, pvals=vp.ascontiguousarray(pvals))
        assert_array_equal(non_contig, contig)


class TestMultivariateHypergeometric(unittest.TestCase):

    def setUp(self):
        self.seed = 8675309

    def __exclude_test_argument_validation(self):
        # Error cases...

        # `colors` must be a 1-d sequence
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      10, 4)

        # Negative nsample
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [2, 3, 4], -1)

        # Negative color
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [-1, 2, 3], 2)

        # nsample exceeds sum(colors)
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [2, 3, 4], 10)

        # nsample exceeds sum(colors) (edge case of empty colors)
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [], 1)

        # Validation errors associated with very large values in colors.
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [999999999, 101], 5, 1, 'marginals')

        int64_info = vp.iinfo(vp.int64)
        max_int64 = int64_info.max
        max_int64_index = max_int64 // int64_info.dtype.itemsize
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [max_int64_index - 100, 101], 5, 1, 'count')

    def __exclude_test_edge_cases(self, method):
        # Set the seed, but in fact, all the results in this test are
        # deterministic, so we don't really need this.
        random = Generator(MT19937(self.seed))

        x = random.multivariate_hypergeometric([0, 0, 0], 0, method=method)
        assert_array_equal(x, [0, 0, 0])

        x = random.multivariate_hypergeometric([], 0, method=method)
        assert_array_equal(x, [])

        x = random.multivariate_hypergeometric([], 0, size=1, method=method)
        assert_array_equal(x, vp.empty((1, 0), dtype=vp.int64))

        x = random.multivariate_hypergeometric([1, 2, 3], 0, method=method)
        assert_array_equal(x, [0, 0, 0])

        x = random.multivariate_hypergeometric([9, 0, 0], 3, method=method)
        assert_array_equal(x, [3, 0, 0])

        colors = [1, 1, 0, 1, 1]
        x = random.multivariate_hypergeometric(colors, sum(colors),
                                               method=method)
        assert_array_equal(x, colors)

        x = random.multivariate_hypergeometric([3, 4, 5], 12, size=3,
                                               method=method)
        assert_array_equal(x, [[3, 4, 5]] * 3)

    # Cases for nsample:
    #     nsample < 10
    #     10 <= nsample < colors.sum()/2
    #     colors.sum()/2 < nsample < colors.sum() - 10
    #     colors.sum() - 10 < nsample < colors.sum()
    def __exclude_test_typical_cases(self, nsample, method, size):
        random = Generator(MT19937(self.seed))

        colors = vp.array([10, 5, 20, 25])
        sample = random.multivariate_hypergeometric(colors, nsample, size,
                                                    method=method)
        if isinstance(size, int):
            expected_shape = (size,) + colors.shape
        else:
            expected_shape = size + colors.shape
        assert_equal(sample.shape, expected_shape)
        assert_((sample >= 0).all())
        assert_((sample <= colors).all())
        assert_array_equal(sample.sum(axis=-1),
                           vp.full(size, fill_value=nsample, dtype=int))
        if isinstance(size, int) and size >= 100000:
            # This sample is large enough to compare its mean to
            # the expected values.
            assert_allclose(sample.mean(axis=0),
                            nsample * colors / colors.sum(),
                            rtol=1e-3, atol=0.005)

    def __exclude_test_repeatability1(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([3, 4, 5], 5, size=5,
                                                    method='count')
        expected = vp.array([[2, 1, 2],
                             [2, 1, 2],
                             [1, 1, 3],
                             [2, 0, 3],
                             [2, 1, 2]])
        assert_array_equal(sample, expected)

    def __exclude_test_repeatability2(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([20, 30, 50], 50,
                                                    size=5,
                                                    method='marginals')
        expected = vp.array([[9, 17, 24],
                             [7, 13, 30],
                             [9, 15, 26],
                             [9, 17, 24],
                             [12, 14, 24]])
        assert_array_equal(sample, expected)

    def __exclude_test_repeatability3(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([20, 30, 50], 12,
                                                    size=5,
                                                    method='marginals')
        expected = vp.array([[2, 3, 7],
                             [5, 3, 4],
                             [2, 5, 5],
                             [5, 3, 4],
                             [1, 5, 6]])
        assert_array_equal(sample, expected)


class TestSetState(unittest.TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.rg = Generator(MT19937(self.seed))
        self.bit_generator = self.rg.bit_generator
        self.state = self.bit_generator.state
        # self.legacy_state = (self.state['bit_generator'],
        #                     self.state['state']['key'],
        #                     self.state['state']['pos'])

    def __exclude_test_gaussian_reset(self):
        # Make sure the cached every-other-Gaussian is reset.
        old = self.rg.standard_normal(size=3)
        self.bit_generator.state = self.state
        new = self.rg.standard_normal(size=3)
        assert_(vp.all(old == new))

    def __exclude_test_gaussian_reset_in_media_res(self):
        # When the state is saved with a cached Gaussian, make sure the
        # cached Gaussian is restored.

        self.rg.standard_normal()
        state = self.bit_generator.state
        old = self.rg.standard_normal(size=3)
        self.bit_generator.state = state
        new = self.rg.standard_normal(size=3)
        assert_(vp.all(old == new))

    def __exclude_test_negative_binomial(self):
        # Ensure that the negative binomial results take floating point
        # arguments without truncation.
        self.rg.negative_binomial(0.5, 0.5)


class TestIntegers(object):
    rfunc = random.integers

    # valid integer/boolean types
    itype = [bool, vp.int8, vp.uint8, vp.int16, vp.uint16,
             vp.int32, vp.uint32, vp.int64, vp.uint64]

    def __exclude_test_unsupported_type(self, endpoint):
        assert_raises(TypeError, self.rfunc, 1, endpoint=endpoint, dtype=float)

    def __exclude_test_bounds_checking(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else vp.iinfo(dt).min
            ubnd = 2 if dt is bool else vp.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, endpoint=endpoint,
                          dtype=dt)

            assert_raises(ValueError, self.rfunc, [lbnd - 1], ubnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd], [ubnd + 1],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [ubnd], [lbnd],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, [0],
                          endpoint=endpoint, dtype=dt)

    def __exclude_test_bounds_checking_array(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else vp.iinfo(dt).min
            ubnd = 2 if dt is bool else vp.iinfo(dt).max + (not endpoint)

            assert_raises(ValueError, self.rfunc, [lbnd - 1] * 2, [ubnd] * 2,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd] * 2,
                          [ubnd + 1] * 2, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, [lbnd] * 2,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [1] * 2, 0,
                          endpoint=endpoint, dtype=dt)

    def __exclude_test_rng_zero_and_extremes(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else vp.iinfo(dt).min
            ubnd = 2 if dt is bool else vp.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            is_open = not endpoint

            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc(tgt, [tgt + is_open], size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], [tgt + is_open],
                                    size=1000, endpoint=endpoint, dtype=dt),
                         tgt)

    def __exclude_test_rng_zero_and_extremes_array(self, endpoint):
        size = 1000
        for dt in self.itype:
            lbnd = 0 if dt is bool else vp.iinfo(dt).min
            ubnd = 2 if dt is bool else vp.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            tgt = ubnd - 1
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

    def __exclude_test_full_range(self, endpoint):

        for dt in self.itype:
            lbnd = 0 if dt is bool else vp.iinfo(dt).min
            ubnd = 2 if dt is bool else vp.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            try:
                self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def __exclude_test_full_range_array(self, endpoint):

        for dt in self.itype:
            lbnd = 0 if dt is bool else vp.iinfo(dt).min
            ubnd = 2 if dt is bool else vp.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            try:
                self.rfunc([lbnd] * 2, [ubnd], endpoint=endpoint, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def __exclude_test_in_bounds_fuzz(self, endpoint):
        # Don't use fixed seed
        # random = Generator(MT19937())

        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = self.rfunc(2, ubnd - endpoint, size=2 ** 16,
                                  endpoint=endpoint, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)

        vals = self.rfunc(0, 2 - endpoint, size=2 ** 16, endpoint=endpoint,
                          dtype=bool)
        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    def __exclude_test_scalar_array_equiv(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else vp.iinfo(dt).min
            ubnd = 2 if dt is bool else vp.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            size = 1000
            random = Generator(MT19937(1234))
            scalar = random.integers(lbnd, ubnd, size=size, endpoint=endpoint,
                                     dtype=dt)

            random = Generator(MT19937(1234))
            scalar_array = random.integers([lbnd], [ubnd], size=size,
                                           endpoint=endpoint, dtype=dt)

            random = Generator(MT19937(1234))
            array = random.integers([lbnd] * size, [ubnd] *
                                    size, size=size, endpoint=endpoint, dtype=dt)
            assert_array_equal(scalar, scalar_array)
            assert_array_equal(scalar, array)

    def __exclude_test_repeatability(self, endpoint):
        import hashlib
        # We use a md5 hash of generated sequences of 1000 samples
        # in the range [0, 6) for all but bool, where the range
        # is [0, 2). Hashes are for little endian numbers.
        tgt = {'bool': 'b3300e66d2bb59e493d255d47c3a6cbe',
               'int16': '39624ead49ad67e37545744024d2648b',
               'int32': '5c4810373f979336c6c0c999996e47a1',
               'int64': 'ab126c15edff26f55c50d2b7e37391ac',
               'int8': 'ba71ccaffeeeb9eeb1860f8075020b9c',
               'uint16': '39624ead49ad67e37545744024d2648b',
               'uint32': '5c4810373f979336c6c0c999996e47a1',
               'uint64': 'ab126c15edff26f55c50d2b7e37391ac',
               'uint8': 'ba71ccaffeeeb9eeb1860f8075020b9c'}

        for dt in self.itype[1:]:
            random = Generator(MT19937(1234))

            # view as little endian for hash
            if sys.byteorder == 'little':
                val = random.integers(0, 6 - endpoint, size=1000, endpoint=endpoint,
                                      dtype=dt)
            else:
                val = random.integers(0, 6 - endpoint, size=1000, endpoint=endpoint,
                                      dtype=dt).byteswap()

            res = hashlib.md5(val.view(vp.int8)).hexdigest()
            assert_(tgt[vp.dtype(dt).name] == res)

        # bools do not depend on endianness
        random = Generator(MT19937(1234))
        val = random.integers(0, 2 - endpoint, size=1000, endpoint=endpoint,
                              dtype=bool).view(vp.int8)
        res = hashlib.md5(val).hexdigest()
        assert_(tgt[vp.dtype(bool).name] == res)

    def __exclude_test_repeatability_broadcasting(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt in (bool, vp.bool_) else vp.iinfo(dt).min
            ubnd = 2 if dt in (bool, vp.bool_) else vp.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            # view as little endian for hash
            random = Generator(MT19937(1234))
            val = random.integers(lbnd, ubnd, size=1000, endpoint=endpoint,
                                  dtype=dt)

            random = Generator(MT19937(1234))
            val_bc = random.integers([lbnd] * 1000, ubnd, endpoint=endpoint,
                                     dtype=dt)

            assert_array_equal(val, val_bc)

            random = Generator(MT19937(1234))
            val_bc = random.integers([lbnd] * 1000, [ubnd] * 1000,
                                     endpoint=endpoint, dtype=dt)

            assert_array_equal(val, val_bc)

    def __exclude_test_int64_uint64_broadcast_exceptions(self, endpoint):
        configs = {vp.uint64: ((0, 2**65), (-1, 2**62), (10, 9), (0, 0)),
                   vp.int64: ((0, 2**64), (-(2**64), 2**62), (10, 9), (0, 0),
                              (-2**63 - 1, -2**63 - 1))}
        for dtype in configs:
            for config in configs[dtype]:
                low, high = config
                high = high - endpoint
                low_a = vp.array([[low] * 10])
                high_a = vp.array([high] * 10)
                assert_raises(ValueError, random.integers, low, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_a,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high_a,
                              endpoint=endpoint, dtype=dtype)

                low_o = vp.array([[low] * 10], dtype=object)
                high_o = vp.array([high] * 10, dtype=object)
                assert_raises(ValueError, random.integers, low_o, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_o,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_o, high_o,
                              endpoint=endpoint, dtype=dtype)

    def __exclude_test_int64_uint64_corner_case(self, endpoint):
        dt = vp.int64
        tgt = vp.iinfo(vp.int64).max
        lbnd = vp.int64(vp.iinfo(vp.int64).max)
        ubnd = vp.uint64(vp.iinfo(vp.int64).max + 1 - endpoint)

        # None of these function calls should
        # generate a ValueError now.
        actual = random.integers(lbnd, ubnd, endpoint=endpoint, dtype=dt)
        assert_equal(actual, tgt)

    def __exclude_test_respect_dtype_singleton(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else vp.iinfo(dt).min
            ubnd = 2 if dt is bool else vp.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            dt = vp.bool_ if dt is bool else dt

            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            assert_equal(sample.dtype, dt)

        for dt in (bool, int, vp.compat.long):
            lbnd = 0 if dt is bool else vp.iinfo(dt).min
            ubnd = 2 if dt is bool else vp.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            # Ensure that we get Python data types
            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            assert not hasattr(sample, 'dtype')
            assert_equal(type(sample), dt)

    def __exclude_test_respect_dtype_array(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else vp.iinfo(dt).min
            ubnd = 2 if dt is bool else vp.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            dt = vp.bool_ if dt is bool else dt

            sample = self.rfunc([lbnd], [ubnd], endpoint=endpoint, dtype=dt)
            assert_equal(sample.dtype, dt)
            sample = self.rfunc([lbnd] * 2, [ubnd] * 2, endpoint=endpoint,
                                dtype=dt)
            assert_equal(sample.dtype, dt)

    def __exclude_test_zero_size(self, endpoint):
        for dt in self.itype:
            sample = self.rfunc(0, 0, (3, 0, 4), endpoint=endpoint, dtype=dt)
            assert sample.shape == (3, 0, 4)
            assert sample.dtype == dt
            assert self.rfunc(0, -10, 0, endpoint=endpoint,
                              dtype=dt).shape == (0,)
            assert_equal(random.integers(0, 0, size=(3, 0, 4)).shape,
                         (3, 0, 4))
            assert_equal(random.integers(0, -10, size=0).shape, (0,))
            assert_equal(random.integers(10, 10, size=0).shape, (0,))

    def __exclude_test_error_byteorder(self):
        other_byteord_dt = '<i4' if sys.byteorder == 'big' else '>i4'
        with pytest.raises(ValueError):
            random.integers(0, 200, size=10, dtype=other_byteord_dt)

    # remove parameter
    def __exclude_test_integers_small_dtype_chisquared(self, sample_size, high,
                                                       dtype, chi2max):
        samples = random.integers(high, size=sample_size, dtype=dtype)

        values, counts = vp.unique(samples, return_counts=True)
        expected = sample_size / high
        chi2 = ((counts - expected)**2 / expected).sum()
        assert chi2 < chi2max


class TestRandomDist(unittest.TestCase):
    # Make sure the random distribution returns the correct value for a
    # given seed

    def setUp(self):
        self.seed = 1234567890

    @testing.multi_ve(multi_ve_node_max)
    def test_integers(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.integers(-99, 99, size=(3, 2))
                desired = vp.array([[-71, -27], [40, 65], [-63, -64]])
                assert_array_equal(actual[ve], desired)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_integers_masked(self):
        # Test masked rejection sampling algorithm to generate array of
        # uint32 in an interval.
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.integers(0, 99, size=(3, 2), dtype=vp.uint32)
                desired = vp.array([[28, 72], [40, 65], [36, 35]], dtype=vp.uint32)
                assert_array_equal(actual[ve], desired)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_integers_closed(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.integers(-99, 99, size=(3, 2), endpoint=True)
                desired = vp.array([[19, 19], [11, 23], [-51, 33]])
                assert_array_equal(actual[ve], desired)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_integers_max_int(self):
        import numpy
        # Tests whether integers with closed=True can generate the
        # maximum allowed Python int that can be converted
        # into a C long. Previous implementations of this
        # method have thrown an OverflowError when attempting
        # to generate this integer.
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                actual[ve] = random.integers(numpy.iinfo('l').max, numpy.iinfo('l').max,
                                             endpoint=True)

                desired = numpy.iinfo('l').max
                assert_equal(actual[ve].get(), desired)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_random(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual1[ve] = random.random((3, 2))
                desired = vp.array([[0.972810894716531, 0.152507473248988],
                                    [0.744906395673752, 0.788559705018997],
                                    [0.612674489850178, 0.044743933016434]]).get()
                assert_array_almost_equal(actual1[ve].get(), desired, decimal=15)

                random = Generator(MT19937(self.seed))
                actual2[ve] = random.random()
                assert_array_almost_equal(actual2[ve].get(), desired[0, 0], decimal=15)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_random_default_rng(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = vp.random.default_rng(self.seed)
                actual1[ve] = random.random((3, 2))
                desired = vp.array([[0.972810894716531, 0.152507473248988],
                                    [0.744906395673752, 0.788559705018997],
                                    [0.612674489850178, 0.044743933016434]]).get()
                assert_array_almost_equal(actual1[ve].get(), desired, decimal=15)

                random = Generator(MT19937(self.seed))
                actual2[ve] = random.random()
                assert_array_almost_equal(actual2[ve].get(), desired[0, 0], decimal=15)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_random_float(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.random((3, 2))
                desired = vp.array([[0.9728109, 0.1525075],
                                    [0.7449064, 0.7885597],
                                    [0.6126745, 0.0447439]])
                assert_array_almost_equal(actual[ve].get(), desired.get(), decimal=7)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_random_float_out(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                _type = vp.float32
                actual1[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.random(out=actual1[ve], dtype=_type)
                desired = vp.array([[0.97281, 0.15251],
                                    [0.74491, 0.78856],
                                    [0.61267, 0.04474]], dtype=_type).get()
                assert_array_almost_equal(actual1[ve].get(), desired, decimal=5)

                actual2[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.random(out=actual2[ve], size=(3, 2), dtype=_type)
                assert_array_almost_equal(actual2[ve].get(), desired, decimal=5)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_random_double_out(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                _type = vp.float64
                actual1[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.random(out=actual1[ve], dtype=_type)
                desired = vp.array([[0.972810894716531, 0.152507473248988],
                                    [0.744906395673752, 0.788559705018997],
                                    [0.612674489850178, 0.044743933016434]],
                                   dtype=_type).get()
                assert_array_almost_equal(actual1[ve].get(), desired, decimal=15)

                actual2[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.random(out=actual2[ve], size=(3, 2), dtype=_type)
                assert_array_almost_equal(actual2[ve].get(), desired, decimal=15)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_random_out_size_mismatch(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                out = vp.zeros(10)
                assert_raises(ValueError, random.random, size=20,
                              out=out)
                assert_raises(ValueError, random.random, size=(10, 1),
                              out=out)

    @testing.multi_ve(multi_ve_node_max)
    def test_random_out_type_mismatch(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                out = numpy.zeros(10)
                assert_raises(ValueError, random.random, size=10,
                              out=out)
                out = vp.zeros(10, dtype=int)
                assert_raises(TypeError, random.random, size=10,
                              out=out, dtype=float)

    def __exclude_test_random_float_scalar(self):
        random = Generator(MT19937(self.seed))
        actual = random.random(dtype=vp.float32)
        desired = 0.0969992
        assert_array_almost_equal(actual.get(), desired, decimal=7)

    @testing.multi_ve(multi_ve_node_max)
    def test_random_type_error(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_raises(TypeError, random.random, dtype='int32')

    def __exclude_test_choice_uniform_replace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 4)
        desired = vp.array([0, 0, 2, 2], dtype=vp.int64)
        assert_array_equal(actual, desired)

    def __exclude_test_choice_nonuniform_replace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        desired = vp.array([0, 1, 0, 1], dtype=vp.int64)
        assert_array_equal(actual, desired)

    def __exclude_test_choice_uniform_noreplace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 3, replace=False)
        desired = vp.array([2, 0, 3], dtype=vp.int64)
        assert_array_equal(actual, desired)
        actual = random.choice(4, 4, replace=False, shuffle=False)
        desired = vp.arange(4, dtype=vp.int64)
        assert_array_equal(actual, desired)

    def __exclude_test_choice_nonuniform_noreplace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 3, replace=False, p=[0.1, 0.3, 0.5, 0.1])
        desired = vp.array([0, 2, 3], dtype=vp.int64)
        assert_array_equal(actual, desired)

    def __exclude_test_choice_noninteger(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(['a', 'b', 'c', 'd'], 4)
        desired = vp.array(['a', 'a', 'c', 'c'])
        assert_array_equal(actual, desired)

    def __exclude_test_choice_multidimensional_default_axis(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 3)
        desired = vp.array([[0, 1], [0, 1], [4, 5]])
        assert_array_equal(actual, desired)

    def __exclude_test_choice_multidimensional_custom_axis(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 1, axis=1)
        desired = vp.array([[0], [2], [4], [6]])
        assert_array_equal(actual, desired)

    def __exclude_test_choice_exceptions(self):
        sample = random.choice
        assert_raises(ValueError, sample, -1, 3)
        assert_raises(ValueError, sample, 3., 3)
        assert_raises(ValueError, sample, [], 3)
        assert_raises(ValueError, sample, [1, 2, 3, 4], 3,
                      p=[[0.25, 0.25], [0.25, 0.25]])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], 2,
                      replace=False, p=[1, 0, 0])

    def __exclude_test_choice_return_shape(self):
        p = [0.1, 0.9]
        # Check scalar
        assert_(vp.isscalar(random.choice(2, replace=True)))
        assert_(vp.isscalar(random.choice(2, replace=False)))
        assert_(vp.isscalar(random.choice(2, replace=True, p=p)))
        assert_(vp.isscalar(random.choice(2, replace=False, p=p)))
        assert_(vp.isscalar(random.choice([1, 2], replace=True)))
        assert_(random.choice([None], replace=True) is None)
        a = vp.array([1, 2])
        arr = vp.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, replace=True) is a)

        # Check 0-d array
        s = tuple()
        assert_(not vp.isscalar(random.choice(2, s, replace=True)))
        assert_(not vp.isscalar(random.choice(2, s, replace=False)))
        assert_(not vp.isscalar(random.choice(2, s, replace=True, p=p)))
        assert_(not vp.isscalar(random.choice(2, s, replace=False, p=p)))
        assert_(not vp.isscalar(random.choice([1, 2], s, replace=True)))
        assert_(random.choice([None], s, replace=True).ndim == 0)
        a = vp.array([1, 2])
        arr = vp.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, s, replace=True).item() is a)

        # Check multi dimensional array
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(random.choice(6, s, replace=True).shape, s)
        assert_equal(random.choice(6, s, replace=False).shape, s)
        assert_equal(random.choice(6, s, replace=True, p=p).shape, s)
        assert_equal(random.choice(6, s, replace=False, p=p).shape, s)
        assert_equal(random.choice(vp.arange(6), s, replace=True).shape, s)

        # Check zero-size
        assert_equal(random.integers(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(random.integers(0, -10, size=0).shape, (0,))
        assert_equal(random.integers(10, 10, size=0).shape, (0,))
        assert_equal(random.choice(0, size=0).shape, (0,))
        assert_equal(random.choice([], size=(0,)).shape, (0,))
        assert_equal(random.choice(['a', 'b'], size=(3, 0, 4)).shape,
                     (3, 0, 4))
        assert_raises(ValueError, random.choice, [], 10)

    def __exclude_test_choice_nan_probabilities(self):
        a = vp.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, random.choice, a, p=p)

    def __exclude_test_choice_p_non_contiguous(self):
        p = vp.ones(10) / 5
        p[1::2] = 3.0
        random = Generator(MT19937(self.seed))
        non_contig = random.choice(5, 3, p=p[::2])
        random = Generator(MT19937(self.seed))
        contig = random.choice(5, 3, p=vp.ascontiguousarray(p[::2]))
        assert_array_equal(non_contig, contig)

    def __exclude_test_choice_return_type(self):
        p = vp.ones(4) / 4.
        actual = random.choice(4, 2)
        assert actual.dtype == vp.int64
        actual = random.choice(4, 2, replace=False)
        assert actual.dtype == vp.int64
        actual = random.choice(4, 2, p=p)
        assert actual.dtype == vp.int64
        actual = random.choice(4, 2, p=p, replace=False)
        assert actual.dtype == vp.int64

    def __exclude_test_choice_large_sample(self):
        import hashlib

        choice_hash = 'd44962a0b1e92f4a3373c23222244e21'
        random = Generator(MT19937(self.seed))
        actual = random.choice(10000, 5000, replace=False)
        if sys.byteorder != 'little':
            actual = actual.byteswap()
        res = hashlib.md5(actual.view(vp.int8)).hexdigest()
        assert_(choice_hash == res)

    @testing.multi_ve(multi_ve_node_max)
    def test_bytes(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.bytes(10)
                desired = b'R\x1e4\x0f\x80\x82!\x04\xa1\x1f'
                assert_equal(actual[ve], desired)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve], actual[ve + 1])

    @testing.multi_ve(multi_ve_node_max)
    def test_shuffle(self):
        # Test lists, arrays (of various dtypes), and multidimensional versions
        # of both, c-contiguous or not:
        for conv in [
            lambda x: vp.array([]),
            #             lambda x: x,
            #             lambda x: vp.asarray(x).astype(vp.int8),
            lambda x: vp.asarray(x).astype(vp.float32),
            #             lambda x: vp.asarray(x).astype(vp.complex64),
            #             lambda x: vp.asarray(x).astype(object),
            #             lambda x: [(i, i) for i in x],
            lambda x: vp.asarray([[i, i] for i in x]),
            #             lambda x: vp.vstack([x, x]).T,
            #             lambda x: (vp.asarray([(i, i) for i in x],
            #                                   [("a", int), ("b", int)])
            #                        .view(vp.recarray)),
            #             lambda x: vp.asarray([(i, i) for i in x],
            #                                  [("a", object), ("b", vp.int32)])
        ]:
            actual = {}
            for ve in range(0, multi_ve_node_max):
                with vp.venode.VE(ve):
                    random = Generator(MT19937(self.seed))
                    alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
                    random.shuffle(alist)
                    actual[ve] = alist
                    # desired = conv([0, 5, 1, 2, 9, 4, 7, 6, 3, 8])
                    desired = conv([0, 1, 9, 6, 2, 4, 5, 8, 7, 3])
                    assert_array_equal(actual[ve], desired)
            for ve in range(0, multi_ve_node_max):
                if ve >= (multi_ve_node_max - 1):
                    break
                assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_shuffle_custom_axis(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = vp.arange(16).reshape((4, 4))
                random.shuffle(actual[ve], axis=1)
                desired = vp.array([[0, 1, 3, 2],
                                    [4, 5, 7, 6],
                                    [8, 9, 11, 10],
                                    [12, 13, 15, 14]])
                assert_array_equal(actual[ve], desired)
                random = Generator(MT19937(self.seed))
                actual[ve] = vp.arange(16).reshape((4, 4))
                random.shuffle(actual[ve], axis=-1)
                assert_array_equal(actual[ve], desired)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_shuffle_axis_nonsquare(self):
        y1 = {}
        y2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                y1[ve] = vp.arange(20).reshape(2, 10)
                y2[ve] = y1[ve].copy()
                random = Generator(MT19937(self.seed))
                random.shuffle(y1[ve], axis=1)
                random = Generator(MT19937(self.seed))
                random.shuffle(y2[ve].T)
                assert_array_equal(y1[ve], y2[ve])
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(y1[ve].get(), y1[ve + 1].get())
            assert_equal(y2[ve].get(), y2[ve + 1].get())

    def __exclude_test_shuffle_masked(self):
        a = vp.ma.masked_values(vp.reshape(range(20), (5, 4)) % 3 - 1, -1)
        b = vp.ma.masked_values(vp.arange(20) % 3 - 1, -1)
        a_orig = a.copy()
        b_orig = b.copy()
        for i in range(50):
            random.shuffle(a)
            assert_equal(
                sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
            random.shuffle(b)
            assert_equal(
                sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))

    def __exclude_test_shuffle_exceptions(self):
        random = Generator(MT19937(self.seed))
        arr = vp.arange(10)
        assert_raises(vp.AxisError, random.shuffle, arr, 1)
        arr = vp.arange(9).reshape((3, 3))
        assert_raises(vp.AxisError, random.shuffle, arr, 3)
        assert_raises(TypeError, random.shuffle, arr, slice(1, 2, None))
        arr = [[1, 2, 3], [4, 5, 6]]
        assert_raises(NotImplementedError, random.shuffle, arr, 1)

    @testing.multi_ve(multi_ve_node_max)
    def test_permutation(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                actual1[ve] = random.permutation(alist)
                desired = [0, 1, 9, 6, 2, 4, 5, 8, 7, 3]
                assert_array_equal(actual1[ve].get(), desired)

                random = Generator(MT19937(self.seed))
                integer_val = 10
                desired = [9, 0, 8, 5, 1, 3, 4, 7, 6, 2]
                actual2[ve] = random.permutation(integer_val)
                assert_array_equal(actual2[ve].get(), desired)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_permutation_custom_axis(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                a = vp.arange(16).reshape((4, 4))
                desired = vp.array([[0, 1, 3, 2],
                                    [4, 5, 7, 6],
                                    [8, 9, 11, 10],
                                    [12, 13, 15, 14]])
                random = Generator(MT19937(self.seed))
                actual1[ve] = random.permutation(a, axis=1)
                assert_array_equal(actual1[ve], desired)
                random = Generator(MT19937(self.seed))
                actual2[ve] = random.permutation(a, axis=-1)
                assert_array_equal(actual2[ve], desired)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    def __exclude_test_permutation_exceptions(self):
        random = Generator(MT19937(self.seed))
        arr = vp.arange(10)
        assert_raises(vp.AxisError, random.permutation, arr, 1)
        arr = vp.arange(9).reshape((3, 3))
        assert_raises(vp.AxisError, random.permutation, arr, 3)
        assert_raises(TypeError, random.permutation, arr, slice(1, 2, None))

    def __exclude_test_beta(self):
        random = Generator(MT19937(self.seed))
        actual = random.beta(.1, .9, size=(3, 2))
        desired = vp.array(
            [[1.083029353267698e-10, 2.449965303168024e-11],
             [2.397085162969853e-02, 3.590779671820755e-08],
             [2.830254190078299e-04, 1.744709918330393e-01]])
        assert_array_almost_equal(actual, desired, decimal=15)

    @testing.multi_ve(multi_ve_node_max)
    def test_binomial(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual1[ve] = random.binomial(100.123, .456, size=(3, 2))
                desired = vp.array([[55, 40],
                                    [49, 50],
                                    [47, 37]])
                assert_array_equal(actual1[ve], desired)

                random = Generator(MT19937(self.seed))
                actual2[ve] = random.binomial(100.123, .456)
                desired = vp.array([55])
                assert_array_equal(actual2[ve], desired)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    def __exclude_test_chisquare(self):
        random = Generator(MT19937(self.seed))
        actual = random.chisquare(50, size=(3, 2))
        desired = vp.array([[32.9850547060149, 39.0219480493301],
                            [56.2006134779419, 57.3474165711485],
                            [55.4243733880198, 55.4209797925213]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def __exclude_test_dirichlet(self):
        random = Generator(MT19937(self.seed))
        alpha = vp.array([51.72840233779265162, 39.74494232180943953])
        actual = random.dirichlet(alpha, size=(3, 2))
        desired = vp.array([[[0.5439892869558927, 0.45601071304410745],
                             [0.5588917345860708, 0.4411082654139292]],
                            [[0.5632074165063435, 0.43679258349365657],
                             [0.54862581112627, 0.45137418887373015]],
                            [[0.49961831357047226, 0.5003816864295278],
                             [0.52374806183482, 0.47625193816517997]]])
        assert_array_almost_equal(actual, desired, decimal=15)
        bad_alpha = vp.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, random.dirichlet, bad_alpha)

        random = Generator(MT19937(self.seed))
        alpha = vp.array([51.72840233779265162, 39.74494232180943953])
        actual = random.dirichlet(alpha)
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def __exclude_test_dirichlet_size(self):
        p = vp.array([51.72840233779265162, 39.74494232180943953])
        assert_equal(random.dirichlet(p, vp.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, vp.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, vp.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, vp.array((2, 2))).shape, (2, 2, 2))

        assert_raises(TypeError, random.dirichlet, p, float(1))

    def __exclude_test_dirichlet_bad_alpha(self):
        alpha = vp.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, random.dirichlet, alpha)

    def __exclude_test_dirichlet_alpha_non_contiguous(self):
        a = vp.array([51.72840233779265162, -1.0, 39.74494232180943953])
        alpha = a[::2]
        random = Generator(MT19937(self.seed))
        non_contig = random.dirichlet(alpha, size=(3, 2))
        random = Generator(MT19937(self.seed))
        contig = random.dirichlet(vp.ascontiguousarray(alpha),
                                  size=(3, 2))
        assert_array_almost_equal(non_contig, contig)

    @testing.multi_ve(multi_ve_node_max)
    def test_exponential(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.exponential(1.1234, size=(3, 2))
                desired = vp.array([[4.04978839, 0.18589266],
                                    [1.53470452, 1.74555309],
                                    [1.06553347, 0.05142458]]).get()
                assert_array_almost_equal(actual[ve].get(), desired, decimal=6)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_exponential_0(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_equal(random.exponential(scale=0).get(), 0)
                # assert_raises(ValueError, random.exponential, scale=-0.)

    def __exclude_test_f(self):
        random = Generator(MT19937(self.seed))
        actual = random.f(12, 77, size=(3, 2))
        desired = vp.array([[0.461720027077085, 1.100441958872451],
                            [1.100337455217484, 0.91421736740018],
                            [0.500811891303113, 0.826802454552058]])
        assert_array_almost_equal(actual, desired, decimal=15)

    @testing.multi_ve(multi_ve_node_max)
    def test_gamma(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.gamma(5, 3, size=(3, 2))
                desired = vp.array([[30.34037139, 8.40365941],
                                    [18.69797822, 19.83477528],
                                    [15.93935665, 5.72660182]]).get()
                assert_array_almost_equal(actual[ve].get(), desired, decimal=6)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_gamma_0(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_equal(random.gamma(shape=0, scale=0).get(), 0)
                # assert_raises(ValueError, random.gamma, shape=-0., scale=-0.)

    @testing.multi_ve(multi_ve_node_max)
    def test_geometric(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.geometric(.123456789, size=(3, 2))
                desired = vp.array([[28, 2],
                                    [11, 12],
                                    [8, 1]]).get()
                assert_array_equal(actual[ve].get(), desired)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    def __exclude_test_geometric_exceptions(self):
        assert_raises(ValueError, random.geometric, 1.1)
        assert_raises(ValueError, random.geometric, [1.1] * 10)
        assert_raises(ValueError, random.geometric, -0.1)
        assert_raises(ValueError, random.geometric, [-0.1] * 10)
        with vp.errstate(invalid='ignore'):
            assert_raises(ValueError, random.geometric, vp.nan)
            assert_raises(ValueError, random.geometric, [vp.nan] * 10)

    @testing.multi_ve(multi_ve_node_max)
    def test_gumbel(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.gumbel(loc=.123456789, scale=2.0, size=(3, 2))
                desired = vp.array([[7.05891882, -1.38657651],
                                    [2.32151809, 2.75132146],
                                    [1.30356344, -2.39064293]]).get()
                assert_array_almost_equal(actual[ve].get(), desired, decimal=6)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_gumbel_0(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_equal(random.gumbel(scale=0).get(), 0)
                # assert_raises(ValueError, random.gumbel, scale=-0.)

    def __exclude_test_hypergeometric(self):
        random = Generator(MT19937(self.seed))
        actual = random.hypergeometric(10.1, 5.5, 14, size=(3, 2))
        desired = vp.array([[9, 9],
                            [9, 9],
                            [10, 9]])
        assert_array_equal(actual, desired)

        # Test nbad = 0
        actual = random.hypergeometric(5, 0, 3, size=4)
        desired = vp.array([3, 3, 3, 3])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(15, 0, 12, size=4)
        desired = vp.array([12, 12, 12, 12])
        assert_array_equal(actual, desired)

        # Test ngood = 0
        actual = random.hypergeometric(0, 5, 3, size=4)
        desired = vp.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(0, 15, 12, size=4)
        desired = vp.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

    def __exclude_test_laplace(self):
        random = Generator(MT19937(self.seed))
        actual = random.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        desired = vp.array([[-3.156353949272393, 1.195863024830054],
                            [-3.435458081645966, 1.656882398925444],
                            [0.924824032467446, 1.251116432209336]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def __exclude_test_laplace_0(self):
        assert_equal(random.laplace(scale=0), 0)
        assert_raises(ValueError, random.laplace, scale=-0.)

    @testing.multi_ve(multi_ve_node_max)
    def test_logistic(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.logistic(loc=.123456789, scale=2.0, size=(3, 2))
                desired = vp.array([[7.27820351, -3.30668005],
                                    [2.26671282, 2.7559877],
                                    [1.04059345, -5.99859037]]).get()
                assert_array_almost_equal(actual[ve].get(), desired, decimal=6)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_lognormal(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
                desired = vp.array([[53.04177796, 0.1454359],
                                    [4.22301457, 5.61995842],
                                    [2.00581375, 0.0379019]]).get()
                assert_array_almost_equal(actual[ve].get(), desired, decimal=6)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_lognormal_0(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_equal(random.lognormal(sigma=0).get(), vp.array([1]))
                # assert_raises(ValueError, random.lognormal, sigma=-0.)

    def __exclude_test_logseries(self):
        random = Generator(MT19937(self.seed))
        actual = random.logseries(p=.923456789, size=(3, 2))
        desired = vp.array([[14, 17],
                            [3, 18],
                            [5, 1]])
        assert_array_equal(actual, desired)

    def __exclude_test_logseries_exceptions(self):
        with vp.errstate(invalid='ignore'):
            assert_raises(ValueError, random.logseries, vp.nan)
            assert_raises(ValueError, random.logseries, [vp.nan] * 10)

    def __exclude_test_multinomial(self):
        random = Generator(MT19937(self.seed))
        actual = random.multinomial(20, [1 / 6.] * 6, size=(3, 2))
        desired = vp.array([[[1, 5, 1, 6, 4, 3],
                             [4, 2, 6, 2, 4, 2]],
                            [[5, 3, 2, 6, 3, 1],
                             [4, 4, 0, 2, 3, 7]],
                            [[6, 3, 1, 5, 3, 2],
                             [5, 5, 3, 1, 2, 4]]])
        assert_array_equal(actual, desired)

    def __exclude_test_multivariate_normal(self, method):
        random = Generator(MT19937(self.seed))
        mean = (.123456789, 10)
        cov = [[1, 0], [0, 1]]
        size = (3, 2)
        actual = random.multivariate_normal(mean, cov, size, method=method)
        desired = vp.array([[[-1.747478062846581, 11.25613495182354],
                             [-0.9967333370066214, 10.342002097029821]],
                            [[0.7850019631242964, 11.181113712443013],
                             [0.8901349653255224, 8.873825399642492]],
                            [[0.7130260107430003, 9.551628690083056],
                             [0.7127098726541128, 11.991709234143173]]])

        assert_array_almost_equal(actual, desired, decimal=15)

        # Check for default size, was raising deprecation warning
        actual = random.multivariate_normal(mean, cov, method=method)
        desired = vp.array([0.233278563284287, 9.424140804347195])
        assert_array_almost_equal(actual, desired, decimal=15)
        # Check that non symmetric covariance input raises exception when
        # check_valid='raises' if using default svd method.
        mean = [0, 0]
        cov = [[1, 2], [1, 2]]
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise')

        # Check that non positive-semidefinite covariance warns with
        # RuntimeWarning
        cov = [[1, 2], [2, 1]]
        assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov)
        assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov,
                     method='eigh')
        assert_raises(LinAlgError, random.multivariate_normal, mean, cov,
                      method='cholesky')

        # and that it doesn't warn with RuntimeWarning check_valid='ignore'
        assert_no_warnings(random.multivariate_normal, mean, cov,
                           check_valid='ignore')

        # and that it raises with RuntimeWarning check_valid='raises'
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise')
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise', method='eigh')

        cov = vp.array([[1, 0.1], [0.1, 1]], dtype=vp.float32)
        with suppress_warnings() as sup:
            random.multivariate_normal(mean, cov, method=method)
            w = sup.record(RuntimeWarning)
            assert len(w) == 0

        mu = vp.zeros(2)
        cov = vp.eye(2)
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='other')
        assert_raises(ValueError, random.multivariate_normal,
                      vp.zeros((2, 1, 1)), cov)
        assert_raises(ValueError, random.multivariate_normal,
                      mu, vp.empty((3, 2)))
        assert_raises(ValueError, random.multivariate_normal,
                      mu, vp.eye(3))

    def __exclude_test_negative_binomial(self):
        random = Generator(MT19937(self.seed))
        actual = random.negative_binomial(n=100, p=.12345, size=(3, 2))
        desired = vp.array([[543, 727],
                            [775, 760],
                            [600, 674]])
        assert_array_equal(actual, desired)

    def __exclude_test_negative_binomial_exceptions(self):
        with vp.errstate(invalid='ignore'):
            assert_raises(ValueError, random.negative_binomial, 100, vp.nan)
            assert_raises(ValueError, random.negative_binomial, 100,
                          [vp.nan] * 10)

    def __exclude_test_noncentral_chisquare(self):
        random = Generator(MT19937(self.seed))
        actual = random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        desired = vp.array([[1.70561552362133, 15.97378184942111],
                            [13.71483425173724, 20.17859633310629],
                            [11.3615477156643, 3.67891108738029]])
        assert_array_almost_equal(actual, desired, decimal=14)

        actual = random.noncentral_chisquare(df=.5, nonc=.2, size=(3, 2))
        desired = vp.array([[9.41427665607629e-04, 1.70473157518850e-04],
                            [1.14554372041263e+00, 1.38187755933435e-03],
                            [1.90659181905387e+00, 1.21772577941822e+00]])
        assert_array_almost_equal(actual, desired, decimal=14)

        random = Generator(MT19937(self.seed))
        actual = random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        desired = vp.array([[0.82947954590419, 1.80139670767078],
                            [6.58720057417794, 7.00491463609814],
                            [6.31101879073157, 6.30982307753005]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def __exclude_test_noncentral_f(self):
        random = Generator(MT19937(self.seed))
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=1,
                                     size=(3, 2))
        desired = vp.array([[0.060310671139, 0.23866058175939],
                            [0.86860246709073, 0.2668510459738],
                            [0.23375780078364, 1.88922102885943]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def __exclude_test_noncentral_f_nan(self):
        random = Generator(MT19937(self.seed))
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=vp.nan)
        assert vp.isnan(actual)

    @testing.multi_ve(multi_ve_node_max)
    def test_normal(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.normal(loc=.123456789, scale=2.0, size=(3, 2))
                desired = vp.array([[3.97107987, -1.92801985],
                                    [1.44054923, 1.72632427],
                                    [0.69604984, -3.27275406]]).get()
                assert_array_almost_equal(actual[ve].get(), desired, decimal=6)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_normal_0(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_equal(random.normal(scale=0).get(), vp.array([0]).get())
                # assert_raises(ValueError, random.normal, scale=-0.)

    def __exclude_test_pareto(self):
        random = Generator(MT19937(self.seed))
        actual = random.pareto(a=.123456789, size=(3, 2))
        desired = vp.array([[1.0394926776069018e+00, 7.7142534343505773e+04],
                            [7.2640150889064703e-01, 3.4650454783825594e+05],
                            [4.5852344481994740e+04, 6.5851383009539105e+07]])
        vp.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)

    @testing.multi_ve(multi_ve_node_max)
    def test_poisson(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.poisson(lam=.123456789, size=(3, 2))
                desired = vp.array([[1, 0],
                                    [0, 0],
                                    [0, 0]]).get()
                assert_array_equal(actual[ve].get(), desired)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    def __exclude_test_poisson_exceptions(self):
        lambig = numpy.iinfo('int64').max
        lamneg = -1
        assert_raises(ValueError, random.poisson, lamneg)
        assert_raises(ValueError, random.poisson, [lamneg] * 10)
        assert_raises(ValueError, random.poisson, lambig)
        assert_raises(ValueError, random.poisson, [lambig] * 10)
        with vp.errstate(invalid='ignore'):
            assert_raises(ValueError, random.poisson, vp.nan)
            assert_raises(ValueError, random.poisson, [vp.nan] * 10)

    def __exclude_test_power(self):
        random = Generator(MT19937(self.seed))
        actual = random.power(a=.123456789, size=(3, 2))
        desired = vp.array([[1.977857368842754e-09, 9.806792196620341e-02],
                            [2.482442984543471e-10, 1.527108843266079e-01],
                            [8.188283434244285e-02, 3.950547209346948e-01]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def __exclude_test_rayleigh(self):
        random = Generator(MT19937(self.seed))
        actual = random.rayleigh(scale=10, size=(3, 2))
        desired = vp.array([[4.51734079831581, 15.6802442485758],
                            [4.19850651287094, 17.08718809823704],
                            [14.7907457708776, 15.85545333419775]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def __exclude_test_rayleigh_0(self):
        assert_equal(random.rayleigh(scale=0), 0)
        assert_raises(ValueError, random.rayleigh, scale=-0.)

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_cauchy(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.standard_cauchy(size=(3, 2))
                desired = vp.array([[-0.08562544, 0.51948822],
                                    [-1.03252734, -0.7829524],
                                    [-2.70604952, 0.14150042]]).get()
                assert_array_almost_equal(actual[ve].get(), desired, decimal=6)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_exponential(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                # actual = random.standard_exponential(size=(3, 2), method='inv')
                # desired = vp.array([[3.604938926967113, 0.165473257759317],
                #                    [1.366124725390337, 1.553812613582069],
                #                    [0.948489827990156, 0.045775841475542]])
                # assert_array_almost_equal(actual.get(), desired.get(), decimal=15)
                assert_raises(NotImplementedError, random.standard_exponential,
                              method='inv')

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_exponential_float(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.standard_exponential(size=(3, 2), dtype=vp.float32)
                desired = vp.array([[3.60494, 0.16547],
                                    [1.36612, 1.55381],
                                    [0.94849, 0.04578]])
                assert_array_almost_equal(actual[ve].get(), desired.get(), decimal=5)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_exponential_float_out(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                _type = vp.float32
                actual1[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_exponential(out=actual1[ve], dtype=_type)
                desired = vp.array([[3.60494, 0.16547],
                                    [1.36612, 1.55381],
                                    [0.94849, 0.04578]], dtype=_type).get()
                assert_array_almost_equal(actual1[ve].get(), desired, decimal=5)

                actual2[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_exponential(out=actual2[ve], size=(3, 2), dtype=_type)
                assert_array_almost_equal(actual2[ve].get(), desired, decimal=5)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_exponential_double_out(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                _type = vp.float64
                actual1[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_exponential(out=actual1[ve], dtype=_type)
                desired = vp.array([[3.604938926967113, 0.165473257759317],
                                    [1.366124725390337, 1.553812613582069],
                                    [0.948489827990156, 0.045775841475542]],
                                   dtype=_type).get()
                assert_array_almost_equal(actual1[ve].get(), desired, decimal=15)

                actual2[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_exponential(out=actual2[ve], size=(3, 2), dtype=_type)
                assert_array_almost_equal(actual2[ve].get(), desired, decimal=15)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_exponential_type_error(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_raises(TypeError, random.standard_exponential, dtype=vp.int32)

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_exponential_out_size_mismatch(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                out = vp.zeros(10)
                assert_raises(ValueError, random.standard_exponential, size=20,
                              out=out)
                assert_raises(ValueError, random.standard_exponential, size=(10, 1),
                              out=out)

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_normal_out_size_mismatch(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                out = vp.zeros(10)
                assert_raises(ValueError, random.standard_normal, size=20,
                              out=out)
                assert_raises(ValueError, random.standard_normal, size=(10, 1),
                              out=out)

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_gamma(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.standard_gamma(shape=3, size=(3, 2))
                desired = vp.array([[7.11449558765142, 1.34138308328934],
                                    [3.88708213595362, 4.19078528596117],
                                    [3.16446924209137, 0.78172091245845]])
                assert_array_almost_equal(actual[ve].get(), desired.get(), decimal=14)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_gamma_scalar_float(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.standard_gamma(3, dtype=vp.float32)
                desired = 7.1144938399353027
                assert_array_almost_equal(actual[ve].get(), desired, decimal=6)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_gamma_float(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.standard_gamma(shape=3, size=(3, 2),
                                                   dtype=vp.float32)
                desired = vp.array([[7.1145, 1.34138],
                                    [3.88708, 4.19079],
                                    [3.16447, 0.78172]])
                assert_array_almost_equal(actual[ve].get(), desired.get(), decimal=5)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_gamma_float_out(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                _type = vp.float32
                actual1[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_gamma(10.0, out=actual1[ve], dtype=_type)
                desired = vp.array([[16.92293, 6.82809],
                                    [11.85711, 12.37146],
                                    [10.58375, 5.31525]], dtype=_type).get()
                assert_array_almost_equal(actual1[ve].get(), desired, decimal=5)

                actual2[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_gamma(10.0, out=actual2[ve], size=(3, 2), dtype=_type)
                assert_array_almost_equal(actual2[ve].get(), desired, decimal=5)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_gamma_double_out(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                _type = vp.float64
                actual1[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_gamma(10.0, out=actual1[ve], dtype=_type)
                desired = vp.array([[16.922929295355765, 6.828085974477842],
                                    [11.857108584627834, 12.371458694938216],
                                    [10.583745848607224, 5.31525417506873]],
                                   dtype=_type).get()
                assert_array_almost_equal(actual1[ve].get(), desired, decimal=15)

                actual2[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_gamma(10.0, out=actual2[ve], size=(3, 2), dtype=_type)
                assert_array_almost_equal(actual2[ve].get(), desired, decimal=15)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_gamma_type_error(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_raises(TypeError, random.standard_gamma, 1.,
                              dtype='int32')

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_gamma_out_size_mismatch(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                out = vp.zeros(10)
                assert_raises(ValueError, random.standard_gamma, 10.0, size=20,
                              out=out)
                assert_raises(ValueError, random.standard_gamma, 10.0, size=(10, 1),
                              out=out)

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_gamma_shape_negative(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                assert_raises(ValueError, random.standard_gamma, -2)

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_gamma_0(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_equal(random.standard_gamma(shape=0).get(), [0])
                assert_equal(random.standard_gamma(shape=-0.).get(), [0])
                # assert_raises(ValueError, random.standard_gamma, shape=-0.)

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_normal(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.standard_normal(size=(3, 2))
                desired = vp.array([[1.923811538759556, -1.025738320166633],
                                    [0.658546218189285, 0.801433738216246],
                                    [0.286296523876628, -1.698105424133927]])
                assert_array_almost_equal(actual[ve].get(), desired.get(), decimal=15)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_normal_float(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.standard_normal(size=(3, 2), dtype=vp.float32)
                desired = vp.array([[1.92381, -1.02574],
                                    [0.65855, 0.80143],
                                    [0.2863, -1.69811]])
                assert_array_almost_equal(actual[ve].get(), desired.get(), decimal=5)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_normal_float_out(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                _type = vp.float32
                actual1[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_normal(out=actual1[ve], dtype=_type)
                desired = vp.array([[1.92381, -1.02574],
                                    [0.65855, 0.80143],
                                    [0.2863, -1.69811]], dtype=_type).get()
                assert_array_almost_equal(actual1[ve].get(), desired, decimal=5)

                actual2[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_normal(out=actual2[ve], size=(3, 2), dtype=_type)
                assert_array_almost_equal(actual2[ve].get(), desired, decimal=5)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_normal_double_out(self):
        actual1 = {}
        actual2 = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                _type = vp.float64
                actual1[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_normal(out=actual1[ve], dtype=_type)
                desired = vp.array([[1.923811538759556, -1.025738320166633],
                                    [0.658546218189285, 0.801433738216246],
                                    [0.286296523876628, -1.698105424133927]],
                                   dtype=_type).get()
                assert_array_almost_equal(actual1[ve].get(), desired, decimal=15)

                actual2[ve] = vp.zeros((3, 2), dtype=_type)
                random = Generator(MT19937(self.seed))
                random.standard_normal(out=actual2[ve], size=(3, 2), dtype=_type)
                assert_array_almost_equal(actual2[ve].get(), desired, decimal=15)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual1[ve].get(), actual1[ve + 1].get())
            assert_equal(actual2[ve].get(), actual2[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_standard_normal_type_error(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                assert_raises(TypeError, random.standard_normal, dtype=vp.int32)

    def __exclude_test_standard_t(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_t(df=10, size=(3, 2))
        desired = vp.array([[-1.484666193042647, 0.30597891831161],
                            [1.056684299648085, -0.407312602088507],
                            [0.130704414281157, -2.038053410490321]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def __exclude_test_triangular(self):
        random = Generator(MT19937(self.seed))
        actual = random.triangular(left=5.12, mode=10.23, right=20.34,
                                   size=(3, 2))
        desired = vp.array([[7.86664070590917, 13.6313848513185],
                            [7.68152445215983, 14.36169131136546],
                            [13.16105603911429, 13.72341621856971]])
        assert_array_almost_equal(actual, desired, decimal=14)

    @testing.multi_ve(multi_ve_node_max)
    def test_uniform(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.uniform(low=1.23, high=10.54, size=(3, 2))
                desired = vp.array([[10.28686943, 2.64984458],
                                    [8.16507854, 8.57149085],
                                    [6.9339995, 1.64656602]]).get()
                assert_array_almost_equal(actual[ve].get(), desired, decimal=6)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    def __exclude_test_uniform_range_bounds(self):
        fmin = vp.finfo('float').min
        fmax = vp.finfo('float').max

        func = random.uniform
        assert_raises(OverflowError, func, -vp.inf, 0)
        assert_raises(OverflowError, func, 0, vp.inf)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, [-vp.inf], [0])
        assert_raises(OverflowError, func, [0], [vp.inf])

        random.uniform(low=vp.nextafter(fmin, 1), high=fmax / 1e17)

    def __exclude_test_scalar_exception_propagation(self):

        class ThrowingFloat(vp.ndarray):
            def __float__(self):
                raise TypeError

        throwing_float = vp.array(1.0).view(ThrowingFloat)
        assert_raises(TypeError, random.uniform, throwing_float,
                      throwing_float)

        class ThrowingInteger(vp.ndarray):
            def __int__(self):
                raise TypeError

        throwing_int = vp.array(1).view(ThrowingInteger)
        assert_raises(TypeError, random.hypergeometric, throwing_int, 1, 1)

    def __exclude_test_vonmises(self):
        random = Generator(MT19937(self.seed))
        actual = random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        desired = vp.array([[1.107972248690106, 2.841536476232361],
                            [1.832602376042457, 1.945511926976032],
                            [-0.260147475776542, 2.058047492231698]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def __exclude_test_vonmises_small(self):
        # check infinite loop.
        random = Generator(MT19937(self.seed))
        r = random.vonmises(mu=0., kappa=1.1e-8, size=10**6)
        assert_(vp.isfinite(r).all())

    def __exclude_test_vonmises_nan(self):
        random = Generator(MT19937(self.seed))
        r = random.vonmises(mu=0., kappa=vp.nan)
        assert_(vp.isnan(r))

    def __exclude_test_wald(self):
        random = Generator(MT19937(self.seed))
        actual = random.wald(mean=1.23, scale=1.54, size=(3, 2))
        desired = vp.array([[0.26871721804551, 3.2233942732115],
                            [2.20328374987066, 2.40958405189353],
                            [2.07093587449261, 0.73073890064369]])
        assert_array_almost_equal(actual, desired, decimal=14)

    @testing.multi_ve(multi_ve_node_max)
    def test_weibull(self):
        actual = {}
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                actual[ve] = random.weibull(a=1.23, size=(3, 2))
                desired = vp.array([[2.83636769, 0.2316431],
                                    [1.28870869, 1.43089792],
                                    [0.9579159, 0.08148665]]).get()
                assert_array_almost_equal(actual[ve].get(), desired, decimal=6)
        for ve in range(0, multi_ve_node_max):
            if ve >= (multi_ve_node_max - 1):
                break
            assert_equal(actual[ve].get(), actual[ve + 1].get())

    @testing.multi_ve(multi_ve_node_max)
    def test_weibull_0(self):
        for ve in range(0, multi_ve_node_max):
            with vp.venode.VE(ve):
                random = Generator(MT19937(self.seed))
                assert_equal(random.weibull(a=0, size=12).get(), vp.zeros(12).get())
                # assert_raises(ValueError, random.weibull, a=-0.)

    def __exclude_test_zipf(self):
        random = Generator(MT19937(self.seed))
        actual = random.zipf(a=1.23, size=(3, 2))
        desired = vp.array([[1, 1],
                            [10, 867],
                            [354, 2]])
        assert_array_equal(actual, desired)


class TestBroadcast(unittest.TestCase):
    # tests that functions that broadcast behave
    # correctly when presented with non-scalar arguments
    def setUp(self):
        self.seed = 123456789

    def __exclude_test_uniform(self):
        random = Generator(MT19937(self.seed))
        low = [0]
        high = [1]
        desired = vp.array([0.16693771389729, 0.19635129550675, 0.75563050964095])

        random = Generator(MT19937(self.seed))
        actual = random.uniform(low * 3, high)
        assert_array_almost_equal(actual, desired, decimal=14)

        random = Generator(MT19937(self.seed))
        actual = random.uniform(low, high * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def __exclude_test_normal(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        random = Generator(MT19937(self.seed))
        desired = vp.array([-0.38736406738527, 0.79594375042255, 0.0197076236097])

        random = Generator(MT19937(self.seed))
        actual = random.normal(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.normal, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        normal = random.normal
        actual = normal(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc, bad_scale * 3)

    def __exclude_test_beta(self):
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        desired = vp.array([0.18719338682602, 0.73234824491364, 0.17928615186455])

        random = Generator(MT19937(self.seed))
        beta = random.beta
        actual = beta(a * 3, b)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a * 3, b)
        assert_raises(ValueError, beta, a * 3, bad_b)

        random = Generator(MT19937(self.seed))
        actual = random.beta(a, b * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def __exclude_test_exponential(self):
        scale = [1]
        bad_scale = [-1]
        desired = vp.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        random = Generator(MT19937(self.seed))
        actual = random.exponential(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.exponential, bad_scale * 3)

    def __exclude_test_standard_gamma(self):
        shape = [1]
        bad_shape = [-1]
        desired = vp.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        random = Generator(MT19937(self.seed))
        std_gamma = random.standard_gamma
        actual = std_gamma(shape * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, std_gamma, bad_shape * 3)

    def __exclude_test_gamma(self):
        shape = [1]
        scale = [2]
        bad_shape = [-1]
        bad_scale = [-2]
        desired = vp.array([1.34491986425611, 0.42760990636187, 1.4355697857258])

        random = Generator(MT19937(self.seed))
        gamma = random.gamma
        actual = gamma(shape * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape * 3, scale)
        assert_raises(ValueError, gamma, shape * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        gamma = random.gamma
        actual = gamma(shape, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape, scale * 3)
        assert_raises(ValueError, gamma, shape, bad_scale * 3)

    def __exclude_test_f(self):
        dfnum = [1]
        dfden = [2]
        bad_dfnum = [-1]
        bad_dfden = [-2]
        desired = vp.array([0.07765056244107, 7.72951397913186, 0.05786093891763])

        random = Generator(MT19937(self.seed))
        f = random.f
        actual = f(dfnum * 3, dfden)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)

        random = Generator(MT19937(self.seed))
        f = random.f
        actual = f(dfnum, dfden * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum, dfden * 3)
        assert_raises(ValueError, f, dfnum, bad_dfden * 3)

    def __exclude_test_noncentral_f(self):
        dfnum = [2]
        dfden = [3]
        nonc = [4]
        bad_dfnum = [0]
        bad_dfden = [-1]
        bad_nonc = [-2]
        desired = vp.array([2.02434240411421, 12.91838601070124, 1.24395160354629])

        random = Generator(MT19937(self.seed))
        nonc_f = random.noncentral_f
        actual = nonc_f(dfnum * 3, dfden, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert vp.all(vp.isnan(nonc_f(dfnum, dfden, [vp.nan] * 3)))

        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)

        random = Generator(MT19937(self.seed))
        nonc_f = random.noncentral_f
        actual = nonc_f(dfnum, dfden * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)

        random = Generator(MT19937(self.seed))
        nonc_f = random.noncentral_f
        actual = nonc_f(dfnum, dfden, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)

    def __exclude_test_noncentral_f_small_df(self):
        random = Generator(MT19937(self.seed))
        desired = vp.array([0.04714867120827, 0.1239390327694])
        actual = random.noncentral_f(0.9, 0.9, 2, size=2)
        assert_array_almost_equal(actual, desired, decimal=14)

    def __exclude_test_chisquare(self):
        df = [1]
        bad_df = [-1]
        desired = vp.array([0.05573640064251, 1.47220224353539, 2.9469379318589])

        random = Generator(MT19937(self.seed))
        actual = random.chisquare(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.chisquare, bad_df * 3)

    def __exclude_test_noncentral_chisquare(self):
        df = [1]
        nonc = [2]
        bad_df = [-1]
        bad_nonc = [-2]
        desired = vp.array([0.07710766249436, 5.27829115110304, 0.630732147399])

        random = Generator(MT19937(self.seed))
        nonc_chi = random.noncentral_chisquare
        actual = nonc_chi(df * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df * 3, nonc)
        assert_raises(ValueError, nonc_chi, df * 3, bad_nonc)

        random = Generator(MT19937(self.seed))
        nonc_chi = random.noncentral_chisquare
        actual = nonc_chi(df, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df, nonc * 3)
        assert_raises(ValueError, nonc_chi, df, bad_nonc * 3)

    def __exclude_test_standard_t(self):
        df = [1]
        bad_df = [-1]
        desired = vp.array([-1.39498829447098, -1.23058658835223, 0.17207021065983])

        random = Generator(MT19937(self.seed))
        actual = random.standard_t(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.standard_t, bad_df * 3)

    def __exclude_test_vonmises(self):
        mu = [2]
        kappa = [1]
        bad_kappa = [-1]
        desired = vp.array([2.25935584988528, 2.23326261461399, -2.84152146503326])

        random = Generator(MT19937(self.seed))
        actual = random.vonmises(mu * 3, kappa)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.vonmises, mu * 3, bad_kappa)

        random = Generator(MT19937(self.seed))
        actual = random.vonmises(mu, kappa * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.vonmises, mu, bad_kappa * 3)

    def __exclude_test_pareto(self):
        a = [1]
        bad_a = [-1]
        desired = vp.array([0.95905052946317, 0.2383810889437, 1.04988745750013])

        random = Generator(MT19937(self.seed))
        actual = random.pareto(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.pareto, bad_a * 3)

    def __exclude_test_weibull(self):
        a = [1]
        bad_a = [-1]
        desired = vp.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        random = Generator(MT19937(self.seed))
        actual = random.weibull(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.weibull, bad_a * 3)

    def __exclude_test_power(self):
        a = [1]
        bad_a = [-1]
        desired = vp.array([0.48954864361052, 0.19249412888486, 0.51216834058807])

        random = Generator(MT19937(self.seed))
        actual = random.power(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.power, bad_a * 3)

    def __exclude_test_laplace(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired = vp.array([-1.09698732625119, -0.93470271947368, 0.71592671378202])

        random = Generator(MT19937(self.seed))
        laplace = random.laplace
        actual = laplace(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        laplace = random.laplace
        actual = laplace(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc, bad_scale * 3)

    def __exclude_test_gumbel(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired = vp.array([1.70020068231762, 1.52054354273631, -0.34293267607081])

        random = Generator(MT19937(self.seed))
        gumbel = random.gumbel
        actual = gumbel(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        gumbel = random.gumbel
        actual = gumbel(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    def __exclude_test_logistic(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired = vp.array([-1.607487640433, -1.40925686003678, 1.12887112820397])

        random = Generator(MT19937(self.seed))
        actual = random.logistic(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.logistic, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        actual = random.logistic(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.logistic, loc, bad_scale * 3)
        assert_equal(random.logistic(1.0, 0.0), 1.0)

    def __exclude_test_lognormal(self):
        mean = [0]
        sigma = [1]
        bad_sigma = [-1]
        desired = vp.array([0.67884390500697, 2.21653186290321, 1.01990310084276])

        random = Generator(MT19937(self.seed))
        lognormal = random.lognormal
        actual = lognormal(mean * 3, sigma)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)

        random = Generator(MT19937(self.seed))
        actual = random.lognormal(mean, sigma * 3)
        assert_raises(ValueError, random.lognormal, mean, bad_sigma * 3)

    def __exclude_test_rayleigh(self):
        scale = [1]
        bad_scale = [-1]
        desired = vp.array([0.60439534475066, 0.66120048396359, 1.67873398389499])

        random = Generator(MT19937(self.seed))
        actual = random.rayleigh(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.rayleigh, bad_scale * 3)

    def __exclude_test_wald(self):
        mean = [0.5]
        scale = [1]
        bad_mean = [0]
        bad_scale = [-2]
        desired = vp.array([0.38052407392905, 0.50701641508592, 0.484935249864])

        random = Generator(MT19937(self.seed))
        actual = random.wald(mean * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.wald, bad_mean * 3, scale)
        assert_raises(ValueError, random.wald, mean * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        actual = random.wald(mean, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.wald, bad_mean, scale * 3)
        assert_raises(ValueError, random.wald, mean, bad_scale * 3)

    def __exclude_test_triangular(self):
        left = [1]
        right = [3]
        mode = [2]
        bad_left_one = [3]
        bad_mode_one = [4]
        bad_left_two, bad_mode_two = right * 2
        desired = vp.array([1.57781954604754, 1.62665986867957, 2.30090130831326])

        random = Generator(MT19937(self.seed))
        triangular = random.triangular
        actual = triangular(left * 3, mode, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two,
                      right)

        random = Generator(MT19937(self.seed))
        triangular = random.triangular
        actual = triangular(left, mode * 3, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3,
                      right)

        random = Generator(MT19937(self.seed))
        triangular = random.triangular
        actual = triangular(left, mode, right * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode, right * 3)
        assert_raises(ValueError, triangular, left, bad_mode_one, right * 3)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two,
                      right * 3)

        assert_raises(ValueError, triangular, 10., 0., 20.)
        assert_raises(ValueError, triangular, 10., 25., 20.)
        assert_raises(ValueError, triangular, 10., 10., 10.)

    def __exclude_test_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        desired = vp.array([0, 0, 1])

        random = Generator(MT19937(self.seed))
        binom = random.binomial
        actual = binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n * 3, p)
        assert_raises(ValueError, binom, n * 3, bad_p_one)
        assert_raises(ValueError, binom, n * 3, bad_p_two)

        random = Generator(MT19937(self.seed))
        actual = random.binomial(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n, p * 3)
        assert_raises(ValueError, binom, n, bad_p_one * 3)
        assert_raises(ValueError, binom, n, bad_p_two * 3)

    def __exclude_test_negative_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        desired = vp.array([0, 2, 1], dtype=vp.int64)

        random = Generator(MT19937(self.seed))
        neg_binom = random.negative_binomial
        actual = neg_binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n * 3, p)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)

        random = Generator(MT19937(self.seed))
        neg_binom = random.negative_binomial
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n, p * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)

    def __exclude_test_poisson(self):

        lam = [1]
        bad_lam_one = [-1]
        desired = vp.array([0, 0, 3])

        random = Generator(MT19937(self.seed))
        max_lam = random._poisson_lam_max
        bad_lam_two = [max_lam * 2]
        poisson = random.poisson
        actual = poisson(lam * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, poisson, bad_lam_one * 3)
        assert_raises(ValueError, poisson, bad_lam_two * 3)

    def __exclude_test_zipf(self):
        a = [2]
        bad_a = [0]
        desired = vp.array([1, 8, 1])

        random = Generator(MT19937(self.seed))
        zipf = random.zipf
        actual = zipf(a * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, zipf, bad_a * 3)
        with vp.errstate(invalid='ignore'):
            assert_raises(ValueError, zipf, vp.nan)
            assert_raises(ValueError, zipf, [0, 0, vp.nan])

    def __exclude_test_geometric(self):
        p = [0.5]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        desired = vp.array([1, 1, 3])

        random = Generator(MT19937(self.seed))
        geometric = random.geometric
        actual = geometric(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, geometric, bad_p_one * 3)
        assert_raises(ValueError, geometric, bad_p_two * 3)

    def __exclude_test_hypergeometric(self):
        ngood = [1]
        nbad = [2]
        nsample = [2]
        bad_ngood = [-1]
        bad_nbad = [-2]
        bad_nsample_one = [-1]
        bad_nsample_two = [4]
        desired = vp.array([0, 0, 1])

        random = Generator(MT19937(self.seed))
        actual = random.hypergeometric(ngood * 3, nbad, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, random.hypergeometric, bad_ngood * 3, nbad, nsample)
        assert_raises(ValueError, random.hypergeometric, ngood * 3, bad_nbad, nsample)
        assert_raises(
            ValueError,
            random.hypergeometric,
            ngood * 3,
            nbad,
            bad_nsample_one)
        assert_raises(
            ValueError,
            random.hypergeometric,
            ngood * 3,
            nbad,
            bad_nsample_two)

        random = Generator(MT19937(self.seed))
        actual = random.hypergeometric(ngood, nbad * 3, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, random.hypergeometric, bad_ngood, nbad * 3, nsample)
        assert_raises(ValueError, random.hypergeometric, ngood, bad_nbad * 3, nsample)
        assert_raises(
            ValueError,
            random.hypergeometric,
            ngood,
            nbad * 3,
            bad_nsample_one)
        assert_raises(
            ValueError,
            random.hypergeometric,
            ngood,
            nbad * 3,
            bad_nsample_two)

        random = Generator(MT19937(self.seed))
        hypergeom = random.hypergeometric
        actual = hypergeom(ngood, nbad, nsample * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)

        assert_raises(ValueError, hypergeom, -1, 10, 20)
        assert_raises(ValueError, hypergeom, 10, -1, 20)
        assert_raises(ValueError, hypergeom, 10, 10, -1)
        assert_raises(ValueError, hypergeom, 10, 10, 25)

        # ValueError for arguments that are too big.
        assert_raises(ValueError, hypergeom, 2**30, 10, 20)
        assert_raises(ValueError, hypergeom, 999, 2**31, 50)
        assert_raises(ValueError, hypergeom, 999, [2**29, 2**30], 1000)

    def __exclude_test_logseries(self):
        p = [0.5]
        bad_p_one = [2]
        bad_p_two = [-1]
        desired = vp.array([1, 1, 1])

        random = Generator(MT19937(self.seed))
        logseries = random.logseries
        actual = logseries(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, logseries, bad_p_one * 3)
        assert_raises(ValueError, logseries, bad_p_two * 3)

    def __exclude_test_multinomial(self):
        random = Generator(MT19937(self.seed))
        actual = random.multinomial([5, 20], [1 / 6.] * 6, size=(3, 2))
        desired = vp.array([[[0, 0, 2, 1, 2, 0],
                             [2, 3, 6, 4, 2, 3]],
                            [[1, 0, 1, 0, 2, 1],
                             [7, 2, 2, 1, 4, 4]],
                            [[0, 2, 0, 1, 2, 0],
                             [3, 2, 3, 3, 4, 5]]], dtype=vp.int64)
        assert_array_equal(actual, desired)

        random = Generator(MT19937(self.seed))
        actual = random.multinomial([5, 20], [1 / 6.] * 6)
        desired = vp.array([[0, 0, 2, 1, 2, 0],
                            [2, 3, 6, 4, 2, 3]], dtype=vp.int64)
        assert_array_equal(actual, desired)


class TestThread(unittest.TestCase):
    # make sure each state produces the same sequence even in threads
    def setUp(self):
        self.seeds = range(4)

    def check_function(self, function, sz):
        from threading import Thread

        out1 = vp.empty((len(self.seeds),) + sz)
        out2 = vp.empty((len(self.seeds),) + sz)

        # threaded generation
        t = [Thread(target=function, args=(Generator(MT19937(s)), o))
             for s, o in zip(self.seeds, out1)]
        [x.start() for x in t]
        [x.join() for x in t]

        # the same serial
        for s, o in zip(self.seeds, out2):
            function(Generator(MT19937(s)), o)

        # these platforms change x87 fpu precision mode in threads
        if vp.intp().dtype.itemsize == 4 and sys.platform == "win32":
            assert_array_almost_equal(out1, out2)
        else:
            assert_array_equal(out1, out2)

    def __exclude_test_normal(self):
        def gen_random(state, out):
            out[...] = state.normal(size=10000)

        self.check_function(gen_random, sz=(10000,))

    def __exclude_test_exp(self):
        def gen_random(state, out):
            out[...] = state.exponential(scale=vp.ones((100, 1000)))

        self.check_function(gen_random, sz=(100, 1000))

    def __exclude_test_multinomial(self):
        def gen_random(state, out):
            out[...] = state.multinomial(10, [1 / 6.] * 6, size=10000)

        self.check_function(gen_random, sz=(10000, 6))


class TestSingleEltArrayInput(unittest.TestCase):
    def setUp(self):
        self.argOne = vp.array([2])
        self.argTwo = vp.array([3])
        self.argThree = vp.array([4])
        self.tgtShape = (1,)

    def __exclude_test_one_arg_funcs(self):
        funcs = (random.exponential, random.standard_gamma,
                 random.chisquare, random.standard_t,
                 random.pareto, random.weibull,
                 random.power, random.rayleigh,
                 random.poisson, random.zipf,
                 random.geometric, random.logseries)

        probfuncs = (random.geometric, random.logseries)

        for func in funcs:
            if func in probfuncs:  # p < 1.0
                out = func(vp.array([0.5]))

            else:
                out = func(self.argOne)

            assert_equal(out.shape, self.tgtShape)

    def __exclude_test_two_arg_funcs(self):
        funcs = (random.uniform, random.normal,
                 random.beta, random.gamma,
                 random.f, random.noncentral_chisquare,
                 random.vonmises, random.laplace,
                 random.gumbel, random.logistic,
                 random.lognormal, random.wald,
                 random.binomial, random.negative_binomial)

        probfuncs = (random.binomial, random.negative_binomial)

        for func in funcs:
            if func in probfuncs:  # p <= 1
                argTwo = vp.array([0.5])

            else:
                argTwo = self.argTwo

            out = func(self.argOne, argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, argTwo[0])
            assert_equal(out.shape, self.tgtShape)

    def __exclude_test_integers(self, endpoint):
        itype = [vp.bool_, vp.int8, vp.uint8, vp.int16, vp.uint16,
                 vp.int32, vp.uint32, vp.int64, vp.uint64]
        func = random.integers
        high = vp.array([1])
        low = vp.array([0])

        for dt in itype:
            out = func(low, high, endpoint=endpoint, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

            out = func(low[0], high, endpoint=endpoint, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

            out = func(low, high[0], endpoint=endpoint, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

    def __exclude_test_three_arg_funcs(self):
        funcs = [random.noncentral_f, random.triangular,
                 random.hypergeometric]

        for func in funcs:
            out = func(self.argOne, self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, self.argTwo[0], self.argThree)
            assert_equal(out.shape, self.tgtShape)

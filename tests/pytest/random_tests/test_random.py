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

from __future__ import division, absolute_import, print_function
import warnings

import nlcpy as np
from numpy.testing import (
    assert_, assert_raises, assert_equal, assert_warns,
    assert_no_warnings, assert_array_equal, assert_array_almost_equal,
    suppress_warnings
)
from nlcpy import random
import sys
import numpy


class TestSeed(object):

    def test_scalar(self):
        s = np.random.RandomState(0)
        assert_equal(s.randint(1000), np.array(694))
        s = np.random.RandomState(4294967295)
        assert_equal(s.randint(1000), np.array(318))

    def test_array(self):
        s = np.random.RandomState(range(10))
        assert_equal(s.randint(1000), np.array(762))
        s = np.random.RandomState(np.arange(10))
        assert_equal(s.randint(1000), np.array(762))
        s = np.random.RandomState([0])
        assert_equal(s.randint(1000), np.array(694))
        s = np.random.RandomState([4294967295])
        assert_equal(s.randint(1000), np.array(318))

    def test_invalid_scalar(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(ValueError, np.random.RandomState, -0.5)
        assert_raises(ValueError, np.random.RandomState, -1)

    def test_invalid_array(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(ValueError, np.random.RandomState, [-0.5])
        assert_raises(ValueError, np.random.RandomState, [-1])
        assert_raises(ValueError, np.random.RandomState, [4294967296])
        assert_raises(ValueError, np.random.RandomState, [1, 2, 4294967296])
        assert_raises(ValueError, np.random.RandomState, [1, -2, 4294967296])

    def test_invalid_array_shape(self):
        assert_raises(ValueError, np.random.RandomState,
                      np.array([], dtype=np.int64))
        assert_raises(ValueError, np.random.RandomState, [[1, 2, 3]])
        assert_raises(ValueError, np.random.RandomState, [[1, 2, 3],
                                                          [4, 5, 6]])

    def test_fixed_reproducibility(self):
        np.random.seed(100)
        nx1 = np.random.rand()
        nx2 = np.random.rand()
        np.random.seed(100)
        nx3 = np.random.rand()
        nx4 = np.random.rand()
        assert_equal(nx1, nx3)
        assert_equal(nx2, nx4)

    def test_invalid_set_state(self):
        assert_raises(TypeError, np.random.RandomState.set_state, None)
        assert_raises(TypeError, np.random.RandomState.set_state, ["dummy"])
        assert_raises(TypeError, np.random.set_state, None)
        assert_raises(TypeError, np.random.set_state, ["dummy"])


class TestBinomial(object):
    def test_n_zero(self):
        # Tests the corner case of n == 0 for the binomial distribution.
        # binomial(0, p) should be zero for any p in [0, 1].
        zeros = np.zeros(2, dtype='int')
        for p in [0, .5, 1]:
            assert_(random.binomial(0, p).get().tolist() == 0)
            # assert_array_equal(
            #   random.binomial(zeros, p).get().tolist(),
            #   zeros.get().tolist())
            assert_array_equal(
                random.binomial(
                    0,
                    p,
                    2).get().tolist(),
                zeros.get().tolist())

    def test_p_is_nan(self):
        assert_raises(ValueError, random.binomial, 1, np.nan)


class TestMultinomial(object):
    def __exclude_test_basic(self):
        random.multinomial(100, [0.2, 0.8])

    def __exclude_test_zero_probability(self):
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def __exclude_test_int_negative_interval(self):
        assert_(-5 <= random.randint(-5, -1) < -1)
        x = random.randint(-5, -1, 5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))

    def __exclude_test_size(self):
        p = [0.5, 0.5]
        assert_equal(np.random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        assert_equal(np.random.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        assert_equal(np.random.multinomial(1, p, np.array((2, 2))).shape,
                     (2, 2, 2))

        assert_raises(TypeError, np.random.multinomial, 1, p,
                      float(1))


class TestSetState(object):
    def setup(self):
        self.seed = 1234567890
        self.prng = random.RandomState(self.seed)
        self.state = self.prng.get_state()

    def test_basic(self):
        old = self.prng.tomaxint(16)
        self.prng.set_state(self.state)
        new = self.prng.tomaxint(16)
        assert_(np.all(old == new))

    def test_gaussian_reset(self):
        # Make sure the cached every-other-Gaussian is reset.
        old = self.prng.standard_normal(size=3)
        self.prng.set_state(self.state)
        new = self.prng.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_gaussian_reset_in_media_res(self):
        # When the state is saved with a cached Gaussian, make sure the
        # cached Gaussian is restored.

        self.prng.standard_normal()
        state = self.prng.get_state()
        old = self.prng.standard_normal(size=3)
        self.prng.set_state(state)
        new = self.prng.standard_normal(size=3)
        assert_(np.all(old == new))

    def __exclude_test_backwards_compatibility(self):
        # Make sure we can accept old state tuples that do not have the
        # cached Gaussian value.
        old_state = self.state[:-2]
        x1 = self.prng.standard_normal(size=16)
        self.prng.set_state(old_state)
        x2 = self.prng.standard_normal(size=16)
        self.prng.set_state(self.state)
        x3 = self.prng.standard_normal(size=16)
        assert_(np.all(x1 == x2))
        assert_(np.all(x1 == x3))

    def __exclude_test_negative_binomial(self):
        # Ensure that the negative binomial results take floating point
        # arguments without truncation.
        self.prng.negative_binomial(0.5, 0.5)


class TestRandint(object):

    rfunc = np.random.randint

    # valid integer/boolean types
    itype = [np.bool_, np.int8, np.uint8, np.int16, np.uint16,
             np.int32, np.uint32, np.int64, np.uint64]

    def __exclude_test_unsupported_type(self):
        assert_raises(TypeError, self.rfunc, 1, dtype=float)

    def __exclude_test_bounds_checking(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, dtype=dt)

    def __exclude_test_rng_zero_and_extremes(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1

            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

    def __exclude_test_full_range(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1

            try:
                self.rfunc(lbnd, ubnd, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def __exclude_test_in_bounds_fuzz(self):
        # Don't use fixed seed
        np.random.seed()

        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = self.rfunc(2, ubnd, size=2**16, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)

        vals = self.rfunc(0, 2, size=2**16, dtype=np.bool_)

        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    def __exclude_test_repeatability(self):
        import hashlib
        # We use a md5 hash of generated sequences of 1000 samples
        # in the range [0, 6) for all but bool, where the range
        # is [0, 2). Hashes are for little endian numbers.
        tgt = {'bool': '7dd3170d7aa461d201a65f8bcf3944b0',
               'int16': '1b7741b80964bb190c50d541dca1cac1',
               'int32': '4dc9fcc2b395577ebb51793e58ed1a05',
               'int64': '17db902806f448331b5a758d7d2ee672',
               'int8': '27dd30c4e08a797063dffac2490b0be6',
               'uint16': '1b7741b80964bb190c50d541dca1cac1',
               'uint32': '4dc9fcc2b395577ebb51793e58ed1a05',
               'uint64': '17db902806f448331b5a758d7d2ee672',
               'uint8': '27dd30c4e08a797063dffac2490b0be6'}

        for dt in self.itype[1:]:
            np.random.seed(1234)

            # view as little endian for hash
            if sys.byteorder == 'little':
                val = self.rfunc(0, 6, size=1000, dtype=dt)
            else:
                val = self.rfunc(0, 6, size=1000, dtype=dt).byteswap()

            res = hashlib.md5(val.view(np.int8)).hexdigest()
            assert_(tgt[np.dtype(dt).name] == res)

        # bools do not depend on endianness
        np.random.seed(1234)
        val = self.rfunc(0, 2, size=1000, dtype=bool).view(np.int8)
        res = hashlib.md5(val).hexdigest()
        assert_(tgt[np.dtype(bool).name] == res)

    def __exclude_test_int64_uint64_corner_case(self):
        # When stored in Numpy arrays, `lbnd` is casted
        # as np.int64, and `ubnd` is casted as np.uint64.
        # Checking whether `lbnd` >= `ubnd` used to be
        # done solely via direct comparison, which is incorrect
        # because when Numpy tries to compare both numbers,
        # it casts both to np.float64 because there is
        # no integer superset of np.int64 and np.uint64. However,
        # `ubnd` is too large to be represented in np.float64,
        # causing it be round down to np.iinfo(np.int64).max,
        # leading to a ValueError because `lbnd` now equals
        # the new `ubnd`.

        dt = np.int64
        tgt = np.iinfo(np.int64).max
        lbnd = np.int64(np.iinfo(np.int64).max)
        ubnd = np.uint64(np.iinfo(np.int64).max + 1)

        # None of these function calls should
        # generate a ValueError now.
        actual = np.random.randint(lbnd, ubnd, dtype=dt)
        assert_equal(actual.get().tolist(), tgt.get().tolist())

    def __exclude_test_respect_dtype_singleton(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1

            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            assert_equal(sample.dtype, np.dtype(dt))

        for dt in (bool, int, np.compat.long):
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1

            # Ensure that we get Python data types
            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            assert_(not hasattr(sample, 'dtype'))
            assert_equal(type(sample), dt)


class TestRandomDist(object):
    # Make sure the random distribution returns the correct value for a
    # given seed

    def setup(self):
        self.seed = 1234567890

    def test_rand(self):
        np.random.seed(1234567890)
        actual = np.random.rand(3, 2)
        desired = np.array([[0.972810894716531, 0.152507473248988],
                            [0.744906395673752, 0.788559705018997],
                            [0.612674489850178, 0.044743933016434]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def test_randn(self):
        np.random.seed(1234567890)
        actual = np.random.randn(3, 2)
        desired = np.array([[1.923811538759556, -1.025738320166633],
                            [0.658546218189285, 0.801433738216246],
                            [0.286296523876628, -1.698105424133927]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def test_randint(self):
        np.random.seed(1234567890)
        actual = np.random.randint(-99, 99, size=(3, 2))
        desired = np.array([[-71, -27],
                            [40, 65],
                            [-63, -64]])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def test_randint_bound(self):
        np.random.seed(1234567890)
        actual = np.random.randint(2, 4, 10)
        desired = np.array([2, 2, 3, 2, 2, 3, 3, 3, 3, 3])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def test_randint_bound_negative(self):
        np.random.seed(1234567890)
        actual = np.random.randint(-4, -2, 10)
        desired = np.array([-4, -4, -3, -4, -4, -3, -3, -3, -3, -3])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def test_random_integers(self):
        np.random.seed(1234567890)
        with suppress_warnings():
            actual = np.random.random_integers(-99, 99, size=(3, 2))
        desired = np.array([[19, 19],
                            [11, 23],
                            [-51, 33]])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def __exclude_test_random_integers_max_int(self):
        # Tests whether random_integers can generate the
        # maximum allowed Python int that can be converted
        # into a C long. Previous implementations of this
        # method have thrown an OverflowError when attempting
        # to generate this integer.
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = np.random.random_integers(numpy.iinfo('l').max,
                                               numpy.iinfo('l').max)
            assert_(len(w) == 1)

        desired = np.iinfo('l').max
        assert_equal(actual.get().tolist(), desired.get().tolist())

    def __exclude_test_random_integers_deprecated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            # DeprecationWarning raised with high == None
            assert_raises(DeprecationWarning,
                          np.random.random_integers,
                          np.iinfo('l').max)

            # DeprecationWarning raised with high != None
            assert_raises(DeprecationWarning,
                          np.random.random_integers,
                          np.iinfo('l').max, np.iinfo('l').max)

    def test_random_integers_bound(self):
        np.random.seed(1234567890)
        actual = np.random.random_integers(2, 4, 10)
        desired = np.array([3, 2, 3, 4, 2, 4, 3, 3, 4, 3])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def test_random_integers_bound_negative(self):
        np.random.seed(1234567890)
        actual = np.random.random_integers(-4, -2, 10)
        desired = np.array([-3, -4, -3, -2, -4, -2, -3, -3, -2, -3])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def test_random(self):
        np.random.seed(1234567890)
        actual = np.random.random((3, 2))
        desired = np.array([[0.972810894716531, 0.152507473248988],
                            [0.744906395673752, 0.788559705018997],
                            [0.612674489850178, 0.044743933016434]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def __exclude_test_choice_uniform_replace(self):
        np.random.seed(1234567890)
        actual = np.random.choice(4, 4)
        desired = np.array([2, 3, 2, 3])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def __exclude_test_choice_nonuniform_replace(self):
        np.random.seed(1234567890)
        actual = np.random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        desired = np.array([1, 1, 2, 2])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def __exclude_test_choice_uniform_noreplace(self):
        np.random.seed(1234567890)
        actual = np.random.choice(4, 3, replace=False)
        desired = np.array([0, 1, 3])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def __exclude_test_choice_nonuniform_noreplace(self):
        np.random.seed(1234567890)
        actual = np.random.choice(4, 3, replace=False,
                                  p=[0.1, 0.3, 0.5, 0.1])
        desired = np.array([2, 3, 1])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def __exclude_test_choice_noninteger(self):
        np.random.seed(1234567890)
        actual = np.random.choice(['a', 'b', 'c', 'd'], 4)
        desired = np.array(['c', 'd', 'c', 'd'])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def __exclude_test_choice_exceptions(self):
        sample = np.random.choice
        assert_raises(ValueError, sample, -1, 3)
        assert_raises(ValueError, sample, 3., 3)
        assert_raises(ValueError, sample, [[1, 2], [3, 4]], 3)
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
        assert_(np.isscalar(np.random.choice(2, replace=True)))
        assert_(np.isscalar(np.random.choice(2, replace=False)))
        assert_(np.isscalar(np.random.choice(2, replace=True, p=p)))
        assert_(np.isscalar(np.random.choice(2, replace=False, p=p)))
        assert_(np.isscalar(np.random.choice([1, 2], replace=True)))
        assert_(np.random.choice([None], replace=True) is None)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(np.random.choice(arr, replace=True) is a)

        # Check 0-d array
        s = tuple()
        assert_(not np.isscalar(np.random.choice(2, s, replace=True)))
        assert_(not np.isscalar(np.random.choice(2, s, replace=False)))
        assert_(not np.isscalar(np.random.choice(2, s, replace=True, p=p)))
        assert_(not np.isscalar(np.random.choice(2, s, replace=False, p=p)))
        assert_(not np.isscalar(np.random.choice([1, 2], s, replace=True)))
        assert_(np.random.choice([None], s, replace=True).ndim == 0)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(np.random.choice(arr, s, replace=True).item() is a)

        # Check multi dimensional array
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(np.random.choice(6, s, replace=True).shape, s)
        assert_equal(np.random.choice(6, s, replace=False).shape, s)
        assert_equal(np.random.choice(6, s, replace=True, p=p).shape, s)
        assert_equal(np.random.choice(6, s, replace=False, p=p).shape, s)
        assert_equal(np.random.choice(np.arange(6), s, replace=True).shape, s)

        # Check zero-size
        assert_equal(np.random.randint(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(np.random.randint(0, -10, size=0).shape, (0,))
        assert_equal(np.random.randint(10, 10, size=0).shape, (0,))
        assert_equal(np.random.choice(0, size=0).shape, (0,))
        assert_equal(np.random.choice([], size=(0,)).shape, (0,))
        assert_equal(np.random.choice(['a', 'b'], size=(3, 0, 4)).shape,
                     (3, 0, 4))
        assert_raises(ValueError, np.random.choice, [], 10)

    def __exclude_test_choice_nan_probabilities(self):
        a = np.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, np.random.choice, a, p=p)

    def test_bytes(self):
        np.random.seed(1234567890)
        actual = np.random.bytes(10)
        desired = b'R\x1e4\x0f\x80\x82!\x04\xa1\x1f'
        assert_equal(actual, desired)

    def test_shuffle(self):
        # Test lists, arrays (of various dtypes), and multidimensional versions
        # of both, c-contiguous or not:
        for conv in [
            lambda x: np.array([]),
            #             lambda x: x,
            #             lambda x: np.asarray(x).astype(np.int8),
            lambda x: np.asarray(x).astype(np.float32),
            #             lambda x: np.asarray(x).astype(np.complex64),
            #             lambda x: np.asarray(x).astype(object),
            #             lambda x: [(i, i) for i in x],
            lambda x: np.asarray([[i, i] for i in x]),
            #             lambda x: np.vstack([x, x]).T,
            #             lambda x: (np.asarray([(i, i) for i in x],
            #                                   [("a", int), ("b", int)])
            #                        .view(np.recarray)),
            #             lambda x: np.asarray([(i, i) for i in x],
            #                                  [("a", object), ("b", np.int32)])
        ]:
            np.random.seed(1234567890)
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            np.random.shuffle(alist)
            actual = alist
            desired = conv([0, 1, 9, 6, 2, 4, 5, 8, 7, 3])
            assert_array_equal(actual, desired)

    def test_shuffle_1d(self):
        np.random.seed(1234567890)
        alist = np.arange(10)
        np.random.shuffle(alist)
        desired1 = np.array([9, 0, 8, 5, 1, 3, 4, 7, 6, 2])
        assert_array_equal(alist, desired1)

        np.random.shuffle(alist)
        desired2 = np.array([6, 3, 9, 8, 4, 7, 0, 1, 5, 2])
        assert_array_equal(alist, desired2)

    def test_shuffle_view(self):
        np.random.seed(1234567890)
        alist = np.arange(20)
        view1 = alist[:10]
        view2 = alist[10:20]

        np.random.shuffle(view1)
        desired1_view1 = np.array([9, 0, 8, 5, 1, 3, 4, 7, 6, 2])
        desired1_view2 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        assert_array_equal(view1, desired1_view1)
        assert_array_equal(view2, desired1_view2)

        desired1 = np.array([9, 0, 8, 5, 1, 3, 4, 7, 6, 2,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        assert_array_equal(alist, desired1)

        np.random.shuffle(view2)
        desired2_view2 = np.array([18, 15, 10, 12, 16, 17, 11, 14, 13, 19])
        assert_array_equal(view1, desired1_view1)
        assert_array_equal(view2, desired2_view2)

        desired2 = np.array([9, 0, 8, 5, 1, 3, 4, 7, 6, 2,
                             18, 15, 10, 12, 16, 17, 11, 14, 13, 19])
        assert_array_equal(alist, desired2)

    def test_shuffle_nd(self):
        np.random.seed(1234567890)
        alist = np.arange(24).reshape(4, 2, 3)
        np.random.shuffle(alist)
        desired1 = np.array([[[0, 1, 2],
                              [3, 4, 5]],

                             [[6, 7, 8],
                              [9, 10, 11]],

                             [[18, 19, 20],
                              [21, 22, 23]],

                             [[12, 13, 14],
                              [15, 16, 17]]])
        assert_array_equal(alist, desired1)

        np.random.shuffle(alist)
        desired2 = np.array([[[12, 13, 14],
                              [15, 16, 17]],

                             [[0, 1, 2],
                              [3, 4, 5]],

                             [[6, 7, 8],
                              [9, 10, 11]],

                             [[18, 19, 20],
                              [21, 22, 23]]])
        assert_array_equal(alist, desired2)

        np.random.shuffle(alist[1])
        np.random.shuffle(alist[2])
        desired3 = np.array([[[12, 13, 14],
                              [15, 16, 17]],

                             [[0, 1, 2],
                              [3, 4, 5]],

                             [[9, 10, 11],
                              [6, 7, 8]],

                             [[18, 19, 20],
                              [21, 22, 23]]])
        assert_array_equal(alist, desired3)

        np.random.shuffle(alist[3][0])
        np.random.shuffle(alist[3][1])
        np.random.shuffle(alist[0][0])
        np.random.shuffle(alist[0][1])
        desired4 = np.array([[[14, 12, 13],
                              [16, 17, 15]],

                             [[0, 1, 2],
                              [3, 4, 5]],

                             [[9, 10, 11],
                              [6, 7, 8]],

                             [[18, 19, 20],
                              [23, 22, 21]]])
        assert_array_equal(alist, desired4)

    def test_shuffle_order_f(self):
        np.random.seed(1234567890)
        # flist = np.arange(9).reshape((3,3), order='F')
        flist = np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]], order='F')
        np.random.shuffle(flist)
        np.random.shuffle(flist)
        assert_array_equal(flist, np.array([[2, 5, 8], [1, 4, 7], [0, 3, 6]], order='F'))

    def __exclude_test_shuffle_masked(self):
        a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)
        b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)
        a_orig = a.copy()
        b_orig = b.copy()
        for i in range(50):
            np.random.shuffle(a)
            assert_equal(
                sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
            np.random.shuffle(b)
            assert_equal(
                sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))

    def test_permutation(self):
        random.seed(self.seed)
        alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        actual = random.permutation(alist)
        desired = [0, 1, 9, 6, 2, 4, 5, 8, 7, 3]
        assert_array_equal(actual.get(), desired)

        # random.seed(self.seed)
        # arr_2d = np.atleast_2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).T
        # actual = random.permutation(arr_2d)
        # assert_array_equal(actual.get(), np.atleast_2d(desired).T)

        # ValueError: Unsupported dtype <U4
        # random.seed(self.seed)
        # bad_x_str = "abcd"
        # assert_raises(IndexError, random.permutation, bad_x_str)

        random.seed(self.seed)
        bad_x_float = 1.2
        assert_raises(IndexError, random.permutation, bad_x_float)

        integer_val = 10
        desired = [9, 0, 8, 5, 1, 3, 4, 7, 6, 2]
        random.seed(self.seed)
        actual = random.permutation(integer_val)
        assert_array_equal(actual.get(), desired)

    def test_permutation_1d(self):
        np.random.seed(1234567890)
        alist = np.arange(10)
        plist = np.random.permutation(alist)
        desired1 = np.array([9, 0, 8, 5, 1, 3, 4, 7, 6, 2])
        assert_array_equal(plist, desired1)

        plist2 = np.random.permutation(plist)
        desired2 = np.array([6, 3, 9, 8, 4, 7, 0, 1, 5, 2])
        assert_array_equal(plist2, desired2)

    def test_permutation_view(self):
        np.random.seed(1234567890)
        alist = np.arange(20)
        view1 = alist[:10]
        view2 = alist[10:20]

        pview1 = np.random.permutation(view1)
        desired1_view1 = np.array([9, 0, 8, 5, 1, 3, 4, 7, 6, 2])
        desired1_view2 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        assert_array_equal(pview1, desired1_view1)
        assert_array_equal(view2, desired1_view2)

        desired1 = np.arange(20)
        assert_array_equal(alist, desired1)
        assert_array_equal(view1, np.arange(10))

        pview2 = np.random.permutation(view2)
        desired2_view2 = np.array([18, 15, 10, 12, 16, 17, 11, 14, 13, 19])
        assert_array_equal(pview1, desired1_view1)
        assert_array_equal(pview2, desired2_view2)

        desired2 = np.arange(20)
        assert_array_equal(alist, desired2)

    def test_permutation_nd(self):
        np.random.seed(1234567890)
        alist = np.arange(24).reshape(4, 2, 3)
        plist1 = np.random.permutation(alist)
        desired1 = np.array([[[0, 1, 2],
                              [3, 4, 5]],

                             [[6, 7, 8],
                              [9, 10, 11]],

                             [[18, 19, 20],
                              [21, 22, 23]],

                             [[12, 13, 14],
                              [15, 16, 17]]])
        assert_array_equal(plist1, desired1)

        plist2 = np.random.permutation(plist1)
        desired2 = np.array([[[12, 13, 14],
                              [15, 16, 17]],

                             [[0, 1, 2],
                              [3, 4, 5]],

                             [[6, 7, 8],
                              [9, 10, 11]],

                             [[18, 19, 20],
                              [21, 22, 23]]])
        assert_array_equal(plist2, desired2)

        plist5 = plist2
        plist3 = np.random.permutation(plist2[1])
        plist5[1] = plist3
        plist4 = np.random.permutation(plist2[2])
        plist5[2] = plist4
        desired3 = np.array([[[12, 13, 14],
                              [15, 16, 17]],

                             [[0, 1, 2],
                              [3, 4, 5]],

                             [[9, 10, 11],
                              [6, 7, 8]],

                             [[18, 19, 20],
                              [21, 22, 23]]])
        assert_array_equal(plist5, desired3)

        plist6 = plist5
        plist6[3][0] = np.random.permutation(plist5[3][0])
        plist6[3][1] = np.random.permutation(plist5[3][1])
        plist6[0][0] = np.random.permutation(plist5[0][0])
        plist6[0][1] = np.random.permutation(plist5[0][1])
        desired4 = np.array([[[14, 12, 13],
                              [16, 17, 15]],

                             [[0, 1, 2],
                              [3, 4, 5]],

                             [[9, 10, 11],
                              [6, 7, 8]],

                             [[18, 19, 20],
                              [23, 22, 21]]])
        assert_array_equal(plist6, desired4)
        assert_array_equal(alist, np.arange(24).reshape(4, 2, 3))

    def test_permutation_order_f(self):
        np.random.seed(1234567890)
        # flist = np.arange(9).reshape((3,3), order='F')
        flist = np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]], order='F')
        plist = np.random.permutation(flist)
        pflist = np.random.permutation(plist)
        assert_array_equal(pflist, np.array(
            [[2, 5, 8], [1, 4, 7], [0, 3, 6]], order='F'))

    def __exclude_test_beta(self):
        np.random.seed(1234567890)
        actual = np.random.beta(.1, .9, size=(3, 2))
        desired = np.array(
            [[1.45341850513746058e-02, 5.31297615662868145e-04],
             [1.85366619058432324e-06, 4.19214516800110563e-03],
             [1.58405155108498093e-04, 1.26252891949397652e-04]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def test_binomial(self):
        np.random.seed(1234567890)
        actual = np.random.binomial(100, .456, size=(3, 2))
        desired = np.array([[55, 40],
                            [49, 50],
                            [47, 37]])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def __exclude_test_chisquare(self):
        np.random.seed(1234567890)
        actual = np.random.chisquare(50, size=(3, 2))
        desired = np.array([[63.87858175501090585, 68.68407748911370447],
                            [65.77116116901505904, 47.09686762438974483],
                            [72.3828403199695174, 74.18408615260374006]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=13)

    def __exclude_test_dirichlet(self):
        np.random.seed(1234567890)
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        actual = np.random.dirichlet(alpha, size=(3, 2))
        desired = np.array([[[0.54539444573611562, 0.45460555426388438],
                             [0.62345816822039413, 0.37654183177960598]],
                            [[0.55206000085785778, 0.44793999914214233],
                             [0.58964023305154301, 0.41035976694845688]],
                            [[0.59266909280647828, 0.40733090719352177],
                             [0.56974431743975207, 0.43025568256024799]]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def __exclude_test_dirichlet_size(self):
        p = np.array([51.72840233779265162, 39.74494232180943953])
        assert_equal(np.random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(np.random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(np.random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))

        assert_raises(TypeError, np.random.dirichlet, p, float(1))

    def __exclude_test_dirichlet_bad_alpha(self):
        alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, np.random.dirichlet, alpha)

    def test_exponential(self):
        np.random.seed(1234567890)
        actual = np.random.exponential(1.1234, size=(3, 2))
        desired = np.array([[4.049788390554855, 0.185892657766817],
                            [1.534704516503505, 1.745553090098096],
                            [1.065533472764141, 0.051424580313624]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def test_exponential_0(self):
        assert_equal(np.random.exponential(scale=0), np.array(0))
        assert_raises(ValueError, np.random.exponential, scale=-0.1)

    def __exclude_test_f(self):
        np.random.seed(1234567890)
        actual = np.random.f(12, 77, size=(3, 2))
        desired = np.array([[1.21975394418575878, 1.75135759791559775],
                            [1.44803115017146489, 1.22108959480396262],
                            [1.02176975757740629, 1.34431827623300415]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def test_gamma(self):
        np.random.seed(1234567890)
        actual = np.random.gamma(5, 3, size=(3, 2))
        desired = np.array([[30.340371387670423, 8.403659413123542],
                            [18.697978217515761, 19.834775281407367],
                            [15.939356649001832, 5.726601821724220]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=14)

    def test_gamma_0(self):
        actual = np.random.gamma(shape=0, scale=0)
        assert_equal(actual.get().tolist(), 0)
        assert_raises(ValueError, np.random.gamma, shape=-0.1, scale=-0.1)

    def test_geometric(self):
        np.random.seed(1234567890)
        actual = np.random.geometric(.123456789, size=(3, 2))
        desired = np.array([[27, 1],
                            [10, 11],
                            [7, 0]])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def test_gumbel(self):
        np.random.seed(1234567890)
        actual = np.random.gumbel(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[7.058918817752728, -1.386576514175936],
                            [2.321518093042787, 2.751321460795623],
                            [1.303563437951490, -2.390642929789188]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def test_gumbel_0(self):
        assert_equal(np.random.gumbel(scale=0), np.array(0))
        assert_raises(ValueError, np.random.gumbel, scale=-0.1)

    def __exclude_test_hypergeometric(self):
        np.random.seed(1234567890)
        actual = np.random.hypergeometric(10, 5, 14, size=(3, 2))
        desired = np.array([[10, 10],
                            [10, 10],
                            [9, 9]])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

        # Test nbad = 0
        actual = np.random.hypergeometric(5, 0, 3, size=4)
        desired = np.array([3, 3, 3, 3])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

        actual = np.random.hypergeometric(15, 0, 12, size=4)
        desired = np.array([12, 12, 12, 12])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

        # Test ngood = 0
        actual = np.random.hypergeometric(0, 5, 3, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

        actual = np.random.hypergeometric(0, 15, 12, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def __exclude_test_laplace(self):
        np.random.seed(1234567890)
        actual = np.random.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.66599721112760157, 0.52829452552221945],
                            [3.12791959514407125, 3.18202813572992005],
                            [-0.05391065675859356, 1.74901336242837324]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def __exclude_test_laplace_0(self):
        assert_equal(np.random.laplace(scale=0), np.array(0))
        assert_raises(ValueError, np.random.laplace, scale=-0.1)

    def test_logistic(self):
        np.random.seed(1234567890)
        actual = np.random.logistic(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[7.278203505945411, -3.306680053929311],
                            [2.266712816006061, 2.755987704763023],
                            [1.040593453487900, -5.998590365057210]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def test_lognormal(self):
        np.random.seed(1234567890)
        actual = np.random.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
        desired = np.array([[53.041777964624508, 0.145435898241472],
                            [4.223014566954349, 5.619958420734945],
                            [2.005813745957540, 0.037901899125987]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=13)

    def test_lognormal_0(self):
        assert_equal(np.random.lognormal(sigma=0), np.array(1))
        assert_raises(ValueError, np.random.lognormal, sigma=-0.1)

    def __exclude_test_logseries(self):
        np.random.seed(1234567890)
        actual = np.random.logseries(p=.923456789, size=(3, 2))
        desired = np.array([[2, 2],
                            [6, 17],
                            [3, 6]])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def __exclude_test_multinomial(self):
        np.random.seed(1234567890)
        actual = np.random.multinomial(20, [1 / 6.] * 6, size=(3, 2))
        desired = np.array([[[4, 3, 5, 4, 2, 2],
                             [5, 2, 8, 2, 2, 1]],
                            [[3, 4, 3, 6, 0, 4],
                             [2, 1, 4, 3, 6, 4]],
                            [[4, 4, 2, 5, 2, 3],
                             [4, 3, 4, 2, 3, 4]]])
        assert_array_equal(actual=actual, desired=desired)

    def __exclude_test_multivariate_normal(self):
        np.random.seed(1234567890)
        mean = (.123456789, 10)
        cov = [[1, 0], [0, 1]]
        size = (3, 2)
        actual = np.random.multivariate_normal(mean, cov, size)
        desired = np.array([[[1.463620246718631, 11.73759122771936],
                             [1.622445133300628, 9.771356667546383]],
                            [[2.154490787682787, 12.170324946056553],
                             [1.719909438201865, 9.230548443648306]],
                            [[0.689515026297799, 9.880729819607714],
                             [-0.023054015651998, 9.201096623542879]]])

        assert_array_almost_equal(actual=actual, desired=desired, decimal=15)

        # Check for default size, was raising deprecation warning
        actual = np.random.multivariate_normal(mean, cov)
        desired = np.array([0.895289569463708, 9.17180864067987])
        assert_array_almost_equal(actual=actual, desired=desired, decimal=15)

        # Check that non positive-semidefinite covariance warns with
        # RuntimeWarning
        mean = [0, 0]
        cov = [[1, 2], [2, 1]]
        assert_warns(RuntimeWarning, np.random.multivariate_normal, mean, cov)

        # and that it doesn't warn with RuntimeWarning check_valid='ignore'
        assert_no_warnings(np.random.multivariate_normal, mean, cov,
                           check_valid='ignore')

        # and that it raises with RuntimeWarning check_valid='raises'
        assert_raises(ValueError, np.random.multivariate_normal, mean, cov,
                      check_valid='raise')

        cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with suppress_warnings() as sup:
            np.random.multivariate_normal(mean, cov)
            w = sup.record(RuntimeWarning)
            assert len(w) == 0

    def __exclude_test_negative_binomial(self):
        np.random.seed(1234567890)
        actual = np.random.negative_binomial(n=100, p=.12345, size=(3, 2))
        desired = np.array([[848, 841],
                            [892, 611],
                            [779, 647]])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def __exclude_test_noncentral_chisquare(self):
        np.random.seed(1234567890)
        actual = np.random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        desired = np.array([[23.91905354498517511, 13.35324692733826346],
                            [31.22452661329736401, 16.60047399466177254],
                            [5.03461598262724586, 17.94973089023519464]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=14)

        actual = np.random.noncentral_chisquare(df=.5, nonc=.2, size=(3, 2))
        desired = np.array([[1.47145377828516666, 0.15052899268012659],
                            [0.00943803056963588, 1.02647251615666169],
                            [0.332334982684171, 0.15451287602753125]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=14)

        np.random.seed(1234567890)
        actual = np.random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        desired = np.array([[9.597154162763948, 11.725484450296079],
                            [10.413711048138335, 3.694475922923986],
                            [13.484222138963087, 14.377255424602957]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=14)

    def __exclude_test_noncentral_f(self):
        np.random.seed(1234567890)
        actual = np.random.noncentral_f(dfnum=5, dfden=2, nonc=1,
                                        size=(3, 2))
        desired = np.array([[1.40598099674926669, 0.34207973179285761],
                            [3.57715069265772545, 7.92632662577829805],
                            [0.43741599463544162, 1.1774208752428319]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=14)

    def test_normal(self):
        np.random.seed(1234567890)
        actual = np.random.normal(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[3.971079866519112, -1.928019851333267],
                            [1.440549225378571, 1.726324265432492],
                            [0.696049836753256, -3.272754059267854]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def test_normal_0(self):
        assert_equal(np.random.normal(scale=0), np.array(0))
        assert_raises(ValueError, np.random.normal, scale=-0.1)

    def __exclude_test_pareto(self):
        np.random.seed(1234567890)
        actual = np.random.pareto(a=.123456789, size=(3, 2))
        desired = np.array(
            [[2.46852460439034849e+03, 1.41286880810518346e+03],
             [5.28287797029485181e+07, 6.57720981047328785e+07],
             [1.40840323350391515e+02, 1.98390255135251704e+05]])
        # For some reason on 32-bit x86 Ubuntu 12.10 the [1, 0] entry in this
        # matrix differs by 24 nulps. Discussion:
        #   https://mail.python.org/pipermail/numpy-discussion/2012-September/063801.html
        # Consensus is that this is probably some gcc quirk that affects
        # rounding but not in any important way, so we just use a looser
        # tolerance on this test:
        np.testing.assert_array_almost_equal_nulp(
            actual.get().tolist(), desired.get().tolist(), nulp=30)

    def test_poisson(self):
        np.random.seed(1234567890)
        actual = np.random.poisson(lam=.123456789, size=(3, 2))
#        desired = np.array([[0, 0],
#                            [1, 0],
#                            [0, 0]])
        desired = np.array([[1, 0],
                            [0, 0],
                            [0, 0]])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())

    def test_poisson_exceptions(self):
        lambig = numpy.iinfo('l').max
        lamneg = -1
        assert_raises(ValueError, np.random.poisson, lamneg)
        # assert_raises(ValueError, np.random.poisson, [lamneg]*10)
        assert_raises(NotImplementedError, np.random.poisson, [lamneg] * 10)
        assert_raises(ValueError, np.random.poisson, lambig)
        # assert_raises(ValueError, np.random.poisson, [lambig]*10)
        assert_raises(NotImplementedError, np.random.poisson, [lambig] * 10)

    def __exclude_test_power(self):
        np.random.seed(1234567890)
        actual = np.random.power(a=.123456789, size=(3, 2))
        desired = np.array([[0.02048932883240791, 0.01424192241128213],
                            [0.38446073748535298, 0.39499689943484395],
                            [0.00177699707563439, 0.13115505880863756]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def __exclude_test_rayleigh(self):
        np.random.seed(1234567890)
        actual = np.random.rayleigh(scale=10, size=(3, 2))
        desired = np.array([[13.8882496494248393, 13.383318339044731],
                            [20.95413364294492098, 21.08285015800712614],
                            [11.06066537006854311, 17.35468505778271009]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=14)

    def __exclude_test_rayleigh_0(self):
        assert_equal(np.random.rayleigh(scale=0), np.array(0))
        assert_raises(ValueError, np.random.rayleigh, scale=-0.1)

    def test_standard_cauchy(self):
        np.random.seed(1234567890)
        actual = np.random.standard_cauchy(size=(3, 2))
        desired = np.array([[-0.085625438121841, 0.519488219321481],
                            [-1.032527339745541, -0.782952395171188],
                            [-2.706049524270479, 0.141500416797786]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def test_standard_exponential(self):
        np.random.seed(1234567890)
        actual = np.random.standard_exponential(size=(3, 2))
        desired = np.array([[3.604938926967113, 0.165473257759317],
                            [1.366124725390337, 1.553812613582069],
                            [0.948489827990156, 0.045775841475542]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def test_standard_gamma(self):
        np.random.seed(1234567890)
        actual = np.random.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[7.114495587651424, 1.341383083289342],
                            [3.887082135953622, 4.190785285961167],
                            [3.164469242091366, 0.781720912458448]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=14)

    def test_standard_gamma_0(self):
        assert_equal(np.random.standard_gamma(shape=0), np.array(0))
        assert_raises(ValueError, np.random.standard_gamma, shape=-0.1)

    def test_standard_normal(self):
        np.random.seed(1234567890)
        actual = np.random.standard_normal(size=(3, 2))
        desired = np.array([[1.923811538759556, -1.025738320166633],
                            [0.658546218189285, 0.801433738216246],
                            [0.286296523876628, -1.698105424133927]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def __exclude_test_standard_t(self):
        np.random.seed(1234567890)
        actual = np.random.standard_t(df=10, size=(3, 2))
        desired = np.array([[0.97140611862659965, -0.08830486548450577],
                            [1.36311143689505321, -0.55317463909867071],
                            [-0.18473749069684214, 0.61181537341755321]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def __exclude_test_triangular(self):
        np.random.seed(1234567890)
        actual = np.random.triangular(left=5.12, mode=10.23, right=20.34,
                                      size=(3, 2))
        desired = np.array([[12.68117178949215784, 12.4129206149193152],
                            [16.20131377335158263, 16.25692138747600524],
                            [11.20400690911820263, 14.4978144835829923]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=14)

    def test_uniform(self):
        np.random.seed(1234567890)
        actual = np.random.uniform(low=1.23, high=10.54, size=(3, 2))
        desired = np.array([[10.286869429810903, 2.649844575948082],
                            [8.165078543722629, 8.571490853726862],
                            [6.933999500505159, 1.646566016383003]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def __exclude_test_uniform_range_bounds(self):
        fmin = numpy.finfo('float').min
        fmax = numpy.finfo('float').max

        func = np.random.uniform
        assert_raises(OverflowError, func, -np.inf, 0)
        assert_raises(OverflowError, func, 0, np.inf)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, [-np.inf], [0])
        assert_raises(OverflowError, func, [0], [np.inf])

        # (fmax / 1e17) - fmin is within range, so this should not throw
        # account for i386 extended precision DBL_MAX / 1e17 + DBL_MAX >
        # DBL_MAX by increasing fmin a bit
        np.random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e17)

    def __exclude_test_scalar_exception_propagation(self):
        # Tests that exceptions are correctly propagated in distributions
        # when called with objects that throw exceptions when converted to
        # scalars.

        class ThrowingFloat(np.ndarray):
            def __float__(self):
                raise TypeError

        throwing_float = np.array(1.0).view(ThrowingFloat)
        assert_raises(TypeError, np.random.uniform, throwing_float,
                      throwing_float)

        class ThrowingInteger(np.ndarray):
            def __int__(self):
                raise TypeError

            __index__ = __int__

        throwing_int = np.array(1).view(ThrowingInteger)
        assert_raises(TypeError, np.random.hypergeometric, throwing_int, 1, 1)

    def __exclude_test_vonmises(self):
        np.random.seed(1234567890)
        actual = np.random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        desired = np.array([[2.28567572673902042, 2.89163838442285037],
                            [0.38198375564286025, 2.57638023113890746],
                            [1.19153771588353052, 1.83509849681825354]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def __exclude_test_vonmises_small(self):
        # check infinite loop.
        np.random.seed(1234567890)
        r = np.random.vonmises(mu=0., kappa=1.1e-8, size=10**6)
        np.testing.assert_(np.isfinite(r).all())

    def __exclude_test_wald(self):
        np.random.seed(1234567890)
        actual = np.random.wald(mean=1.23, scale=1.54, size=(3, 2))
        desired = np.array([[3.82935265715889983, 5.13125249184285526],
                            [0.35045403618358717, 1.50832396872003538],
                            [0.24124319895843183, 0.22031101461955038]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_weibull(self):
        np.random.seed(1234567890)
        actual = np.random.weibull(a=1.23, size=(3, 2))
        desired = np.array([[2.836367691007171, 0.231643097460156],
                            [1.288708691186699, 1.430897920624153],
                            [0.957915896173428, 0.081486647850046]])
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=15)

    def test_weibull_0(self):
        np.random.seed(1234567890)
        actual = np.random.weibull(a=0, size=12)
        desired = np.zeros(12)
        assert_equal(actual.get().tolist(), desired.get().tolist())
        assert_raises(ValueError, np.random.weibull, a=-0.1)

    def __exclude_test_zipf(self):
        np.random.seed(1234567890)
        actual = np.random.zipf(a=1.23, size=(3, 2))
        desired = np.array([[66, 29],
                            [1, 1],
                            [3, 13]])
        assert_array_equal(actual.get().tolist(), desired.get().tolist())


class TestBroadcast(object):
    # tests that functions that broadcast behave
    # correctly when presented with non-scalar arguments
    def setup(self):
        self.seed = 123456789

    def setSeed(self):
        np.random.seed(1234567890)

    def __exclude_test_uniform(self):
        low = [0]
        high = [1]
        uniform = np.random.uniform
        desired = np.array([0.53283302478975902,
                            0.53413660089041659,
                            0.50955303552646702])

        self.setSeed()
        actual = uniform(low * 3, high)
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=14)

        self.setSeed()
        actual = uniform(low, high * 3)
        assert_array_almost_equal(actual=actual, desired=desired, decimal=14)

    def __exclude_test_normal(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        normal = np.random.normal
        desired = np.array([2.2129019979039612,
                            2.1283977976520019,
                            1.8417114045748335])

        self.setSeed()
        actual = normal(loc * 3, scale)
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=14)
        assert_raises(ValueError, normal, loc * 3, bad_scale)

        self.setSeed()
        actual = normal(loc, scale * 3)
        assert_array_almost_equal(
            actual.get().tolist(),
            desired.get().tolist(),
            decimal=14)
        assert_raises(ValueError, normal, loc, bad_scale * 3)

    def __exclude_test_beta(self):
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        beta = np.random.beta
        desired = np.array([0.19843558305989056,
                            0.075230336409423643,
                            0.24976865978980844])

        self.setSeed()
        actual = beta(a * 3, b)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a * 3, b)
        assert_raises(ValueError, beta, a * 3, bad_b)

        self.setSeed()
        actual = beta(a, b * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a, b * 3)
        assert_raises(ValueError, beta, a, bad_b * 3)

    def __exclude_test_exponential(self):
        scale = [1]
        bad_scale = [-1]
        exponential = np.random.exponential
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        self.setSeed()
        actual = exponential(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, exponential, bad_scale * 3)

    def __exclude_test_standard_gamma(self):
        shape = [1]
        bad_shape = [-1]
        std_gamma = np.random.standard_gamma
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        self.setSeed()
        actual = std_gamma(shape * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, std_gamma, bad_shape * 3)

    def __exclude_test_gamma(self):
        shape = [1]
        scale = [2]
        bad_shape = [-1]
        bad_scale = [-2]
        gamma = np.random.gamma
        desired = np.array([1.5221370731769048,
                            1.5277256455738331,
                            1.4248762625178359])

        self.setSeed()
        actual = gamma(shape * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape * 3, scale)
        assert_raises(ValueError, gamma, shape * 3, bad_scale)

        self.setSeed()
        actual = gamma(shape, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape, scale * 3)
        assert_raises(ValueError, gamma, shape, bad_scale * 3)

    def __exclude_test_f(self):
        dfnum = [1]
        dfden = [2]
        bad_dfnum = [-1]
        bad_dfden = [-2]
        f = np.random.f
        desired = np.array([0.80038951638264799,
                            0.86768719635363512,
                            2.7251095168386801])

        self.setSeed()
        actual = f(dfnum * 3, dfden)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)

        self.setSeed()
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
        nonc_f = np.random.noncentral_f
        desired = np.array([9.1393943263705211,
                            13.025456344595602,
                            8.8018098359100545])

        self.setSeed()
        actual = nonc_f(dfnum * 3, dfden, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)

        self.setSeed()
        actual = nonc_f(dfnum, dfden * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)

        self.setSeed()
        actual = nonc_f(dfnum, dfden, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)

    def __exclude_test_noncentral_f_small_df(self):
        self.setSeed()
        desired = np.array([6.869638627492048, 0.785880199263955])
        actual = np.random.noncentral_f(0.9, 0.9, 2, size=2)
        assert_array_almost_equal(actual, desired, decimal=14)

    def __exclude_test_chisquare(self):
        df = [1]
        bad_df = [-1]
        chisquare = np.random.chisquare
        desired = np.array([0.57022801133088286,
                            0.51947702108840776,
                            0.1320969254923558])

        self.setSeed()
        actual = chisquare(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, chisquare, bad_df * 3)

    def __exclude_test_noncentral_chisquare(self):
        df = [1]
        nonc = [2]
        bad_df = [-1]
        bad_nonc = [-2]
        nonc_chi = np.random.noncentral_chisquare
        desired = np.array([9.0015599467913763,
                            4.5804135049718742,
                            6.0872302432834564])

        self.setSeed()
        actual = nonc_chi(df * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df * 3, nonc)
        assert_raises(ValueError, nonc_chi, df * 3, bad_nonc)

        self.setSeed()
        actual = nonc_chi(df, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df, nonc * 3)
        assert_raises(ValueError, nonc_chi, df, bad_nonc * 3)

    def __exclude_test_standard_t(self):
        df = [1]
        bad_df = [-1]
        t = np.random.standard_t
        desired = np.array([3.0702872575217643,
                            5.8560725167361607,
                            1.0274791436474273])

        self.setSeed()
        actual = t(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, t, bad_df * 3)

    def __exclude_test_vonmises(self):
        mu = [2]
        kappa = [1]
        bad_kappa = [-1]
        vonmises = np.random.vonmises
        desired = np.array([2.9883443664201312,
                            -2.7064099483995943,
                            -1.8672476700665914])

        self.setSeed()
        actual = vonmises(mu * 3, kappa)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu * 3, bad_kappa)

        self.setSeed()
        actual = vonmises(mu, kappa * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu, bad_kappa * 3)

    def __exclude_test_pareto(self):
        a = [1]
        bad_a = [-1]
        pareto = np.random.pareto
        desired = np.array([1.1405622680198362,
                            1.1465519762044529,
                            1.0389564467453547])

        self.setSeed()
        actual = pareto(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, pareto, bad_a * 3)

    def __exclude_test_weibull(self):
        a = [1]
        bad_a = [-1]
        weibull = np.random.weibull
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        self.setSeed()
        actual = weibull(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, weibull, bad_a * 3)

    def __exclude_test_power(self):
        a = [1]
        bad_a = [-1]
        power = np.random.power
        desired = np.array([0.53283302478975902,
                            0.53413660089041659,
                            0.50955303552646702])

        self.setSeed()
        actual = power(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, power, bad_a * 3)

    def __exclude_test_laplace(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        laplace = np.random.laplace
        desired = np.array([0.067921356028507157,
                            0.070715642226971326,
                            0.019290950698972624])

        self.setSeed()
        actual = laplace(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc * 3, bad_scale)

        self.setSeed()
        actual = laplace(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc, bad_scale * 3)

    def __exclude_test_gumbel(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        gumbel = np.random.gumbel
        desired = np.array([0.2730318639556768,
                            0.26936705726291116,
                            0.33906220393037939])

        self.setSeed()
        actual = gumbel(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)

        self.setSeed()
        actual = gumbel(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    def __exclude_test_logistic(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        logistic = np.random.logistic
        desired = np.array([0.13152135837586171,
                            0.13675915696285773,
                            0.038216792802833396])

        self.setSeed()
        actual = logistic(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc * 3, bad_scale)

        self.setSeed()
        actual = logistic(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc, bad_scale * 3)

    def __exclude_test_lognormal(self):
        mean = [0]
        sigma = [1]
        bad_sigma = [-1]
        lognormal = np.random.lognormal
        desired = np.array([9.1422086044848427,
                            8.4013952870126261,
                            6.3073234116578671])

        self.setSeed()
        actual = lognormal(mean * 3, sigma)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)

        self.setSeed()
        actual = lognormal(mean, sigma * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean, bad_sigma * 3)

    def __exclude_test_rayleigh(self):
        scale = [1]
        bad_scale = [-1]
        rayleigh = np.random.rayleigh
        desired = np.array([1.2337491937897689,
                            1.2360119924878694,
                            1.1936818095781789])

        self.setSeed()
        actual = rayleigh(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, rayleigh, bad_scale * 3)

    def __exclude_test_wald(self):
        mean = [0.5]
        scale = [1]
        bad_mean = [0]
        bad_scale = [-2]
        wald = np.random.wald
        desired = np.array([0.11873681120271318,
                            0.12450084820795027,
                            0.9096122728408238])

        self.setSeed()
        actual = wald(mean * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean * 3, scale)
        assert_raises(ValueError, wald, mean * 3, bad_scale)

        self.setSeed()
        actual = wald(mean, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean, scale * 3)
        assert_raises(ValueError, wald, mean, bad_scale * 3)
        assert_raises(ValueError, wald, 0.0, 1)
        assert_raises(ValueError, wald, 0.5, 0.0)

    def __exclude_test_triangular(self):
        left = [1]
        right = [3]
        mode = [2]
        bad_left_one = [3]
        bad_mode_one = [4]
        bad_left_two, bad_mode_two = right * 2
        triangular = np.random.triangular
        desired = np.array([2.03339048710429,
                            2.0347400359389356,
                            2.0095991069536208])

        self.setSeed()
        actual = triangular(left * 3, mode, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two,
                      right)

        self.setSeed()
        actual = triangular(left, mode * 3, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3,
                      right)

        self.setSeed()
        actual = triangular(left, mode, right * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode, right * 3)
        assert_raises(ValueError, triangular, left, bad_mode_one, right * 3)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two,
                      right * 3)

    def __exclude_test_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        binom = np.random.binomial
        desired = np.array([1, 1, 1])

        self.setSeed()
        actual = binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n * 3, p)
        assert_raises(ValueError, binom, n * 3, bad_p_one)
        assert_raises(ValueError, binom, n * 3, bad_p_two)

        self.setSeed()
        actual = binom(n, p * 3)
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
        neg_binom = np.random.negative_binomial
        desired = np.array([1, 0, 1])

        self.setSeed()
        actual = neg_binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n * 3, p)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)

        self.setSeed()
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n, p * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)

    def __exclude_test_poisson(self):
        max_lam = np.random.RandomState()._poisson_lam_max

        lam = [1]
        bad_lam_one = [-1]
        bad_lam_two = [max_lam * 2]
        poisson = np.random.poisson
        desired = np.array([1, 1, 0])

        self.setSeed()
        actual = poisson(lam * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, poisson, bad_lam_one * 3)
        assert_raises(ValueError, poisson, bad_lam_two * 3)

    def __exclude_test_zipf(self):
        a = [2]
        bad_a = [0]
        zipf = np.random.zipf
        desired = np.array([2, 2, 1])

        self.setSeed()
        actual = zipf(a * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, zipf, bad_a * 3)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, zipf, np.nan)
            assert_raises(ValueError, zipf, [0, 0, np.nan])

    def __exclude_test_geometric(self):
        p = [0.5]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        geom = np.random.geometric
        desired = np.array([2, 2, 2])

        self.setSeed()
        actual = geom(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, geom, bad_p_one * 3)
        assert_raises(ValueError, geom, bad_p_two * 3)

    def __exclude_test_hypergeometric(self):
        ngood = [1]
        nbad = [2]
        nsample = [2]
        bad_ngood = [-1]
        bad_nbad = [-2]
        bad_nsample_one = [0]
        bad_nsample_two = [4]
        hypergeom = np.random.hypergeometric
        desired = np.array([1, 1, 1])

        self.setSeed()
        actual = hypergeom(ngood * 3, nbad, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood * 3, nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, bad_nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_two)

        self.setSeed()
        actual = hypergeom(ngood, nbad * 3, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_two)

        self.setSeed()
        actual = hypergeom(ngood, nbad, nsample * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)

    def __exclude_test_logseries(self):
        p = [0.5]
        bad_p_one = [2]
        bad_p_two = [-1]
        logseries = np.random.logseries
        desired = np.array([1, 1, 1])

        self.setSeed()
        actual = logseries(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, logseries, bad_p_one * 3)
        assert_raises(ValueError, logseries, bad_p_two * 3)


class _TestThread(object):
    # make sure each state produces the same sequence even in threads
    def setup(self):
        self.seeds = range(4)

    def check_function(self, function, sz):
        from threading import Thread

        out1 = np.empty((len(self.seeds),) + sz)
        out2 = np.empty((len(self.seeds),) + sz)

        # threaded generation
        t = [Thread(target=function, args=(np.random.RandomState(s), o))
             for s, o in zip(self.seeds, out1)]
        [x.start() for x in t]
        [x.join() for x in t]

        # the same serial
        for s, o in zip(self.seeds, out2):
            function(np.random.RandomState(s), o)

        # these platforms change x87 fpu precision mode in threads
        if np.intp().dtype.itemsize == 4 and sys.platform == "win32":
            assert_array_almost_equal(out1, out2)
        else:
            assert_array_equal(out1, out2)

    def __exclude_test_normal(self):
        def gen_random(state, out):
            out[...] = state.normal(size=10000)
        self.check_function(gen_random, sz=(10000,))

    def __exclude_test_exp(self):
        def gen_random(state, out):
            out[...] = state.exponential(scale=np.ones((100, 1000)))
        self.check_function(gen_random, sz=(100, 1000))

    def __exclude_test_multinomial(self):
        def gen_random(state, out):
            out[...] = state.multinomial(10, [1 / 6.] * 6, size=10000)
        self.check_function(gen_random, sz=(10000, 6))


class TestSingleEltArrayInput(object):
    def setup(self):
        self.argOne = np.array([2])
        self.argTwo = np.array([3])
        self.argThree = np.array([4])
        self.tgtShape = (1,)

    def SEG_test_one_arg_funcs(self):
        funcs = (np.random.exponential, np.random.standard_gamma,
                 np.random.chisquare, np.random.standard_t,
                 np.random.pareto, np.random.weibull,
                 np.random.power, np.random.rayleigh,
                 np.random.poisson, np.random.zipf,
                 np.random.geometric, np.random.logseries)

        probfuncs = (np.random.geometric, np.random.logseries)

        for func in funcs:
            if func in probfuncs:  # p < 1.0
                out = func(np.array([0.5]))

            else:
                out = func(self.argOne)

            assert_equal(out.shape, self.tgtShape)

    def __exclude_test_two_arg_funcs(self):
        funcs = (np.random.uniform, np.random.normal,
                 np.random.beta, np.random.gamma,
                 np.random.f, np.random.noncentral_chisquare,
                 np.random.vonmises, np.random.laplace,
                 np.random.gumbel, np.random.logistic,
                 np.random.lognormal, np.random.wald,
                 np.random.binomial, np.random.negative_binomial)

        probfuncs = (np.random.binomial, np.random.negative_binomial)

        for func in funcs:
            if func in probfuncs:  # p <= 1
                argTwo = np.array([0.5])

            else:
                argTwo = self.argTwo

            out = func(self.argOne, argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, argTwo[0])
            assert_equal(out.shape, self.tgtShape)

    def __exclude_test_three_arg_funcs(self):
        funcs = [np.random.noncentral_f, np.random.triangular,
                 np.random.hypergeometric]

        for func in funcs:
            out = func(self.argOne, self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, self.argTwo[0], self.argThree)
            assert_equal(out.shape, self.tgtShape)


class TestNotImpl(object):
    def test_normal_tupple(self):
        assert_raises(NotImplementedError, np.random.normal, [1, 2], 1)
        assert_raises(NotImplementedError, np.random.normal, 1, [1, 2])

    def test_gamma_tupple(self):
        assert_raises(NotImplementedError, np.random.gamma, [1, 2], 1)
        assert_raises(NotImplementedError, np.random.gamma, 1, [1, 2])

    def test_poisson_tupple(self):
        assert_raises(NotImplementedError, np.random.poisson, [1, 2])

    def test_logistic_tupple(self):
        assert_raises(NotImplementedError, np.random.logistic, [1, 2], 1)
        assert_raises(NotImplementedError, np.random.logistic, 1, [1, 2])

    def test_weibull_tupple(self):
        assert_raises(NotImplementedError, np.random.weibull, [1, 2])

    def test_exponential_tupple(self):
        assert_raises(NotImplementedError, np.random.exponential, [1, 2])

    def test_lognormal_tupple(self):
        assert_raises(NotImplementedError, np.random.lognormal, [1, 2], 1)
        assert_raises(NotImplementedError, np.random.lognormal, 1, [1, 2])

    def test_gumbel_tupple(self):
        assert_raises(NotImplementedError, np.random.gumbel, [1, 2], 1)
        assert_raises(NotImplementedError, np.random.gumbel, 1, [1, 2])

    def test_geometric_tupple(self):
        assert_raises(NotImplementedError, np.random.geometric, [1, 2])

    def test_binomial_tupple(self):
        assert_raises(NotImplementedError, np.random.binomial, [1, 2], 1)
        assert_raises(NotImplementedError, np.random.binomial, 1, [1, 2])

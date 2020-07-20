#
# * The source code in this file is based on the soure code of NumPy and CuPy.
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
#     * Neither the name of the NumPy Developers nor the names of any contributors may be
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
# # CuPy License #
#
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#     THE SOFTWARE.
#

import numpy
import nlcpy
import sys


from nlcpy.random.generator import RandomState
from nlcpy.random.generator import _integers_types

_asl_seed_max = numpy.iinfo(numpy.uint32).max
_generator_supported_type = {'float32', 'float64'}


class SeedSequence:
    """SeedSequence mixes sources of entropy in a reproducible way to set the initial
    state for independent and very probably non-overlapping BitGenerators.

    Args:
        entropy : None or int or sequence[int], optional
            The entropy for creating a SeedSequence.

    Note:
        Best practice for achieving reproducible bit streams is to use the default None
        for the initial entropy, and then use SeedSequence. entropy to log/pickle the
        entropy for reproducibility:
        >>> import nlcpy as vp
        >>> sq1 = vp.random.SeedSequence()
        >>> sq1.entropy
        >>> array([3641776954])

    """
    def __init__(self, entropy=None):
        if entropy is None:
            self.entropy = nlcpy.random.randint(0, _asl_seed_max)
        else:
            s = nlcpy.asarray(entropy)
            if s.dtype.name not in _integers_types:
                raise TypeError(
                    'SeedSequence expects int or sequence of ints for '
                    'entropy not {}' .format(entropy))
            if nlcpy.any(s < 0) or nlcpy.any(s > _asl_seed_max):
                raise ValueError('expected non-negative integer')
            self.entropy = s


class BitGenerator:
    """Base Class for generic BitGenerators, which provide a stream of random bits based
    on different algorithms. Must be overridden.

    Args:
        seed : None or int or array_like[ints], optional
            A seed to initialize the BitGenerator. If None, then fresh, unpredictable
            entropy will be pulled from the OS.

    """
    def __init__(self, seed=None):
        if seed is None:
            seed = SeedSequence()
            self.entropy = seed.entropy
        elif hasattr(seed, "entropy"):
            self.entropy = seed.entropy
        else:
            s = SeedSequence(seed)
            self.entropy = s.entropy


class MT19937(BitGenerator):
    """Container for the Mersenne Twister pseudo-random number generator.

    MT19937 is BitGenerator.

    """
    pass


class Generator:
    """Container for the BitGenerators.

    Generator exposes a number of methods for generating random numbers drawn from a
    variety of probability distributions. In addition to the distribution-specific
    arguments, each method takes a keyword argument size that defaults to None. If size
    is None, then a single value is generated and returned. If size is an integer, then a
    1-D array filled with generated values is returned. If size is a tuple, then an array
    with that shape is filled and returned. The function `nlcpy.random.default_rng` will
    instantiate a Generator with default BitGenerator.

    Args:
        bit_generator : None or BitGenerator or int or array_like[ints], optional
            BitGenerator to use as the core generator.

    See Also:
        `default_rng` : Recommended constructor for Generator.

    """
    def __init__(self, bit_generator):
        if not isinstance(bit_generator, BitGenerator):
            raise TypeError('Generator expects BitGenerator')
        if hasattr(bit_generator, "entropy"):
            self._rand = RandomState(bit_generator.entropy)
            self.bit_generator = bit_generator
        else:
            raise RuntimeError('Generator has not entropy')

    def integers(self, low, high=None, size=None,
                 dtype='int64', endpoint=False):
        """Returns random integers from low (inclusive) to high (exclusive), or if
        endpoint=True, low (inclusive) to high (inclusive).

        Replaces RandomState.randint (with endpoint=False) and
        RandomState.random_integers (with endpoint=True) Returns random integers from the
        "discrete uniform" distribution of the specified dtype. If high is None (the
        default), then results are from 0 to low.

        Args:
            low : int
                Lowest (signed) integer to be drawn from the distribution (unless
                high=None, in which case this parameter is 0 and this value is used for
                high).
            high : int, optional
                If provided, one above the largest (signed) integer to be drawn from the
                distribution (see above for behavior if high=None).
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.
            dtype : str or dtype, optional
                Desired dtype of the result. All dtypes are determined by their name,
                i.e., 'int64', 'int', etc, so byteorder is not available. The default
                value is 'nlcpy.int64'.
            endpoint : bool, optional
                If true, sample from the interval [low, high] instead of the default
                [low, high). Defaults to False.

        Returns:
            out : `ndarray` of ints
                size-shaped array of random integers from the appropriate distribution.

        Raises:
            low is neither a scalar nor None : NotImplementedError occurs
            high is neither a scalar nor None : NotImplementedError occurs

        Note:
            When using broadcasting with uint64 dtypes, the maximum value (2**64) cannot
            be represented as a standard integer type. The high array (or low if high is
            None) must have object dtype, e.g., array([2**64]).

        Examples:
            >>> import nlcpy as vp
            >>> rng = vp.random.default_rng()
            >>> rng.integers(2, size=10)
            array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])  # random
            >>> rng.integers(1, size=10)
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        """
        self._rand._is_number(low)
        if high is None:
            high = low
            low = 0
        else:
            self._rand._is_number(high)
        if endpoint == 1:
            high += 1
        return self._rand.randint(low, high, size=size, dtype=dtype)

    def random(self, size=None, dtype='d', out=None):
        """Returns random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the stated interval.
        To sample :math:`Unif[a, b), b > a` multiply the output of random by (b-a) and
        add a:
        (b - a) * random() + a

        Args:
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.  Default is None, in which case a single value is
                returned.
            dtype : str or dtype, optional
                Desired dtype of the result, either 'd' (or 'float64') or 'f' (or
                'float32'). All dtypes are determined by their name. The default value is
                'd'.
            out : `ndarray`, optional
                Alternative output array in which to place the result. If size is not
                None,
                it must have the same shape as the provided size and must match the type
                of the output values.

        Returns:
            out : `ndarray` of floats
                Array of random floats of shape size.

        Examples:
            >>> import nlcpy as vp
            >>> rng = vp.random.default_rng()
            >>> rng.random()
            array([0.47108547995356098]) # random
            >>> type(rng.random())
            <class 'nlcpy.core.core.ndarray'>
            >>> rng.random((5,))
            array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])

        """
        dt = numpy.dtype(dtype)
        key = dt.name
        if key not in _generator_supported_type:
            raise TypeError('Unsupported dtype "{0}" for {1} '.
                            format(key, sys._getframe().f_code.co_name))
        if out is None:
            return self._rand._generate_random_uniform(size=size, dtype=dtype)
        else:
            self._outarg_check(dt, size, out)
            self._rand._generate_random_uniform_for_generator(
                size=size, out=out)

    def bytes(self, length):
        """Returns random bytes.

        Args:
            length : int
                Number of random bytes.

        Returns:
            out : str
                String of length length.

        Examples:
            >>> import nlcpy as vp
            >>> vp.random.default_rng().bytes(10)
            b'\\x9fW\\xc5\\x12\\x95\\xfd\\xba\\x0fd\\xff' # random

        """
        return self._rand.bytes(length)

    def shuffle(self, x):
        """Modifies a sequence in-place by shuffling its contents.

        This function only shuffles the array along the first axis of a multi-dimensional
        array. The order of sub-arrays is changed but their contents remains the same.

        Args:
            x : array_like
                The array or list to be shuffled.

        Returns:
            None

        Examples:
            >>> import nlcpy as vp
            >>> rng = vp.random.default_rng()
            >>> arr = vp.arange(10)
            >>> rng.shuffle(arr)
            >>> arr
            array([7, 1, 5, 6, 0, 8, 4, 2, 9, 3]) # random
            # Multi-dimensional arrays are only shuffled along the first axis:
            >>> arr = vp.arange(9).reshape((3, 3))
            >>> rng.shuffle(arr)
            >>> arr
            array([[3, 4, 5], # random
                   [6, 7, 8],
                   [0, 1, 2]])

        """
        return self._rand.shuffle(x)

    def permutation(self, x):
        """Randomly permutes a sequence, or returns a permuted range.

        If x is a multi-dimensional array, it is only shuffled along it first index.

        Args:
            x : int or array_like
                If x is an integer, randomly permute vp.arange(x). If x is an array, make
                a copy and shuffle the elements randomly.

        Returns:
            out : `ndarray`
                Permuted sequence or array range.

        Examples:
            >>> import nlcpy as vp
            >>> rng = vp.random.default_rng()
            >>> rng.permutation(10)
            array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random
            >>> rng.permutation([1, 4, 9, 12, 15])
            array([15,  1,  9,  4, 12]) # random
            >>> arr = vp.arange(9).reshape((3, 3))
            >>> rng.permutation(arr)
            array([[6, 7, 8], # random
                   [0, 1, 2],
                   [3, 4, 5]])

        """
        return self._rand.permutation(x)

    def binomial(self, n, p, size=None):
        """Draws samples from a binomial distribution.

        Samples are drawn from a binomial distribution with specified parameters, n
        trials and p probability of success where n an integer >= 0 and p is in the
        interval [0,1]. (n may be input as a float, but it is truncated to an integer in
        use)

        Args:
            n : int
                Parameter of the distribution, >= 0. Floats are also accepted, but they
                will be truncated to integers.
            p : float
                Parameter of the distribution, >= 0 and <=1.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized binomial distribution, where each
                sample is equal to the number of successes over the n trials.

        Raises:
            n is neither a scalar nor None : NotImplementedError occurs.
            p is neither a scalar nor None : NotImplementedError occurs.

        Note:
            The probability density for the binomial distribution is:math:` P(N) = {n
            \\choose N}p^N(1-p)^{n-N}, `
            where :math:`n` is the number of trials, :math:`p` is the probability of
            success, and :math:`N` is the number of successes.
            When estimating the standard error of a proportion in a population by using a
            random sample, the normal distribution works well unless the product p*n <=5,
            where p = population proportion estimate, and n = number of samples, in which
            case the binomial distribution is used instead.
            For example, a sample of 15 people shows 4 who are left handed, and 11 who
            are right handed. Then p = 4/15 = 27. 0.27*15 = 4, so the binomial
            distribution should be used in this case.

        Examples:
            >>> import nlcpy as vp
            # Draw samples from the distribution:
            >>> rng = vp.random.default_rng()
            >>> n, p = 10, .5  # number of trials, probability of each trial
            >>> s = rng.binomial(n, p, 1000)
            # result of flipping a coin 10 times, tested 1000 times.
            # A real world example. A company drills 9 wild-cat oil exploration
            # wells, each with an estimated probability of success of 0.1. All nine
            # wells fail. What is the probability of that happening?
            # Let's do 20,000 trials of the model, and count the number that
            # generate zero positive results.
            >>> sum(rng.binomial(9, 0.1, 20000) == 0)/20000.
            # array(0.38625), or 38%.

        """
        return self._rand.binomial(n, p, size=size)

    def exponential(self, scale, size=None):
        """Draws samples from an exponential distribution.

        Its probability density function is:math:` f(x; \\frac{1}{\\beta}) =
        \\frac{1}{\\beta} \\exp(-\\frac{x}{\\beta}), `
        for x > 0 and 0 elsewhere. :math:`\\beta` is the scale parameter, which is the
        inverse of the rate parameter :math:` \\lambda = 1/\\beta `.

        Args:
            scale : float
                The scale parameter, :math:` \\beta = 1/\\lambda `. Must be non-negative.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized exponential distribution.

        Raises:
            scale is neither a scalar nor None : NotImplementedError occurs.

        """
        return self._rand.exponential(scale, size=size)

    def gamma(self, shape, scale=1.0, size=None):
        """Draws samples from a Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters, shape
        (sometimes designated "k") and scale (sometimes designated "theta"), where both
        parameters are > 0.

        Args:
            shape : float
                The shape of the gamma distribution. Must be non-negative.
            scale : float, optional
                The scale of the gamma distribution. Must be non-negative. Default is
                equal to 1.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized gamma distribution.

        Raises:
            shape is neither a scalar nor None : NotImplementedError occurs.
            scale is neither a scalar nor None : NotImplementedError occurs.

        Note:
            The probability density for the Gamma distribution is:math:` p(x) =
            x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)}, `
            where :math:`k` is the shape and :math:`\\theta` the scale, and
            :math:`\\Gamma` is the Gamma function.

        Examples:
            >>> import nlcpy as vp
            # Draw samples from the distribution:
            >>> shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
            >>> s = vp.random.default_rng().gamma(shape, scale, 1000)

        """
        return self._rand.gamma(shape, scale, size=size)

    def geometric(self, p, size=None):
        """Draws samples from a geometric distribution.

        Bernoulli trials are experiments with one of two outcomes: success or failure (an
        example of such an experiment is flipping a coin).  The geometric distribution
        models the number of trials that must be run in order to achieve success.
        It is therefore supported on the positive integers, k = 1, 2, ....
        The probability mass function of the geometric distribution is
        :math:` f(k) = (1 - p)^{k - 1} p `
        where p is the probability of success of an individual trial.

        Args:
            p : float
                The probability of success of an individual trial.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized geometric distribution.

        Raises:
            p is neither a scalar nor None : NotImplementedError occurs.

        Examples:
            >>> import nlcpy as vp
            # Draw ten thousand values from the geometric distribution,
            # with the probability of an individual success equal to 0.35:
            >>> z = vp.random.default_rng().geometric(p=0.35, size=10000)
            # How many trials succeeded after a single run?
            >>> (z.get() == 1).sum() / 10000.
            0.2235  # random

        """
        return self._rand.geometric(p, size=size)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        """Draws samples from a Gumbel distribution.

        Draws samples from a Gumbel distribution with specified location and scale.  For
        more information on the Gumbel distribution, see Notes and References below.

        Args:
            loc : float, optional
                The location of the mode of the distribution. Default is 0.
            scale : float, optional
                The scale parameter of the distribution. Default is 1. Must be
                non-negative.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized Gumbel distribution.

        Raises:
            loc is neither a scalar nor None : NotImplementedError occurs.
            scale is neither a scalar nor None : NotImplementedError occurs.

        Note:
            The probability density for the Gumbel distribution is
            :math:` p(x) = \\frac{e^{-(x - \\mu)/ \\beta}}{\\beta} e^{ -e^{-(x - \\mu)/
            \\beta}}, `
            where :math:`\\mu` is the mode, a location parameter, and :math:`\\beta` is
            the scale parameter.
            The function has a mean of :math:` \\mu + 0.57721\\beta ` and a variance of
            :math:` \\frac{\\pi^2}{6}\\beta^2 `.

        See Also:
            `weibull` : Draws samples from a Weibull distribution.

        Examples:
            >>> import nlcpy as vp
            # Draw samples from the distribution:
            >>> rng = vp.random.default_rng()
            >>> mu, beta = 0, 0.1 # location and scale
            >>> s = rng.gumbel(mu, beta, 1000)

        """
        return self._rand.gumbel(loc, scale, size=size)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        """Draws samples from a logistic distribution.

        Samples are drawn from a logistic distribution with specified parameters, loc
        (location or mean, also median), and scale (>0).

        Args:
            loc : float, optional
                Parameter of the distribution. Default is 0.
            scale : float, optional
                Parameter of the distribution. Must be non-negative. Default is 1.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized logistic distribution.

        Raises:
            loc is neither a scalar nor None : NotImplementedError occurs.
            scale is neither a scalar nor None : NotImplementedError occurs.

        Note:
            The probability density for the Logistic distribution is:math:` P(x) =
            \\frac{e^{-(x-\\mu)/s}}{s(1+e^{-(x-\\mu)/s})^2}, `
            where :math:`\\mu` = location and :math:`s` = scale.

        Examples:
            >>> import nlcpy as vp
            # Draw samples from the distribution:
            >>> loc, scale = 10, 1
            >>> s = vp.random.default_rng().logistic(loc, scale, 10000)

        """
        return self._rand.logistic(loc, scale, size=size)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        """Draws samples from a log-normal distribution.

        Draws samples from a log-normal distribution with specified mean, standard
        deviation, and array shape.  Note that the mean and standard deviation are not
        the values for the distribution itself, but of the underlying normal distribution
        it is derived from.

        Args:
            mean : float, optional
                Mean value of the underlying normal distribution. Default is 0.
            sigma : float, optional
                Standard deviation of the underlying normal distribution. Must be
                non-negative. Default is 1.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized log-normal distribution.

        Raises:
            mean is neither a scalar nor None : NotImplementedError occurs.
            sigma is neither a scalar nor None : NotImplementedError occurs.

        Note:
            A variable x has a log-normal distribution if log(x) is normally distributed.
            The probability density function for the log-normal distribution is:
            :math:` p(x) = \\frac{1}{\\sigma x
            \\sqrt{2\\pi}}e^{(-\\frac{(ln(x)-\\mu)^2}{2\\sigma^2})} `
            where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation
            of the normally distributed logarithm of the variable.

        Examples:
            >>> import nlcpy as vp
            # Draw samples from the distribution:
            >>> rng = vp.random.default_rng()
            >>> mu, sigma = 3., 1. # mean and standard deviation
            >>> s = rng.lognormal(mu, sigma, 1000)

        """
        return self._rand.lognormal(mean, sigma, size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Draws random samples from a normal (Gaussian) distribution.

        The probability density function of the normal distribution, first derived by De
        Moivre and 200 years later by both Gauss and Laplace independently, is often
        called the bell curve because of its characteristic shape (see the example
        below).
        The normal distributions occurs often in nature.  For example, it describes the
        commonly occurring distribution of samples influenced by a large number of tiny,
        random disturbances, each with its own unique distribution.

        Args:
            loc : float
                Mean ("centre") of the distribution.
            scale : float
                Standard deviation (spread or "width") of the distribution. Must be
                non-negative.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized normal distribution.

        Raises:
            loc is neither a scalar nor None : NotImplementedError occurs.
            scale is neither a scalar nor None : NotImplementedError occurs.

        Note:
            The probability density for the Gaussian distribution is:math:` p(x) =
            \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}e^{ - \\frac{ (x - \\mu)^2 } {2
            \\sigma^2} }, `
            where :math:`\\mu` is the mean and :math:`\\sigma` the standard
            deviation. The square of the standard deviation, :math:`\\sigma^2`, is called
            the variance.
            The function has its peak at the mean, and its "spread" increases with the
            standard deviation (the function reaches 0.607 times its maximum at
            numpy.random.normal is more likely to return samples lying close to the mean,
            rather than those far away.

        Examples:
            >>> import nlcpy as vp
            # Draw samples from the distribution:
            >>> mu, sigma = 0, 0.1 # mean and standard deviation
            >>> s = vp.random.default_rng().normal(mu, sigma, 1000)
            # Verify the mean and the variance:
            >>> abs(mu - vp.mean(s))
            array(0.00206415)  # may vary
            >>> abs(sigma - vp.std(s, ddof=1))
            array(0.00133596)  # may vary
            # Two-by-four array of samples from N(3, 6.25):
            >>> vp.random.default_rng().normal(3, 2.5, size=(2, 4))
            array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
                   [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random

        """
        return self._rand.normal(loc=loc, scale=scale, size=size)

    def poisson(self, lam=1.0, size=None):
        """Draws samples from a Poisson distribution.

        The Poisson distribution is the limit of the binomial distribution for large N.

        Args:
            lam : float
                Expectation of interval, must be >= 0. A sequence of expectation
                intervals must be broadcastable over the requested size.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized Poisson distribution.

        Raises:
            lam is neither a scalar nor None : NotImplementedError occurs.

        Note:
            The Poisson distribution:math:` f(k; \\lambda)=\\frac{\\lambda^k
            e^{-\\lambda}}{k!} `
            For events with an expected separation :math:`\\lambda` the Poisson
            distribution :math:`f(k; \\lambda)` describes the probability of :math:`k`
            events occurring within the observed interval :math:`\\lambda`.
            Because the output is limited to the range of the C int64 type, a ValueError
            is raised when lam is within 10 sigma of the maximum representable value.

        Examples:
            >>> import nlcpy as vp
            # Draw samples from the distribution:
            >>> rng = vp.random.default_rng()
            >>> s = rng.poisson(5, 10000)

        """
        return self._rand.poisson(lam, size=size)

    def standard_cauchy(self, size=None):
        """Draws samples from a standard Cauchy distribution with mode = 0.

        Also known as the Lorentz distribution.

        Args:
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            samples : `ndarray`
                The drawn samples.

        Note:
            The probability density function for the full Cauchy distribution is:math:`
            P(x; x_0, \\gamma) = \\frac{1}{\\pi \\gamma \\bigl[
            1+(\\frac{x-x_0}{\\gamma})^2 \\bigr] } `
            and the Standard Cauchy distribution just sets :math:` x_0=0 ` and

        Examples:
            >>> import nlcpy as vp
            # Draw samples and plot the distribution:
            >>> s = vp.random.default_rng().standard_cauchy(1000000)

        """
        return self._rand.standard_cauchy(size=size)

    def standard_exponential(
            self, size=None, dtype='d', method=None, out=None):
        """Draws samples from a standard exponential distribution.

        standard_exponential is identical to the exponential distribution with a scale
        parameter of 1.

        Args:
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.
            dtype : dtype, optional
                Desired dtype of the result, either 'd' (or 'float64') or 'f' (or
                'float32'). All dtypes are determined by their name. The default value is
                'd'.
            method : str, optional
                Function by this argument is not implemented.
            out : `ndarray`, optional
                Alternative output array in which to place the result. If size is not
                None, it must have the same shape as the provided size and must match the
                type of the output values.

        Returns:
            out : `ndarray`
                Drawn samples.

        Examples:
            >>> import nlcpy as vp
            >>> n = vp.random.default_rng().standard_exponential((3, 8000))

        """
        if method is not None:
            raise NotImplementedError('method is None only')
        dt = numpy.dtype(dtype)
        key = dt.name
        if key not in _generator_supported_type:
            raise TypeError('Unsupported dtype "{0}" for {1} '.
                            format(key, sys._getframe().f_code.co_name))

        if out is None:
            return self._rand._generate_random_exponential(
                1.0, size=size, dtype=float)
        else:
            self._outarg_check(dt, size, out)
            self._rand._generate_random_exponential_for_generator(
                size=out.size, out=out)

    def standard_gamma(self, shape, size=None, dtype='d', out=None):
        """Draws samples from a standard Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters,shape
        (sometimes designated "k") and scale=1.

        Args:
            shape : float
                Parameter, must be non-negative.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.
            dtype : str or dtype, optional
                Desired dtype of the result, either 'd' (or 'float64') or 'f' (or
                'float32'). All dtypes are determined by their name. The default value is
                'd'.
            out : `ndarray`, optional
                Alternative output array in which to place the result. If size is not
                None, it must have the same shape as the provided size and must match the
                type of the output values.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized standard gamma distribution.

        Raises:
            shape is neither a scalar nor None : NotImplementedError occurs

        Note:
            The probability density for the Gamma distribution is :math:` p(x) =
            x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)}, ` where :math:`k` is the
            shape and :math:`\\theta` the scale, and :math:`\\Gamma` is the Gamma
            function.
            The Gamma distribution is often used to model the times to failure of
            electronic components, and arises naturally in processes for which the
            waiting times between Poisson distributed events are relevant.

        Examples:
            # Draw samples from the distribution.
            >>> import nlcpy as vp
            >>> shape, scale = 2., 1. # mean and width
            >>> s = vp.random.default_rng().standard_gamma(shape, 1000000)
            >>> import matplotlib.pyplot as plt
            >>> import scipy.special as sps # doctest: +SKIP
            >>> count, bins, ignored = plt.hist(s, 50, density=True)
            >>> y = bins**(shape-1) * ((vp.exp(-bins/scale))/ # doctest: +SKIP
            ... (sps.gamma(shape) * scale**shape))
            >>> plt.plot(bins, y, linewidth=2, color='r') # doctest: +SKIP
            >>> plt.show()

        """
        self._rand._is_number(shape)
        if shape < 0:
            raise ValueError('shape < 0')

        dt = numpy.dtype(dtype)
        key = dt.name
        if key not in _generator_supported_type:
            raise TypeError('Unsupported dtype "{0}" for {1} '.
                            format(key, sys._getframe().f_code.co_name))

        if out is None:
            return self._rand._generate_random_gamma(
                shape, scale=1.0, size=size, dtype=dtype)
        else:
            self._outarg_check(dt, size, out)
            self._rand._generate_random_gamma_for_generator(
                shape, scale=1.0, size=out.size, out=out)

    def standard_normal(self, size=None, dtype='d', out=None):
        """Draws samples from a standard Normal distribution (mean=0, stdev=1).

        Args:
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.
            dtype : str or dtype, optional
                Desired dtype of the result, either 'd' (or 'float64') or 'f' (or
                'float32'). All dtypes are determined by their name. The default value is
                'd'.
            out : `ndarray`, optional
                Alternative output array in which to place the result. If size is not
                None, it must have the same shape as the provided size and must match the
                type of the output values.

        Returns:
            out : `ndarray`
                A floating-point array of shape size of drawn samples, if size was not
                specified.

        Note:
            For random samples from :math:`N(\\mu, \\sigma^2)`, use one of:mu + sigma *
            gen.standard_normal(size=...)
            gen.normal(mu, sigma, size=...)

        See Also:
            `normal` : Draws random samples from a normal (Gaussian) distribution.

        Examples:
            >>> import nlcpy as vp
            >>> rng = vp.random.default_rng()
            >>> rng.standard_normal()
            array([2.1923875335537315]) # random

        """
        dt = numpy.dtype(dtype)
        key = dt.name
        if key not in _generator_supported_type:
            raise TypeError('Unsupported dtype "{0}" for {1} '.
                            format(key, sys._getframe().f_code.co_name))
        if out is None:
            return self._rand._generate_random_normal(
                0.0, scale=1.0, size=size, dtype=float)
        else:
            self._outarg_check(dt, size, out)
            self._rand._generate_random_normal_for_generator(
                size=out.size, out=out)

    def uniform(self, low=0.0, high=1.0, size=None):
        """Draws samples from a uniform distribution.

        Samples are uniformly distributed over the half-open interval [low, high)
        (includes low, but excludes high).  In other words,
        any value within the given interval is equally likely to be drawn by uniform.

        Args:
            low : float, optional
                Lower boundary of the output interval.  All values generated will be
                greater than or equal to low.  The default value is 0.
            high : float
                Upper boundary of the output interval.  All values generated will be less
                than high.  The default value is 1.0.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized uniform distribution.

        Raises:
            low is neither a scalar nor None : NotImplementedError occurs.
            high is neither a scalar nor None : NotImplementedError occurs.

        Note:
            The probability density function of the uniform distribution is:math:` p(x) =
            \\frac{1}{b - a} `
            anywhere within the interval [a, b), and zero elsewhere.
            When high == low, values of low will be returned.

        See Also:
            `integers` : Returns random integers from low (inclusive) to high
                (exclusive), or if endpoint=True, low (inclusive) to high (inclusive).
            `random` : Returns random floats in the half-open interval [0.0, 1.0).

        Examples:
            >>> import nlcpy as vp
            # Draw samples from the distribution:
            >>> s = vp.random.default_rng().uniform(-1,0,1000)
            # All values are within the given interval:
            >>> vp.all(s >= -1)
            array(True)
            >>> vp.all(s < 0)
            array(True)

        """
        return self._rand.uniform(low, high, size=size)

    def weibull(self, a, size=None):
        """Draws samples from a Weibull distribution.

        Draws samples from a 1-parameter Weibull distribution with the given shape
        parameter a.
        :math:` X = (-ln(U))^{1/a} `
        Here, U is drawn from the uniform distribution over (0,1].
        The more common 2-parameter Weibull, including a scale parameter

        Args:
            a : float
                Shape parameter of the distribution.  Must be nonnegative.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn.

        Returns:
            out : `ndarray`
                Drawn samples from the parameterized Weibull distribution.

        Raises:
            a is neither a scalar nor None : NotImplementedError occurs.

        Note:
            The probability density for the Weibull distribution is:math:` p(x) =
            \\frac{a}{\\lambda}(\\frac{x}{\\lambda})^{a-1}e^{-(x/\\lambda)^a}, `
            where :math:`a` is the shape and :math:`\\lambda` the scale.
            The function has its peak (the mode) at
            :math:` \\lambda(\\frac{a-1}{a})^{1/a}. `
            When a = 1, the Weibull distribution reduces to the exponential distribution.

        See Also:
            `gumbel` : Draws samples from a Gumbel distribution.

        Examples:
            >>> import nlcpy as vp
            # Draw samples from the distribution:
            >>> rng = vp.random.default_rng()
            >>> a = 5. # shape
            >>> s = rng.weibull(a, 1000)

        """
        return self._rand.weibull(a, size=size)

    def _outarg_check(self, dt, size, nda):
        if not isinstance(nda, nlcpy.core.core.ndarray):
            raise ValueError(
                "Supplied output array is not contiguous, writable or aligned.")

        if numpy.dtype(nda.dtype) != dt:
            raise TypeError(
                "Supplied output array has the wrong type,"
                " Expected {0}, got {1}".format(
                    dt, nda.dtype.name))
        if size is not None:
            if nda.shape != size and nda.shape != (size,):
                raise ValueError(
                    "size must match out.shape when used together")


# lazy initialize (To avoid a long execution time when import )
_default_rng = None


def default_rng(seed=None):
    """Constructs a new nlcpy.random.Generator with the default BitGenerator (MT19937).

    Args:
        seed : None or int or array_like[ints], optional

    Note:
        When seed is omitted or None, a new BitGenerator and Generator will be
        instantiated each time.
        This function does not manage a default global instance.

    """
    global _default_rng
    if _default_rng is None or seed is not None:
        _default_rng = Generator(MT19937(seed))

    return _default_rng

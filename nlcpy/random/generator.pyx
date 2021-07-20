#
# * The source code in this file is based on the soure code of NumPy and CuPy.
#
# # NLCPy License #
#
#     Copyright (c) 2020-2021 NEC Corporation
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

from nlcpy.random.libgenerator cimport *
import numbers
import copy
import numpy
import nlcpy

from nlcpy import veo

from nlcpy import ndarray
from nlcpy import asarray
from nlcpy import array
from nlcpy import empty
from nlcpy import any
from nlcpy import nan
from nlcpy.request import request
from nlcpy import AxisError

# change in the future
from numpy import broadcast

_integers_types = {
    'bool',
    'int',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64'}
_unsigned_integers_types = {'bool', 'uint8', 'uint16', 'uint32', 'uint64'}


def _get_rand():
    global _rand
    if _rand is None:
        _rand = RandomState()
    return _rand


def get_state():
    """Returns an ndarray representing the internal state of the generator.

    For more details, see set_state.

    Returns
    -------
    out : ndarray
        An ndarray containing seeds to be required for generating random numbers.

    Note
    ----

    set_state and get_state are not needed to work with any of therandom distributions in
    NLCPy.

    """
    rs = _get_rand()
    return rs.get_state()


def set_state(state):
    """Sets the internal state of the generator from an ndarray.

    For use if one has reason to manually (re-)set the internal state ofthe bit generator
    used by the RandomState instance.

    Parameters
    ----------
    state : ndarrayi
        The state ndarray has the following items:
        1. seeds for ASL Unified Interface.

    Returns
    -------
    out : None
        Returns 'None' on success.

    Note
    ----

    set_state and get_state are not needed to work with any of therandom distributions in
    NLCPy.

    See Also
    --------
    get_state : Returns a tuple representing the internal state of the generator.

    """
    rs = _get_rand()
    rs.set_state(state)
    return


def seed(seed=None):
    """Reseeds a MT19937 BitGenerator

    Examples
    --------
    >>> import nlcpy
    >>> from nlcpy.random import MT19937
    >>> from nlcpy.random import RandomState, SeedSequence
    >>> rs = RandomState(MT19937(SeedSequence(123456789)))
    # Later, you want to restart the stream
    >>> rs = RandomState(MT19937(SeedSequence(987654321)))

    """
    rs = _get_rand()
    rs.seed(seed)
    return


class RandomState():
    """Container for the slow Mersenne Twister pseudo-random number generator.

    RandomState and Generator expose a number of methods for generating random numbers
    drawn from a variety of probability distributions.
    In addition to the distribution-specific arguments, each method takes a keyword
    argument size that defaults to ``None``. If size is ``None``,
    then a single value is generated and returned. If size is an integer,
    then a 1-D array filled with generated values is returned.
    If size is a tuple, then an array with that shape is filled and returned.

    Parameters
    ----------
    seed : None or int or array_like, optional
        Random seed used to initialize the pseudo-random number generator or an
        instantized BitGenerator.  If an integer or array, used as a seed for the MT19937
        BitGenerator. Values can be any integer between 0 and 2**32 - 1 inclusive, an
        array (or other sequence) of such integers, or ``None`` (the default).

    """
    _asl_seed_max = numpy.iinfo(numpy.uint32).max

    def __init__(self, seed=None):
        self.seed(seed)

    def get_state(self, legacy=True):
        """Returns an ndarray representing the internal state of the generator.

        For more details, see set_state.

        Parameters
        ----------
        legacy : bool
            Not used in NLCPy.

        Returns
        -------
        out : ndarray
            An ndarray containing seeds to be required for generating random numbers.

        Note
        ----
        :func:`RandomState.set_state` and :func:`RandomState.get_state` are
        not needed to work with any of the random distributions in NLCPy.

        See Also
        --------
        RandomState.set_state : Sets the internal state of the generator from an ndarray.
        """
        return self._asl_get_state()

    def set_state(self, state):
        """Sets the internal state of the generator from an ndarray.

        For use if one has reason to manually (re-)set the internal state of the bit
        generator used by the RandomState instance.

        Parameters
        ----------
        state : ndarray
            An ndarray containing seeds to be required for generating random numbers.

        Returns
        -------
        out : None
            Returns 'None' on success.

        Note
        ----
        :func:`RandomState.set_state` and :func:`RandomState.get_state` are
        not needed to work with any of therandom distributions in NLCPy.

        See Also
        --------
        RandomState.get_state : Returns a tuple representing the internal
                                state of the generator.
        """
        if not isinstance(state, ndarray):
            raise TypeError('state is not valid')
        else:
            if not numpy.dtype(state.dtype).name == 'uint32':
                raise TypeError(
                    'Unsupported dtype {} for set_state' .format(
                        numpy.dtype(
                            state.dtype).name))
            self._asl_set_state(state)

    def _get_seed(self):
        return self._ve_seed.get().tolist()

    def tomaxint(self, size):
        """Random integers between 0 and ``nlcpy.iinfo(nlcpy.int).max``, inclusive.

        Return a sample of uniformly distributed random integers in the interval
        ``[0, nlcpy.iinfo(nlcpy.int).max]``.
        The nlcpy.int64/nlcpy.int32 type translates to the C long integer type,
        which is int64_t in NLCPy.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape size.

        Examples
        --------
        >>> import numpy as np
        >>> import nlcpy as vp
        >>> rs = vp.random.RandomState() # need a RandomState object
        >>> rs.tomaxint((2,2,2))    # doctest: +SKIP
        array([[[1170048599, 1600360186], # random
                [ 739731006, 1947757578]],
        <BLANKLINE>
               [[1871712945,  752307660],
                [1601631370, 1479324245]]])
        >>> rs.tomaxint((2,2,2)) < vp.iinfo(vp.int).max
        array([[[ True,  True],
                [ True,  True]],
        <BLANKLINE>
               [[ True,  True],
                [ True,  True]]])

        """
        return self.randint(0, 2**31 - 1, size=size)

    def rand(self, size):
        """Random values in a given shape.

        Create an array of the given shape and populate it with random samples from a
        uniform distribution over ``[0, 1)``.

        Parameters
        ----------
        size : int or tuple of ints, optional
            The dimensions of the returned array, must be non-negative.

        Returns
        -------
        out : ndarray
            Random values, with shape *size.*

        See Also
        --------
        RandomState.random : Returns random floats in the half-open
                             interval ``[0.0, 1.0)``.

        Examples
        --------
        >>> import nlcpy as vp
        >>> vp.random.rand(3,2)  # doctest: +SKIP
        array([[0.2501974 , 0.01560572],  # random
               [0.93670877, 0.6073555 ],  # random
               [0.18378925, 0.22068119]]) # random
        """
        return self.random_sample(size)

    def randn(self, size):
        """Returns a sample (or samples) from the "standard normal" distribution.

        If positive int_like arguments are provided, randn generates an array of shape
        ``(d0, d1, ..., dn)``, filled with random floats sampled from a univariate
        "normal" (Gaussian) distribution of mean 0 and variance 1.

        Parameters
        ----------
        size : int or tuple of ints, optional
            The dimensions of the returned array, must be non-negative.

        Returns
        -------
        Z : ndarray
            A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from the
            standard normal distribution, or a single such float if no parameters were
            supplied.

        Note
        ----
        For random samples from :math:`N(\\mu, \\sigma^2)`, use::

            sigma * vp.random.randn(...) + mu

        See Also
        --------
        RandomState.standard_normal : Draws samples from a standard Normal
                                      distribution (mean=0, stdev=1).
        RandomState.normal : Draws random samples from a normal (Gaussian) distribution.

        Examples
        --------
        >>> import nlcpy as vp
        >>> vp.random.randn()   # doctest: +SKIP
        array(0.54214143)  # random

        Two-by-four array of samples from N(3, 6.25):

        >>> 3 + 2.5 * vp.random.randn(2, 4) # doctest: +SKIP
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random
        """
        return self.normal(0.0, 1.0, size=size)

    def randint(self, low, high=None, size=None, dtype=int):
        """Returns random integers from *low* (inclusive) to *high* (exclusive).

        Returns random integers from the "discrete uniform" distribution of the specified
        dtype in the "half-open" interval ``[low, high)``. If *high* is None
        (the default), then results are from ``[0, low)``.

        Parameters
        ----------
        low : int
            array_like of ints is not implemented. Lowest (signed) integers to be drawn
            from the distribution (unless ``high=None``, in which case this parameter is
            one above the *highest* such integer).
        high : int , optional
            array_like of ints is not implemented. If provided, one above the largest
            (signed) integer to be drawn from the distribution (see above for behavior if
            ``high=None``).
        size : int or ints, optional
            Output shape. If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn. Default is None, in which case a single value
            is returned.
        dtype : dtype, optional
            Desired dtype of the result. All dtypes are determined by their name, i.e.,
            'int64', 'int', etc, so byteorder is not available and a specific precision
            may have different C types depending on the platform.

        Returns
        -------
        out : ndarray of ints
            size-shaped array of random integers from the appropriate distribution, or a
            single such random int if size not provided.

        See Also
        --------
        RandomState.random_integers : Random integers of type nlcpy.int64/nlcpy.int32
                                      between low and high, inclusive.

        Examples
        --------
        >>> import nlcpy as vp
        >>> vp.random.randint(2, size=10)   # doctest: +SKIP
        array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random
        >>> vp.random.randint(1, size=10)   # doctest: +SKIP
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        Generate a 2 x 4 array of ints between 0 and 4, inclusive:

        >>> vp.random.randint(5, size=(2, 4)) # doctest: +SKIP
        array([[4, 0, 2, 1], # random
               [3, 2, 2, 0]])

        Generate a 1 x 3 array with 3 different upper bounds

        >>> vp.random.randint(5, size=(2, 4))  # doctest: +SKIP
        array([[4, 0, 2, 1], # random
               [3, 2, 2, 0]])
        """

        # AttributeError: module 'nlcpy' has no attribute 'dtype'
        dt = numpy.dtype(dtype)
        key = dt.name
        if key not in _integers_types:
            raise TypeError(
                'Unsupported dtype {} for randint' .format(
                    numpy.dtype(dtype).name))

        if high is None:
            lo = 0
            hi = low
        else:
            lo = low
            hi = high
        if lo >= hi:
            raise ValueError('low >= high')

        if key == 'bool':
            if lo < 0:
                raise ValueError(
                    'low is out of bounds for {}'.format(
                        numpy.dtype(dtype).name))
            if hi > 2:
                raise ValueError(
                    'high is out of bounds for {}'.format(
                        numpy.dtype(dtype).name))
        else:
            if lo < numpy.iinfo(dtype).min:
                raise ValueError(
                    'low is out of bounds for {}'.format(
                        numpy.dtype(dtype).name))
            if hi > numpy.iinfo(dtype).max + 1:
                raise ValueError(
                    'high is out of bounds for {}'.format(
                        numpy.dtype(dtype).name))

        return self._generate_random_integers(
            lo, high=hi, size=size, dtype=dtype)

    def ranf(self, size):
        """This is an alias of random_sample.

        See random_sample for the complete documentation.
        """
        return self.random_sample(size)

    def sample(self, size):
        """This is an alias of random_sample.

        See random_sample for the complete documentation.
        """
        return self.random_sample(size)

    def random(self, size=None):
        """Returns random floats in the half-open interval ``[0.0, 1.0)``.

        Alias for random_sample to ease forward-porting to the new random API.
        """
        return self.random_sample(size)

    def random_integers(self, low, high=None, size=None):
        """Random integers of type nlcpy.int64/nlcpy.int32 between *low* and *high*,
        inclusive.

        Return random integers of type nlcpy.int64/nlcpy.int32 from the "discrete
        uniform" distribution in the closed interval [*low*, *high*].  If *high* is None
        (the default), then results are from [1, *low*].

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is the *highest* such integer).
        high : int, optional
            If provided, the largest (signed) integer to be drawn from the distribution
            (see above for behavior if ``high=None``).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray of ints
            size-shaped array of random integers from the appropriate distribution.

        Note
        ----
        To sample from N evenly spaced floating-point numbers between a and b, use::

            a + (b - a) * (vp.random.random_integers(N) - 1) / (N - 1.)

        See Also
        --------
        RandomState.randint :  Returns random integers from *low* (inclusive)
                               to *high* (exclusive).

        Examples
        --------
        .. plot::
            :align: center

            >>> import nlcpy as vp
            >>> vp.random.random_integers(5)    # doctest: +SKIP
            array(2) # random
            >>> type(vp.random.random_integers(5))
            <class 'nlcpy.core.core.ndarray'>
            >>> vp.random.random_integers(5, size=(3,2))  # doctest: +SKIP
            array([[5, 4], # random
                   [3, 3],
                   [4, 5]])

            Choose five random numbers from the set of five evenly-spaced
            numbers between 0 and 2.5, inclusive (*i.e.*, from the set
            {0, 5/8, 10/8, 15/8, 20/8}):

            >>> 2.5 * (vp.random.random_integers(5, size=(5,)) - 1) / 4. # doctest: +SKIP
            array([ 0.625,  1.25 ,  0.625,  0.625,  2.5  ]) # random

            Roll two six sided dice 1000 times and sum the results:

            >>> d1 = vp.random.random_integers(1, 6, 1000)
            >>> d2 = vp.random.random_integers(1, 6, 1000)
            >>> dsums = d1 + d2

            Display results as a histogram:

            >>> import matplotlib.pyplot as plt
            >>> count, bins, ignored = plt.hist(dsums.get(), 11, density=True)
            >>> plt.show()
        """
        if high is None:
            """
            warnings.warn(("This function is deprecated. Please call "
                          "randint(1, {low} + 1) instead".format(low=low)).
                          DeprecationWarning)
            """
            high = low
            low = 1
        else:
            """
            warnings.warn(("This function is deprecated. Please call "
                           "randint({low}, {high} + 1) "
                           "instead".format(low=low, high=high)).
                          DeprecationWarning)
            """

        return self.randint(low, int(high) + 1, size=size, dtype='l')

    def random_sample(self, size=None):
        """Returns random floats in the half-open interval ``[0.0, 1.0)``.

        Results are from the "continuous uniform" distribution over the stated interval.
        To sample :math:`Unif[a, b), b > a` multiply the output of random_sample by
        *(b-a)* and add *a*::

            (b - a) * random_sample() + a

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape. If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn. Default is None, in which case a single value is
            returned.

        Returns
        -------
        out : ndarray of floats
            Array of random floats of shape size.

        Examples
        --------
        >>> import nlcpy as vp
        >>> vp.random.random_sample()            # doctest: +SKIP
        array(0.80430306) # random
        >>> type(vp.random.random_sample())
        <class 'nlcpy.core.core.ndarray'>
        >>> vp.random.random_sample((5,))        # doctest: +SKIP
        array([0.56570372, 0.13436335, 0.62341754, 0.88471288, 0.13366607])

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * vp.random.random_sample((3, 2)) - 5  # doctest: +SKIP
        array([[-3.99149989, -0.52338984], # random
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])
        """
        return self._generate_random_uniform(size=size, dtype=float)

    def bytes(self, length):
        """Returns random bytes.

        Parameters
        ----------
        length : int
            Number of random bytes.

        Returns
        -------
        out : str
            String of length length.

        Examples
        --------
        >>> import nlcpy as vp
        >>> vp.random.bytes(10)     # doctest: +SKIP
        b'\\x9fW\\xc5\\x12\\x95\\xfd\\xba\\x0fd\\xff' # random

        """
        n_uint32 = ((length - 1) // 4 + 1)
        # '<' is little endian, 'u4' is unsigned 4byte = u32
        return self.randint(0, 424967296, size=n_uint32,
                            dtype='uint32').astype('<u4').tobytes()[:length]

    def shuffle(self, x):
        """Modifies a sequence in-place by shuffling its contents.

        This function only shuffles the array along the first axis of a multi-dimensional
        array. The order of sub-arrays is changed but their contents remains the same.

        Parameters
        ----------
        x : array_like
            The array or list to be shuffled.

        Returns
        -------
        None

        Examples
        --------
        >>> import nlcpy as vp
        >>> arr = vp.arange(10)
        >>> vp.random.shuffle(arr)
        >>> arr            # doctest: +SKIP
        array([7, 1, 5, 6, 0, 8, 4, 2, 9, 3]) # random

        Multi-dimensional arrays are only shuffled along the first axis:

        >>> arr = vp.arange(9).reshape((3, 3))
        >>> vp.random.shuffle(arr)
        >>> arr           # doctest: +SKIP
        array([[3, 4, 5], # random
               [6, 7, 8],
               [0, 1, 2]])
        """
        n = len(x)
        if isinstance(x, nlcpy.ndarray):
            self._generate_random_shuffle(x)
        else:
            # Untyped path.
            random_interval = self.randint(0, high=n, size=n)
            for i in reversed(range(0, n)):
                j = random_interval.get().flat[i]
                x[i], x[j] = x[j], x[i]
        return

    def permutation(self, x):
        """Randomly permutes a sequence, or returns a permuted range.

        If *x* is a multi-dimensional array, it is only shuffled along it first index.

        Parameters
        ----------
        x : int or array_like
            If *x* is an integer, randomly permute ``vp.arange(x)``. If *x*
            is an array, make a copy and shuffle the elements randomly.

        Returns
        -------
        out : ndarray
            Permuted sequence or array range.

        Examples
        --------
        >>> import nlcpy as vp
        >>> vp.random.permutation(10) # doctest: +SKIP
        array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random

        >>> vp.random.permutation([1, 4, 9, 12, 15]) # doctest: +SKIP
        array([15,  1,  9,  4, 12]) # random

        >>> arr = vp.arange(9).reshape((3, 3))
        >>> vp.random.permutation(arr) # doctest: +SKIP
        array([[6, 7, 8], # random
               [0, 1, 2],
               [3, 4, 5]])
        """
        return self._generate_random_permutation(x)

    def binomial(self, n, p, size=None):
        """Draws samples from a binomial distribution.

        Samples are drawn from a binomial distribution with specified parameters, n
        trials and p probability of success where n an integer >= 0 and p is in the
        interval ``[0,1]``. (n may be input as a float, but it is truncated to an integer
        in use)

        Parameters
        ----------
        n : int
            Parameter of the distribution, >= 0. Floats are also accepted, but they will
            be truncated to integers.
        p : float
            Parameter of the distribution, >= 0 and <=1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized binomial distribution, where each sample
            is equal to the number of successes over the n trials.

        Restriction
        -----------
        * If *n* is neither a scalar nor None : *NotImplementedError* occurs.
        * If *p* is neither a scalar nor None : *NotImplementedError* occurs.

        Note
        ----
        The probability density for the binomial distribution is

        .. math::
            P(N) = {n \\choose N}p^N(1-p)^{n-N},

        where :math:`n` is the number of trials, :math:`p` is the probability of success,
        and :math:`N` is the number of successes.

        When estimating the standard error of a proportion in a population by using a
        random sample, the normal distribution works well unless the product p*n <=5,
        where p = population proportion estimate, and n = number of samples, in which
        case the binomial distribution is used instead.

        For example, a sample of 15 people shows 4 who are left handed, and 11 who are
        right handed. Then p = 4/15 = 27. 0.27*15 = 4, so the binomial distribution
        should be used in this case.

        Examples
        --------
        Draw samples from the distribution:

        >>> import nlcpy as vp
        >>> n, p = 10, .5  # number of trials, probability of each trial
        >>> s = vp.random.binomial(n, p, 1000)  # doctest: +SKIP
        # result of flipping a coin 10 times, tested 1000 times.

        A real world example. A company drills 9 wild-cat oil exploration
        wells, each with an estimated probability of success of 0.1. All nine
        wells fail. What is the probability of that happening?
        Let's do 20,000 trials of the model, and count the number that
        generate zero positive results.

        >>> sum(vp.random.binomial(9, 0.1, 20000) == 0)/20000.   # doctest: +SKIP
        array(0.38815)  #  or 38%.

       """
        self._is_number(n, p)
        if p < 0 or p > 1 or p is nan:
            raise ValueError('p < 0, p > 1 or p is NaN')
        _size = size if size is not None else broadcast(
            asarray(n), asarray(p)).shape
        return self._generate_random_binomial(n, p, size=_size, dtype=int)

    def exponential(self, scale=1.0, size=None):
        """Draws samples from an exponential distribution.

        Its probability density function is

        .. math::
            f(x; \\frac{1}{\\beta}) =
            \\frac{1}{\\beta} \\exp(-\\frac{x}{\\beta}),

        for ``x > 0`` and 0 elsewhere. :math:`\\beta` is the scale parameter,
        which is the inverse of the rate parameter :math:`\\lambda = 1/\\beta`.

        Parameters
        ----------
        scale : float
            The scale parameter, :math:`\\beta = 1/\\lambda`. Must be non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized exponential distribution.

        Restriction
        -----------
        * If *scale* is neither a scalar nor None : *NotImplementedError* occurs.
        """
        self._is_number(scale)
        if scale < 0:
            raise ValueError('scale < 0')

        _size = size if size is not None else asarray(scale).shape
        return self._generate_random_exponential(
            scale, size=_size, dtype=float)

    def gamma(self, shape, scale=1.0, size=None):
        """Draws samples from a Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters, *shape*
        (sometimes designated "k") and *scale* (sometimes designated "theta"),
        where both parameters are > 0.

        Parameters
        ----------
        shape : float
            The shape of the gamma distribution. Must be non-negative.
        scale : float, optional
            The scale of the gamma distribution. Must be non-negative. Default is equal
            to 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized gamma distribution.

        Restriction
        -----------
        * If *shape* is neither a scalar nor None : *NotImplementedError* occurs.
        * If *scale* is neither a scalar nor None : *NotImplementedError* occurs.

        Note
        ----
        The probability density for the Gamma distribution is

        .. math::
            p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},

        where :math:`k` is the shape and :math:`\\theta` the scale, and :math:`\\Gamma`
        is the Gamma function.

        Examples
        --------
        .. plot::
            :align: center

            Draw samples from the distribution:

            >>> import nlcpy as vp
            >>> shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
            >>> s = vp.random.gamma(shape, scale, 1000)

            >>> import matplotlib.pyplot as plt
            >>> import scipy.special as sps
            >>> count, bins, ignored = plt.hist(s.get(), 50, density=True)
            >>> y = bins**(shape-1)*(vp.exp(-bins/scale)/(sps.gamma(shape)*scale**shape))
            >>> plt.plot(bins, y, linewidth=2, color='r') # doctest: +SKIP
            >>> plt.show()
        """
        self._is_number(shape, scale)
        if shape < 0:
            raise ValueError('shape < 0')
        if scale < 0:
            raise ValueError('scale < 0')
        _size = size if size is not None else broadcast(
            asarray(shape), asarray(scale)).shape
        return self._generate_random_gamma(
            shape, scale, size=_size, dtype=float)

    def geometric(self, p, size=None):
        """Draws samples from a geometric distribution.

        Bernoulli trials are experiments with one of two outcomes: success or failure (an
        example of such an experiment is flipping a coin).  The geometric distribution
        models the number of trials that must be run in order to achieve success.
        It is therefore supported on the positive integers, ``k = 1, 2, ...``.
        The probability mass function of the geometric distribution is

        .. math:: f(k) = (1 - p)^{k - 1} p

        where *p* is the probability of success of an individual trial.

        Parameters
        ----------
        p : float
            The probability of success of an individual trial.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized geometric distribution.

        Restriction
        -----------
        * If *p* is neither a scalar nor None : *NotImplementedError* occurs.

        Examples
        --------
        Draw ten thousand values from the geometric distribution,
        with the probability of an individual success equal to 0.35:

        >>> import nlcpy as vp
        >>> z = vp.random.geometric(p=0.35, size=10000)

        How many trials succeeded after a single run?

        >>> vp.sum(z == 1) / 10000  # doctest: +SKIP
        array(0.3527)  # random
        """
        self._is_number(p)
        if p <= 0 or p > 1:
            raise ValueError('p <= 0, p > 1 or p contains NaNs')
        _size = size if size is not None else asarray(p).shape
        return self._generate_random_geometric(p, size=size, dtype=int)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        """Draws samples from a Gumbel distribution.

        Draws samples from a Gumbel distribution with specified location and scale.  For
        more information on the Gumbel distribution, see Notes and References below.

        Parameters
        ----------
        loc : float, optional
            The location of the mode of the distribution. Default is 0.
        scale : float, optional
            The scale parameter of the distribution. Default is 1. Must be non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized Gumbel distribution.

        Restriction
        -----------
        * If *loc* is neither a scalar nor None : *NotImplementedError* occurs.
        * If *scale* is neither a scalar nor None : *NotImplementedError* occurs.

        Note
        ----
        The probability density for the Gumbel distribution is

        .. math::
            p(x) = \\frac{e^{-(x - \\mu)/ \\beta}}{\\beta}
                   e^{ -e^{-(x - \\mu)/ \\beta}},

        where :math:`\\mu` is the mode, a location parameter, and :math:`\\beta` is the
        scale parameter.

        The function has a mean of :math:`\\mu + 0.57721\\beta` and a variance of
        :math:`\\frac{\\pi^2}{6}\\beta^2`.

        See Also
        --------
        RandomState.weibull : Draws samples from a Weibull distribution.

        Examples
        --------
        .. plot::
            :align: center

            Draw samples from the distribution:

            >>> import nlcpy as vp
            >>> mu, beta = 0, 0.1 # location and scale
            >>> s = vp.random.gumbel(mu, beta, 1000)

            Display the histogram of the samples, along with the probability
            density function:

            >>> import matplotlib.pyplot as plt
            >>> count, bins, ignored = plt.hist(s.get(), 30, density=True)
            >>> plt.plot(bins, (1/beta)*vp.exp(-(bins - mu)/beta)*
            ... vp.exp( -vp.exp( -(bins - mu) /beta) ),
            ... linewidth=2, color='r') # doctest: +SKIP
            >>> plt.show()
        """
        self._is_number(loc, scale)
        if scale < 0:
            raise ValueError('scale < 0')

        # ASL Lib is different numpy : add [Result * -1]
        ret = self._generate_random_gumbel(loc, scale, size=size, dtype=float)
        return (ret * -1 if ret.size > 1 or ret != 0 else ret)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        """Draws samples from a logistic distribution.

        Samples are drawn from a logistic distribution with specified parameters, loc
        (location or mean, also median), and scale (>0).

        Parameters
        ----------
        loc : float, optional
            Parameter of the distribution. Default is 0.
        scale : float, optional
            Parameter of the distribution. Must be non-negative. Default is 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized logistic distribution.

        Restriction
        -----------
        * If *loc* is neither a scalar nor None : *NotImplementedError* occurs.
        * If *scale* is neither a scalar nor None : *NotImplementedError* occurs.

        Note
        ----
        The probability density for the Logistic distribution is

        .. math::
            P(x) = \\frac{e^{-(x-\\mu)/s}}{s(1+e^{-(x-\\mu)/s})^2},

        where :math:`\\mu` = location and :math:`s` = scale.

        Examples
        --------
        .. plot::
            :align: center

            Draw samples from the distribution:

            >>> import nlcpy as vp
            >>> loc, scale = 10, 1
            >>> s = vp.random.logistic(loc, scale, 10000)
            >>> import matplotlib.pyplot as plt
            >>> count, bins, ignored = plt.hist(s.get(), bins=50)

            Plot against distribution

            >>> def logist(x, loc, scale):
            ...     return vp.exp((loc-x)/scale)/(scale*(1+vp.exp((loc-x)/scale))**2)
            >>> lgst_val = logist(bins, loc, scale)
            >>> plt.plot(bins, lgst_val * count.max() / lgst_val.max()) # doctest: +SKIP
            >>> plt.show()
        """
        self._is_number(loc, scale)
        _size = size if size is not None else broadcast(
            asarray(loc), asarray(scale)).shape
        return self._generate_random_logistic(
            loc, scale, size=_size, dtype=float)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        """Draws samples from a log-normal distribution.

        Draws samples from a log-normal distribution with specified mean, standard
        deviation, and array shape.  Note that the mean and standard deviation are not
        the values for the distribution itself, but of the underlying normal distribution
        it is derived from.

        Parameters
        ----------
        mean : float, optional
            Mean value of the underlying normal distribution. Default is 0.
        sigma : float, optional
            Standard deviation of the underlying normal distribution. Must be
            non-negative. Default is 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized log-normal distribution.

        Restriction
        -----------
        * If *mean* is neither a scalar nor None : *NotImplementedError* occurs.
        * If *sigma* is neither a scalar nor None : *NotImplmentedError* occurs.

        Note
        ----
        A variable x has a log-normal distribution if log(x) is normally distributed.
        The probability density function for the log-normal distribution is:

        .. math::
            p(x) = \\frac{1}{\\sigma x
                   \\sqrt{2\\pi}}e^{(-\\frac{(ln(x)-\\mu)^2}{2\\sigma^2})}

        where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation of
        the normally distributed logarithm of the variable.

        Examples
        --------
        .. plot::
            :align: center

            Draw samples from the distribution:

            >>> import nlcpy as vp
            >>> mu, sigma = 3., 1. # mean and standard deviation
            >>> s = vp.random.lognormal(mu, sigma, 1000)

            Display the histogram of the samples, along with the probability
            density function:

            >>> import matplotlib.pyplot as plt
            >>> count, bins, ignored = plt.hist(s.get(), 100, density=True, align='mid')

            >>> x = vp.linspace(min(bins), max(bins), 10000)
            >>> pdf = (vp.exp(-(vp.log(x) - mu)**2 / (2 * sigma**2))
            ...        / (x * sigma * vp.sqrt(2 * vp.pi)))

            >>> plt.plot(x, pdf, linewidth=2, color='r')  # doctest: +SKIP
            >>> plt.axis('tight')   # doctest: +SKIP
            >>> plt.show()
        """
        self._is_number(mean, sigma)
        if sigma < 0:
            raise ValueError('sigma < 0')
        return self._generate_random_lognormal(
            mean, sigma, size=size, dtype=float)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Draws random samples from a normal (Gaussian) distribution.

        The probability density function of the normal distribution, first derived by De
        Moivre and 200 years later by both Gauss and Laplace independently, is often
        called the bell curve because of its characteristic shape (see the example
        below).
        The normal distributions occurs often in nature.  For example, it describes the
        commonly occurring distribution of samples influenced by a large number of tiny,
        random disturbances, each with its own unique distribution.

        Parameters
        ----------
        loc : float
            Mean ("centre") of the distribution.
        scale : float
            Standard deviation (spread or "width") of the distribution. Must be
            non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized normal distribution.

        Restriction
        -----------
        * If *loc* is neither a scalar nor None : *NotImplementedError* occurs.
        * If *scale* is neither a scalar nor None : *NotImplementedError* occurs.

        Note
        ----
        The probability density for the Gaussian distribution is

        .. math::
            p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}
                   e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2}},

        where :math:`\\mu` is the mean and :math:`\\sigma` the standard deviation.
        The square of the standard deviation, :math:`\\sigma^2`, is called the
        variance.

        The function has its peak at the mean, and its "spread" increases with the
        standard deviation (the function reaches 0.607 times its maximum at
        nlcpy.random.normal is more likely to return samples lying close to the mean,
        rather than those far away.

        Examples
        --------
        .. plot::
            :align: center

            Draw samples from the distribution:

            >>> import nlcpy as vp
            >>> mu, sigma = 0, 0.1 # mean and standard deviation
            >>> s = vp.random.normal(mu, sigma, 1000)

            Verify the mean and the variance:

            >>> abs(mu - vp.mean(s)) # doctest: +SKIP
            array(0.00206415)  # may vary
            >>> abs(sigma - vp.std(s, ddof=1)) # doctest: +SKIP
            array(0.00133596)  # may vary

            Display the histogram of the samples, along with the probability
            density function:

            >>> import matplotlib.pyplot as plt
            >>> count, bins, ignored = plt.hist(s.get(), 30, density=True)
            >>> plt.plot(bins, 1/(sigma * vp.sqrt(2 * vp.pi)) *
            ...          vp.exp( - (bins - mu)**2 / (2 * sigma**2) ),
            ...          linewidth=2, color='r') # doctest: +SKIP
            >>> plt.show()

            Two-by-four array of samples from N(3, 6.25):

            >>> vp.random.normal(3, 2.5, size=(2, 4))    # doctest: +SKIP
            array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
                   [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random


        """
        self._is_number(loc, scale)
        if scale < 0:
            raise ValueError('scale < 0')
        return self._generate_random_normal(loc, scale, size=size, dtype=float)

    def poisson(self, lam=1.0, size=None):
        """Draws samples from a Poisson distribution.

        The Poisson distribution is the limit of the binomial distribution for large N.

        Parameters
        ----------
        lam : float
            Expectation of interval, must be >= 0. A sequence of expectation intervals
            must be broadcastable over the requested size.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized Poisson distribution.

        Restriction
        -----------
        * If *lam* is neither a scalar nor None : *NotImplementedError* occurs.

        Note
        ----
        The Poisson distribution

        .. math::
            f(k; \\lambda)=\\frac{\\lambda^ke^{-\\lambda}}{k!}

        For events with an expected separation :math:`\\lambda` the Poisson distribution
        :math:`f(k; \\lambda)` describes the probability of :math:`k` events occurring
        within the observed interval :math:`\\lambda`.

        Because the output is limited to the range of the C int64 type, a ValueError is
        raised when lam is within 10 sigma of the maximum representable value.

        Examples
        --------
        .. plot::
            :align: center

            Draw samples from the distribution:

            >>> import nlcpy as vp
            >>> s = vp.random.poisson(5, 10000)

            Display histogram of the sample:

            >>> import matplotlib.pyplot as plt
            >>> count, bins, ignored = plt.hist(s.get(), 14, density=True)
            >>> plt.show()
        """
        self._is_number(lam)
        if lam < 0 or lam is nan:
            raise ValueError('lam < 0 or lam is NaN')
        if lam > numpy.iinfo('l').max - numpy.sqrt(numpy.iinfo('l').max) * 10:
            raise ValueError('lam value too large')
        _size = size if size is not None else asarray(lam).shape
        return self._generate_random_poisson(lam, size=_size, dtype=int)

    def standard_cauchy(self, size=None):
        """Draws samples from a standard Cauchy distribution with mode = 0.

        Also known as the Lorentz distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        samples : ndarray
            The drawn samples.

        Note
        ----
        The probability density function for the full Cauchy distribution is

        .. math::
            P(x; x_0, \\gamma) = \\frac{1}{\\pi \\gamma
                                 \\bigl[ 1+(\\frac{x-x_0}{\\gamma})^2
                                 \\bigr] }

        and the Standard Cauchy distribution just sets :math:`x_0 = 0` and
        :math:`\\gamma = 1`

        Examples
        --------
        .. plot::
            :align: center

            Draw samples and plot the distribution:

            >>> import nlcpy as vp
            >>> import matplotlib.pyplot as plt
            >>> s = vp.random.standard_cauchy(1000000)

            >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
            >>> plt.hist(s.get(), bins=100) # doctest: +SKIP
            >>> plt.show()
        """
        return self._generate_random_cauchy(
            a=0.0, b=1.0, size=size, dtype=float)

    def standard_exponential(self, size=None):
        """Draws samples from a standard exponential distribution.

        standard_exponential is identical to the exponential distribution with a scale
        parameter of 1.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples.

        See Also
        --------
        RandomState.exponential : Draws samples from an exponential distribution.

        Examples
        --------
        Output a 3x8000 array:

        >>> import nlcpy as vp
        >>> n = vp.random.standard_exponential((3, 8000))
        """
        return self.exponential(scale=1.0, size=size)

    def standard_gamma(self, shape, size=None):
        """Draws samples from a standard Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters, shape
        (sometimes designated "k") and scale=1.

        Parameters
        ----------
        shape : float
            Parameter, must be non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized standard gamma distribution.

        Restriction
        -----------
        * If *shape* is neither a scalar nor None : *NotImplementedError* occurs.

        Note
        ----
        The probability density for the Gamma distribution is

        .. math::
            p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},

        where :math:`k` is the shape and :math:`\\theta` the scale, and :math:`\\Gamma`
        is the Gamma function.

        Examples
        --------
        .. plot::
            :align: center

            Draw samples from the distribution:

            >>> import nlcpy as vp
            >>> shape, scale = 2., 1. # mean and width
            >>> s = vp.random.standard_gamma(shape, 1000000)

            Display the histogram of the samples, along with the probability
            density function:

            >>> import matplotlib.pyplot as plt
            >>> import scipy.special as sps
            >>> count, bins, ignored = plt.hist(s.get(), 50, density=True)
            >>> y = bins**(shape-1) * ((vp.exp(-bins/scale))/
            ...                       (sps.gamma(shape) * scale**shape))
            >>> plt.plot(bins, y, linewidth=2, color='r') # doctest: +SKIP
            >>> plt.show()
        """
        return self.gamma(shape, scale=1.0, size=size)

    def standard_normal(self, size=None):
        """Draws samples from a standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            A floating-point array of shape ``size`` of drawn samples, or a single sample
            if ``size`` was not specified.

        Note
        ----
        For random samples from :math:`N(\\mu, \\sigma^2)`, use one of::

            vp mu + sigma * vp.random.standard_normal(size=...)
            vp.random.normal(mu, sigma, size=...)

        See Also
        --------
        RandomState.normal : Draws random samples from a normal (Gaussian) distribution.

        Examples
        --------
        >>> import nlcpy as vp
        >>> vp.random.standard_normal()  # doctest: +SKIP
        array(2.96222821)
        >>> s = vp.random.standard_normal(8000)
        >>> s   # doctest: +SKIP
        array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311,  # random
               -0.38672696, -0.4685006 ])                                # random
        >>> s.shape
        (8000,)
        >>> s = vp.random.standard_normal(size=(3, 4, 2))
        >>> s.shape
        (3, 4, 2)

        Two-by-four array of samples from N(3, 6.25):

        >>> 3 + 2.5 * vp.random.standard_normal(size=(2, 4))   # doctest: +SKIP
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random
        """
        return self.normal(loc=0.0, scale=1.0, size=size)

    def uniform(self, low=0.0, high=1.0, size=None):
        """Draws samples from a uniform distribution.

        Samples are uniformly distributed over the half-open interval ``[low, high)``
        (includes low, but excludes high).  In other words,
        any value within the given interval is equally likely to be drawn by uniform.

        Parameters
        ----------
        low : float, optional
            Lower boundary of the output interval.  All values generated will be greater
            than or equal to low.  The default value is 0.
        high : float
            Upper boundary of the output interval.  All values generated will be less
            than high.  The default value is 1.0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized uniform distribution.

        Restriction
        -----------
        * If *low* is neither a scalar nor None : *NotImplementedError* occurs.
        * If *high* is neither a scalar nor None : *NotImplementedError* occurs.

        Note
        ----
        The probability density function of the uniform distribution is

        .. math::
            p(x) = \\frac{1}{b - a}

        anywhere within the interval ``[a, b)``, and zero elsewhere.
        When ``high`` == ``low``, values of ``low`` will be returned.

        See Also
        --------
        RandomState.randint : Returns random integers from low (inclusive)
                              to high (exclusive).
        RandomState.random_integers : Random integers of type vp.int between
                                      low and high, inclusive.
        RandomState.random_sample : Returns random floats in the half-open
                                    interval ``[0.0, 1.0)``.
        RandomState.random : Returns random floats in the half-open interval
                             ``[0.0, 1.0)``.
        RandomState.rand : Random values in a given shape.

        Examples
        --------
        .. plot::
            :align: center

            Draw samples from the distribution:

            >>> import nlcpy as vp
            >>> s = vp.random.uniform(-1,0,1000)

            All values are within the given interval:

            >>> vp.all(s >= -1)
            array(True)
            >>> vp.all(s < 0)
            array(True)

            Display the histogram of the samples, along with the probability
            density function:

            >>> import matplotlib.pyplot as plt
            >>> count, bins, ignored = plt.hist(s.get(), 15, density=True)
            >>> plt.plot(bins, vp.ones_like(bins),
            ... linewidth=2, color='r') # doctest: +SKIP
            >>> plt.show()
        """
        rand = self._generate_random_uniform(size=size, dtype=float)

        # isscalar is not nlcpy. use numpy.
        if not numpy.isscalar(low):
            low = ndarray(low, float)
        if not numpy.isscalar(high):
            high = ndarray(high, float)
        rand = (high - low) * rand + low
        return rand

    def weibull(self, a, size=None):
        """Draws samples from a Weibull distribution.

        Draws samples from a 1-parameter Weibull distribution with the given shape
        parameter *a*.

        .. math:: X = (-ln(U))^{1/a}

        Here, U is drawn from the uniform distribution over ``(0,1]``.
        The more common 2-parameter Weibull, including a scale parameter
        :math:`\\lambda` is just

        .. math:: X = \\lambda(-ln(U))^{1/a}

        Parameters
        ----------
        a : float
            Shape parameter of the distribution.  Must be nonnegative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
            samples are drawn.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized Weibull distribution.

        Restriction
        -----------
        * If *a* is neither a scalar nor None : *NotImplementedError* occurs.

        Note
        ----
        The probability density for the Weibull distribution is

        .. math::
            p(x) = \\frac{a}{\\lambda}(\\frac{x}{\\lambda})^{a-1}
                   e^{-(x/\\lambda)^a},

        where :math:`a` is the shape and :math:`\\lambda` the scale.

        The function has its peak (the mode) at :math:`\\lambda(\\frac{a-1}{a})^{1/a}`.

        When ``a = 1``, the Weibull distribution reduces to the exponential distribution.

        See Also
        --------
        RandomState.gumbel : Draws samples from a Gumbel distribution.

        Examples
        --------
        .. plot::
            :align: center

            Draw samples from the distribution:

            >>> import nlcpy as vp
            >>> a = 5. # shape
            >>> s = vp.random.weibull(a, 1000)

            Display the histogram of the samples, along with the probability
            density function:

            >>> import matplotlib.pyplot as plt
            >>> x = vp.arange(1,100.)/50.
            >>> def weib(x,n,a):
            ...     return (a / n) * (x / n)**(a - 1) * vp.exp(-(x / n)**a)

            >>> count, bins, ignored = plt.hist(s.get())
            >>> x = vp.arange(1,100.)/50.
            >>> scale = count.max()/weib(x, 1., 5.).max()
            >>> plt.plot(x, weib(x, 1., 5.)*scale) # doctest: +SKIP
            >>> plt.show()
        """
        self._is_number(a)
        if a < 0:
            raise ValueError('a < 0')

        _size = size if size is not None else asarray(a).shape
        return self._generate_random_weibull(a, b=1.0, size=size, dtype=float)

    def seed(self, seed=None):
        """Reseeds a default bit generator(MT19937), which provide a stream of random
        bits.

        Note
        ----
        This is a convenience, legacy function. The best practice is to **not** reseed a
        BitGenerator, rather to recreate a new one. This method is here for legacy
        reasons. This example demonstrates best practice.

        Examples
        --------
        >>> import nlcpy as vp
        >>> rs = vp.random.RandomState(123456789) # doctest: +SKIP
        # Later, you want to restart the stream
        >>> rs.seed(987654321)     # doctest: +SKIP

        """
        if seed is None:
            import random
            r = random.randint(0, self._asl_seed_max)
            self._ve_seed = nlcpy.array(r, dtype='u4')
        else:
            if isinstance(seed, nlcpy.random.BitGenerator):
                self._ve_seed = nlcpy.asarray(seed.entropy)
            else:
                self._ve_seed = nlcpy.asarray(seed)
            if self._ve_seed.size == 0:
                raise ValueError("Seed must be non-empty")
            if self._ve_seed.ndim > 1:
                raise ValueError("Seed array must be 1-d")
            if nlcpy.any(self._ve_seed < 0) or nlcpy.any(
                    self._ve_seed > self._asl_seed_max):
                raise ValueError('Seed must be between 0 and 2**32 - 1')
            self._ve_seed = self._ve_seed.astype(dtype='u4', copy=False)

        fpe = request._get_fpe_flag()
        args = (self._ve_seed._ve_array,
                veo.OnStack(fpe, inout=veo.INTENT_OUT))
        request._push_and_flush_request(
            'nlcpy_random_set_seed',
            args,
            callback=self._asl_error_check
        )
        self._vh_seed = self._ve_seed.get()

    def _update_vh_seed(self, size):
        self._vh_seed = self._vh_seed + size
        self._vh_seed %= self._asl_seed_max

    def _is_number(self, *param):
        for p in param:
            if not isinstance(p, numbers.Number):
                raise NotImplementedError('Array is not supported')

    def _asl_error_check(self, ret):
        if ret != ASL_ERROR_OK:
            if ret == ASL_ERROR_ARGUMENT:
                raise RuntimeError('[ASL_ERR] incorrect argument')
            elif ret == ASL_ERROR_LIBRARY_UNINITIALIZED:
                raise RuntimeError('[ASL_ERR] library not initialized')
            elif ret == ASL_ERROR_RANDOM_INVALID:
                raise RuntimeError(
                    '[ASL_ERR] invalid random number generator handle')
            elif ret == ASL_ERROR_MEMORY:
                raise RuntimeError('[ASL_ERR] out of memory')
            elif ret == ASL_ERROR_MPI:
                raise RuntimeError(
                    '[ASL_ERR] MPI error (only when parallelly excuted)')
            elif ret == ASL_ERROR_RANDOM_INCOMPATIBLE_CALL:
                raise RuntimeError(
                    '[ASL_ERR] not in a continuous distribution')
            else:
                raise RuntimeError(
                    '[ASL_ERR] unexpected err code=%s' %
                    str(ret))

    def _asl_get_state(self):
        shape = request._push_and_flush_request(
            'nlcpy_random_get_state_size',
            (),
            callback=None,
            sync=True
        )

        state = ndarray(shape, dtype=nlcpy.uint32)

        fpe = request._get_fpe_flag()
        args = (state._ve_array, veo.OnStack(fpe, inout=veo.INTENT_OUT))
        request._push_and_flush_request(
            'nlcpy_random_save_state',
            args,
            callback=self._asl_error_check,
        )
        return state

    def _asl_set_state(self, state):
        fpe = request._get_fpe_flag()
        args = (state._ve_array, veo.OnStack(fpe, inout=veo.INTENT_OUT))
        request._push_and_flush_request(
            'nlcpy_random_restore_state',
            args,
            callback=self._asl_error_check
        )
        return

    def _generate_random_uniform(self, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        out = ndarray(shape=size, dtype=dtype)

        fpe = request._get_fpe_flag()
        args = (out._ve_array, veo.OnStack(fpe, inout=veo.INTENT_OUT))
        request._push_and_flush_request(
            'nlcpy_random_generate_uniform_f64',
            args,
            callback=self._asl_error_check
        )
        return out

    def _generate_random_integers(self, low, high, size=None, dtype=int):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        out = ndarray(shape=size, dtype=dtype)
        work = ndarray(shape=size, dtype=nlcpy.uint64)

        fpe = request._get_fpe_flag()

        args = (
            out._ve_array,
            work._ve_array,
            low,
            high - low,
            veo.OnStack(fpe, inout=veo.INTENT_OUT))
        name = 'nlcpy_random_generate_integers' \
            if numpy.dtype(dtype).name not in _unsigned_integers_types \
            else 'nlcpy_random_generate_unsigned_integers'

        request._push_and_flush_request(
            name,
            args,
            callback=self._asl_error_check,
        )

        return out

    def _generate_random_normal(self, loc, scale, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        if scale == 0.:
            return nlcpy.zeros(size, dtype=int)

        out = ndarray(shape=size, dtype=dtype)
        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            loc,
            scale,
            veo.OnStack(fpe, inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_normal_f64',
            args,
            callback=self._asl_error_check
        )
        return out

    def _generate_random_gamma(self, shape, scale, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        if shape == 0. or scale == 0.:
            return nlcpy.zeros(size, dtype=int)

        out = ndarray(shape=size, dtype=dtype)

        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            shape,
            1 / scale,
            veo.OnStack(fpe, inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_gamma_f64',
            args,
            callback=self._asl_error_check
        )

        return out

    def _generate_random_poisson(self, lam, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        if lam == 0:
            return nlcpy.zeros(size, int)

        out = ndarray(shape=size, dtype=dtype)

        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            lam,
            veo.OnStack(fpe, inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_poisson_f64',
            args,
            callback=self._asl_error_check
        )

        return out

    def _generate_random_logistic(self, loc, scale, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        out = ndarray(shape=size, dtype=dtype)
        fpe = request._get_fpe_flag()

        args = (
            out._ve_array,
            loc,
            scale,
            veo.OnStack(
                fpe,
                inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_logistic_f64',
            args,
            callback=self._asl_error_check
        )

        return out

    def _generate_random_weibull(self, a, b, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        if a == 0.:
            return nlcpy.zeros(size, dtype=int)

        out = ndarray(shape=size, dtype=dtype)

        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            a,
            b,
            veo.OnStack(
                fpe,
                inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_weibull_f64',
            args,
            callback=self._asl_error_check
        )

        return out

    def _generate_random_exponential(self, scale=1.0, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        if scale == 0.:
            return nlcpy.zeros(size, dtype=int)

        out = ndarray(shape=size, dtype=dtype)

        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            scale,
            veo.OnStack(
                fpe,
                inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_exponential_f64',
            args,
            callback=self._asl_error_check
        )

        return out

    def _generate_random_cauchy(self, a=0.0, b=1.0, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        out = ndarray(shape=size, dtype=dtype)

        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            a,
            b,
            veo.OnStack(
                fpe,
                inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_cauchy_f64',
            args,
            callback=self._asl_error_check
        )

        return out

    def _generate_random_lognormal(
            self, mean=0.0, sigma=1.0, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        if sigma == 0.:
            return nlcpy.ones(size, dtype)

        out = ndarray(shape=size, dtype=dtype)

        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            mean,
            sigma,
            veo.OnStack(
                fpe,
                inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_lognormal_f64',
            args,
            callback=self._asl_error_check
        )

        return out

    def _generate_random_gumbel(
            self, loc=0.0, scale=1.0, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        if scale == 0.:
            return nlcpy.zeros(size, dtype=int)

        out = ndarray(shape=size, dtype=dtype)

        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            loc,
            scale,
            veo.OnStack(
                fpe,
                inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_gumbel_f64',
            args,
            callback=self._asl_error_check
        )

        return out

    def _generate_random_geometric(self, p, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        out = ndarray(shape=size, dtype=dtype)

        fpe = request._get_fpe_flag()
        args = (out._ve_array, p, veo.OnStack(fpe, inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_geometric_f64',
            args,
            callback=self._asl_error_check
        )

        return out + 1

    def _generate_random_binomial(self, n, p, size=None, dtype=float):
        if size is None:
            size = ()
        if not numpy.isscalar(size) and len(size) == 0:
            size = ()

        if n == 0 or p == 0 or p == 1:
            return nlcpy.zeros(size, int)

        out = ndarray(shape=size, dtype=dtype)

        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            n,
            p,
            veo.OnStack(
                fpe,
                inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_binomial_f64',
            args,
            callback=self._asl_error_check
        )

        return out

    def _generate_random_uniform_for_generator(self, size=None, out=None):
        fpe = request._get_fpe_flag()
        args = (out._ve_array, veo.OnStack(fpe, inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_uniform_f64',
            args,
            callback=self._asl_error_check
        )

        return

    def _generate_random_exponential_for_generator(
            self, scale=1.0, size=None, out=None):
        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            scale,
            veo.OnStack(
                fpe,
                inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_exponential_f64',
            args,
            callback=self._asl_error_check
        )

        return

    def _generate_random_normal_for_generator(
            self, loc=0.0, scale=1.0, size=None, out=None):
        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            loc,
            scale,
            veo.OnStack(
                fpe,
                inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_normal_f64',
            args,
            callback=self._asl_error_check
        )

        return

    def _generate_random_gamma_for_generator(
            self, shape, scale=1.0, size=None, out=None):
        fpe = request._get_fpe_flag()
        args = (
            out._ve_array,
            shape,
            1 / scale,
            veo.OnStack(
                fpe,
                inout=veo.INTENT_OUT))

        request._push_and_flush_request(
            'nlcpy_random_generate_gamma_f64',
            args,
            callback=self._asl_error_check
        )

        return

    def _generate_random_shuffle(self, x, axis=0):
        if x.ndim == 0 or x.size == 1:
            return
        for i in range(x.ndim):
            if x._shape[i] == 0:
                return

        fpe = request._get_fpe_flag()
        if axis < 0:
            axis += x.ndim
        if axis < -x.ndim or axis >= x.ndim:
            raise AxisError(
                f'axis {axis} is out of bounds for array of dimension {x.ndim}')

        np_state = numpy.random.get_state()
        numpy.random.seed(self._vh_seed)
        self._update_vh_seed(x.size)

        shuffle_idx = numpy.arange(x.shape[axis], dtype='i8')
        numpy.random.shuffle(shuffle_idx)
        shuffle_idx = nlcpy.array(shuffle_idx, dtype='i8')
        shuffle_work = x.copy()

        numpy.random.set_state(np_state)

        request._push_request(
            "nlcpy_random_shuffle",
            "random_op",
            (x, shuffle_idx, shuffle_work, axis),
        )
        return

    def _generate_random_permutation(self, x, axis=0):
        if isinstance(x, (int, nlcpy.integer)):
            arr = nlcpy.arange(x)
            self._generate_random_shuffle(arr, axis)
            return arr

        arr = nlcpy.asarray(x)
        if arr.ndim < 1:
            raise IndexError("x must be an integer or at least 1-dimensional")

        if nlcpy.may_share_memory(arr, x):
            arr = nlcpy.array(arr)
        self._generate_random_shuffle(arr, axis)
        return arr


_rand = RandomState()

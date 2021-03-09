# distutils: language = c++
#
# * The source code in this file is based on the soure code of NumPy.
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
import numpy
import numbers
import warnings
import copy
import nlcpy
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport internal
from nlcpy.core.core cimport MemoryLocation
from nlcpy.core cimport core
from nlcpy.core cimport manipulation
from nlcpy.core cimport broadcast
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.core cimport dtype as _dtype
from nlcpy import veo
from nlcpy.manipulation.shape import reshape
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.statistics.function_base import *
from nlcpy.wrapper.numpy_wrap import numpy_wrap
cimport cython
cimport cpython


@numpy_wrap
def histogram(a, bins=10, range=None, weights=None, density=None):
    """Computes the histogram of a set of data.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
        If *bins* is an int, it defines the number of equal-width bins in the given range
        (10, by default). If *bins* is a sequence, it defines a monotonically increasing
        array of bin edges, including the rightmost edge, allowing for non-uniform bin
        widths. If *bins* is a string, it defines the method used to calculate the
        optimal bin width, as defined by :func:`histogram_bin_edges`.
    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is simply
        ``(a.min(), a.max()).`` Values outside the range are ignored. The first element
        of the range must be less than or equal to the second. *range* affects the
        automatic bin computation as well. While bin width is computed to be optimal
        based on the actual data within *range*, the bin count will fill the entire
        range including portions containing no data.
    weights : array_like, optional
        An array of weights, of the same shape as *a*. Each value in a only contributes
        its associated weight towards the bin count (instead of 1). If *density* is True,
        the weights are normalized, so that the integral of the density over the range
        remains 1.
    density : bool, optional
        If ``False``, the result will contain the number of samples in each bin. If
        ``True``, the result is the value of the probability *density* function at the
        bin, normalized such that the integral over the range is 1. Note that the sum of
        the histogram values will not be equal to 1 unless bins of unity width are
        chosen; it is not a probability mass function.

    Returns
    -------
    hist :  ndarray
        The values of the histogram. See *density* and *weights* for a description of the
        possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.histogram`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    Note
    ----
    All but the last (righthand-most) bin is half-open. In other words, if *bins* is::

        [1, 2, 3, 4]

    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and the second ``[2,
    3)``. The last bin, however, is ``[3, 4]``, which includes 4.

    See Also
    --------
    histogramdd : Computes the multidimensional histogram of some data.
    bincount : Counts number of occurrences of each value in array of non-negative ints.
    digitize : Returns the indices of the bins to which each value in input array
        belongs.
    histogram_bin_edges : Function to calculate only the edges of
        the bins used by the histogram function.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> vp.histogram(vp.arange(4), bins=vp.arange(5), density=True)
    (array([0.25, 0.25, 0.25, 0.25]), array([0, 1, 2, 3, 4]))
    >>> vp.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
    (array([1, 4, 1]), array([0, 1, 2, 3]))
    >>> a = vp.arange(5)
    >>> hist, bin_edges = vp.histogram(a, density=True)
    >>> hist   # doctest: +SKIP
    array([0.5, 0. , 0.5, 0. , 0. , 0.5, 0. , 0.5, 0. , 0.5])
    >>> vp.sum(hist * vp.diff(bin_edges))  # doctest: +SKIP
    array(1.)

    """
    raise NotImplementedError


@numpy_wrap
def histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None):
    """Computes the bi-dimensional histogram of two data samples.

    Parameters
    ----------
    x : array_like,shape(N,)
        An array containing the x coordinates of the points to be histogrammed.
    y : array_like,shape(N,)
        An array containing the y coordinates of the points to be histogrammed.
    bins : int or array_like or [int, int] or [array, array], optional
        The bin specification:

        * If int, the number of bins for the two dimensions (nx=ny=bins).
        * If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        * If [int, int], the number of bins in each dimension (nx, ny = bins).
        * If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        * A combination [int, array] or [array, int], where int is the number of bins
          and array is the bin edges.
    range : array_like, shape(2,2), optional
        The leftmost and rightmost edges of the bins along each dimension (if not
        specified explicitly in the *bins* parameters):
        ``[[xmin, xmax], [ymin, ymax]]``.
        All values outside of this range will be considered outliers and not tallied in
        the histogram.
    normed : bool, optional
        An alias for the density argument that behaves identically. To avoid confusion
        with the broken normed argument to :func:`histogram`, *density* should be
        preferred.
    weights : array_like, shape(N,),optional
        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``. Weights are
        normalized to 1 if *normed* is True. If normed is False, the values of the
        returned histogram are equal to the sum of the weights belonging to the samples
        falling into each bin.
    density :  bool, optional
        If False, the default, returns the number of samples in each bin. If True,
        returns the probability density function at the bin,
        ``bin_count / sample_count / bin_area.``

    Returns
    -------
    H : ndarray,shape(nx, ny)
        The bi-dimensional histogram of samples *x* and *y*. Values in *x* are
        histogrammed along the first dimension and values in *y* are histogrammed along
        the second dimension.
    xedges : ndarray, shape(nx+1,)
        The bin edges along the first dimension.
    yedges : ndarray, shape(ny+1,)
        The bin edges along the second dimension.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.histogram2d`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    Note
    ----
    When normed is True, then the returned histogram is the sample density, defined such
    that the sum over bins of the product ``bin_value * bin_area`` is 1.

    Please note that the histogram does not follow the Cartesian convention where *x*
    values are on the abscissa and *y* values on the ordinate axis. Rather, *x* is
    histogrammed along the first dimension of the array (vertical), and *y* along the
    second dimension of the array (horizontal). This ensures compatibility with
    :func:`histogramdd`.

    See Also
    --------
    histogram : Computes the histogram of a set of data.
    histogramdd : Computes the multidimensional histogram of some data.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.random.seed(42)
    >>> z = vp.random.randn(2,50)
    >>> H, xedges,yedges = vp.histogram2d(z[0],z[1], bins=5)
    >>> H.shape, xedges[0].size, yedges[0].size
    ((5, 5), 1, 1)

    """
    raise NotImplementedError


@numpy_wrap
def histogramdd(sample, bins=10, range=None, normed=None, weights=None, density=None):
    """Computes the multidimensional histogram of some data.

    Parameters
    ----------
    sample : (N,D)array,or(D,N)array_like
        The data to be histogrammed.
        Note the unusual interpretation of sample when an array_like:

        * When an array, each row is a coordinate in a D-dimensional space - such as
          ``histogramdd(vp.array([p1, p2, p3]))``.
        * When an array_like, each element is the list of values for single coordinate
          - such as ``histogramdd((X, Y, Z))``.

        The first form should be preferred.
    bins : sequence or int, optional
        The bin specification:

        * A sequence of arrays describing the monotonically increasing bin edges along
          each dimension.
        * The number of bins for each dimension (nx, ny...=bins)
        * The number of bins for all dimensions (nx=ny=...=bins)
    range : sequence, optional
        A sequence of length D, each an optional (lower, upper) tuple giving the outer
        bin edges to be used if the edges are not given explicitly in *bins*. An entry
        of None in the sequence results in the minimum and maximum values being used for
        the corresponding dimension. The default, None, is equivalent to passing a tuple
        of D None values.
    density : bool, optional
        If False, the default, returns the number of samples in each bin. If True,
        returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_volume``.
    normed : bool, optional
        An alias for the density argument that behaves identically. To avoid confusion
        with the broken normed argument to :func:`histogram`, *density* should be
        preferred
    weights : (N,) array_like, optional
        An array of values *w_i* weighing each sample (*x_i*, *y_i*, *z_i*, ...).
        Weights are normalized to 1 if normed is True. If normed is False, the values of
        the returned histogram are equal to the sum of the weights belonging to the
        samples falling into each bin.

    Returns
    -------
    H : (N,) array_like, optional
        The multidimensional histogram of sample x. See normed and weights for the
        different possible semantics.
    edges : list
        A list of D arrays describing the bin edges for each dimension.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.histogramdd`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    See Also
    --------
    histogram : Computes the histogram of a set of data.
    histogram2d : Computes the bi-dimensional histogram of two data samples.

    Examples
    --------
    >>> import nlcpy as vp
    >>> r = vp.random.randn(100,3)
    >>> H, edges = vp.histogramdd(r, bins = (5, 8, 4))
    >>> H.shape, edges[0].size, edges[1].size, edges[2].size
    ((5, 8, 4), 6, 9, 5)

    """
    raise NotImplementedError


@numpy_wrap
def bincount(x, weights=None, minlength=0):
    """Counts number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest value in *x*. If
    *minlength* is specified, there will be at least this number of bins in the output
    array (though it will be longer if necessary, depending on the contents of *x*).
    Each bin gives the number of occurrences of its index value in *x*. If weights is
    specified the input array is weighted by it, i.e. if a value ``n`` is found at
    position ``i, out[n] += weight[i] instead of out[n] += 1``.

    Parameters
    ----------
    x : array_like, 1 dimension, nonnegative ints
        Input array.
    weights : array_like, optional
        Weights, array of the same shape as *x*.
    minlength : int, optional
        A minimum number of bins for the output array.

    Returns
    -------
    out : ndarray of ints
        The result of binning the input array. The length of out is equal to
        ``vp.amax(x)+1``.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.bincount`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    See Also
    --------
    histogram : Computes the histogram of a set of data.
    digitize : Return the indices of the bins to which each value in input array belongs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.bincount(vp.arange(5))
    array([1, 1, 1, 1, 1])
    >>> vp.bincount(vp.array([0, 1, 1, 3, 2, 1, 7]))
    array([1, 3, 1, 1, 0, 0, 0, 1])
    >>> x = vp.array([0, 1, 1, 3, 2, 1, 7, 23])
    >>> vp.bincount(x).size == vp.amax(x)+1
    array(True)

    The input array needs to be of integer dtype, otherwise a TypeError is raised:

    >>> vp.bincount(vp.arange(5, dtype=float))   # doctest: +SKIP
    Traceback (most recent call last):
     ...
    TypeError: Cannot cast array data from dtype('float64') to dtype('int64')
    according to the rule 'safe'

    A possible use of ``bincount`` is to perform sums over variable-size chunks of an
    array, using the ``weights`` keyword.

    >>> w = vp.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
    >>> x = vp.array([0, 1, 1, 2, 2, 2])
    >>> vp.bincount(x,  weights=w)
    array([0.3, 0.7, 1.1])

    """
    raise NotImplementedError


@numpy_wrap
def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """Function to calculate only the edges of the bins used by :func:`histogram` function.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str,  optional
        If bins is an int, it defines the number of equal-width bins in the given range
        (10, by default). If *bins* is a sequence, it defines the bin edges, including
        the rightmost edge, allowing for non-uniform bin widths. If *bins* is a string
        from the list below, :func:`histogram_bin_edges` will use the method chosen to
        calculate the optimal bin width and consequently the number of bins (see Notes
        for more detail on the estimators) from the data that falls within the requested
        range. While the bin width will be optimal for the actual data in the range, the
        number of bins will be computed to fill the entire range, including the empty
        portions. For visualisation, using the 'auto' option is suggested. Weighted data
        is not supported for automated bin size selection.

        'auto'
            Maximum of the 'sturges' and 'fd' estimators. Provides good all
            around performance.
        'fd'(Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into account data
            variability and data size.
        'doane'
            An improved version of Sturges' estimator that works better with
            non-normal datasets.
        'scott'
            Less robust estimator that that takes into account data
            variability and data size.
        'stone'
            Estimator based on leave-one-out cross-validation estimate of the
            integrated squared error. Can be regarded as a generalization of Scott's
            rule.
        'rice'
            Estimator does not take variability into account, only data size.
            Commonly overestimates number of bins required.
        'sturges'
            R's default method, only accounts for data size. Only optimal
            for gaussian data and underestimates number of bins for large non-gaussian
            datasets.
        'sqrt'
            Square root (of data size) estimator, used by Excel and other
            programs for its speed and simplicity.

    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is simply
        ``(a.min(), a.max())``. Values outside the range are ignored. The first element
        of the range must be less than or equal to the second. *range* affects the
        automatic bin computation as well. While bin width is computed to be optimal
        based on the actual data within range, the bin count will fill the entire range
        including portions containing no data.
    weights : array_like, optional
        An array of weights, of the same shape as *a*. Each value in a only contributes
        its associated weight towards the bin count (instead of 1). This is currently not
        used by any of the bin estimators, but may be in the future.

    Returns
    -------
    bin_edges : array of dtype float
        The edges to pass into :func:`histogram`

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.histogram_bin_edges`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    Note
    ----
    The methods to estimate the optimal number of bins are well founded in literature,
    and are inspired by the choices R provides for histogram visualisation. Note that
    having the number of bins proportional to :math:`n^{1/3}` is asymptotically optimal,
    which is why it appears in most estimators. These are simply plug-in methods that
    give good starting points for number of bins. In the equations below, recast to bin
    width using the :func:`ptp`  of the data. The final bin count is obtained from
    ``vp.round(vp.ceil(range / h))``.

    * **'auto'(maximum of the 'sturges' and 'fd'estimators)**

      A compromise to get a good value. For small datasets the Sturges value will usually
      be chosen, while larger datasets will usually default to FD. Avoids the overly
      conservative behaviour of FD and Sturges for small and large datasets respectively.
      Switchover point is usually a.size :math:`\\approx` 1000.

    * **'fd'(Freedman Diaconis Estimator)**

      .. math::
          h=2\\frac{IQR}{n^{1/3}}

      The binwidth is proportional to the interquartile range (IQR) and inversely
      proportional to cube root of a.size. Can be too conservative for small datasets,
      but is quite good for large datasets. The IQR is very robust to outliers.

    * **'scott'**

      .. math::
          h=\\sigma\\sqrt[3]{\\frac{24*\\sqrt{n}}{n}}

      The binwidth is proportional to the standard deviation of the data and inversely
      proportional to cube root of ``x.size``. Can be too conservative for small
      datasets, but is quite good for large datasets. The standard deviation is not very
      robust to outliers. Values are very similar to the Freedman-Diaconis estimator in
      the absence of outliers.

    * **'rice'**

      .. math::
          n_{h}=2n^{1/3}

      The number of bins is only proportional to cube root of ``a.size``. It tends to
      overestimate the number of bins and it does not take into account data variability.

    * **'sturges'**

      .. math::
          n_{h}=\\log_{2}{n}+1

      The number of bins is the base 2 log of ``a.size``. This estimator assumes
      normality of data and is too conservative for larger, non-normal datasets. This is
      the default method in R's ``hist`` method.

    * **'doane'**

      .. math::
          n_{h}= 1 + \\log_{2}{(n)} + \\log_{2}{(1+\\frac{|g_{1}|}{\\sigma_{g1}}})
          g_{1}=mean[(\\frac{x-\\mu}{\\sigma})^{3}]
          \\sigma_{g1}=\\sqrt{\\frac{6(n-2)}{(n+1)(n+3)}}

      An improved version of Sturges' formula that produces better estimates for
      non-normal datasets. This estimator attempts to account for the skew of the data.

    * **'sqrt'**

      .. math::
          n_{h}=\\sqrt{n}

      The simplest and fastest estimator. Only takes into account the data size.

    See Also
    --------
    histogram : Computes the histogram of a set of data.

    Examples
    --------
    >>> import nlcpy as vp
    >>> arr = vp.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
    >>> vp.histogram_bin_edges(arr, bins='auto', range=(0, 1))
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> vp.histogram_bin_edges(arr, bins=2)
    array([0. , 2.5, 5. ])

    For consistency with histogram, an array of pre-computed bins is passed through
    unmodified:

    >>> vp.histogram_bin_edges(arr, [1, 2])
    array([1, 2])

    This function allows one set of bins to be computed, and reused across multiple
    histograms:

    >>> shared_bins = vp.histogram_bin_edges(arr, bins='auto')
    >>> shared_bins
    array([0., 1., 2., 3., 4., 5.])

    """
    raise NotImplementedError


@numpy_wrap
def digitize(x, bins, right=False):
    """Return is the indices of the bins to which each value in input array belongs.

    .. csv-table::
        :header: right, order of bins, returned index i satisfies

        ``False``, increasing, ``bins[i-1] <= x < bins[i]``
        ``True``, increasing, ``bins[i-1] < x <= bins[i]``
        ``False``, decreasing, ``bins[i-1] < x <= bins[i]``
        ``True``, decreasing, ``bins[i-1] < x <= bins[i]``

    If values in *x* are beyond the bounds of *bins*, 0 or ``len(bins)`` is returned as
    appropriate.


    Parameters
    ----------
    x : array_like
        Input array to be binned.
    bins : array_like
        Array of bins. It has to be 1-dimensional and monotonic.
    right : bool, optional
        Indicating whether the intervals include the right or the left bin edge. Default
        behavior is (right==False) indicating that the interval does not include the
        right edge. The left bin end is open in this case, i.e., bins[i-1] <= x < bins[i]
        is the default behavior for monotonically increasing bins.

    Returns
    -------
    indices : ndarray of ints
        Output array of indices, of same shape as *x*.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.digitize`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    Note
    ----
    If values in *x* are such that they fall outside the bin range, attempting to index
    bins with the indices that :func:`digitize` returns will result in an *IndexError*.

    For monotonically _increasing_ *bins*, the following are equivalent::

        vp.digitize(x, bins, right=True)

    Note that as the order of the arguments are reversed, the side must be too.

    See Also
    --------
    bincount : Counts number of occurrences of each value in array of non-negative ints.
    histogram : Computes the histogram of a set of data.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.array([0.2, 6.4, 3.0, 1.6])
    >>> bins = vp.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> inds = vp.digitize(x, bins)
    >>> inds
    array([1, 4, 3, 2])
    >>> for n in range(x.size):
    ...     print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])
    ...
    0.0 <= 0.2 < 1.0
    4.0 <= 6.4 < 10.0
    2.5 <= 3.0 < 4.0
    1.0 <= 1.6 < 2.5
    >>> x = vp.array([1.2, 10.0, 12.4, 15.5, 20.])
    >>> bins = vp.array([0, 5, 10, 15, 20])
    >>> vp.digitize(x,bins,right=True)
    array([1, 2, 3, 4, 4])
    >>> vp.digitize(x,bins,right=False)
    array([1, 3, 3, 4, 5])

    """
    raise NotImplementedError

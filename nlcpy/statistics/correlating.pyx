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


cpdef corrcoef(a, y=None, rowvar=True, bias=nlcpy._NoValue, ddof=nlcpy._NoValue):
    """Returns Pearson product-moment correlation coefficients.

    Please refer to the documentation for cov for more detail. The relationship between
    the correlation coefficient matrix, *R*, and the covariance matrix, *C*, is

    .. math::
        R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }

    The values of *R* are between -1 and 1, inclusive.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations. Each row of
        *x* represents a variable, and each column a single observation of all those
        variables. Also see *rowvar* below.
    y : array_like, optional
        An additional set of variables and observations. *y* has the same shape as *x*.
    rowvar : bool, optional
        If *rowvar* is True (default), then each row represents a variable, with
        observations in the columns. Otherwise, the relationship is transposed: each
        column represents a variable, while the rows contain observations.
    bias : _NoValue, optional
        Has no effect, do not use.
    ddof : _NoValue, optional
        Has no effect, do not use.

    Returns
    -------
    R : ndarray
        The correlation coefficient matrix of the variables.

    Restriction
    -----------
    * For complex numbers : *NotImplementedError* occurs.

    Note
    ----
    Due to floating point rounding the resulting array may not be Hermitian, the diagonal
    elements may not be 1, and the elements may not satisfy the inequality abs(a) <= 1.
    This function accepts but discards arguments bias and *ddof*.

    See Also
    --------
    cov : Covariance matrix

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.array([[1,2,1,9,10,3,2,6,7],[2,1,8,3,7,5,10,7,2]])
    >>> vp.corrcoef(x)   # doctest: +SKIP
    array([[ 1.        , -0.05640533],
           [-0.05640533,  1.        ]])
    >>> y = vp.array([2,1,1,8,9,4,3,5,7])
    >>> vp.corrcoef(x,y) # doctest: +SKIP
    array([[ 1.        , -0.05640533,  0.97094584],
           [-0.05640533,  1.        , -0.01315587],
           [ 0.97094584, -0.01315587,  1.        ]])

    """
    if bias is not nlcpy._NoValue or ddof is not nlcpy._NoValue:
        warnings.warn('bias and ddof have no effect and are deprecated',
                      DeprecationWarning)

    if not isinstance(a, nlcpy.core.core.ndarray):
        a = core.argument_conversion(a)

    if y is not None and not isinstance(y, nlcpy.core.core.ndarray):
        y = core.argument_conversion(y)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    cv_data = nlcpy.cov(a, y, rowvar=rowvar)

    if not isinstance(cv_data, nlcpy.core.core.ndarray):
        cv_data = core.argument_conversion(cv_data)

    nlcpy_chk_type(a)
    nlcpy_chk_type(y)

    try:
        diag_data = nlcpy.diag(cv_data)
    except ValueError:
        return cv_data / cv_data

    stddev = nlcpy.sqrt(diag_data)
    tmp_c1 = cv_data / stddev[:, None]
    tmp_c2 = tmp_c1 / stddev[None, :]

    ans = tmp_c2

    if ans.size == 1:
        ans_cnv1 = ans.ravel()
        ans_cnv2 = nlcpy.where(ans_cnv1 >= 1, 1, ans_cnv1)
        ans_cnv3 = nlcpy.where(ans_cnv2 <= -1, -1, ans_cnv2)
        ret = ans_cnv3[0]

    else:
        ret = nlcpy.fmin(1.0, nlcpy.fmax(-1.0, ans))

    ret = nlcpy.squeeze(ret)

    return ret


cpdef cov(a, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    """Estimates a covariance matrix, given data and weights.

    Covariance indicates the level to which two variables vary together. If we examine
    N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`, then the covariance
    matrix element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
    The element :math:`C_{ii}` is the variance of :math:`x_i`. See the notes for an
    outline of the algorithm.

    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations. Each row of
        *m* represents a variable, and each column a single observation of all those
        variables. Also see *rowvar* below.
    y : array_like, optional
        An additional set of variables and observations. *y* has the same form as that
        of *m*.
    rowvar : bool, optional
        If *rowvar* is True (default), then each row represents a variable, with
        observations in the columns. Otherwise, the relationship is transposed: each
        column represents a variable, while the rows contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the number of
        observations given (unbiased estimate). These arguments had no effect on the
        return values of the function and can be safely ignored in this and previous
        versions of nlcpy.
        If bias is True, then normalization is by ``N``.
    ddof : int, optional
        If not None the default value implied by bias is overridden. Note that ``ddof=1``
        will return the unbiased estimate, even if both fweights and aweights are
        specified, and ``ddof=0`` will return the simple average. See the notes for the
        details. The default value is None.
    fweights : array_like, int, optional
        1-D array of integer frequency weights; the number of times each observation
        vector should be repeated.
    aweights : array_like, optional
        1-D array of observation vector weights. These relative weights are typically
        large for observations considered "important" and smaller for observations
        considered less "important". If ``ddof=0`` the array of weights can be used to
        assign probabilities to observation vectors.

    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.

    Note
    ----
    Assume that the observations are in the columns of the observation array *m* and let
    ``f = fweights`` and ``a = aweights`` for brevity. The steps to compute the weighted
    covariance are as follows:

    >>> import nlcpy as vp
    >>> m = vp.arange(10, dtype=vp.float64)
    >>> f = vp.arange(10) * 2
    >>> a = vp.arange(10) ** 2.
    >>> ddof = 9 # N - 1
    >>> w = f * a
    >>> v1 = vp.sum(w)
    >>> v2 = vp.sum(w * a)
    >>> m -= vp.sum(m * w, axis=None, keepdims=True) / v1
    >>> cov = vp.dot(m * w, m.T) * v1 / (v1**2 - ddof * v2)

    Note that when ``a == 1``, the normalization factor ``v1 / (v1**2 - ddof * v2)`` goes
    over to ``1 / (vp.sum(f) - ddof)`` as it should.

    See Also
    --------
    corrcoef : Normalized covariance matrix

    Examples
    --------
    Consider two variables, :math:`x_0` and :math:`x_1`, which correlate perfectly,
    but in opposite directions:

    >>> import nlcpy as vp
    >>> x = vp.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    array([[0, 1, 2],
           [2, 1, 0]])

    Note how :math:`x_0` increases while :math:`x_1` decreases.
    The covariance matrix shows this clearly:

    >>> vp.cov(x) # doctest: +SKIP
    array([[ 1., -1.],
           [-1.,  1.]])

    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.

    Further, note how *x* and *y* are combined:

    >>> x = [-2.1, -1,  4.3]
    >>> y = [3,  1.1,  0.12]
    >>> vp.cov(x, y) # doctest: +SKIP
    array([[11.71      , -4.286     ],
           [-4.286     ,  2.14413333]])
    >>> vp.cov(x)    # doctest: +SKIP
    array(11.71)

    """

    if ddof is not None and isinstance(ddof, int) is False:
        raise ValueError("cov ddof({}) must be integer({})".format(ddof, type(ddof)))

    nlcpy_chk_type(a)

    if not isinstance(a, nlcpy.core.core.ndarray):
        a = core.argument_conversion(a)

    if y is not None and not isinstance(y, nlcpy.core.core.ndarray):
        y = core.argument_conversion(y)

    in_a = a
    if in_a.ndim > 2:
        raise ValueError("a has more than 2 dimensions")

    if y is not None:
        y = nlcpy.asarray(y)
        y = core.argument_conversion(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")

    X = nlcpy.array(in_a, ndmin=2, dtype=numpy.float64)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return nlcpy.array([]).reshape(0, 0)

    if y is not None:
        y = nlcpy.array(y, copy=False, ndmin=2, dtype=numpy.float64)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = nlcpy.concatenate((X, y), axis=0)

    if ddof is None:
        if bias is False:
            ddof = 1
        else:
            ddof = 0

    w = None
    if fweights is not None:
        fweights = nlcpy.asarray(fweights, dtype=numpy.float64)
        if fweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError("fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = nlcpy.asarray(aweights, dtype=numpy.float64)
        if aweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError("aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = nlcpy.average(X, axis=1, weights=w, returned=True)
    if w_sum.size > 1:
        w_sum = w_sum[0]

    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * nlcpy.sum(w * aweights) / w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=3)
        fact = 0.0

    if isinstance(avg, nlcpy.core.core.ndarray) is False:
        avg = core.argument_conversion(avg)

    if isinstance(w_sum, nlcpy.core.core.ndarray) is False:
        w_sum = core.argument_conversion(w_sum)

    if avg.ndim > 0:
        X -= avg[:, None]
    else:
        X -= avg

    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T

    cnv_data = nlcpy.dot(X, nlcpy.conjugate(X_T))
    ret = cnv_data * nlcpy.true_divide(1, fact)

    if ret.size == 1:
        ans_cnv1 = ret.ravel()
        ans_cnv2 = ans_cnv1[0]
    else:
        ans_cnv2 = ret

    ans_cnv2 = nlcpy.squeeze(ans_cnv2)

    return ans_cnv2


@numpy_wrap
def correlate(a, v, mode='valid'):
    """Cross-correlation of two 1-dimensional sequences.

    This function computes the correlation as generally defined in signal processing
    texts::

        c_{av}[k] = sum_n a[n+k] * conj(v[n])

    with *a* and *v* sequences being zero-padded where necessary and conj being the
    conjugate.

    Parameters
    ----------
    a,v : array_like
        Input sequences.
    mode : {'valid','same','full'}, optional

        * 'full' : By default, mode is 'full'. This returns the convolution at each point
          of overlap, with an output shape of (N+M-1,). At the end-points of the
          convolution, the signals do not overlap completely, and boundary effects may be
          seen.
        * 'same' : Mode 'same' returns output of length ``max(M, N)``. Boundary
          effects are still visible.
        * 'valid': Mode 'valid' returns output of length ``max(M, N) - min(M, N) + 1``.
          The convolution product is only given for points where the signals overlap
          completely. Values outside the signal boundary have no effect.

    Returns
    -------
    out : ndarray
        Discrete, linear convolution of *a* and *v*.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.correlate`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    Note
    ----
    The definition of correlation above is not unique and sometimes correlation may be
    defined differently. Another common definition is::

        c'_{av}[k]=sum_n a[n] conj(v[n+k])

    which is related to ``c_{av}[k] by c'_{av}[k] = c_{av}[-k]``.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.correlate([1, 2, 3], [0, 1, 0.5])
    array([3.5])
    >>> vp.correlate([1, 2, 3], [0, 1, 0.5], "same")
    array([2. , 3.5, 3. ])
    >>> vp.correlate([1, 2, 3], [0, 1, 0.5], "full")
    array([0.5, 2. , 3.5, 3. , 0. ])

    Using complex sequences:

    >>> vp.correlate([1+1j, 2, 3-1j], [0, 1, 0.5j], 'full')
    array([0.5-0.5j, 1. +0.j , 1.5-1.5j, 3. -1.j , 0. +0.j ])

    Note that you get the time reversed, complex conjugated result when the two input
    sequences change places, i.e., ``c_{va}[k] = c^{*}_{av}[-k]``:

    >>> vp.correlate([0, 1, 0.5j], [1+1j, 2, 3-1j], 'full')
    array([0. +0.j , 3. +1.j , 1.5+1.5j, 1. +0.j , 0.5+0.5j])

    """
    raise NotImplementedError

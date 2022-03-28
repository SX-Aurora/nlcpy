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
import numpy

import nlcpy
from nlcpy import core
from nlcpy.request import request


def where(condition, x=None, y=None):
    """Returns elements chosen from *x* or *y* depending on *condition*.

    Note
    ----
    When only condition is provided, this function is a shorthand for
    ``nlcpy.asarray(condition).nonzero()``. Using nonzero directly should be preferred,
    as it behaves correctly for subclasses. The rest of this documentation covers only
    the case where all three arguments are provided.

    Parameters
    ----------
    condition : array_like, bool
        Where True, yield *x*, otherwise yield *y*.
    x, y : array_like
        Values from which to choose. *x*, *y* and *condition* need to be broadcastable to
        some shape.

    Returns
    -------
    out : ndarray
        An array with elements from *x* where *condition* is True, and elements from *y*
        elsewhere.

    Note
    ----
    If all the arrays are 1-D, :func:`where` is equivalent to::

        [xv if c else yv for c, xv, yv in zip(condition, x, y)]

    See Also
    --------
    nonzero : Returns the indices of the elements
        that are non-zero.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> vp.where(a < 5, a, 10*a)
    array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

    This can be used on multidimensional arrays too:

    >>> vp.where([[True, False], [True, True]],
    ...          [[1, 2], [3, 4]],
    ...          [[9, 8], [7, 6]])
    array([[1, 8],
           [3, 4]])

    The shapes of x, y, and the condition are broadcast together:

    >>> x = vp.arange(3).reshape([3,1])
    >>> y = vp.arange(4).reshape([1,4])
    >>> vp.where(x < y, x, 10 + y)  # both x and 10+y are broadcast
    array([[10,  0,  0,  0],
           [10, 11,  1,  1],
           [10, 11, 12,  2]])
    >>> a = vp.array([[0, 1, 2],
    ...               [0, 2, 4],
    ...               [0, 3, 6]])
    >>> vp.where(a < 4, a, -1)  # -1 is broadcast
    array([[ 0,  1,  2],
           [ 0,  2, -1],
           [ 0,  3, -1]])

    """

    if condition is None:
        condition = False
    arr = nlcpy.asarray(condition)
    if x is None and y is None:
        return nlcpy.nonzero(arr)

    if x is None or y is None:
        raise ValueError("either both or neither of x and y should be given")

    if not isinstance(x, nlcpy.ndarray):
        x = numpy.asarray(x)
    if not isinstance(y, nlcpy.ndarray):
        y = numpy.asarray(y)
    ret_type = numpy.result_type(x, y)

    arr_x = nlcpy.asarray(x, dtype=ret_type)
    arr_y = nlcpy.asarray(y, dtype=ret_type)

    if arr.dtype != bool:
        arr = (arr != 0)

    values, shape = core._broadcast_core((arr, arr_x, arr_y))
    ret = nlcpy.ndarray(shape=shape, dtype=ret_type)
    request._push_request(
        "nlcpy_where",
        "indexing_op",
        (ret, values[0], values[1], values[2]),)

    return ret


def diag_indices(n, ndim=2):
    """Returns the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main diagonal of an
    array *a* with ``a.ndim >= 2`` dimensions and shape (n, n, ..., n).
    For ``a.ndim = 2`` this is the usual diagonal, for ``a.ndim > 2`` this is the set of
    indices to access ``a[i, i, ..., i]`` for ``i = [0..n-1]``.

    Parameters
    ----------
    n : int
        The size, along each dimension, of the arrays for which the returned indices can
        be used.
    ndim : int, optional
        The number of dimensions.

    Examples
    --------
    Create a set of indices to access the diagonal of a (4, 4) array:

    >>> import nlcpy as vp
    >>> di = vp.diag_indices(4)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    >>> a = vp.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> a[di] = 100
    >>> a
    array([[100,   1,   2,   3],
           [  4, 100,   6,   7],
           [  8,   9, 100,  11],
           [ 12,  13,  14, 100]])

    Now, we create indices to manipulate a 3-D array:

    >>> d3 = vp.diag_indices(2, 3)
    >>> d3
    (array([0, 1]), array([0, 1]), array([0, 1]))

    And use it to set the diagonal of an array of zeros to 1:

    >>> a = vp.zeros((2, 2, 2), dtype=int)
    >>> a[d3] = 1
    >>> a
    array([[[1, 0],
            [0, 0]],
    <BLANKLINE>
           [[0, 0],
            [0, 1]]])
    """
    idx = nlcpy.arange(n)
    return (idx,) * ndim

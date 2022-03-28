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

import nlcpy
import numpy
from nlcpy.core import manipulation


def concatenate(arrays, axis=0, out=None):
    """Joins a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays : sequence of array_like
        The arrays must have the same shape, except in the dimension corresponding to
        *axis* (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined. If axis is None, arrays are
        flattened before use. Default is 0.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be correct,
        matching that of what concatenate would have returned if no out argument were
        specified.

    Returns
    -------
    res : ndarray
        The concatenated array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1, 2], [3, 4]])
    >>> b = vp.array([[5, 6]])
    >>> vp.concatenate((a, b), axis=0)
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> vp.concatenate((a, b.T), axis=1)
    array([[1, 2, 5],
           [3, 4, 6]])
    >>> vp.concatenate((a, b), axis=None)
    array([1, 2, 3, 4, 5, 6])
    """
    return manipulation._ndarray_concatenate(arrays, axis, out)


def stack(arrays, axis=0, out=None):
    """Joins a sequence of arrays along a new axis.

    The axis parameter specifies the index of the new axis in the dimensions of the
    result. For example, if axis=0 it will be the first dimension and if axis=-1 it will
    be the last dimension.

    Parameters
    ----------
    arrays : sequence of array_like
        Each array must have the same shape.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    out :  `ndarray`, optional
        If provided, the destination to place the result. The shape must be correct,
        matching that of what stack would have returned if no out argument were
        specified.

    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.

    See Also
    --------
    concatenate : Joins a sequence of arrays along an existing axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> arrays = [vp.random.randn(3, 4) for _ in range(10)]
    >>> vp.stack(arrays, axis=0).shape
    (10, 3, 4)
    >>> vp.stack(arrays, axis=1).shape
    (3, 10, 4)
    >>> vp.stack(arrays, axis=2).shape
    (3, 4, 10)
    >>> a = vp.array([1, 2, 3])
    >>> b = vp.array([2, 3, 4])
    >>> vp.stack((a, b))
    array([[1, 2, 3],
           [2, 3, 4]])
    >>> vp.stack((a, b), axis=-1)
    array([[1, 2],
           [2, 3],
           [3, 4]])

    """
    arrays = [nlcpy.asanyarray(arr) for arr in arrays]
    if not arrays:
        raise ValueError('need at least one array to stack')

    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError('all input arrays must have the same shape')

    if isinstance(axis, numpy.ndarray) or isinstance(axis, nlcpy.ndarray):
        axis = int(axis)
    elif not isinstance(axis, int):
        raise TypeError("an integer is required (got type {})"
                        .format(type(axis).__name__))

    result_ndim = arrays[0].ndim + 1
    if axis >= result_ndim or axis < -result_ndim:
        raise numpy.AxisError("axis {} is out of bounds for array of dimension {}"
                              .format(axis, result_ndim))
    axis = axis if axis >= 0 else result_ndim + axis

    sl = (slice(None),) * axis + (None,)
    expanded_arrays = [arr[sl] for arr in arrays]
    return nlcpy.concatenate(expanded_arrays, axis=axis, out=out)


def vstack(tup):
    """Stacks arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D arrays of shape
    (N,) have been reshaped to (1,N). Rebuilds arrays divided by vsplit.
    This function makes most sense for arrays with up to 3 dimensions. For instance, for
    pixel-data with a height (first axis), width (second axis), and r/g/b channels (third
    axis). The functions concatenate, stack and block provide more general stacking and
    concatenation operations.

    Parameters
    ----------
    tup : sequence of array_like
        The arrays must have the same shape along all but the first axis. 1-D arrays
        must have the same length.

    Returns
    -------
    stacked : `ndarray`
        The array formed by stacking the given arrays, will be at least 2-D.

    See Also
    --------
    stack : Joins a sequence of arrays along a new axis.
    hstack : Stacks arrays in sequence horizontally (column wise).
    concatenate : Joins a sequence of arrays along an existing axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([1, 2, 3])
    >>> b = vp.array([2, 3, 4])
    >>> vp.vstack((a,b))
    array([[1, 2, 3],
           [2, 3, 4]])
    >>> a = vp.array([[1], [2], [3]])
    >>> b = vp.array([[2], [3], [4]])
    >>> vp.vstack((a,b))
    array([[1],
           [2],
           [3],
           [2],
           [3],
           [4]])

    """
    if hasattr(tup, '__iter__'):
        arrays = []
        for array in tup:
            array = nlcpy.asanyarray(array)
            if array.ndim < 2:
                array = array.reshape([1, array.size])
            arrays.append(array)
    else:
        arrays = tup

    return manipulation._ndarray_concatenate(arrays, 0, None)


def hstack(tup):
    """Stacks arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis, except for 1-D arrays
    where it concatenates along the first axis. Rebuilds arrays divided by hsplit.
    This function makes most sense for arrays with up to 3 dimensions. For instance, for
    pixel-data with a height (first axis), width (second axis), and r/g/b channels (third
    axis). The functions concatenate, stack and block provide more general stacking and
    concatenation operations.

    Parameters
    ----------
    tup : sequence of array_like
        The arrays must have the same shape along all but the second axis, except 1-D
        arrays which can be any length.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.

    See Also
    --------
    stack : Joins a sequence of arrays along a new axis.
    vstack : Stacks arrays in sequence vertically (row wise).
    concatenate : Joins a sequence of arrays along an existing axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([1, 2, 3])
    >>> b = vp.array([2, 3, 4])
    >>> vp.hstack((a,b))
    array([1, 2, 3, 2, 3, 4])
    >>> a = vp.array([[1], [2], [3]])
    >>> b = vp.array([[2], [3], [4]])
    >>> vp.hstack((a,b))
    array([[1, 2],
           [2, 3],
           [3, 4]])

    """
    if hasattr(tup, '__iter__'):
        arrays = []
        for array in tup:
            array = nlcpy.asanyarray(array)
            if array.ndim == 0:
                array = array.reshape(1)
            arrays.append(array)
    else:
        arrays = tup

    if arrays[0].ndim == 1:
        return manipulation._ndarray_concatenate(arrays, 0, None)
    else:
        return manipulation._ndarray_concatenate(arrays, 1, None)


def block(arrays):
    """Assembles an nd-array from nested lists of blocks.

    Blocks in the innermost lists are concatenated (see :func:`concatenate`) along the
    last dimension (-1), then these are concatenated along the second-last dimension
    (-2), and so on until the outermost list is reached.

    Blocks can be of any dimension, but will not be broadcasted using the normal rules.
    Instead, leading axes of size 1 are inserted, to make ``block.ndim`` the same for
    all blocks. This is primarily useful for working with scalars, and means that code
    like ``vp.block([v, 1])`` is valid, where ``v.ndim == 1``.

    When the nested list is two levels deep, this allows block matrices to be
    constructed from their components.

    Parameters
    ----------
    arrays : nested list of array_like or scalars (but not tuples)
        If passed a single ndarray or scalar (a nested list of depth 0), this is
        returned unmodified (and not copied).
        Elements shapes must match along the appropriate axes (without broadcasting),
        but leading 1s will be prepended to the shape as necessary to make the
        dimensions match.

    Returns
    -------
    block_array : ndarray
        The array assembled from the given blocks.
        The dimensionality of the output is equal to the greatest of:
        - the dimensionality of all the inputs
        - the depth to which the input list is nested

    See Also
    --------
    vsplit : Splits an array into multiple sub-arrays vertically (row-wise).
    concatenate : Joins a sequence of arrays along an existing axis.
    stack : Joins a sequence of arrays along a new axis.
    hstack : Stacks arrays in sequence horizontally (column wise).
    vstack : Stacks arrays in sequence vertically (row wise).

    Note
    ----
    When called with only scalars, ``vp.block`` is equivalent to an ndarray call.
    So ``vp.block([[1, 2], [3, 4]])`` is equivalent to ``vp.array([[1, 2], [3, 4]])``.

    This function does not enforce that the blocks lie on a fixed grid.
    ``vp.block([[a, b], [c, d]])`` is not restricted to arrays of the form::

        AAAbb
        AAAbb
        cccDD

    But is also allowed to produce, for some ``a, b, c, d``::

        AAAbb
        AAAbb
        cDDDD

    Since concatenation happens along the last axis first, `block` is **not**
    capable of producing the following directly::

        AAAbb
        cccbb
        cccDD

    Matlab's "square bracket stacking", ``[A, B, ...; p, q, ...]``, is equivalent to
    ``vp.block([[A, B, ...], [p, q, ...]])``.

    Examples
    --------
    The most common use of this function is to build a block matrix

    >>> import nlcpy as vp
    >>> A = vp.eye(2) * 2
    >>> B = vp.eye(3) * 3
    >>> vp.block([
    ...     [A,               vp.zeros((2, 3))],
    ...     [vp.ones((3, 2)), B               ]
    ... ])
    array([[2., 0., 0., 0., 0.],
           [0., 2., 0., 0., 0.],
           [1., 1., 3., 0., 0.],
           [1., 1., 0., 3., 0.],
           [1., 1., 0., 0., 3.]])

    With a list of depth 1, block can be used as :func:`hstack`

    >>> vp.block([1, 2, 3])              # hstack([1, 2, 3])
    array([1, 2, 3])

    >>> a = vp.array([1, 2, 3])
    >>> b = vp.array([2, 3, 4])
    >>> vp.block([a, b, 10])             # hstack([a, b, 10])
    array([ 1,  2,  3,  2,  3,  4, 10])

    >>> A = vp.ones((2, 2), int)
    >>> B = 2 * A
    >>> vp.block([A, B])                 # hstack([A, B])
    array([[1, 1, 2, 2],
           [1, 1, 2, 2]])

    With a list of depth 2, block can be used in place of :func:`vstack`:

    >>> a = vp.array([1, 2, 3])
    >>> b = vp.array([2, 3, 4])
    >>> vp.block([[a], [b]])             # vstack([a, b])
    array([[1, 2, 3],
           [2, 3, 4]])

    >>> A = vp.ones((2, 2), int)
    >>> B = 2 * A
    >>> vp.block([[A], [B]])             # vstack([A, B])
    array([[1, 1],
           [1, 1],
           [2, 2],
           [2, 2]])

    It can also be used in places of :func:`atleast_1d` and :func:`atleast_2d`

    >>> a = vp.array(0)
    >>> b = vp.array([1])
    >>> vp.block([a])                    # atleast_1d(a)
    array([0])
    >>> vp.block([b])                    # atleast_1d(b)
    array([1])

    >>> vp.block([[a]])                  # atleast_2d(a)
    array([[0]])
    >>> vp.block([[b]])                  # atleast_2d(b)
    array([[1]])
    """
    return manipulation._block(arrays)

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
import operator
import numpy
import nlcpy
from nlcpy.request import request
from numpy import AxisError


def flip(m, axis=None):
    """Reverses the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    m : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to flip over. The default, axis=None, will flip over
        all of the axes of the input array. If axis is negative it counts from the
        last to the first axis.
        If axis is a tuple of ints, flipping is performed on all of the axes
        specified in the tuple.

    Returns
    -------
    out : ndarray
        A view of m with the entries of axis reversed. Since a view is returned, this
        operation is done in constant time.

    Note
    ----
    flip(m, 0) is equivalent to flipud(m).

    flip(m, 1) is equivalent to fliplr(m).

    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.

    flip(m) corresponds to ``m[::-1,::-1,...,::-1]`` with ``::-1`` at all positions.

    flip(m, (0, 1)) corresponds to ``m[::-1,::-1,...]`` with ``::-1`` at position 0 and
    position 1.

    See Also
    --------
    flipud : Flips array in the up/down direction.
    fliplr : Flips array in the left/right direction.

    Examples
    --------
    >>> import nlcpy as vp
    >>> A = vp.arange(8).reshape((2,2,2))
    >>> A
    array([[[0, 1],
            [2, 3]],
    <BLANKLINE>
           [[4, 5],
            [6, 7]]])
    >>> vp.flip(A, 0)
    array([[[4, 5],
            [6, 7]],
    <BLANKLINE>
           [[0, 1],
            [2, 3]]])
    >>> vp.flip(A, 1)
    array([[[2, 3],
            [0, 1]],
    <BLANKLINE>
           [[6, 7],
            [4, 5]]])
    >>> vp.flip(A)
    array([[[7, 6],
            [5, 4]],
    <BLANKLINE>
           [[3, 2],
            [1, 0]]])
    >>> vp.flip(A, (0, 2))
    array([[[5, 4],
            [7, 6]],
    <BLANKLINE>
           [[1, 0],
            [3, 2]]])
    >>> A = vp.random.randn(3, 4, 5)
    >>> vp.all(vp.flip(A, 2) == A[:, :, ::-1, ...])
    array(True)

    """
    m = nlcpy.asanyarray(m)
    if axis is None:
        indexer = (slice(None, None, -1),) * m.ndim
    else:
        if type(axis) is nlcpy.ndarray:
            axis = axis.get()
        if type(axis) not in (tuple, list):
            try:
                axis = [operator.index(axis)]
            except TypeError:
                pass
        _axis = []
        for ax in axis:
            if type(ax) is nlcpy.ndarray:
                ax = ax.get()
            if type(ax) is numpy.ndarray:
                if ax.size > 1:
                    raise TypeError(
                        'only size-1 arrays can be converted to Python scalars')
                else:
                    ax = ax.item()
            _axis.append(ax + m.ndim if ax < 0 else ax)
        axis = _axis
        if len(axis) != len(set(axis)):
            raise ValueError('repeated axis')
        indexer = [slice(None) for i in range(m.ndim)]
        for ax in axis:
            if ax >= m.ndim or ax < 0:
                raise AxisError(
                    'axis {0} is out of bounds for array of dimension {1}'
                    .format(ax, m.ndim))
            indexer[ax] = slice(None, None, -1)
        indexer = tuple(indexer)
    return m[indexer]


def flipud(m):
    """Flips array in the up/down direction.

    Flip the entries in each column in the up/down direction. Rows are preserved, but
    appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input array.

    Returns
    -------
    out : `ndarray`
        A view of *m* with the rows reversed. Since a view is returned, this operation
        is done in constant time.

    Note
    ----
    Equivalent to ``m[::-1,...]``. Does not require the array to be two-dimensional.

    See Also
    --------
    fliplr : Flips array in the left/right direction.

    Examples
    --------
    >>> import nlcpy as vp
    >>> A = vp.diag([1.0, 2, 3])
    >>> A
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]])
    >>> vp.flipud(A)
    array([[0., 0., 3.],
           [0., 2., 0.],
           [1., 0., 0.]])
    >>> A = vp.random.randn(2, 3, 5)
    >>> vp.all(vp.flipud(A) == A[::-1, ...])
    array(True)
    >>> vp.flipud([1,2])
    array([2, 1])

    """
    m = nlcpy.asanyarray(m)
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return m[::-1, ...]


def fliplr(m):
    """Flips array in the left/right direction.

    Flip the entries in each row in the left/right direction. Columns are preserved, but
    appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input array, must be at least 2-D.

    Returns
    -------
    out : ndarray
        A view of *m* with the columns reversed. Since a view is returned, this
        operation is done in constant time.

    Note
    ----
    Equivalent to ``m[:,::-1]``. Requires the array to be at least 2-D.

    See Also
    --------
    flipud : Flips array in the up/down direction.

    Examples
    --------
    >>> import nlcpy as vp
    >>> A = vp.diag([1., 2., 3.])
    >>> A
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]])
    >>> vp.fliplr(A)
    array([[0., 0., 1.],
           [0., 2., 0.],
           [3., 0., 0.]])
    >>> A = vp.random.randn(2, 3, 5)
    >>> vp.all(vp.fliplr(A) == A[:, ::-1, ...])
    array(True)

    """
    m = nlcpy.asanyarray(m)
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return m[:, ::-1]


def roll(a, shift, axis=None):
    """Rolls array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at the first.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int or tuple of ints
        The number of places by which elements are shifted. If a tuple, then *axis* must
        be a tuple of the same size, and each of the given axes is shifted by the
        corresponding number. If an int while *axis* is a tuple of ints, then the
        same value is used for all given axes.
    axis : int or tuple of ints, optional
        Axis or axes along which elements are shifted. By default, the array is
        flattened before shifting, after which the original shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as *a*.

    See Also
    --------
    rollaxis : Rolls the specified axis backwards, until it lies in a given position.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(10)
    >>> vp.roll(x, 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> vp.roll(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])

    >>> x2 = vp.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> vp.roll(x2, 1)
    array([[9, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> vp.roll(x2, -1)
    array([[1, 2, 3, 4, 5],
           [6, 7, 8, 9, 0]])
    >>> vp.roll(x2, 1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> vp.roll(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> vp.roll(x2, 1, axis=1)
    array([[4, 0, 1, 2, 3],
           [9, 5, 6, 7, 8]])
    >>> vp.roll(x2, -1, axis=1)
    array([[1, 2, 3, 4, 0],
           [6, 7, 8, 9, 5]])
    """
    a = nlcpy.asanyarray(a)
    if axis is None:
        return roll(a.ravel(), shift, 0).reshape(a.shape)

    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass

    _axis = axis.get() if isinstance(axis, nlcpy.ndarray) else axis
    axis = [ax + a.ndim if ax < 0 else ax for ax in _axis]
    for ax in axis:
        if ax < 0 or ax >= a.ndim:
            raise AxisError(
                'axis {} is out of bounds for array of dimension {}'
                .format(ax, a.ndim))

    shift = nlcpy.asanyarray(shift)
    axis = nlcpy.asanyarray(axis)
    if shift.ndim > 1 or axis.ndim > 1:
        raise ValueError(
            "'shift' and 'axis' should be scalars or 1D sequences")

    if shift.size > axis.size:
        axis = nlcpy.broadcast_to(axis, shift.shape)
    else:
        shift = nlcpy.broadcast_to(shift, axis.shape)

    shift = nlcpy.array(shift, dtype='l')
    axis = nlcpy.array(axis, dtype='l')
    result = nlcpy.empty(a.shape, dtype=a.dtype)
    work = nlcpy.zeros(a.ndim, dtype='l')
    request._push_request(
        'nlcpy_roll',
        'manipulation_op',
        (a, shift, axis, work, result)
    )

    return result

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
from nlcpy.core import manipulation
import functools


def moveaxis(a, source, destination):
    """Moves axes of an array to new positions.

    Other axes remain in their original order.

    Parameters
    ----------
    a : ndarray
        The array whose axes should be reordered.
    source : int or sequence of ints
        Original positions of the axes to move. These must be unique.
    destination : int or sequence of ints
        Destination positions for each of the original axes. These must also be unique.

    Returns
    -------
    result : ndarray
        Array with moved axes. This array is a view of the input array.

    See Also
    --------
    transpose : Permutes the dimensions of an array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.zeros((3, 4, 5))
    >>> vp.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> vp.moveaxis(x, -1, 0).shape
    (5, 3, 4)

    # These all achieve the same result:

    >>> vp.transpose(x).shape
    (5, 4, 3)
    >>> vp.moveaxis(x, [0, 1], [-1, -2]).shape
    (5, 4, 3)
    >>> vp.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
    (5, 4, 3)

    """
    a = nlcpy.asanyarray(a)
    return manipulation._moveaxis(a, source, destination)


def rollaxis(a, axis, start=0):
    """Rolls the specified axis backwards, until it lies in a given position.

    This function is implemented for backward compatibility of numpy. You should use
    :func:`moveaxis`.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int
        The axis to roll backwards. The positions of the other axes do not change
        relative to one another.
    start : int, optional
        The axis is rolled until it lies before this position. The default, 0, results in
        a "complete" roll.

    Returns
    -------
    res : ndarray
        Array with moved axes. This array is a view of the input array.

    See Also
    --------
    moveaxis : Moves axes of an array to new positions.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.ones((3,4,5,6))
    >>> vp.rollaxis(a, 3, 1).shape
    (3, 6, 4, 5)
    >>> vp.rollaxis(a, 2).shape
    (5, 3, 4, 6)
    >>> vp.rollaxis(a, 1, 4).shape
    (3, 5, 6, 4)

    """
    a = nlcpy.asanyarray(a)
    if isinstance(axis, numpy.ndarray) or isinstance(axis, nlcpy.ndarray):
        axis = int(axis)
    return manipulation._rollaxis(a, axis, start=start)


def _transpose_wrapper(func):
    @functools.wraps(func)
    def _transpose_dispatcher(a, axes=None):
        return func(a, axes)
    return _transpose_dispatcher


@_transpose_wrapper
def transpose(a, axes=None):
    """Permutes the dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axes : list of ints, optional
        By default, reverse the dimensions, otherwise permute the axes according to the
        values given.

    Returns
    -------
    p : ndarray
        *a* with its axes permuted. A view is returned whenever possible.

    Note
    ----

    Use `transpose(a, argsort(axes))` to invert the transposition of tensors when using
    the `axes` keyword argument. Transposing a 1-D array returns an unchanged view of the
    original array.

    See Also
    --------
    moveaxis : Moves axes of an array to new positions.
    argsort : Returns the indices that would sort an array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(4).reshape((2,2))
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> vp.transpose(x)
    array([[0, 2],
           [1, 3]])
    >>> x = vp.ones((1, 2, 3))
    >>> vp.transpose(x, (1, 0, 2)).shape
    (2, 1, 3)

    """
    a = nlcpy.asanyarray(a)
    return a.transpose(axes)


def swapaxes(a, axis1, axis2):
    """ Interchanges two axes of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : ndarray
        If *a* is an ndarray, then a view of *a* is returned;
        otherwise a new array is created.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.array([[1, 2, 3]])
    >>> vp.swapaxes(x, 0, 1)
    array([[1],
           [2],
           [3]])
    >>> x = vp.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> x
    array([[[0, 1],
            [2, 3]],
    <BLANKLINE>
           [[4, 5],
            [6, 7]]])
    >>> vp.swapaxes(x,0,2)
    array([[[0, 4],
            [2, 6]],
    <BLANKLINE>
           [[1, 5],
            [3, 7]]])
    """
    a = nlcpy.asanyarray(a)
    return a.swapaxes(axis1, axis2)

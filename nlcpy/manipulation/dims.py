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
import nlcpy
from nlcpy import core


# ----------------------------------------------------------------------------
# Array manipulation routines
# see: https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html
# ----------------------------------------------------------------------------


def broadcast_to(array, shape, subok=False):
    """Broadcasts an array to a new shape.

    Parameters
    ----------
    array : array_like
        The array to broadcast.
    shape : sequence of ints
        The shape of the desired array.
    subok : bool, optional
        Not implemented.

    Returns
    -------
    broadcast : ndarray
        A readonly view on the original array with the given shape. It is typically not
        contiguous. Furthermore, more than one element of a broadcasted array may refer
        to a single memory location.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.array([1, 2, 3])
    >>> vp.broadcast_to(x, (3, 3))
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    """
    if subok:
        raise NotImplementedError('subok in broadcast_to is not implemented yet.')
    array = nlcpy.asanyarray(array)
    return core.broadcast_to(array, shape)


def expand_dims(a, axis):
    """Expands the shape of an array.

    Insert a new axis that will appear at the *axis* position in the expanded
    array shape.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or tuple of ints
        Position in the expanded axes where the new axis is placed.

    Returns
    -------
    res : ndarray
        View of *a* with the number of dimensions increased by one.

    See Also
    --------
    squeeze : Removes single-dimensional entries from the shape
        of an array.
    reshape : Gives a new shape to an array without
        changing its data.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.array([1,2])
    >>> x.shape
    (2,)

    The following is equivalent to x[vp.newaxis,:] or x[vp.newaxis]:

    >>> y = vp.expand_dims(x, axis=0)
    >>> y
    array([[1, 2]])
    >>> y.shape
    (1, 2)
    >>> y = vp.expand_dims(x, axis=1)  # Equivalent to x[:,vp.newaxis]
    >>> y
    array([[1],
           [2]])
    >>> y.shape
    (2, 1)

    axis may also be a tuple:
    >>> y = vp.expand_dims(x, axis=(0, 1))
    >>> y
    array([[[1, 2]]])

    >>> y = vp.expand_dims(x, axis=(2, 0))
    >>> y
    array([[[1],
            [2]]])

    Note that some examples may use None instead of vp.newaxis. These are the same
    objects:

    >>> vp.newaxis is None
    True
    """
    a = nlcpy.asanyarray(a)
    return core.manipulation._expand_dims(a, axis)


def squeeze(a, axis=None):
    """Removes single-dimensional entries from the shape of an array.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Selects a subset of the single-dimensional entries in the shape. If an axis is
        selected with shape entry greater than one, an error is raised.

    Returns
    -------
    squeezed : ndarray
        The input array, but with all or a subset of the dimensions of length 1 removed.
        This is always *a* itself or a view into *a*.

    See Also
    --------
    expand_dims : Expands the shape of an array.
    reshape : Gives a new shape to an array
        without changing its data.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> vp.squeeze(x).shape
    (3,)
    >>> vp.squeeze(x, axis=0).shape
    (3, 1)
    >>> vp.squeeze(x, axis=1).shape   # doctest: +SKIP
    Traceback (most recent call last):
    ...
    ValueError: cannot select an axis to squeeze out which has size not equal to one
    >>> vp.squeeze(x, axis=2).shape
    (1, 3)
    """
    a = nlcpy.asanyarray(a)
    return a.squeeze(axis)


def broadcast_arrays(*args, **kwargs):
    """Broadcasts any number of arrays against each other.

    Parameters
    ----------
    *args : array_likes
        The arrays to broadcast.
    subok : bool, optional
        Not implemented.

    Returns
    -------
    broadcasted : list of arrays
        These arrays are views on the original arrays. They are typically not
        contiguous. Furthermore, more than one element of a broadcasted array may refer
        to a single memory location. If you need to write to the arrays, make copies
        first. While you can set the ``writable`` flag True, writing to a single output
        value may end up changing more than one location in the output array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.array([[1,2,3]])
    >>> y = vp.array([[4],[5]])
    >>> vp.broadcast_arrays(x, y)
    [array([[1, 2, 3],
           [1, 2, 3]]), array([[4, 4, 4],
           [5, 5, 5]])]

    Here is a useful idiom for getting contiguous copies instead of non-contiguous views.

    >>> [vp.array(a) for a in vp.broadcast_arrays(x, y)]
    [array([[1, 2, 3],
           [1, 2, 3]]), array([[4, 4, 4],
           [5, 5, 5]])]
    """
    subok = kwargs.pop('subok', False)
    if subok:
        raise NotImplementedError('subok in broadcast_arrays is not implemented yet.')
    if kwargs:
        raise TypeError('broadcast_arrays() got an unexpected keyword '
                        'argument {!r}'.format(list(kwargs.keys())[0]))

    args = [nlcpy.array(_m, copy=False) for _m in args]
    return core._broadcast_core([*args])[0]


def atleast_1d(*arys):
    """Converts inputs to arrays with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional
    inputs are preserved.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more input arrays.

    Returns
    -------
    ret : ndarrays
        An array, or list of arrays, each with ``ndim >= 1``.
        Copies are made only if necessary.

    See Also
    --------
    atleast_2d : Views inputs as arrays with at least two dimensions.
    atleast_3d : Views inputs as arrays with at least three dimensions.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.atleast_1d(1.0)
    array([1.])

    >>> x = vp.arange(9.0).reshape(3,3)
    >>> vp.atleast_1d(x)
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])
    >>> vp.atleast_1d(x) is x
    True

    >>> vp.atleast_1d(1, [3, 4])
    [array([1]), array([3, 4])]
    """
    res = []
    for ary in arys:
        ary = nlcpy.asanyarray(ary)
        if ary.ndim == 0:
            res.append(ary.reshape(1))
        else:
            res.append(ary)
    if len(res) == 1:
        return res[0]
    else:
        return res


def atleast_2d(*arys):
    """Views inputs as arrays with at least two dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences. Non-array inputs are converted to arrays.
        Arrays that already have two or more dimensions are preserved.

    Returns
    -------
    res1, res2, ... : ndarray
        An array, or list of arrays, each with ``ndim >= 2``. Copies are avoided
        where possible, and views with two or more dimensions are returned.

    See Also
    --------
    atleast_1d : Converts inputs to arrays with at least one dimension.
    atleast_3d : Views inputs as arrays with at least three dimensions.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.atleast_2d(3.0)
    array([[3.]])

    >>> x = vp.arange(3.0)
    >>> vp.atleast_2d(x)
    array([[0., 1., 2.]])
    >>> vp.atleast_2d(x).base is x
    True

    >>> vp.atleast_2d(1, [1, 2], [[1, 2]])
    [array([[1]]), array([[1, 2]]), array([[1, 2]])]
    """
    res = []
    for ary in arys:
        ary = nlcpy.asanyarray(ary)
        if ary.ndim == 0:
            res.append(ary.reshape(1, 1))
        elif ary.ndim == 1:
            res.append(ary[nlcpy.newaxis, :])
        else:
            res.append(ary)
    if len(res) == 1:
        return res[0]
    else:
        return res


def atleast_3d(*arys):
    """Views inputs as arrays with at least three dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences. Non-array inputs are converted to arrays.
        Arrays that already have three or more dimensions are preserved.

    Returns
    -------
    res1, res2, ... : ndarray
        An array, or list of arrays, each with ``ndim >= 3``. Copies are avoided where
        possible, and views with three or more dimensions are returned. For example,
        a 1-D array of shape ``(N,)`` becomes a view of shape ``(1, N, 1)``, and a 2-D
        array of shape ``(M, N)`` becomes a view of shape ``(M, N, 1)``.

    See Also
    --------
    atleast_1d : Converts inputs to arrays with at least one dimension.
    atleast_2d : Views inputs as arrays with at least two dimensions.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.atleast_3d(3.0)
    array([[[3.]]])

    >>> x = vp.arange(3.0)
    >>> vp.atleast_3d(x).shape
    (1, 3, 1)

    >>> x = vp.arange(12.0).reshape(4,3)
    >>> vp.atleast_3d(x).shape
    (4, 3, 1)
    >>> vp.atleast_3d(x).base is x.base  # x is a reshape, so not base itself
    True

    >>> for arr in vp.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):
    ...     print(arr, arr.shape) # doctest: +SKIP
    ...
    [[[1]
      [2]]] (1, 2, 1)
    [[[1]
      [2]]] (1, 2, 1)
    [[[1 2]]] (1, 1, 2)
    """
    res = []
    for ary in arys:
        ary = nlcpy.asanyarray(ary)
        if ary.ndim == 0:
            res.append(ary.reshape(1, 1, 1))
        elif ary.ndim == 1:
            res.append(ary[nlcpy.newaxis, :, nlcpy.newaxis])
        elif ary.ndim == 2:
            res.append(ary[:, :, nlcpy.newaxis])
        else:
            res.append(ary)
    if len(res) == 1:
        return res[0]
    else:
        return res

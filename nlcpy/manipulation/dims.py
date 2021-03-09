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
    axis : int
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

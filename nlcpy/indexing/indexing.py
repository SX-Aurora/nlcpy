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
from nlcpy.wrapper.numpy_wrap import numpy_wrap


@numpy_wrap
def take(a, indices, axis=None, out=None, mode='wrap'):
    """Takes elements from an array along an axis.

    When axis is not None, this function does the same thing as "fancy" indexing
    (indexing arrays using arrays); however, it can be easier to use if you need elements
    along a given axis. A call such as ``arr.take(indices, axis=3)`` is equivalent to
    ``arr[:,:,:,indices,...]``.

    Parameters
    ----------
    a : array_like
        The source array.
    indices : array_like
        The indices of the values to extract. Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened input array is
        used.
    out : ndarray, optional
        If provided, the result will be placed in this array. It should be of the
        appropriate shape and dtype.
    mode : {'wrap', raise', 'clip'}, optional
        In the current NLCPy, this argument is not supported. The default is 'wrap'.

    Returns
    -------
    out : ndarray

    Restriction
    -----------
    - *mode* != 'wrap': *NotImplementedError* occurs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([4, 3, 5, 7, 6, 8])
    >>> indices = [0, 1, 4]
    >>> vp.take(a, indices)
    array([4, 3, 6])

    In this example, "fancy" indexing can be used.

    >>> a[indices]
    array([4, 3, 6])

    If indices is not one dimensional, the output also has these dimensions.

    >>> vp.take(a, [[0, 1], [2, 3]])
    array([[4, 3],
           [5, 7]])

    """
    if mode != 'wrap':
        raise NotImplementedError('mode is not supported yet')
    a = nlcpy.asarray(a)
    return a.take(indices, axis, out, mode=mode)


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Returns specified diagonals.

    If *a* is 2-D, returns the diagonal of *a* with the given offset, i.e.,
    the collection of elements of the form ``a[i, i+offset]``.
    If *a* has more than two dimensions, then the axes specified by *axis1*
    and *axis2* are used to determine the 2-D sub-array whose diagonal is
    returned.
    The shape of the resulting array can be determined by removing *axis1*
    and *axis2* and appending an index to the right equal to the size of the
    resulting diagonals. This function returns a writable view of the original array.

    Parameters
    ----------
    a : array_like
        Array from which the diagonals are taken.

    offset : int, optional
        Offset of the diagonal from the main diagonal.
        Can be positive or negative. Defaults to main diagonal (0).

    axis1 : int, optional
        Axis to be used as the first axis of the 2-D sub-arrays from which
        the diagonals should be taken. Defaults to first axis (0).

    axis2 : int, optional
        Axis to be used as the second axis of the 2-D sub-arrays from which
        the diagonals should be taken. Defaults to second axis (1).

    Returns
    -------
    array_of_diagonals : ndarray
        If *a* is 2-D, then a 1-D array containing the diagonal and of the same type as
        *a* is returned. If ``a.ndim > 2``, then the dimensions specified by *axis1*
        and *axis2* are removed, and a new axis inserted at the end corresponding to
        the diagonal.

    Note
    ----
        ``a.ndim < 2`` : *ValueError* occurs.

    See Also
    --------
        diag : Extracts a diagonal or constructs a diagonal array.

    Examples
    --------

    >>> import nlcpy as vp
    >>> a = vp.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> a.diagonal()
    array([0, 3])
    >>> a.diagonal(1)
    array([1])

    A 3-D example:

    >>> a = vp.arange(8).reshape(2,2,2); a
    array([[[0, 1],
            [2, 3]],
    <BLANKLINE>
           [[4, 5],
            [6, 7]]])
    >>> a.diagonal(0,  # Main diagonals of two arrays created by skipping
    ...            0,  # across the outer(left)-most axis last and
    ...            1)  # the "middle" (row) axis first.
    array([[0, 6],
           [1, 7]])

    The sub-arrays whose main diagonals we just obtained;
    note that each corresponds to fixing the right-most (column) axis,
    and that the diagonals are “packed” in rows.

    >>> a[:,:,0]  # main diagonal is [0 6]
    array([[0, 2],
           [4, 6]])
    >>> a[:,:,1]  # main diagonal is [1 7]
    array([[1, 3],
           [5, 7]])

    """
    a = nlcpy.asarray(a)
    return a.diagonal(offset, axis1, axis2)


def select(condlist, choicelist, default=0):
    """Returns an array drawn from elements in choicelist, depending on conditions.

    Parameters
    ----------
    condlist : list of bool ndarrays
        The list of conditions which determine from which array in *choicelist* the
        output elements are taken. When multiple conditions are satisfied, the first
        one encountered in *condlist* is used.
    choicelist : list of ndarrays
        The list of arrays from which the output elements are taken.
        It has to be of the same length as *condlist*.
    default : scalar, optional
        The element inserted in *output* when all conditions evaluate to False.

    Returns
    -------
    output : ndarray
        The output at position m is the m-th element of the array in *choicelist* where
        the m-th element of the corresponding array in *condlist* is True.

    See Also
    --------
    where : Returns elements chosen from *x* or *y* depending on *condition*.
    take : Takes elements from an array along an axis.
    diag : Extracts a diagonal or construct a diagonal array.
    diagonal : Returns specified diagonals.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(10)
    >>> condlist = [x<3, x>5]
    >>> choicelist = [x, x**2]
    >>> vp.select(condlist, choicelist)
    array([ 0,  1,  2,  0,  0,  0, 36, 49, 64, 81])
    """
    if len(condlist) != len(choicelist):
        raise ValueError(
            'list of cases must be same length as list of conditions')

    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")

    choicelist = list(choicelist)
    if numpy.isscalar(default):
        default = numpy.asarray(default)
    else:
        default = nlcpy.asarray(default)
    choicelist.append(default)
    dtype = numpy.result_type(*choicelist)

    condlist = nlcpy.broadcast_arrays(*condlist)
    choicelist = nlcpy.broadcast_arrays(*choicelist)

    for i in range(len(condlist)):
        cond = condlist[i]
        if cond.dtype.type is not nlcpy.bool_:
            raise TypeError(
                'invalid entry {} in condlist: should be boolean ndarray'.format(i))

    if choicelist[0].ndim == 0:
        result_shape = condlist[0].shape
    else:
        result_shape = nlcpy.broadcast_arrays(condlist[0], choicelist[0])[0].shape

    result = nlcpy.full(result_shape, choicelist[-1], dtype=dtype)

    choicelist = choicelist[-2::-1]
    condlist = condlist[::-1]
    for choice, cond in zip(choicelist, condlist):
        nlcpy.copyto(result, choice, where=cond)
    return result

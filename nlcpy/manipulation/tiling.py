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
from nlcpy.core import core
from nlcpy.core.core import ndarray, array
from nlcpy.request import request

import functools
import operator


def tile(A, reps):
    """Constructs an array by repeating A the number of times given by reps.

    If *reps* has length ``d``, the result will have dimension of ``max(d, A.ndim)``.

    If ``A.ndim < d`` , *A* is promoted to be d-dimensional by prepending new axes.
    So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape
    (1, 1, 3) for 3-D replication.
    If this is not the desired behavior, promote *A* to d-dimensions
    manually before calling this function.

    If ``A.ndim > d``, *reps* is promoted to *A.ndim* by pre-pending 1's to it.
    Thus for an *A* of shape (2, 3, 4, 5), a *reps* of (2, 2) is treated as
    (1, 1, 2, 2).

    Parameters
    ----------
    A : array_like
        The input array.
    reps : array_like
        The number of repetitions of *A* along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.

    Note
    ----

    Although tile may be used for broadcasting, it is strongly recommended to use nlcpy's
    broadcasting operations and functions.

    See Also
    --------
    broadcast_to : Broadcasts an array to a new shape.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([0, 1, 2])
    >>> vp.tile(a, 2)
    array([0, 1, 2, 0, 1, 2])
    >>> vp.tile(a, (2, 2))
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])
    >>> vp.tile(a, (2, 1, 2))
    array([[[0, 1, 2, 0, 1, 2]],
    <BLANKLINE>
           [[0, 1, 2, 0, 1, 2]]])
    >>> b = vp.array([[1, 2], [3, 4]])
    >>> vp.tile(b, 2)
    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])
    >>> vp.tile(b, (2, 1))
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])
    >>> c = vp.array([1,2,3,4])
    >>> vp.tile(c,(4,1))
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])
    """

    if not isinstance(A, ndarray):
        A = core.argument_conversion(A)

    # TODO: numpy.isscalar -> nlcpy.isscalar
    if numpy.isscalar(reps) or reps is None:
        shape_reps = (reps,)
        dim_reps = len(shape_reps)
    elif isinstance(reps, (ndarray, numpy.ndarray)):
        if reps.ndim == 0:
            shape_reps = (reps,)
        elif reps.ndim == 1:
            shape_reps = tuple([reps[i] for i in range(reps.size)])
        elif reps.ndim > 1:
            raise ValueError("The truth value of an array with more than"
                             + " one element is ambiguous. "
                             + "Use a.any() or a.all()")
        dim_reps = reps.size
    elif isinstance(reps, (list, tuple)):
        reps_size = 1
        if len(reps) <= 0:
            reps = (1,)
        else:
            if A.ndim in [0, 1, 2]:
                inner_cnt = 1
            else:
                inner_cnt = functools.reduce(operator.mul, A.shape[0:-1])

            for i in range(len(reps)):
                if isinstance(reps[i], (list, tuple)):
                    if len(reps[i]) <= 0:
                        raise ValueError("operands could not be broadcast"
                                         + " together with shape ("
                                         + str(reps_size * inner_cnt)
                                         + ",) (0,)")
                    if len(reps[i]) == 1:
                        if isinstance(reps[i][0], (list, tuple)):
                            raise ValueError(
                                "object too deep for desired array")
                        else:
                            raise TypeError("'%s' object cannot be"
                                            " interpreted as an integer"
                                            % (type(reps[i]).__name__))
                    elif len(reps[i]) > 1:
                        list_flg = False
                        scal_flg = False
                        for j in range(len(reps[i])):
                            if isinstance(reps[i][j], (list, tuple)):
                                list_flg = True
                            # TODO: numpy.isscalar -> nlcpy.isscalar
                            elif numpy.isscalar(reps[i][j]):
                                scal_flg = True
                            elif isinstance(reps[i][j], (ndarray, numpy.ndarray)):
                                if reps[i][j].size == 1:
                                    scal_flg = True
                                elif reps[i][j].size <= 0 or reps[i][j].size >= 2:
                                    list_flg = True

                        if list_flg is True and scal_flg is True:
                            raise ValueError(
                                "setting an array element with a sequence.")
                        elif not list_flg and scal_flg:
                            raise ValueError("operands could not be broadcast"
                                             + " together with shape ("
                                             + str(reps_size * inner_cnt)
                                             + ",) (" + str(len(reps[i]))
                                             + ",)")
                        elif list_flg and not scal_flg:
                            raise ValueError(
                                "object too deep for desired array")

                elif isinstance(reps[i], (ndarray, numpy.ndarray)):
                    if reps[i].size > 1 and reps.ndim > 0:
                        raise ValueError(
                            "The truth value of an array with more than"
                            + " one element is ambiguous."
                            + " Use a.any() or a.all()")
                    else:
                        if reps[i].ndim == 0:
                            shape_reps = (reps[i],)
                        elif reps[i].ndim == 1:
                            shape_reps = (reps[i],)

                elif reps[i] is not None and not isinstance(reps[i], int):
                    if isinstance(reps[i], complex):
                        reps_size *= int(reps[i].real)
                    else:
                        reps_size *= int(reps[i])

        shape_reps = tuple(reps)
        dim_reps = len(shape_reps)

    if A.ndim < dim_reps:
        A = array(A, ndmin=dim_reps)

    shape_A = A.shape
    shape_reps = (1,) * (A.ndim - dim_reps) + shape_reps
    shape = tuple(s * t for s, t in zip(shape_A, shape_reps))
    ret = ndarray(shape=shape, dtype=A.dtype)

    if ret.size > 0:
        request._push_request(
            'nlcpy_tile',
            'manipulation_op',
            (A, ret)
        )
    return ret


def repeat(a, repeats, axis=None):
    """Repeats elements of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    repeats : int or sequence of ints
        The number of repetitions for each element. *repeats* is broadcasted to fit the
        shape of the given axis.
    axis : int, optional
        The axis along which to repeat values. By default, use the flattened input
        array, and return a flat output array.

    Returns
    -------
    c : ndarray
        Output array which has the same shape as a, except along the given axis.

    See Also
    --------
    tile : Constructs an array by repeating A the number of times given by reps.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.repeat(3, 4)
    array([3, 3, 3, 3])
    >>> x = vp.array([[1, 2], [3, 4]])
    >>> vp.repeat(x, 2)
    array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> vp.repeat(x, 3, axis=1)
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]])
    >>> vp.repeat(x, [1, 2], axis=0)
    array([[1, 2],
           [3, 4],
           [3, 4]])

    """
    a = nlcpy.asanyarray(a)
    return a.repeat(repeats, axis)

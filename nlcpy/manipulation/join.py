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

from nlcpy.core import manipulation


def concatenate(arrays, axis=0, out=None):
    """Joins a sequence of arrays along an existing axis.

    Args:
        arrays : sequence of array_like
            The arrays must have the same shape, except in the dimension corresponding to
            axis (the first, by default).
        axis : int, optional
            The axis along which the arrays will be joined. If axis is None, arrays are
            flattened before use. Default is 0.
        out : `ndarray`, optional
            If provided, the destination to place the result. The shape must be correct,
            matching that of what concatenate would have returned if no out argument were
            specified.

    Returns:
        res : `ndarray`
            The concatenated array.

    Examples:
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

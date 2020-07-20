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


def take(a, indices, axis=None, out=None):
    """Takes elements from an array along an axis.

    When axis is not None, this function does the same thing as "fancy" indexing
    (indexing arrays using arrays); however, it can be easier to use if you need elements
    along a given axis. A call such as arr.take(indices, axis=3) is equivalent to
    arr[:,:,:,indices,...].

    Args:
        a : array_like
            The source array.
        indices : array_like
            The indices of the values to extract. Also allow scalars for indices.
        axis : int, optional
            The axis over which to select values. By default, the flattened input array
            is used.
        out : `ndarray`, optional
            If provided, the result will be placed in this array. It should be of the
            appropriate shape and dtype. Note that out is always buffered if
            mode='raise'; use other modes for better performance.

    Returns:
        out : `ndarray`
            This function does not support mode argument. If indices are out-of-bounds,
            this function always wraps around them.

    Examples:
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
    a = nlcpy.asarray(a)
    return a.take(indices, axis, out)


def diagonal(a, offset=0, axis1=0, axis2=1):
    a = nlcpy.asarray(a)
    return a.diagonal(offset, axis1, axis2)

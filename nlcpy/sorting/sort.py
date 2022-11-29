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
from nlcpy.wrapper.numpy_wrap import numpy_wrap


@numpy_wrap
def sort(a, axis=-1, kind=None, order=None):
    """Returns a sorted copy of an array.

    Parameters
    ----------
    a : array_like
        Array to be sorted.
    axis : int or None, optional
        Axis along which to sort. If None, the array is flattened before sorting. The
        default is -1, which sorts along the last axis.
    kind : {'None','stable'}, optional
        Sorting algorithm. The default is 'stable', kind only supported 'stable'. ('None'
        is treated as 'stable'.)
    order : str or list of str, optional
        In the current NLCPy, This argument is not supported. The default is 'None'.

    Returns
    -------
    sorted_array : ndarray
        Array of the same type and shape as *a*.

    Restriction
    -----------
    *NotImplementedError*:

      - If *kind* is not None and ``kind != 'stable'``.
      - If *order* is not None.
      - If 'c' is contained in *a.dtype.kind*.

    Note
    ----

    'stable' uses the radix sort for all data types.

    See Also
    --------
    ndarray.sort : Method to sort an array in-place.
    argsort : Indirect sort.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1,4],[3,1]])
    >>> vp.sort(a)                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> vp.sort(a, axis=None)     # sort the flattened array
    array([1, 1, 3, 4])
    >>> vp.sort(a, axis=0)        # sort along the first axis
    array([[1, 1],
           [3, 4]])

    """
    a = nlcpy.asarray(a)
    if kind is not None and kind not in 'stable':
        raise NotImplementedError('kind only supported \'stable\'.')
    if order is not None:
        raise NotImplementedError('order is not implemented.')
    if a.dtype.kind in ('c',):
        raise NotImplementedError('Unsupported dtype %s' % a.dtype)
    if axis is None:
        ret = a.flatten()
        axis = -1
    else:
        ret = a.copy()
    ret.sort(axis=axis, kind=kind, order=order)
    return ret


@numpy_wrap
def argsort(a, axis=-1, kind=None, order=None):
    """Returns the indices that would sort an array.

    Perform an indirect sort along the given axis using the radix sort. It returns an
    array of indices of the same shape as *a* that index data along the given axis in
    sorted order.

    Parameters
    ----------
    a : array_like
        Array to sort.
    axis : int or None, optional
        Axis along which to sort. The default is -1 (the last axis). If None, the
        flattened array is used.
    kind : {'None','stable'}, optional
        Sorting algorithm. The default is 'stable', kind only supported 'stable'. ('None'
        is treated as 'stable'.)
    order : str or list of str, optional
        This argument is not supported. The default is 'None'.

    Returns
    -------
    index_array : ndarray
        Array of indices that sort *a* along the specified *axis*. If a is
        one-dimensional, ``a[index_array]`` yields a sorted *a*. More generally,

    Restriction
    -----------
    *NotImplementedError*:

      - If *kind* is not None and ``kind != 'stable'``.
      - If *order* is not None.
      - If 'c' is contained in *a.dtype.kind*.

    See Also
    --------
    sort : Describes sorting algorithms used.
    ndarray.sort : Method to sort an array in-place.

    Examples
    --------

    One dimensional array:

    >>> import nlcpy as vp
    >>> x = vp.array([3, 1, 2])
    >>> vp.argsort(x)
    array([1, 2, 0])

    Two-dimensional array:

    >>> x = vp.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])
    >>> ind = vp.argsort(x, axis=0)  # sorts along first axis (down)
    >>> ind
    array([[0, 1],
           [1, 0]])
    >>> ind = vp.argsort(x, axis=1)  # sorts along last axis (across)
    >>> ind
    array([[0, 1],
           [0, 1]])

    """
    a = nlcpy.asarray(a)
    if kind is not None and kind not in 'stable':
        raise NotImplementedError('kind only supported \'stable\'.')
    if order is not None:
        raise NotImplementedError('order is not implemented.')
    if a.dtype.kind in ('c',):
        raise NotImplementedError('Unsupported dtype %s' % a.dtype)
    ret = a.argsort(axis=axis, kind=kind, order=order)
    return ret

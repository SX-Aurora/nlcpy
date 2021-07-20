#
# * The source code in this file is based on the soure code of CuPy.
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
# # CuPy License #
#
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

# distutils: language = c++
import numpy
import nlcpy
import numbers
import warnings
import ctypes

import nlcpy
from nlcpy import veo
from nlcpy.core cimport core
from nlcpy.core cimport broadcast
from nlcpy.core.core cimport *
from nlcpy.manipulation.shape import reshape
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.request cimport request
from nlcpy.request.ve_kernel cimport *

cimport numpy as cnp

# @name Sorting, Searching, and Counting
# @{


# ----------------------------------------------------------------------------
# argmax
# see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
# ----------------------------------------------------------------------------
cpdef argmax(a, axis=None, out=None):
    """Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified
        axis.
    out : ndarray, optional
        If provided, the result will be inserted into this array. It should be of the
        appropriate shape and dtype.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as *a.shape* with the
        dimension along *axis* removed.

    Note
    ----

    In case of multiple occurrences of the maximum values, the indices corresponding to
    the first occurrence are returned.

    See Also
    --------
    argmin : Returns the indices of the minimum values along an axis.
    amax : Returns the maximum of an array or maximum along an axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.arange(6).reshape(2,3) + 10
    >>> a
    array([[10, 11, 12],
           [13, 14, 15]])
    >>> vp.argmax(a)
    array(5)
    >>> vp.argmax(a, axis=0)
    array([1, 1, 1])
    >>> vp.argmax(a, axis=1)
    array([2, 2])

    >>> b = vp.arange(6)
    >>> b[1] = 5
    >>> b
    array([0, 5, 2, 3, 4, 5])
    >>> vp.argmax(b)  # Only the first occurrence is returned.
    array(1)

    """
    # check None
    if a is None:
        if out is not None:
            out = None
        return 0

    # convert to nlcpy.ndarray
    arr = core.argument_conversion(a)

    ########################################################################
    # TODO: VE-VH collaboration
    if arr._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError("argmax on VH is not yet implemented.")

    if out is not None:
        if out._memloc in {on_VH, on_VE_VH}:
            raise NotImplementedError("argmax on VH is not yet implemented.")

    ########################################################################
    # axis
    axis_save = axis

    # convert to list for assignment
    if axis is not None and not isinstance(axis, int):
        raise TypeError("'%s' object cannot be interpreted as an integer"
                        %(type(axis).__name__))

    else:
        axis = (axis,)

    if axis_save is not None and axis_save != NLCPY_MAXNDIM:
        axis = list(axis)
        # if negative, counts from the last to the first axis
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] = arr.ndim + axis[i]

        # check axis range
        for i in range(len(axis)):
            if axis[i] < 0 or axis[i] > arr.ndim-1:
                raise AxisError(
                    'axis '+str(axis[i])
                    +' is out of bounds for array of dimension '+str(arr.ndim))
    else:
        # all dimensions are reduced
        axis = [-1, ]

    if axis[0] == -1 and arr.size < 1 or \
       axis[0] != -1 and arr.shape[axis[0]] < 1:
        raise ValueError("attempt to get argmax of an empty sequence")

    ########################################################################
    # determine output shape
    if axis_save is None:
        if out is not None:
            raise ValueError("output array does not match result of argmax")
        lst=[1, ]
    else:
        lst = list(arr.shape)
        for i, axis_i in enumerate(axis):
            lst.pop(axis_i-i if axis_i>=i else axis_i)
    shape_out = tuple(lst)

    if out is not None and out.shape != shape_out:
        raise ValueError("output array does not match result of np.argmax.")

    if arr.dtype.char == '?':
        initial = nlcpy.array(0, dtype=arr.dtype)
    elif arr.dtype.char in ('ilIL'):
        initial = nlcpy.array(nlcpy.iinfo(arr.dtype).min, dtype=arr.dtype)
    elif arr.dtype.char in ('dfDF'):
        initial = nlcpy.array(nlcpy.finfo(arr.dtype).min, dtype=arr.dtype)

    ########################################################################
    # check order
    if arr._f_contiguous and not arr._c_contiguous:
        order_out = 'F'
    else:
        order_out = 'C'

    ########################################################################
    # use axis
    x = arr
    for axis_i in axis:
        shape = []
        if x.ndim != 0:
            for i in range(x.ndim):
                if i != axis_i and axis_i != -1:
                    shape.append(x.shape[i])
                else:
                    shape.append(1)
        else:
            shape = [1, ]

        # y = ndarray(shape=shape, dtype=x.dtype, order=order_out)
        if out is not None and axis_i == axis[-1]:
            if out._f_contiguous and not out._c_contiguous:
                y = ndarray(shape=shape, dtype=x.dtype, order='F')
                out = out.reshape(shape)
                z = broadcast.broadcast_to(out, shape)
            else:
                y = ndarray(shape=shape, dtype=x.dtype, order='C')
                out = out.reshape(shape)
                z = broadcast.broadcast_to(out, shape)

        else:
            # z = ndarray(shape=shape, dtype=numpy.int64, order=order_out)
            y = ndarray(shape=shape, dtype=x.dtype, order='C')
            z = ndarray(shape=shape, dtype=numpy.int64, order='C')
            if z.ve_adr == 0:
                raise MemoryError()

        # call argmax function on VE
        request._push_request(
            "nlcpy_argmax",
            "searching_op",
            (x, y, z, initial,
             <int64_t>1 if order_out == 'C' else 0,
             <int64_t>axis_i),
        )
    # return array indices
    if axis_save is None:
        # TODO: currently, nlcpy.ndarray class cannot be cast to int.
        # as a workaround, cast to str first and then cast to int.
        ret = reshape(z, shape_out)
        return ret[0]
    else:
        return reshape(z, shape_out)


# ----------------------------------------------------------------------------
# argmin
# see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html
# ----------------------------------------------------------------------------
cpdef argmin(a, axis=None, out=None):
    """Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified
        axis.
    out : ndarray, optional
        If provided, the result will be inserted into this array. It should be of the
        appropriate shape and dtype.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as *a.shape* with the
        dimension along *axis* removed.

    Note
    ----

    In case of multiple occurrences of the minimum values, the indices corresponding to
    the first occurrence are returned.

    See Also
    --------
    argmax : Returns the indices of the maximum values along an axis.
    amin : Returns the minimum of an array or minimum along an axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.arange(6).reshape(2,3) + 10
    >>> a
    array([[10, 11, 12],
           [13, 14, 15]])
    >>> vp.argmin(a)
    array(0)
    >>> vp.argmin(a, axis=0)
    array([0, 0, 0])
    >>> vp.argmin(a, axis=1)
    array([0, 0])

    >>> b = vp.arange(6) + 10
    >>> b[4] = 10
    >>> b
    array([10, 11, 12, 13, 10, 15])
    >>> vp.argmin(b)  # Only the first occurrence is returned.
    array(0)

    """
    # check None
    if a is None:
        if out is not None:
            out = None
        return 0

    # convert to nlcpy.ndarray
    arr = core.argument_conversion(a)

    ########################################################################
    # TODO: VE-VH collaboration
    if arr._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError("argmin on VH is not yet implemented.")

    if out is not None:
        if out._memloc in {on_VH, on_VE_VH}:
            raise NotImplementedError("argmin on VH is not yet implemented.")

    ########################################################################
    # axis
    axis_save = axis

    # convert to list for assignment
    if axis is not None and not isinstance(axis, int):
        raise TypeError("'%s' object cannot be interpreted as an integer"
                        %(type(axis).__name__))

    else:
        axis = (axis,)

    if axis_save is not None and axis_save != NLCPY_MAXNDIM:
        axis = list(axis)
        # if negative, counts from the last to the first axis
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] = arr.ndim + axis[i]

        # check axis range
        for i in range(len(axis)):
            if axis[i] < 0 or axis[i] > arr.ndim-1:
                raise AxisError(
                    'axis '+str(axis[i])
                    +' is out of bounds for array of dimension '+str(arr.ndim))
    else:
        # all dimensions are reduced
        axis = [-1, ]

    if axis[0] == -1 and arr.size < 1 or \
       axis[0] != -1 and arr.shape[axis[0]] < 1:
        raise ValueError("attempt to get argmin of an empty sequence")

    ########################################################################
    # determine output shape
    if axis_save is None:
        if out is not None:
            raise ValueError("output array does not match result of argmin")
        lst=[1, ]
    else:
        lst = list(arr.shape)
        for i, axis_i in enumerate(axis):
            lst.pop(axis_i-i if axis_i>=i else axis_i)
    shape_out = tuple(lst)

    if out is not None and out.shape != shape_out:
        raise ValueError("output array does not match result of np.argmin.")

    if arr.dtype.char == '?':
        initial = nlcpy.array(1, dtype=arr.dtype)
    elif arr.dtype.char in ('ilIL'):
        initial = nlcpy.array(nlcpy.iinfo(arr.dtype).max, dtype=arr.dtype)
    elif arr.dtype.char in ('dfDF'):
        initial = nlcpy.array(nlcpy.finfo(arr.dtype).max, dtype=arr.dtype)

    ########################################################################
    # check order
    if arr._f_contiguous and not arr._c_contiguous:
        order_out = 'F'
    else:
        order_out = 'C'

    ########################################################################
    # use axis
    x = arr
    for axis_i in axis:
        shape = []
        if x.ndim != 0:
            for i in range(x.ndim):
                if i != axis_i and axis_i != -1:
                    shape.append(x.shape[i])
                else:
                    shape.append(1)
        else:
            shape = [1, ]

        # y = ndarray(shape=shape, dtype=x.dtype, order=order_out)
        if out is not None and axis_i == axis[-1]:
            if out._f_contiguous and not out._c_contiguous:
                y = ndarray(shape=shape, dtype=x.dtype, order='F')
                out = out.reshape(shape)
                z = broadcast.broadcast_to(out, shape)
            else:
                y = ndarray(shape=shape, dtype=x.dtype, order='C')
                out = out.reshape(shape)
                z = broadcast.broadcast_to(out, shape)

        else:
            # z = ndarray(shape=shape, dtype=numpy.int64, order=order_out)
            y = ndarray(shape=shape, dtype=x.dtype, order='C')
            z = ndarray(shape=shape, dtype=numpy.int64, order='C')
            if z.ve_adr == 0:
                raise MemoryError()

        # call argmin function on VE
        request._push_request(
            "nlcpy_argmin",
            "searching_op",
            (x, y, z, initial,
             <int64_t>1 if order_out == 'C' else 0,
             <int64_t>axis_i),
        )
    # return array indices
    if axis_save is None:
        # TODO: currently, nlcpy.ndarray class cannot be cast to int.
        # as a workaround, cast to str first and then cast to int.
        ret = reshape(z, shape_out)
        return ret[0]
    else:
        return reshape(z, shape_out)


# ----------------------------------------------------------------------------
# nonzero
# see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html
# ----------------------------------------------------------------------------
cpdef nonzero(a):
    """Returns the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of *a*, containing the indices of
    the non-zero elements in that dimension. The values in *a* are always tested and
    returned in row-major, C-style order.
    To group the indices by element, rather than dimension, use :func:`argwhere`,
    which returns a row for each non-zero element.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.

    Note
    ----
    While the nonzero values can be obtained with ``a[nonzero(a)]``, it is recommended to
    use ``x[x.astype(bool)]`` or ``x[x != 0]`` instead, which will correctly handle 0-d
    arrays.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    >>> x
    array([[3, 0, 0],
           [0, 4, 0],
           [5, 6, 0]])
    >>> vp.nonzero(x)
    (array([0, 1, 2, 2]), array([0, 1, 0, 1]))
    >>> x[vp.nonzero(x)]
    array([3, 4, 5, 6])

    A common use for ``nonzero`` is to find the indices of an array, where a condition is
    True. Given an array *a*, the condition *a* > 3 is a boolean array and since False is
    interpreted as 0, ``nlcpy.nonzero(a > 3)`` yields the indices of the *a* where the
    condition is true.

    >>> a = vp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> a > 3
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]])
    >>> vp.nonzero(a > 3)
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    Using this result to index *a* is equivalent to using the mask directly:

    >>> a[vp.nonzero(a > 3)]
    array([4, 5, 6, 7, 8, 9])
    >>> a[a > 3]  # prefer this spelling
    array([4, 5, 6, 7, 8, 9])

    nonzero can also be called as a method of the array.

    >>> (a > 3).nonzero()
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    """
    errcnt = 0
    if a is not None:
        arr = core.argument_conversion(a)
        if arr.ndim == 0:
            arr = core.array([arr.get()])
        elif arr.size < 1:
            ret_ndim = arr.ndim
            errcnt += 1
    else:
        ret_ndim = 1
        errcnt += 1

    if errcnt != 0:
        ret = []
        for _ in range(ret_ndim):
            ret.append(core.array([], dtype="int64"))
        return tuple(ret)

    ########################################################################
    # TODO: VE-VH collaboration
    if arr._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError("nonzero on VH is not yet implemented.")

    ########################################################################
    # check order
    if arr._f_contiguous and not arr._c_contiguous:
        order_out = 'F'
    else:
        order_out = 'C'

    ########################################################################
    # main : outputs in list format and then converts into tuple format.
    ret = []

    arr_bool = nlcpy.array(arr, dtype=bool)
    arr_bool_cnt = arr_bool[arr_bool].size
    out_shape = arr_bool_cnt

    if(out_shape <= 0):
        ret = []
        ret.append(core.array([], dtype=numpy.int64))
        return tuple(ret)

    # loop by dimension num to put togther indexes by dimensions of an input array.
    for axis in range(arr_bool.ndim):

        # creates an output array with non-zero size because it is 1-dimention array.
        x = ndarray(shape=(out_shape), dtype="int64")
        request._push_request(
            "nlcpy_nonzero",
            "searching_op",
            (arr_bool, x, int(axis)),
        )
        ret.append(x)

    return tuple(ret)

# ----------------------------------------------------------------------------
# argwhere
# see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argwhere.html
# ----------------------------------------------------------------------------
cpdef ndarray argwhere(a):
    """Finds the indices of array elements that are non-zero, grouped by element.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    index_array : ndarray
        Indices of elements that are non-zero. Indices are grouped by element.

    Note
    ----

    The output of argwhere is not suitable for indexing arrays. For this purpose use
    :func:`nonzero` instead.

    See Also
    --------
    where : Returns elements chosen from x or y depending on condition.
    nonzero : Returns the indices of the elements that are non-zero.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(6).reshape(2,3)
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> vp.argwhere(x>1)
    array([[0, 2],
           [1, 0],
           [1, 1],
           [1, 2]])

    """
    errcnt = 0
    if a is not None:
        if type(a) not in (ndarray, numpy.ndarray):
            # TODO: using isscalar. needs replacing when implemented.
            if numpy.isscalar(a) is True:
                arr = core.array([a])
                shape = (0 if a == 0 else 1, 0)
                errcnt += 1
            elif isinstance(a, list) and len(a) >= 1:
                arr = core.argument_conversion(a)
            elif isinstance(a, tuple) and len(a) >= 1:
                arr = core.argument_conversion(a)
            elif isinstance(a, numpy.ndarray):
                arr = core.argument_conversion(a)
            else:
                shape = (0, 1)
                errcnt += 1
        else:
            if a.size == 1:
                arr = core.array([a.get()])
                shape = (0 if a == 0 else 1, 0)
                errcnt += 1
            elif a.size > 1:
                arr = core.argument_conversion(a)
            else:
                shape = (0, a.ndim)
                errcnt += 1
    else:
        errcnt += 1

    if errcnt != 0:
        return core.array([], dtype=numpy.int64).reshape(shape)

    ########################################################################
    # TODO: VE-VH collaboration
    if arr._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError("argwhere on VH is not yet implemented.")

    ########################################################################
    # check order
    if arr._f_contiguous and not arr._c_contiguous:
        order_out = 'F'
    else:
        order_out = 'C'

    ########################################################################
    arr_bool = nlcpy.array(arr, dtype=bool)
    arr_bool_cnt = arr_bool[arr_bool].size
    out_shape = arr_bool_cnt

    if arr_bool_cnt != 0:
        # determins a shape of the output array,
        # where shape is (num of elements of True, a.ndim)
        out_shape = (arr_bool_cnt, arr.ndim)
    else:
        return core.array([], dtype=numpy.int64).reshape((0, arr.ndim))

    # argwhere is in F order
    x = ndarray(shape=out_shape, dtype=numpy.int64, order='F')

    request._push_request(
        "nlcpy_argwhere",
        "searching_op",
        (arr_bool, x),
    )

    return x

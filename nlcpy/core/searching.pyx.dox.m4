#
# * The source code in this file is based on the soure code of CuPy.
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
include(macros.m4)dnl
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

define(<--@reduction_function@-->,<--@
# ----------------------------------------------------------------------------
# $1
# see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.$1.html
# ----------------------------------------------------------------------------
## @fn $1(a, axis=None, out=None)
# @rename searching.$1 $1 nlcpy.$1
# @ingroup Searching
# @if(lang_ja)
# @else
# @pydoc
ifelse($1,argmax,<--@dnl
# @brief Returns the indices of the maximum values along an axis.
@-->,<--@dnl
# @brief Returns the indices of the minimum values along an axis.
@-->)dnl
#
# @details
# @param a : <em>array_like</em> @n
#   Input array.
#
# @param axis : <em>int, @b optional</em> @n
#   By default, the index is into the flattened array, otherwise along the specified axis.
#
# @param out : <em>@ref n-dimensional_array "ndarray", @b optional</em> @n
#   If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.
#
# @retval index_array : <em>@ref n-dimensional_array "ndarray" of ints</em> @n
#   Array of indices into the array. It has the same shape as @em a.shape with the dimension along @em axis removed.
#
# @sa
ifelse($1,argmax,<--@dnl
# @li @ref argmin "argmin" : Returns the indices of the minimum values along an axis.
# @li @ref order.amax "amax" : Returns the maximum of an array or maximum along an axis.
@-->,<--@dnl
# @li @ref argmax "argmax" : Returns the indices of the maximum values along an axis.
# @li @ref order.amin "amin" : Returns the minimum of an array or minimum along an axis.
@-->)dnl
#
# @note
ifelse($1,argmax,<--@dnl
#   In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
@-->,<--@dnl
#   In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned.
@-->)dnl
#
# @par Example
ifelse($1,argmax,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> a = vp.arange(6).reshape(2,3) + 10
# >>> a
# array([[10, 11, 12],
#        [13, 14, 15]])
# >>> vp.argmax(a)
# array(5)
# >>> vp.argmax(a, axis=0)
# array([1, 1, 1])
# >>> vp.argmax(a, axis=1)
# array([2, 2])
# @endcode @n
# @code
# >>> b = vp.arange(6)
# >>> b[1] = 5
# >>> b
# array([0, 5, 2, 3, 4, 5])
# >>> vp.argmax(b)  # Only the first occurrence is returned.
# array(1)
# @endcode
@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> a = vp.arange(6).reshape(2,3) + 10
# >>> a
# array([[10, 11, 12],
#        [13, 14, 15]])
# >>> vp.argmin(a)
# array(0)
# >>> vp.argmin(a, axis=0)
# array([0, 0, 0])
# >>> vp.argmin(a, axis=1)
# array([0, 0])
# @endcode @n
# @code
# >>> b = vp.arange(6) + 10
# >>> b[4] = 10
# >>> b
# array([10, 11, 12, 13, 10, 15])
# >>> vp.argmin(b)  # Only the first occurrence is returned.
# array(0)
# @endcode
@-->)dnl
#
# @endpydoc
# @endif
cpdef $1(a, axis=None, out=None):
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
        raise NotImplementedError("$1 on VH is not yet implemented.")

    if out is not None:
        if out._memloc in {on_VH, on_VE_VH}:
            raise NotImplementedError("$1 on VH is not yet implemented.")

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
        raise ValueError("attempt to get $1 of an empty sequence")

    ########################################################################
    # determine output shape
    if axis_save is None:
        if out is not None:
            raise ValueError("output array does not match result of $1")
        lst=[1, ]
    else:
        lst = list(arr.shape)
        for i, axis_i in enumerate(axis):
            lst.pop(axis_i-i if axis_i>=i else axis_i)
    shape_out = tuple(lst)

    if out is not None and out.shape != shape_out:
        raise ValueError("output array does not match result of np.$1.")

ifelse($1,argmax,<--@
    if arr.dtype.char == '?':
        initial = nlcpy.array(0, dtype=arr.dtype)
    elif arr.dtype.char in ('ilIL'):
        initial = nlcpy.array(nlcpy.iinfo(arr.dtype).min, dtype=arr.dtype)
    elif arr.dtype.char in ('dfDF'):
        initial = nlcpy.array(nlcpy.finfo(arr.dtype).min, dtype=arr.dtype)
@-->,<--@
    if arr.dtype.char == '?':
        initial = nlcpy.array(1, dtype=arr.dtype)
    elif arr.dtype.char in ('ilIL'):
        initial = nlcpy.array(nlcpy.iinfo(arr.dtype).max, dtype=arr.dtype)
    elif arr.dtype.char in ('dfDF'):
        initial = nlcpy.array(nlcpy.finfo(arr.dtype).max, dtype=arr.dtype)
@-->)dnl

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

        # call $1 function on VE
        request._push_request(
            "nlcpy_$1",
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

@-->)dnl
reduction_function(argmax)dnl
reduction_function(argmin)dnl

# ----------------------------------------------------------------------------
# nonzero
# see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html
# ----------------------------------------------------------------------------
## @fn nonzero(a)
# @rename searching.nonzero nonzero nlcpy.nonzero
# @ingroup Searching
# @if(lang_ja)
# @else
# @pydoc
# @brief Returns the indices of the elements that are non-zero.
#
# @details
#   Returns a tuple of arrays, one for each dimension of @em a, containing the indices of the non-zero elements
#   in that dimension. The values in @em a are always tested and returned in row-major, C-style order. @n
#   To group the indices by element, rather than dimension, use @ref nlcpy.argwhere "argwhere",
#   which returns a row for each non-zero element. @n
#
# @param a : <em>array_like</em> @n
#   Input array.
#
# @retval tuple_of_arrays : <em>tuple</em> @n
#   Indices of elements that are non-zero.
#
# @note
#   While the nonzero values can be obtained with <span class="pre">a[nonzero(a)]</span>, it is recommended to use
#   <span class="pre">x[x.astype(bool)]</span> or <span class="pre">x[x != 0]</span> instead,
#   which will correctly handle 0-d arrays.
#
# @par Example
# @code
# >>> import nlcpy as vp
# >>> x = vp.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
# >>> x
# array([[3, 0, 0],
#        [0, 4, 0],
#        [5, 6, 0]])
# >>> vp.nonzero(x)
# (array([0, 1, 2, 2]), array([0, 1, 0, 1]))
# @endcode @n
# @code
# >>> x[vp.nonzero(x)]
# array([3, 4, 5, 6])
# @endcode @n
# A common use for <span class="pre">nonzero</span> is to find the indices of an array, where a condition is True.
# Given an array @em a, the condition @em a > 3 is a boolean array and since False is interpreted as 0,
# np.nonzero(a > 3) yields the indices of the @em a where the condition is true.
# @code
# >>> a = vp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# >>> a > 3
# array([[False, False, False],
#        [ True,  True,  True],
#        [ True,  True,  True]])
# >>> vp.nonzero(a > 3)
# (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
# @endcode @n
# Using this result to index @em a is equivalent to using the mask directly: @n
# @code
# >>> a[vp.nonzero(a > 3)]
# array([4, 5, 6, 7, 8, 9])
# >>> a[a > 3]  # prefer this spelling
# array([4, 5, 6, 7, 8, 9])
# @endcode @n
# nonzero can also be called as a method of the array. @n
# @code
# >>> (a > 3).nonzero()
# (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
# @endcode
#
# @endpydoc
# @endif
cpdef nonzero(a):
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
## @fn argwhere(a)
# @rename searching.argwhere argwhere nlcpy.argwhere
# @ingroup Searching
# @if(lang_ja)
# @else
# @pydoc
# @brief Finds the indices of array elements that are non-zero, grouped by element.
#
# @details
#
# @param a : <em>array_like</em> @n
#   Input data.
#
# @retval index_array : <em>@ref n-dimensional_array "ndarray"</em> @n
#   Indices of elements that are non-zero. Indices are grouped by element.
#
# @sa
#  @li @ref generate.where "where" : Returns elements chosen from x or y depending on condition.
#  @li @ref nonzero "nonzero" : Returns the indices of the elements that are non-zero.
#
# @note
#  The output of argwhere is not suitable for indexing arrays. For this purpose use nonzero(a) instead.
#
# @par Example
# @code
# >>> import nlcpy as vp
# >>> x = vp.arange(6).reshape(2,3)
# >>> x
# array([[0, 1, 2],
#        [3, 4, 5]])
# >>> vp.argwhere(x>1)
# array([[0, 2],
#        [1, 0],
#        [1, 1],
#        [1, 2]])
# @endcode
#
# @endpydoc
# @endif
cpdef ndarray argwhere(a):
    errcnt = 0
    if a is not None:
        if type(a) is not ndarray:
            # TODO: using isscalar. needs replacing when implemented.
            if numpy.isscalar(a) is True:
                arr = core.array([a])
            elif isinstance(a, list) and len(a) >= 1:
                arr = core.argument_conversion(a)
            elif isinstance(a, tuple) and len(a) >= 1:
                arr = core.argument_conversion(a)
            elif isinstance(a, numpy.ndarray):
                arr = core.argument_conversion(a)
            else:
                dim = 1
                errcnt += 1
        else:
            if a.ndim == 0:
                arr = core.array([a.get()])
            elif a.size >= 1:
                arr = core.argument_conversion(a)
            else:
                dim = 1
                errcnt += 1
    else:
        errcnt += 1

    if errcnt != 0:
        return core.array([], dtype=numpy.int64).reshape((0, dim))

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
## @}

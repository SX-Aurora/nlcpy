# distutils: language = c++
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
import warnings
import numpy
from nlcpy.wrapper.numpy_wrap import numpy_wrap
cimport cython
cimport cpython


def amax(a, axis=None, out=None, keepdims=nlcpy._NoValue,
         initial=nlcpy._NoValue, where=nlcpy._NoValue):
    """Returns the maximum of an array or maximum along an axis.

    Args:
        a : array_like
            Array containing numbers whose maximum is desired. If a is not an array, a
            conversion is attempted.
        axis : None or int or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
            If this is a tuple of ints, the maximum is selected over multiple axes.
        out : `ndarray`, optional
            Alternative output array in which to place the result. Must be of the same
            shape and buffer length as the expected output.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the input array.
        initial : scalar, optional
            The maximum value of an output element. Must be present to allow computation
            on empty slice. See `reduce` for details.
        where : array_like of bool, optional
            Elements to compare for the maximum. See `reduce` for details.

    Returns:
        amax : `ndarray`
            An array with the same shape as a, with the specified axis removed. If a is a
            0-d array, or if axis is None, an ndarray scalar is returned. The same dtype
            as a is returned.

    Note:
        NaN values are propagated, that is if at least one item is NaN, the corresponding
        max value will be NaN as well. To ignore NaN values, please use nanmax.
        Don't use amax for element-wise comparison of 2 arrays; when a.shape[0] is 2,
        maximum(a[0], a[1]) is faster than amax(a, axis=0).

    See Also:
        amin : Returns the minimum of an array or minimum along an axis.
        nanmax : Returns maximum of an array or maximum along an axis, ignoring any NaNs.
        ufuncs.maximum : Element-wise maximum of array elements.
        ufuncs.fmax : Element-wise maximum of array elements.
        searching.argmax : Returns the indices of the maximum values along an axis.
        nanmin : Returns the minimum of an array or minimum along an axis, ignoring any
            NaNs.
        ufuncs.minimum : Element-wise minimum of array elements.
        ufuncs.fmin : Element-wise minimum of array elements.

    Examples:
        >>> import nlcpy as vp
        >>> a = vp.arange(4).reshape((2,2))
        >>> a
        array([[0, 1],
               [2, 3]])
        >>> vp.amax(a)           # Maximum of the flattened array
        array(3)
        >>> vp.amax(a, axis=0)   # Maxima along the first axis
        array([2, 3])
        >>> vp.amax(a, axis=1)   # Maxima along the second axis
        array([1, 3])
        >>> vp.amax(a, where=[False, True], initial=-1, axis=0)
        array([-1,  3])
        >>> b = vp.arange(5, dtype=float)
        >>> b[2] = vp.NaN
        >>> vp.amax(b)
        array(nan)
        >>> vp.amax(b, where=~vp.isnan(b), initial=-1)
        array(4.0)
        >>> vp.nanmax(b)
        array(4.0)

    """
    args = dict()
    if keepdims is not nlcpy._NoValue:
        args["keepdims"] = keepdims
    if initial is not nlcpy._NoValue:
        args["initial"] = initial
    if where is not nlcpy._NoValue:
        args["where"] = where
    return nlcpy.maximum.reduce(a, axis=axis, out=out, **args)


def amin(a, axis=None, out=None, keepdims=nlcpy._NoValue,
         initial=nlcpy._NoValue, where=nlcpy._NoValue):
    """Returns the minimum of an array or minimum along an axis.

    Args:
        a : array_like
            Array containing numbers whose minimum is desired. If a is not an array, a
            conversion is attempted.
        axis : None or int or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
            If this is a tuple of ints, the minimum is selected over multiple axes.
        out : `ndarray`, optional
            Alternative output array in which to place the result. Must be of the same
            shape and buffer length as the expected output.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the input array.
        initial : scalar, optional
            The maximum value of an output element. Must be present to allow computation
            on empty slice. See `reduce` for details.
        where : array_like of bool, optional
            Elements to compare for the minimum. See `reduce` for details.

    Returns:
        amin : `ndarray`
            An array with the same shape as a, with the specified axis removed. If a is a
            0-d array, or if axis is None, an ndarray scalar is returned. The same dtype
            as a is returned.

    Note:
        NaN values are propagated, that is if at least one item is NaN, the corresponding
        min value will be NaN as well. To ignore NaN values, please use nanmin.
        Don't use amin for element-wise comparison of 2 arrays; when a.shape[0] is 2,
        minimum(a[0], a[1]) is faster than amin(a, axis=0).

    See Also:
        amax : Returns the maximum of an array or maximum along an axis.
        nanmin : Returns minimum of an array or minimum along an axis, ignoring any NaNs.
        ufuncs.minimum : Element-wise minimum of array elements.
        ufuncs.fmin : Element-wise minimum of array elements.
        searching.argmin : Returns the indices of the minimum values along an axis.
        nanmax : Returns the maximum of an array or maximum along an axis, ignoring any
            NaNs.
        ufuncs.maximum : Element-wise maximum of array elements.
        ufuncs.fmax : Element-wise maximum of array elements.

    Examples:
        >>> import nlcpy as vp
        >>> a = vp.arange(4).reshape((2,2))
        >>> a
        array([[0, 1],
               [2, 3]])
        >>> vp.amin(a)           # Minimum of the flattened array
        array(0)
        >>> vp.amin(a, axis=0)   # Minima along the first axis
        array([0, 1])
        >>> vp.amin(a, axis=1)   # Minima along the second axis
        array([0, 2])
        >>> vp.amin(a, where=[False, True], initial=10, axis=0)
        array([10,  1])
        >>> b = vp.arange(5, dtype=float)
        >>> b[2] = vp.NaN
        >>> vp.amin(b)
        array(nan)
        >>> vp.amin(b, where=~vp.isnan(b), initial=10)
        array(0.)
        >>> vp.nanmin(b)
        array(0.)

    """
    args = dict()
    if keepdims is not nlcpy._NoValue:
        args["keepdims"] = keepdims
    if initial is not nlcpy._NoValue:
        args["initial"] = initial
    if where is not nlcpy._NoValue:
        args["where"] = where
    return nlcpy.minimum.reduce(a, axis=axis, out=out, **args)


def nanmax(a, axis=None, out=None, keepdims=nlcpy._NoValue):
    """Returns maximum of an array or maximum along an axis, ignoring any NaNs.

    When all-NaN slices are encountered a RuntimeWarning is raised and Nan is returned
    for that slice.

    Args:
        a : array_like
            Array containing numbers whose maximum is desired. If a is not an array, a
            conversion is attempted.
        axis : None or int or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
            If this is a tuple of ints, the maximum is selected over multiple axes.
        out : `ndarray`, optional
            Alternative output array in which to place the result. Must be of the same
            shape and buffer length as the expected output.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the input array.

    Returns:
        nanmax : `ndarray`
            An array with the same shape as a, with the specified axis removed. If a is a
            0-d array, or if axis is None, an ndarray scalar is returned. The same dtype
            as a is returned.

    Note:
        NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
        This means that Not a Number is not equivalent to infinity. Positive infinity is
        treated as a very large number and negative infinity is treated as a very small
        (i.e. negative) number.
        If the input has a integer type the function is equivalent to nlcpy.amax.

    See Also:
        nanmin : Returns the minimum of an array or minimum along an axis, ignoring any
            NaNs.
        amax : Returns the maximum of an array or maximum along an axis.
        ufuncs.fmax : Element-wise maximum of array elements.
        ufuncs.maximum : Element-wise maximum of array elements.
        ufuncs.isnan : Tests element-wise for NaN and return result as a boolean array.
        ufuncs.isfinite : Tests element-wise for finiteness (not infinity or not Not a
            Number).
        amin : Returns the minimum of an array or maximum along an axis.
        ufuncs.fmin : Element-wise minimum of array elements.
        ufuncs.minimum : Element-wise minimum of array elements.

    Examples:
        >>> import nlcpy as vp
        >>> a = vp.array([[1, 2], [3, vp.nan]))
        >>> vp.nanmax(a)
        array(3.)
        >>> vp.nanmax(a, axis=0)
        array([3., 2.])
        >>> vp.nanmax(a, axis=1)
        array([2., 3.])
        When positive infinity and negative infinity are present:
        >>> vp.nanmax([1, 2, vp.nan, vp.inf])
        array(2.)
        >>> vp.nanmax([1, 2, vp.nan, vp.NINF])
        array(inf)

    """
    a = nlcpy.core.argument_conversion(a)
    args = dict()
    if keepdims is not nlcpy._NoValue:
        args["keepdims"] = keepdims
    res = nlcpy.fmax.reduce(a, axis=axis, out=out, **args)
    if type(res) is nlcpy.ndarray and nlcpy.any(nlcpy.isnan(res)):
        warnings.warn("All-NaN slice encountered", RuntimeWarning)
    return res


def nanmin(a, axis=None, out=None, keepdims=nlcpy._NoValue):
    """Returns minimum of an array or minimum along an axis, ignoring any NaNs.

    When all-NaN slices are encountered a RuntimeWarning is raised and Nan is returned
    for that slice.

    Args:
        a : array_like
            Array containing numbers whose minimum is desired. If a is not an array, a
            conversion is attempted.
        axis : None or int or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
            If this is a tuple of ints, the minimum is selected over multiple axes.
        out : `ndarray`, optional
            Alternative output array in which to place the result. Must be of the same
            shape and buffer length as the expected output.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the input array.

    Returns:
        nanmin : `ndarray`
            An array with the same shape as a, with the specified axis removed. If a is a
            0-d array, or if axis is None, an ndarray scalar is returned. The same dtype
            as a is returned.

    Note:
        NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
        This means that Not a Number is not equivalent to infinity. Positive infinity is
        treated as a very large number and negative infinity is treated as a very small
        (i.e. negative) number.
        If the input has a integer type the function is equivalent to nlcpy.amin.

    See Also:
        nanmax : Returns the maximum of an array or maximum along an axis, ignoring any
            NaNs.
        amin : Returns the minimum of an array or maximum along an axis.
        ufuncs.fmin : Element-wise minimum of array elements.
        ufuncs.minimum : Element-wise minimum of array elements.
        ufuncs.isnan : Tests element-wise for NaN and return result as a boolean array.
        ufuncs.isfinite : Tests element-wise for finiteness (not infinity or not Not a
            Number).
        amax : Returns the maximum of an array or maximum along an axis.
        ufuncs.fmax : Element-wise maximum of array elements.
        ufuncs.maximum : Element-wise maximum of array elements.

    Examples:
        >>> import nlcpy as vp
        >>> a = vp.array([[1, 2], [3, vp.nan]))
        >>> vp.nanmin(a)
        array(1.)
        >>> vp.nanmin(a, axis=0)
        array([1., 2.])
        >>> vp.nanmin(a, axis=1)
        array([1., 3.])
        When positive infinity and negative infinity are present:
        >>> vp.nanmin([1, 2, vp.nan, vp.inf])
        array(1.)
        >>> vp.nanmin([1, 2, vp.nan, vp.NINF])
        array(-inf)

    """
    a = nlcpy.core.argument_conversion(a)
    args = dict()
    if keepdims is not nlcpy._NoValue:
        args["keepdims"] = keepdims
    res = nlcpy.fmin.reduce(a, axis=axis, out=out, **args)
    if type(res) is nlcpy.ndarray and nlcpy.any(nlcpy.isnan(res)):
        warnings.warn("All-NaN slice encountered", RuntimeWarning)
    return res

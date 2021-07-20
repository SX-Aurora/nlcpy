# distutils: language = c++
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
import numpy
import numbers
import warnings
import copy
import nlcpy
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport internal
from nlcpy.core.core cimport MemoryLocation
from nlcpy.core cimport core
from nlcpy.core cimport manipulation
from nlcpy.core cimport broadcast
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.core cimport dtype as _dtype
from nlcpy import veo
from nlcpy.manipulation.shape import reshape
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.statistics.function_base import *
from nlcpy.wrapper.numpy_wrap import numpy_wrap
cimport cython
cimport cpython

cpdef average(a, axis=None, weights=None, returned=False):
    """Computes the weighted average along the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing data to be averaged. If *a* is not an array, a conversion is
        attempted.
    axis : None or int, optional
        Axis along which to average *a*. The default, axis=None, will average over all of
        the elements of the input array. If axis is negative it counts from the last to
        the first axis.  tuple of axis not supported.
    weights : array_like, optional
        An array of weights associated with the values in *a*. Each value in *a*
        contributes to the average according to its associated weight. The weights array
        can either be 1-D (in which case its length must be the size of *a* along the
        given axis) or of the same shape as *a*. If *weights=None*, then all data in
        *a* are assumed to have a weight equal to one.
    returned : bool, optional
        Default is False. If True, the tuple average, *sum_of_weights* is returned,
        otherwise only the average is returned. If *weights=None*, *sum_of_weights* is
        equivalent to the number of elements over which the average is taken.

    Returns
    -------
    retval, [sum_of_weights] : ndarray
        Return the average along the specified axis. When *returned* is True, return a
        tuple with the average as the first element and the sum of the weights as the
        second element.
        *sum_of_weights* is of the same type as *retval*. The result dtype follows a
        general pattern.
        If *weights* is None, the result dtype will be that of *a* , or ``float64``
        if *a* is integral.
        Otherwise, if *weights* is not None and *a* is non-integral, the result type
        will be the type of lowest precision capable of representing values of both
        *a* and *weights*. If *a* happens to be integral, the previous rules still
        applies but the result dtype will at least be ``float64``.

    Restriction
    -----------
    * If *axis* is neither a scalar nor None : *NotImplementedError* occurs.
    * For complex numbers : *NotImplementedError* occurs.

    See Also
    --------
    mean : Computes the arithmetic mean along the specified axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> data = list(range(1,5))
    >>> data
    [1, 2, 3, 4]
    >>> vp.average(data)
    array(2.5)
    >>> vp.average(range(1,11), weights=range(10,0,-1))
    array(4.)
    >>> data = vp.arange(6).reshape((3,2))
    >>> data
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> vp.average(data, axis=1, weights=[1./4, 3./4])
    array([0.75, 2.75, 4.75])
    >>> vp.average(data, weights=[1./4, 3./4])
    Traceback (most recent call last):
        ...
    TypeError: Axis must be specified when shapes of a and weights differ.

    """
    if a is None:
        return None
    else:
        a = core.argument_conversion(a)

    nlcpy_chk_axis(a, axis=axis)

    nlcpy_chk_type(a)

    result_dtype = None

    if weights is None:
        avg = nlcpy.mean(a, axis)
        scl = avg.dtype.type(a.size / avg.size)
    else:
        wgt = core.argument_conversion(weights)

        if issubclass(a.dtype.type, (nlcpy.integer, nlcpy.bool_)):
            result_dtype = numpy.result_type(a.dtype, wgt.dtype, 'f8')

        else:
            result_dtype = a.dtype

        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")

            wgt = nlcpy.broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)

        scl = nlcpy.add.reduce(wgt, axis=axis, dtype=result_dtype)
        if nlcpy.any(scl == 0.0):
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")

        mul = nlcpy.multiply(a, wgt)
        tmp_cal = nlcpy.add.reduce(mul, axis=axis)
        avg = tmp_cal / scl

    avg = nlcpy.squeeze(avg)

    if returned:
        if scl.shape != avg.shape:
            if not isinstance(scl, nlcpy.core.core.ndarray):
                scl = nlcpy.array([scl])

            scl = nlcpy.broadcast_to(scl, avg.shape).copy()
            return avg, scl
        else:
            return avg, scl
    else:
        return avg


cpdef mean(a, axis=None, dtype=None, out=None, keepdims=nlcpy._NoValue):
    """Computes the arithmetic mean along the specified axis.

    Returns the average of the array elements. The average is taken over the flattened
    array by default, otherwise over the specified axis. **float64** intermediate and
    return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If *a* is not an array, a
        conversion is attempted.
    axis : None or int , optional
        Axis along which the means are computed. The default is to compute the mean of
        the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean. For integer inputs, the default is
        **float64**; for floating point inputs, it is the same as the input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result. The default is ``None``;
        if provided, it must have the same shape as the expected output, but the type
        will be cast if necessary. See :ref:`ufuncs <ufuncs>` for details.
    keepdims : bool, optional
        If this is set to True, the axis which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If *out=None*, returns a new array containing the mean values, otherwise a
        reference to the output array is returned.

    Restriction
    -----------
    * If *axis* is neither a scalar nor None : *NotImplementedError* occurs.
    * For complex numbers, *NotImplementedError* occurs.

    Note
    ----
    The arithmetic mean is the sum of the elements along the axis divided by the number
    of elements.

    Note that for floating-point input, the mean is computed using the same
    precision the input has. Depending on the input data, this can cause the results to
    be inaccurate, especially for **float32** (see example below). Specifying a
    higher-precision accumulator using the dtype keyword can alleviate this issue.

    See Also
    --------
    average : Weighted average
    std : Computes the standard deviation along the specified axis.
    var : Computes the variance along the specified axis.
    nanmean : Computes the arithmetic mean along the specified axis, ignoring NaNs.
    nanstd : Computes the standard deviation along the specified axis, while ignoring
        NaNs.
    nanvar : Computes the variance along the specified axis, while ignoring NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1, 2], [3, 4]])
    >>> vp.mean(a)
    array(2.5)
    >>> vp.mean(a, axis=0)
    array([2., 3.])
    >>> vp.mean(a, axis=1)
    array([1.5, 3.5])

    In single precision, mean can be inaccurate:

    >>> a = vp.zeros((2, 512*512), dtype=vp.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> vp.mean(a)
    array(0.5500002, dtype=float32)

    Computing the mean in float64 is more accurate:

    >>> vp.mean(a, dtype=vp.float64) # doctest: +SKIP
    array(0.55)

    """
    if a is None:
        if out is not None:
            out = None
        return None

    keepdims = True if keepdims is True else False

    a = nlcpy.asarray(a)

    if isinstance(axis, (list, tuple)):
        raise NotImplementedError('multiple axis is not implemented.')

    nlcpy_chk_axis(a, axis=axis)

    nlcpy_chk_type(a)

    if axis is not None:
        keep_shape = list(a.shape)
        keep_shape[axis] = 1
    else:
        keep_shape = (1,) * a.ndim

    a = core.argument_conversion(a)

    if dtype is None:
        if issubclass(a.dtype.type, (nlcpy.integer, nlcpy.bool_)):
            result_dtype = numpy.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    if axis is None:
        a_size = a.size
        flat_a = a.ravel()
        var_calc = nlcpy.add.reduce(flat_a, dtype=result_dtype,
                                    out=out, keepdims=keepdims)
        asf = var_calc / a_size
        asf = asf.astype(result_dtype)
    else:
        var_calc = nlcpy.add.reduce(a, axis=axis, dtype=result_dtype,
                                    out=out, keepdims=keepdims)
        asf = var_calc / a.shape[axis]
        asf = asf.astype(result_dtype)

    ans = asf

    if keepdims is True:
        ans = ans.reshape(keep_shape)
    else:
        ans = nlcpy.squeeze(ans)

    if out is not None:
        if isinstance(out, nlcpy.core.core.ndarray) is True:
            ans = nlcpy.squeeze(ans)
            if out.ndim != ans.ndim:
                raise TypeError(
                    'out is wrong dim(input={} output={})'.format(out.ndim,
                                                                  ans.ndim))
            if out.shape != ans.shape:
                raise TypeError(
                    'out is wrong shape(input={} output={})'.format(out.shape,
                                                                    ans.shape))

            try:
                out[...] = ans
            except Exception as e:
                out = 0
                raise TypeError('out shapes is wrong:{} '.format(e))

            out = out.astype(result_dtype)
            return out
        else:
            try:
                out = ans
            except Exception as e:
                out = 0
                raise TypeError('out shapes is wrong:{}'.format(e))

            out = out.astype(result_dtype)
            return out

    return ans


cpdef median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Computes the median along the specified axis.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or scalar that can be converted to an array.
    axis : int, None, optional
        Axis along which the medians are computed. The default is to compute the median
        along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape and buffer length as the expected output, but the type (of the output) will
        be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow use of memory of input array *a* for calculations. The input
        array will be modified by the call to median. This will save memory when you do
        not need to preserve the contents of the input array. Treat the input as
        undefined, but it will probably be fully or partially sorted. Default is False.
        If overwrite_input is True and *a* is not already a :obj:`ndarray`, an error will
        be raised.
    keepdims : bool, optional
        If this is set to True, the axis which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the original array.

    Returns
    -------
    median : ndarray
        A new array holding the result. If the input contains integers or floats smaller
        than ``float64``, then the output data-type is ``nlcpy.float64``. Otherwise, the
        data-type of the output is the same as that of the input. If *out* is specified,
        that array is returned instead.

    Restriction
    -----------
    * If *axis* is neither a scalar nor None : *NotImplementedError* occurs.
    * For complex numbers, *NotImplementedError* occurs.

    Note
    ----
    Given a vector ``V`` of length ``N``, the median of V is the middle value of a sorted
    copy of ``V, V_sorted`` - i.e., ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the
    average of the two middle values of ``V_sorted`` when ``N`` is even.

    See Also
    --------
    mean : Computes the arithmetic mean along the specified axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> vp.median(a)
    array(3.5)
    >>> vp.median(a, axis=0)
    array([6.5, 4.5, 2.5])
    >>> vp.median(a, axis=1)
    array([7., 2.])
    >>> m = vp.median(a, axis=0)
    >>> out = vp.zeros_like(m)
    >>> vp.median(a, axis=0, out=m)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])
    >>> b = a.copy()
    >>> vp.median(b, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not vp.all(a==b)
    >>> b = a.copy()
    >>> vp.median(b, axis=None, overwrite_input=True)
    array(3.5)
    >>> assert not vp.all(a==b)

    """
    if a is None:
        return None
    else:
        a = core.argument_conversion(a)

    nlcpy_chk_type(a)

    nlcpy_chk_axis(a, axis=axis)

    if axis is not None:
        keep_shape = list(a.shape)
        keep_shape[axis] = 1
        cp_a = a
    else:
        keep_shape = (1,) * a.ndim
        cp_a = a.ravel()
        axis = 0

    if overwrite_input:
        cp_a.sort(axis=axis)
        sort_a = cp_a
    else:
        sort_a = nlcpy.sort(cp_a, axis=axis)

    if sort_a.shape == ():
        return sort_a.item()

    if sort_a.ndim == 0:
        ret = sort_a.copy()
    else:

        if sort_a.shape[axis] % 2 == 1:
            hf = sort_a.shape[axis] // 2
            ret = sort_a.take(hf, axis=axis)
        else:
            hf1 = sort_a.shape[axis] // 2
            hf2 = hf1 - 1
            ret = (sort_a.take(hf1, axis=axis) + sort_a.take(hf2, axis=axis)) / 2

        new_a = nlcpy_median_nancheck(sort_a, ret, axis)
        ret = new_a

    if ret.dtype.kind in ('i', 'u', 'f'):
        if ret.dtype != numpy.dtype('f4'):
            ret = ret.astype('f8', copy=False)

    if keepdims:
        ret = ret.reshape(keep_shape)
    else:
        ret = nlcpy.squeeze(ret)

    if out is not None:
        if isinstance(out, nlcpy.core.core.ndarray) is True:
            ret = nlcpy.squeeze(ret)
            if out.ndim != ret.ndim:
                raise TypeError(
                    'median out is wrong dim(input={} output={})'.format(out.ndim,
                                                                         ret.ndim))

            if out.shape != ret.shape:
                raise TypeError(
                    'median out is wrong shape(input={} output={})'.format(out.shape,
                                                                           ret.shape))

            try:
                out[...] = ret
            except Exception as e:
                out = 0
                raise TypeError('out shapes is wrong:{}'.format(e))

            out = out.astype(ret.dtype)

            return out
        else:
            try:
                out = ret
            except Exception as e:
                out = 0
                raise TypeError('out shapes is wrong:{} '.format(e))

            out = out.astype(ret.dtype)
            return out

    return ret


cpdef nanmean(a, axis=None, dtype=None, out=None, keepdims=nlcpy._NoValue):
    """Computes the arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements. The average is taken over the flattened
    array by default, otherwise over the specified axis. **float64** intermediate and
    return values are used for integer inputs. For all-NaN slices, NaN is returned and a
    *RuntimeWarning* is raised.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If *a* is not an array, a
        conversion is attempted.
    axis : int, None, optional
        Axis along which the means are computed. The default is to compute the mean of
        the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean. For integer inputs, the default is
        **float64**; for inexact inputs, it is the same as the input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result. The default is ``None``; if
        provided, it must have the same shape as the expected output, but the type will
        be cast if necessary. See :ref:`ufuncs <ufuncs>` for details.
    keepdims : bool, optional
        If this is set to True, the axis which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the original *a*.

    Returns
    -------
    m : ndarray
        If *out=None*, returns a new array containing the mean values, otherwise a
        reference to the output array is returned.
        Nan is returned for slices that contain only NaNs.

    Restriction
    -----------
    * If *axis* is neither a scalar nor None : *NotImplementedError* occurs.
    * For complex numbers, *NotImplementedError* occurs.

    Note
    ----
    The arithmetic mean is the sum of the non-NaN elements along the axis divided by the
    number of non-NaN elements. Note that for floating-point input, the mean is computed
    using the same precision the input has. Depending on the input data, this can cause
    the results to be inaccurate, especially for **float32**. Specifying a
    higher-precision accumulator using the dtype keyword can alleviate this issue.

    See Also
    --------
    average : Weighted average
    mean : Arithmetic mean taken while not ignoring NaNs
    var : Computes the variance along the specified axis.
    nanvar : Computes the variance along the specified axis, while ignoring NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1, vp.nan], [3, 4]])
    >>> vp.nanmean(a)   # doctest: +SKIP
    array(2.66666667)
    >>> vp.nanmean(a, axis=0)  # doctest: +SKIP
    array([2., 4.])
    >>> vp.nanmean(a, axis=1)  # doctest: +SKIP
    array([1. , 3.5])


    """
    if a is None:
        return None

    a = core.argument_conversion(a)

    nlcpy_chk_axis(a, axis=axis)

    nlcpy_chk_type(a)

    keepdims = True if keepdims is True else False

    if a.size < 1:
        raise ValueError(
            "zero-size array to reduction operation maximum which has no identity")

    if axis is not None:
        keep_shape = list(a.shape)
        keep_shape[axis] = 1
    else:
        keep_shape = (1,) * a.ndim

    if dtype is None:
        if issubclass(a.dtype.type, (nlcpy.integer, nlcpy.bool_)):
            result_dtype = numpy.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    a_in = a

    cnt = nlcpy.where(nlcpy.isnan(a_in), 1, 0)
    if nlcpy.sum(cnt) == 0:
        if axis is None:
            ret = nlcpy.mean(a_in, axis=None, out=out, keepdims=keepdims)
            ans = ret
        else:
            ret = nlcpy.mean(a_in, axis=axis, out=out, keepdims=keepdims)
            ans = ret

        if keepdims is True:
            ans = ans.reshape(keep_shape)
        else:
            ans = nlcpy.squeeze(ans)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                ans = nlcpy.squeeze(ans)
                if out.ndim != ans.ndim or out.shape != ans.shape:
                    raise TypeError('out is wrong dim(input={} output={})'.format(
                        out.ndim, ans.ndim))

                try:
                    out[...] = ans
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)

                return out
            else:
                try:
                    out = ans
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)

                return out

    else:
        arr, mask = nlcpy_replace_nan(a, 0)

        h_mask = nlcpy_hatmask(mask)
        cnt = nlcpy.add.reduce(h_mask, axis=axis, dtype=nlcpy.intp, keepdims=keepdims)
        tot = nlcpy.add.reduce(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        avg = nlcpy_divide_by_count(tot, cnt, out=out)

        ans = avg

        if axis is not None:
            keep_shape = list(a.shape)
            keep_shape[axis] = 1
        else:
            keep_shape = (1,) * a.ndim

        if keepdims is True:
            ans = ans.reshape(keep_shape)
        else:
            ans = nlcpy.squeeze(ans)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                ans = nlcpy.squeeze(ans)
                if out.ndim != ans.ndim:
                    raise TypeError('out is wrong dim(input={} output={})'.format(
                        out.ndim, ans.ndim))

                if out.shape != ans.shape:
                    raise TypeError('out is wrong shape(input={} output={})'.format(
                        out.shape, ans.shape))

                try:
                    out[...] = ans
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)
                return out
            else:
                try:
                    out = ans
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)
                return out

    ans = ans.astype(result_dtype)

    return ans


@numpy_wrap
def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=nlcpy._NoValue):
    """Computes the median along the specified axis, while ignoring NaNs.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {int, sequenceof int, None}, optional
        Axis or axes along which the medians are computed. The default is to compute the
        median along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape as the expected output but the type (of the calculated values) will be cast
        if necessary.
    overwrite_input : bool, optional
        If True, then allow use of memory of input array *a* for calculations. The input
        array will be modified by the call to median. This will save memory when you do
        not need to preserve the contents of the input array. Treat the input as
        undefined, but it will probably be fully or partially sorted. Default is False.
        If *overwrite_input* is True and *a* is not already an ndarray, an error will be
        raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the original array.

    Returns
    -------
    median : ndarray
        A new array holding the result. If the input contains integers or floats smaller
        than ``float64``, then the output data-type is ``nlcpy.float64``. Otherwise, the
        data-type of the output is the same as that of the input. If out is specified,
        that array is returned instead.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.nanmedian`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    Note
    ----
    Given a vector ``V`` of length ``N``, the median of ``V`` is the middle value of a
    sorted copy of ``V, V_sorted`` - i.e., ``V_sorted[(N-1)/2]``, when ``N`` is odd,
    and the average of the two middle values of ``V_sorted`` when ``N`` is even.

    See Also
    --------
    mean : Computes the arithmetic mean along the specified axis.
    median : Compute the median along the specified axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a  = vp.array([[10.0, 7, 4], [3, 2, 1]])
    >>> a[0, 1] = vp.nan
    >>> a
    array([[10., nan,  4.],
           [ 3.,  2.,  1.]])
    >>> vp.median(a)
    array(nan)
    >>> vp.nanmedian(a)
    array(3.)
    >>> vp.nanmedian(a, axis=0)
    array([6.5, 2. , 2.5])
    >>> vp.median(a, axis=1)
    array([nan,  2.])
    >>> b = a.copy()
    >>> vp.nanmedian(b, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not vp.all(a==b)  # doctest: +SKIP
    >>> b = a.copy()
    >>> vp.nanmedian(b, axis=None, overwrite_input=True)
    array(3.)
    >>> assert not vp.all(a==b)  # doctest: +SKIP

    """
    raise NotImplementedError


cpdef nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=nlcpy._NoValue):
    """Computes the standard deviation along the specified axis, while ignoring NaNs.

    Returns the standard deviation, a measure of the spread of a distribution, of the
    non-NaN array elements. The standard deviation is computed for the flattened array by
    default, otherwise over the specified axis. For all-NaN slices or slices with zero
    degrees of freedom, NaN is returned and a *RuntimeWarning* is raised.

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of the non-NaN values.
    axis : int,  None, optional
        Axis along which the standard deviation is computed. The default is to compute
        the standard deviation of the flattened array.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of integer type the
        default is float64, for arrays of float types it is the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape as the expected output but the type (of the calculated values) will be cast
        if necessary.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations is ``N - ddof``,
        where ``N`` represents the number of non-NaN elements. By default, *ddof* is
        zero.
    keepdims : bool, optional
        If this is set to True, the axis which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the original *a*.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If out is None, return a new array containing the standard deviation, otherwise
        return a reference to the output array. If ddof is >= the number of non-NaN
        elements in a slice or the slice contains only NaNs, then the result for that
        slice is NaN.

    Restriction
    -----------
    * If *axis* is neither a scalar nor None : *NotImplementedError* occurs.
    * For complex numbers : *NotImplementedError* occurs.

    Note
    ----
    The standard deviation is the square root of the average of the squared deviations
    from the mean: ``std = sqrt(mean(abs(x - x.mean())**2))``.

    The average squared deviation is normally calculated as ``x.sum() / N``, where
    ``N = len(x)``. If, however, ddof is specified, the divisor ``N - ddof`` is used
    instead. In standard statistical practice, ``ddof=1`` provides an unbiased estimator
    of the variance of the infinite population. ``ddof=0`` provides a maximum likelihood
    estimate of the variance for normally distributed variables. The standard deviation
    computed in this function is the square root of the estimated variance, so even with
    ``ddof=1``, it will not be an unbiased estimate of the standard deviation per se.

    For floating-point input, the std is computed using the same precision the input has.
    Depending on the input data, this can cause the results to be inaccurate, especially
    for float32 (see example below). Specifying a higher-accuracy accumulator using the
    dtype keyword can alleviate this issue.

    See Also
    --------
    var : Computes the variance along the specified axis.
    mean : Computes the arithmetic mean along the specified axis.
    std : Computes the standard deviation along the specified axis.
    nanvar : Computes the variance along the specified axis, while ignoring NaNs.
    nanmean : Computes the arithmetic mean along the specified axis, ignoring NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1, vp.nan], [3, 4]])
    >>> vp.nanstd(a)    # doctest: +SKIP
    array(1.24721913)
    >>> vp.nanstd(a, axis=0)
    array([1., 0.])
    >>> vp.nanstd(a, axis=1)
    array([0. , 0.5])

    """
    if a is None:
        return None

    a = core.argument_conversion(a)

    nlcpy_chk_axis(a, axis=axis)

    nlcpy_chk_type(a)

    if a.size < 1:
        raise ValueError(
            "zero-size array to reduction operation maximum which has no identity")

    if ddof is not None and isinstance(ddof, int) is False:
        raise ValueError("ddof must be integer")

    if axis is not None:
        keep_shape = list(a.shape)
        keep_shape[axis] = 1
    else:
        keep_shape = (1,) * a.ndim

    if dtype is None:
        if issubclass(a.dtype.type, (nlcpy.integer, nlcpy.bool_)):
            result_dtype = numpy.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    a_in = a

    cnt = nlcpy.where(nlcpy.isnan(a_in), 1, 0)
    if nlcpy.sum(cnt) == 0:
        if axis is None:
            ret = nlcpy.std(a_in, axis=None, dtype=dtype,
                            out=out, ddof=ddof, keepdims=keepdims)
            ans = ret
        else:
            ret = nlcpy.std(a_in, axis=axis, dtype=dtype,
                            out=out, ddof=ddof, keepdims=keepdims)
            ans = ret

        if keepdims is True:
            ans = ans.reshape(keep_shape)
        else:
            ans = nlcpy.squeeze(ans)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                ans = nlcpy.squeeze(ans)
                if out.ndim != ans.ndim or out.shape != ans.shape:
                    raise TypeError('out is wrong dim(input={} output={})'.format(
                        out.ndim, ans.ndim))
                try:
                    out[...] = ans
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)
                return out
            else:
                try:
                    out = ans
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)
                return out
    else:
        var_data = nlcpy.nanvar(a, axis=axis, dtype=dtype, out=out,
                                ddof=ddof, keepdims=keepdims)
        if isinstance(var_data, nlcpy.core.core.ndarray):
            std = nlcpy.sqrt(var_data, out=var_data)
        else:
            tmp = nlcpy.sqrt(var_data)
            std = nlcpy.dtype.type(tmp)

        ans = std

        if axis is not None:
            keep_shape = list(a.shape)
            keep_shape[axis] = 1
        else:
            keep_shape = (1,) * a.ndim

        if keepdims is True:
            ans = ans.reshape(keep_shape)
        else:
            ans = nlcpy.squeeze(ans)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                ans = nlcpy.squeeze(ans)
                if out.ndim != ans.ndim:
                    raise TypeError('out is wrong dim(input={} output={})'.format(
                        out.ndim, ans.ndim))

                if out.shape != ans.shape:
                    raise TypeError('out is wrong shape(input={} output={})'.format(
                        out.shape, ans.shape))

                try:
                    out[...] = ans
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)
                return out
            else:
                try:
                    out = ans
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)
                return out

    ans = ans.astype(result_dtype)

    return ans


cpdef nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=nlcpy._NoValue):
    """Computes the variance along the specified axis, while ignoring NaNs.

    Returns the variance of the array elements, a measure of the spread of a
    distribution. The variance is computed for the flattened array by default, otherwise
    over the specified axis. For all-NaN slices or slices with zero degrees of freedom,
    NaN is returned and a *RuntimeWarning* is raised.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired. If *a* is not an array, a
        conversion is attempted.
    axis : int, None, optional
        Axis along which the variance is computed. The default is to compute the variance
        of the flattened array.
    dtype : data-type, optional
        Type to use in computing the variance. For arrays of integer type the default is
        float32; for arrays of float types it is the same as the array type.
    out : ndarray, optional
        Alternate output array in which to place the result. It must have the same shape
        as the expected output, but the type is cast if necessary.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is ``N - ddof``,
        where ``N`` represents the number of non-NaN elements. By default, *ddof* is
        zero.
    keepdims : bool, optional
        If this is set to True, the axis which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the original *a*.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If *out* is None, return a new array containing the variance, otherwise return a
        reference to the output array. If ddof is >= the number of non-NaN elements in a
        slice or the slice contains only NaNs, then the result for that slice is NaN.

    Restriction
    -----------
    * If *axis* is neither a scalar nor None : *NotImplementedError* occurs.
    * For complex numbers, *NotImplementedError* occurs.

    Note
    ----
    The variance is the average of the squared deviations from the mean, i.e.,
    ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``. If,
    however, *ddof* is specified, the divisor ``N - ddof`` is used instead. In standard
    statistical practice, ``ddof=1`` provides an unbiased estimator of the variance of
    a hypothetical infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for normally
    distributed variables.

    For floating-point input, the variance is computed using the same precision the input
    has. Depending on the input data, this can cause the results to be inaccurate,
    especially for float32 (see example below). Specifying a higher-accuracy accumulator
    using the ``dtype`` keyword can alleviate this issue.

    See Also
    --------
    std : Standard deviation
    mean : Average
    var : Variance while not ignoring NaNs
    nanstd : Computes the standard deviation along the specified axis, while ignoring
        NaNs.
    nanmean : Computes  the arithmetic mean along the specified axis, ignoring NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1, vp.nan], [3, 4]])
    >>> vp.nanvar(a)    # doctest: +SKIP
    array(1.55555556)
    >>> vp.nanvar(a, axis=0)
    array([1., 0.])
    >>> vp.nanvar(a, axis=1)
    array([0.  , 0.25])

    """
    if a is None:
        return None

    a = core.argument_conversion(a)

    nlcpy_chk_axis(a, axis=axis)

    nlcpy_chk_type(a)

    keepdims = True if keepdims is True else False

    if a.size < 1:
        raise ValueError(
            "zero-size array to reduction operation maximum which has no identity")

    if ddof is not None and isinstance(ddof, int) is False:
        raise ValueError("ddof must be integer")

    if axis is not None:
        keep_shape = list(a.shape)
        keep_shape[axis] = 1
    else:
        keep_shape = (1,) * a.ndim

    if dtype is None:
        if issubclass(a.dtype.type, (nlcpy.integer, nlcpy.bool_)):
            result_dtype = numpy.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    a_in = a

    cnt = nlcpy.where(nlcpy.isnan(a_in), 1, 0)
    if nlcpy.sum(cnt) == 0:
        if axis is None:
            ret = nlcpy.var(a_in, axis=None, dtype=result_dtype,
                            out=out, ddof=ddof, keepdims=keepdims)
            var_data = ret
        else:
            ret = nlcpy.var(a_in, axis=axis, dtype=result_dtype,
                            out=out, ddof=ddof, keepdims=keepdims)
            var_data = ret

        if keepdims is True:
            var_data = var.reshape(keep_shape)
        else:
            var_data = nlcpy.squeeze(var_data)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                var_data = nlcpy.squeeze(var_data)
                if out.ndim != var_data.ndim or out.shape != var_data.shape:
                    raise TypeError('out is wrong dim(input={} output={})'.format(
                        out.ndim, var_data.ndim))
                try:
                    out[...] = var_data
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)
                return out
            else:
                try:
                    out = var_data
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)
                return out

    else:

        arr, mask = nlcpy_replace_nan(a, 0)
        if mask is None:
            return nlcpy.var(arr, axis=axis, dtype=result_dtype, out=out,
                             ddof=ddof, keepdims=keepdims)

        if out is not None and not issubclass(out.dtype.type, nlcpy.inexact):
            raise TypeError("If a is inexact, then out must be inexact")

        _keepdims = True

        h_mask = nlcpy_hatmask(mask)
        cnt = nlcpy.add.reduce(h_mask, axis=axis, dtype=nlcpy.intp, keepdims=_keepdims)
        avg = nlcpy.add.reduce(arr, axis=axis, dtype=arr.dtype, keepdims=_keepdims)
        avg = nlcpy_divide_by_count(avg, cnt)

        arr = nlcpy.subtract(arr, avg)
        arr = nlcpy_copyto(arr, 0, mask)
        if issubclass(arr.dtype.type, nlcpy.complexfloating):
            sqr = nlcpy.multiply(arr, arr.conj()).real
        else:
            sqr = nlcpy.multiply(arr, arr)

        var_data = nlcpy.add.reduce(sqr, axis=axis, dtype=sqr.dtype,
                                    out=None, keepdims=keepdims)
        if var_data.ndim < cnt.ndim:
            cnt = nlcpy_wrapit(cnt, var_data.shape)

        dof = cnt - ddof
        var_data = nlcpy_divide_by_count(var_data, dof)

        is_bad = (dof <= 0)
        if nlcpy.any(is_bad):
            warnings.warn("Degrees of freedom <= 0 for slice.",
                          RuntimeWarning, stacklevel=3)
            var_data = nlcpy_copyto(var_data, nlcpy.nan, is_bad)

        if axis is not None:
            keep_shape = list(a.shape)
            keep_shape[axis] = 1
        else:
            keep_shape = (1,) * a.ndim

        if keepdims is True:
            var_data = var_data.reshape(keep_shape)
        else:
            var_data = nlcpy.squeeze(var_data)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                var_data = nlcpy.squeeze(var_data)
                if out.ndim != var_data.ndim:
                    raise TypeError('out is wrong dim(input={} output={})'.format(
                        out.ndim, var_data.ndim))

                if out.shape != var_data.shape:
                    raise TypeError('out is wrong shape(input={} output={})'.format(
                        out.shape, var_data.shape))

                try:
                    out[...] = var_data
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)
                return out
            else:
                try:
                    out = var_data
                except Exception as e:
                    out = 0
                    raise TypeError('out shapes is wrong:{} '.format(e))

                out = out.astype(result_dtype)
                return out

    var_data = var_data.astype(result_dtype)

    return var_data


cpdef std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=nlcpy._NoValue):
    """Computes the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution, of the
    array elements. The standard deviation is computed for the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of these values.
    axis : None or int, optional
        Axis along which the standard deviation is computed. The default is to compute
        the standard deviation of the flattened array.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of integer type the
        default is float64, for arrays of float types it is the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape as the expected output but the type (of the calculated values) will be cast
        if necessary.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations is ``N - ddof``,
        where ``N`` represents the number of elements. By default, *ddof* is zero.
    keepdims : bool, optional
        If this is set to True, the axis which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If *out* is None, return a new array containing the standard deviation, otherwise
        return a reference to the output array.

    Restriction
    -----------
    * If *axis* is neither a scalar nor None : *NotImplementedError* occurs.
    * For complex numbers : *NotImplementedError* occurs.

    Note
    ----
    The standard deviation is the square root of the average of the squared deviations
    from the mean, i.e., ``std = sqrt(mean(abs(x - x.mean())**2))``.

    The average squared deviation is normally calculated as ``x.sum() / N``, where
    ``N = len(x)``. If, however, ``ddof=1`` provides an unbiased estimator of the
    variance of the infinite population. ``ddof=0`` provides a maximum likelihood
    estimate of the variance for normally distributed variables. The standard deviation
    computed in this function is the square root of the estimated variance, so even with
    ``ddof=1``, it will not be an unbiased estimate of the standard deviation per se.

    For floating-point input, the std is computed using the same precision the input has.
    Depending on the input data, this can cause the results to be inaccurate, especially
    for float32 (see example below). Specifying a higher-accuracy accumulator using the
    dtype keyword can alleviate this issue.

    See Also
    --------
    var : Computes the variance along the specified axis.
    mean : Computes the arithmetic mean along the specified axis.
    nanmean : Computes the arithmetic mean along the specified axis, ignoring NaNs.
    nanstd : Computes the standard deviation along the specified axis, while ignoring
        NaNs.
    nanvar : Computes the variance along the specified axis, while ignoring NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1, 2], [3, 4]])
    >>> vp.std(a)    # doctest: +SKIP
    array(1.11803399)
    >>> vp.std(a, axis=0)  # doctest: +SKIP
    array([1., 1.])
    >>> vp.std(a, axis=1)  # doctest: +SKIP
    array([0.5, 0.5])

    In single precision, std() can be inaccurate:

    >>> a = vp.zeros((2, 512*512), dtype=vp.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> vp.std(a)      # doctest: +SKIP
    array(0.45002034, dtype=float32)

    Computing the standard deviation in float64 is more accurate:

    >>> vp.std(a, dtype=vp.float64)  # doctest: +SKIP
    array(0.45)

    """
    if a is None:
        return None

    a = core.argument_conversion(a)

    nlcpy_chk_axis(a, axis=axis)

    nlcpy_chk_type(a)

    if a.size < 1:
        raise ValueError(
            "zero-size array to reduction operation maximum which has no identity")

    if ddof is not None and isinstance(ddof, int) is False:
        raise ValueError(
            "std ddof({}) must be integer({})".format(ddof, type(ddof)))

    axis_save = axis

    if dtype is None:
        if issubclass(a.dtype.type, (nlcpy.integer, nlcpy.bool_)):
            result_dtype = numpy.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    var_s = nlcpy.var(a, axis=axis, dtype=result_dtype,
                      out=out, ddof=ddof, keepdims=False)
    ans = nlcpy.sqrt(var_s, out=out)

    if axis is not None:
        keep_shape = list(a.shape)
        keep_shape[axis] = 1
    else:
        keep_shape = (1,) * a.ndim

    if keepdims is True:
        ans = ans.reshape(keep_shape)
    else:
        ans = nlcpy.squeeze(ans)

    if out is not None:
        if isinstance(out, nlcpy.core.core.ndarray) is True:
            ans = nlcpy.squeeze(ans)
            if out.ndim != ans.ndim:
                raise TypeError(
                    'out is wrong dim(input={} output={})'.format(
                        out.ndim, ans.ndim))

            if out.shape != ans.shape:
                raise TypeError(
                    'out is wrong shape(input={} output={})'.format(
                        out.shape, ans.shape))

            try:
                ans1 = ans.astype(result_dtype)
                out[...] = ans1
            except Exception as e:
                out = 0
                raise TypeError('out shapes is wrong:{} '.format(e))

            out = out.astype(result_dtype)
            return out
        else:
            try:
                ans1 = ans.astype(result_dtype)
                out = ans1
            except Exception as e:
                out = 0
                raise TypeError('out shapes is wrong:{} '.format(e))

            out = out.astype(result_dtype)
            return out

    ans = ans.astype(result_dtype)

    return ans


cpdef var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=nlcpy._NoValue):
    """Computes the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution. The variance is computed for the flattened array by default, otherwise
    over the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired. If *a* is not an array, a
        conversion is attempted.
    axis : None or int, optional
        Axis  along which the variance is computed. The default is to compute the
        variance of the flattened array.
    dtype : data-type, optional
        Type to use in computing the variance. For arrays of integer type the default is
        float32; for arrays of float types it is the same as the array type.
    out : ndarray, optional
        Alternate output array in which to place the result. It must have the same shape
        as the expected output, but the type is cast if necessary.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is ``N - ddof``,
        where ``N`` represents the number of elements. By default, *ddof* is zero. The
        array or list to be shuffled.
    keepdims : bool, optional
        If this is set to True, the axis which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If ``out=None``, returns a new array containing the variance; otherwise, a
        reference to the output array is returned.

    Restriction
    -----------
    * If *axis* is neither a scalar nor None : *NotImplementedError* occurs.
    * For complex numbers, *NotImplementedError* occurs.

    Note
    ----
    The variance is the average of the squared deviations from the mean, i.e.,
    ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``. If,
    however, *ddof* is specified, the divisor ``N - ddof`` is used instead. In standard
    statistical practice, ``ddof=1`` provides an unbiased estimator of the variance of a
    hypothetical infinite population. ``ddof=0`` provides a maximum likelihood estimate
    of the variance for normally distributed variables.

    For floating-point input, the variance is computed using the same precision the input
    has. Depending on the input data, this can cause the results to be inaccurate,
    especially for float32 (see example below). Specifying a higher-accuracy accumulator
    using the ``dtype`` keyword can alleviate this issue.

    See Also
    --------
    std : Computes the standard deviation along the specified axis.
    mean : Computes the arithmetic mean along the specified axis.
    nanmean : Computes the arithmetic mean along the specified axis, ignoring NaNs.
    nanstd : Computes the standard deviation along the specified axis, while ignoring
        NaNs.
    nanvar : Computes the variance along the specified axis, while ignoring NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1, 2], [3, 4]])
    >>> vp.var(a)
    array(1.25)
    >>> vp.var(a, axis=0)
    array([1., 1.])
    >>> vp.var(a, axis=1)
    array([0.25, 0.25])

    In single precision, var() can be inaccurate:

    >>> a = vp.zeros((2, 512*512), dtype=vp.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> vp.var(a)
    array(0.2024999, dtype=float32)

    Computing the variance in float64 is more accurate:

    >>> vp.var(a, dtype=vp.float64)   # doctest: +SKIP
    array(0.2025) # may vary
    >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
    0.2025

    """
    if a is None:
        if out is not None:
            return None

    if not isinstance(a, nlcpy.core.core.ndarray):
        a = nlcpy.array(a)

    if a.size < 1:
        raise ValueError(
            "zero-size array to reduction operation maximum which has no identity")

    if ddof is not None and isinstance(ddof, int) is False:
        raise ValueError(
            "var ddof({}) must be integer({})".format(ddof, type(ddof)))

    nlcpy_chk_axis(a, axis=axis)

    if axis is not None:
        keep_shape = list(a.shape)
        keep_shape[axis] = 1
    else:
        keep_shape = (1,) * a.ndim

    nlcpy_chk_type(a)

    if dtype is None:
        if issubclass(a.dtype.type, (nlcpy.integer, nlcpy.bool_)):
            result_dtype = numpy.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    if axis is None:
        data_mean = nlcpy.mean(a, dtype=result_dtype)
        total = a.size
    else:
        data_mean = nlcpy.mean(a, axis=axis, dtype=result_dtype)
        total = a.shape[axis]

    if not isinstance(data_mean, nlcpy.core.core.ndarray):
        data_mean = nlcpy.array(data_mean)

    ans = nlcpy.mean(a * a, axis=axis, dtype=result_dtype,
                     keepdims=False) - (data_mean * data_mean)

    if dtype is not None:
        result_dtype = dtype
    else:
        result_dtype = ans.dtype

    if (ddof != 0):
        ans = ans * total / (total - ddof)

    if keepdims is True:
        ans = ans.reshape(keep_shape)
    else:
        ans = nlcpy.squeeze(ans)

    if out is not None:
        if isinstance(out, nlcpy.core.core.ndarray) is True:
            ans = nlcpy.squeeze(ans)
            if out.ndim != ans.ndim:
                raise TypeError(
                    'out is wrong dim(input={} output={})'.format(
                        out.ndim, ans.ndim))

            if out.shape != ans.shape:
                raise TypeError(
                    'out is wrong shape(input={} output={})'.format(
                        out.shape, ans.shape))

            try:
                ans1 = ans.astype(result_dtype)
                out[...] = ans1
                return out
            except Exception as e:
                out = 0
                raise TypeError('out shapes is wrong:{} '.format(e))

        else:
            try:
                ans1 = ans.astype(result_dtype)
                out = ans1
                return out
            except Exception as e:
                out = 0
                raise TypeError('out shapes is wrong:{} '.format(e))

    ans = ans.astype(result_dtype)
    return ans

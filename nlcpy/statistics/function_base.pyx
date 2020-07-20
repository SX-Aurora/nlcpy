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
from cpython cimport array
import numpy
import nlcpy as ny
import numbers
import warnings
import numpy as np
import ctypes
import numpy.ctypeslib as npct
import copy

import nlcpy
from nlcpy import veo
from nlcpy.core cimport core
from nlcpy.core cimport broadcast
from nlcpy.core.core cimport *
from nlcpy.manipulation.shape import reshape
from nlcpy.core.error import _AxisError as AxisError
cimport numpy as cnp

cpdef average(a, axis=None, weights=None, returned=False):
    """Computes the weighted average along the specified axis.

    Args:
        a : array_like
            Array containing data to be averaged. If a is not an array, a conversion is
            attempted.
        axis : None or int, optional
            Axis along which to average a. The default, axis=None, will average over all
            of the elements of the input array. If axis is negative it counts from the
            last to the first axis.  tuple of axis not supported.
        weights : array_like, optional
            An array of weights associated with the values in a. Each value in a
            contributes to the average according to its associated weight. The weights
            array can either be 1-D (in which case its length must be the size of a along
            the given axis) or of the same shape as a. If weights=None, then all data in
            a are assumed to have a weight equal to one.
        returned : bool, optional
            Default is False. If True, the tuple `average`, sum_of_weights is returned,
            otherwise only the average is returned. If weights=None, sum_of_weights is
            equivalent to the number of elements over which the average is taken.

    Returns:
        retval, [sum_of_weights] : `ndarray`
            Return the average along the specified axis. When returned is True, return a
            tuple with the average as the first element and the sum of the weights as the
            second element.
            sum_of_weights is of the same type as retval. The result dtype follows a
            general pattern.
            If weights is None, the result dtype will be that of a , or float64 if a is
            integral.
            Otherwise, if weights is not None and a is non-integral, the result type will
            be the type of lowest precision capable of representing values of both a and
            weights. If a happens to be integral, the previous rules still applies but
            the result dtype will at least be float64.

    Raises:
        axis is neither a scalar nor None : NotImplementedError occurs.
        For complex numbers, NotImplementedError occurs.

    See Also:
        mean : Computes the arithmetic mean along the specified axis.

    Examples:
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
        avg = ny.mean(a, axis)
        scl = avg.dtype.type(a.size / avg.size)
    else:
        wgt = core.argument_conversion(weights)

        if issubclass(a.dtype.type, (ny.integer, ny.bool_)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')

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

            wgt = ny.broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)

        scl = ny.add.reduce(wgt, axis=axis, dtype=result_dtype)
        if ny.any(scl == 0.0):
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")

        mul = ny.multiply(a, wgt)
        tmp_cal = ny.add.reduce(mul, axis=axis)
        avg = tmp_cal / scl

    avg = ny.squeeze(avg)

    if returned:
        if scl.shape != avg.shape:
            if not isinstance(scl, nlcpy.core.core.ndarray):
                scl = ny.array([scl])

            scl = ny.broadcast_to(scl, avg.shape).copy()
            return avg, scl
        else:
            return avg, scl
    else:
        return avg

cpdef mean(a, axis=None, dtype=None, out=None, keepdims=nlcpy._NoValue):
    """Computes the arithmetic mean along the specified axis.

    Returns the average of the array elements. The average is taken over the flattened
    array by default, otherwise over the specified axis. float64 intermediate and return
    values are used for integer inputs.

    Args:
        a : array_like
            Array containing numbers whose mean is desired. If a is not an array, a
            conversion is attempted.
        axis : None or int , optional
            Axis along which the means are computed. The default is to compute the mean
            of the flattened array.
        dtype : data-type, optional
            Type to use in computing the mean. For integer inputs, the default is
            float64; for floating point inputs, it is the same as the input dtype.
        out : `ndarray`, optional
            Alternate output array in which to place the result. The default is None; if
            provided, it must have the same shape as the expected output, but the type
            will be cast if necessary. See `ufuncs` for details.
        keepdims : bool, optional
            If this is set to True, the axis which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the input array.

    Returns:
        m : `ndarray`, see dtype parameter above
            If out=None, returns a new array containing the mean values, otherwise a
            reference to the output array is returned.

    Raises:
        axis is neither a scalar nor None : NotImplementedError occurs.
        For complex numbers, NotImplementedError occurs.

    Note:
        The arithmetic mean is the sum of the elements along the axis divided by the
        number of elements. Note that for floating-point input, the mean is computed
        using the same precision the input has. Depending on the input data, this can
        cause the results to be inaccurate, especially for float32 (see example below).
        Specifying a higher-precision accumulator using the dtype keyword can alleviate
        this issue.

    See Also:
        average : Weighted average
        std : Computes the standard deviation along the specified axis.
        var : Computes the variance along the specified axis.
        nanmean : Computes the arithmetic mean along the specified axis, ignoring NaNs.
        nanstd : Computes the standard deviation along the specified axis, while ignoring
            NaNs.
        nanvar : Computes the variance along the specified axis, while ignoring NaNs.

    Examples:
        >>> import nlcpy as vp
        >>> a = vp.array([[1, 2], [3, 4]])
        >>> vp.mean(a)
        array(2.5)
        >>> vp.mean(a, axis=0)
        array([2., 3.])
        >>> vp.mean(a, axis=1)
        array([1.5, 3.5])
        In single precision, `mean` can be inaccurate:
        >>>
        >>> a = vp.zeros((2, 512*512), dtype=vp.float32)
        >>> a[0, :] = 1.0
        >>> a[1, :] = 0.1
        >>> vp.mean(a)
        array(0.5499878, dtype=float32)
        Computing the mean in float64 is more accurate:
        >>>
        >>> vp.mean(a, dtype=vp.float64)
        array(0.55 )# may vary

    """
    if a is None:
        if out is not None:
            out = None
        return None

    keepdims = True if keepdims is True else False

    a = ny.asarray(a)

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
        if issubclass(a.dtype.type, (ny.integer, ny.bool_)):
            result_dtype = np.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    if axis is None:
        a_size = a.size
        flat_a = a.ravel()
        var_calc = ny.add.reduce(flat_a, dtype=result_dtype, out=out, keepdims=keepdims)
        asf = var_calc / a_size
        asf = asf.astype(result_dtype)
    else:
        var_calc = ny.add.reduce(a, axis=axis, dtype=result_dtype,
                                 out=out, keepdims=keepdims)
        asf = var_calc / a.shape[axis]
        asf = asf.astype(result_dtype)

    ans = asf

    if keepdims is True:
        ans = ans.reshape(keep_shape)
    else:
        ans = ny.squeeze(ans)

    if out is not None:
        if isinstance(out, nlcpy.core.core.ndarray) is True:
            ans = ny.squeeze(ans)
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

    Args:
        a : array_like
            Input array or scalar that can be converted to an array.
        axis : int, None, optional
            Axis along which the medians are computed. The default is to compute the
            median along a flattened version of the array.
        out : `ndarray`, optional
            Alternative output array in which to place the result. It must have the same
            shape and buffer length as the expected output, but the type (of the output)
            will be cast if necessary.
        overwrite_input : bool, optional
            If True, then allow use of memory of input array a for calculations. The
            input array will be modified by the call to `median`. This will save memory
            when you do not need to preserve the contents of the input array. Treat the
            input as undefined, but it will probably be fully or partially sorted.
            Default is False. If overwrite_input is True and a is not already a
            `ndarray`, an error will be raised.
        keepdims : bool, optional
            If this is set to True, the axis which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the original array.

    Returns:
        median : `ndarray`
            A new array holding the result. If the input contains integers or floats
            smaller than float64, then the output data-type is nlcpy.float64. Otherwise,
            the data-type of the output is the same as that of the input. If out is
            specified, that array is returned instead.

    Raises:
        axis is neither a scalar nor None : NotImplementedError occurs.
        For complex numbers, NotImplementedError occurs.

    Note:
        Given a vector V of length N, the median of V is the middle value of a sorted
        copy of V, V_sorted - i.e., V_sorted[(N-1)/2], when N is odd, and the average of
        the two middle values of V_sorted when N is even.

    See Also:
        mean : Computes the arithmetic mean along the specified axis.

    Examples:
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
        array([7.,  2.])
        >>> m = vp.median(a, axis=0)
        >>> out = vp.zeros_like(m)
        >>> vp.median(a, axis=0, out=m)
        array([6.5,  4.5,  2.5])
        >>> m
        array([6.5,  4.5,  2.5])
        >>> b = a.copy()
        >>> vp.median(b, axis=1, overwrite_input=True)
        array([7.,  2.])
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
        sort_a = ny.sort(cp_a, axis=axis)

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
        ret = ny.squeeze(ret)

    if out is not None:
        if isinstance(out, nlcpy.core.core.ndarray) is True:
            ret = ny.squeeze(ret)
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
    array by default, otherwise over the specified axis. float64 intermediate and return
    values are used for integer inputs. For all-NaN slices, NaN is returned and a
    RuntimeWarning is raised.

    Args:
        a : array_like
            Array containing numbers whose mean is desired. If a is not an array, a
            conversion is attempted.
        axis : int, None, optional
            Axis along which the means are computed. The default is to compute the mean
            of the flattened array.
        dtype : data-type, optional
            Type to use in computing the mean. For integer inputs, the default is
            float64; for inexact inputs, it is the same as the input dtype.
        out : `ndarray`, optional
            Alternate output array in which to place the result. The default is None; if
            provided, it must have the same shape as the expected output, but the type
            will be cast if necessary. See `ufuncs` for details.
        keepdims : bool, optional
            If this is set to True, the axis which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the original a. If out=None, returns a new array containing
            the mean values, otherwise a reference to the output array is returned. Nan
            is returned for slices that contain only NaNs.

    Raises:
        axis is neither a scalar nor None : NotImplementedError occurs.
        For complex numbers, NotImplementedError occurs.

    Note:
        The arithmetic mean is the sum of the non-NaN elements along the axis divided by
        the number of non-NaN elements. Note that for floating-point input, the mean is
        computed using the same precision the input has. Depending on the input data,
        this can cause the results to be inaccurate, especially for float32. Specifying a
        higher-precision accumulator using the dtype keyword can alleviate this issue.

    See Also:
        average : Weighted average
        mean : Arithmetic mean taken while not ignoring NaNs
        var : Computes the variance along the specified axis.
        nanvar : Computes the variance along the specified axis, while ignoring NaNs.

    Examples:
        >>> import nlcpy as vp
        >>> a = vp.array([[1, vp.nan], [3, 4]])
        >>> vp.nanmean(a)
        array(2.66666667)
        >>> vp.nanmean(a, axis=0)
        array([2.,  4.])
        >>> vp.nanmean(a, axis=1)
        array([1.,  3.5]) # may vary

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

    if a._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError('nanmean on VH is not yet implemented.')

    if out is not None:
        if out._memloc in {on_VH, on_VE_VH}:
            raise NotImplementedError('nanmean on VH is not yet implemented.')

    if axis is not None:
        keep_shape = list(a.shape)
        keep_shape[axis] = 1
    else:
        keep_shape = (1,) * a.ndim

    if dtype is None:
        if issubclass(a.dtype.type, (ny.integer, ny.bool_)):
            result_dtype = np.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    a_in = a

    cnt = ny.where(ny.isnan(a_in), 1, 0)
    if ny.sum(cnt) == 0:
        if axis is None:
            ret = ny.mean(a_in, axis=None, out=out, keepdims=keepdims)
            ans = ret
        else:
            ret = ny.mean(a_in, axis=axis, out=out, keepdims=keepdims)
            ans = ret

        if keepdims is True:
            ans = ans.reshape(keep_shape)
        else:
            ans = ny.squeeze(ans)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                ans = ny.squeeze(ans)
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
        cnt = ny.add.reduce(h_mask, axis=axis, dtype=ny.intp, keepdims=keepdims)
        tot = ny.add.reduce(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
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
            ans = ny.squeeze(ans)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                ans = ny.squeeze(ans)
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


cpdef nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=nlcpy._NoValue):
    """Computes the standard deviation along the specified axis, while ignoring NaNs.

    Returns the standard deviation, a measure of the spread of a distribution, of the
    non-NaN array elements. The standard deviation is computed for the flattened array by
    default, otherwise over the specified axis. For all-NaN slices or slices with zero
    degrees of freedom, NaN is returned and a RuntimeWarning is raised.

    Args:
        a : array_like
            Calculate the standard deviation of the non-NaN values.
        axis : int,  None, optional
            Axis along which the standard deviation is computed. The default is to
            compute the standard deviation of the flattened array.
        dtype : dtype, optional
            Type to use in computing the standard deviation. For arrays of integer type
            the default is float64, for arrays of float types it is the same as the array
            type.a
        out : `ndarray`, optional
            Alternative output array in which to place the result. It must have the same
            shape as the expected output but the type (of the calculated values) will be
            cast if necessary.
        ddof : int, optional
            Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of non-NaN elements. By default, ddof is zero.
        keepdims : bool, optional
            If this is set to True, the axis which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the original a.

    Returns:
        standard_deviation : `ndarray`, see dtype parameter above.
            If out is None, return a new array containing the standard deviation,
            otherwise return a reference to the output array. If ddof is >= the number of
            non-NaN elements in a slice or the slice contains only NaNs, then the result
            for that slice is NaN.

    Raises:
        axis is neither a scalar nor None : NotImplementedError occurs.
        For complex numbers, NotImplementedError occurs.

    Note:
        The standard deviation is the square root of the average of the squared
        deviations from the mean: std = sqrt(mean(abs(x - x.mean())**2)). The average
        squared deviation is normally calculated as x.sum() / N, where N = len(x). If,
        however, ddof is specified, the divisor N - ddof is used instead. In standard
        statistical practice, ddof=1 provides an unbiased estimator of the variance of
        the infinite population. ddof=0 provides a maximum likelihood estimate of the
        variance for normally distributed variables. The standard deviation computed in
        this function is the square root of the estimated variance, so even with ddof=1,
        it will not be an unbiased estimate of the standard deviation per se. For
        floating-point input, the std is computed using the same precision the input has.
        Depending on the input data, this can cause the results to be inaccurate,
        especially for float32 (see example below). Specifying a higher-accuracy
        accumulator using the dtype keyword can alleviate this issue.

    See Also:
        var : Computes the variance along the specified axis.
        mean : Computes the arithmetic mean along the specified axis.
        std : Computes the standard deviation along the specified axis.
        nanvar : Computes the variance along the specified axis, while ignoring NaNs.
        nanmean : Computes the arithmetic mean along the specified axis, ignoring NaNs.

    Examples:
        >>> import nlcpy as vp
        >>> a = vp.array([[1, vp.nan], [3, 4]])
        >>> vp.nanstd(a)
        array(1.24721913)
        >>> vp.nanstd(a, axis=0)
        array([1., 0.])
        >>> vp.nanstd(a, axis=1)
        array([0.,  0.5]) # may vary

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

    if a._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError('nanstd on VH is not yet implemented.')

    if out is not None:
        if out._memloc in {on_VH, on_VE_VH}:
            raise NotImplementedError('nanstd on VH is not yet implemented.')

    if axis is not None:
        keep_shape = list(a.shape)
        keep_shape[axis] = 1
    else:
        keep_shape = (1,) * a.ndim

    if dtype is None:
        if issubclass(a.dtype.type, (ny.integer, ny.bool_)):
            result_dtype = np.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    a_in = a

    cnt = ny.where(ny.isnan(a_in), 1, 0)
    if ny.sum(cnt) == 0:
        if axis is None:
            ret = ny.std(a_in, axis=None, dtype=dtype,
                         out=out, ddof=ddof, keepdims=keepdims)
            ans = ret
        else:
            ret = ny.std(a_in, axis=axis, dtype=dtype,
                         out=out, ddof=ddof, keepdims=keepdims)
            ans = ret

        if keepdims is True:
            ans = ans.reshape(keep_shape)
        else:
            ans = ny.squeeze(ans)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                ans = ny.squeeze(ans)
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
        var_data = ny.nanvar(a, axis=axis, dtype=dtype, out=out,
                             ddof=ddof, keepdims=keepdims)
        if isinstance(var_data, nlcpy.core.core.ndarray):
            std = ny.sqrt(var_data, out=var_data)
        else:
            tmp = ny.sqrt(var_data)
            std = ny.dtype.type(tmp)

        ans = std

        if axis is not None:
            keep_shape = list(a.shape)
            keep_shape[axis] = 1
        else:
            keep_shape = (1,) * a.ndim

        if keepdims is True:
            ans = ans.reshape(keep_shape)
        else:
            ans = ny.squeeze(ans)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                ans = ny.squeeze(ans)
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
    NaN is returned and a RuntimeWarning is raised.

    Args:
        a : array_like
            Array containing numbers whose variance is desired. If a is not an array, a
            conversion is attempted.
        axis : int, None, optional
            Axis along which the variance is computed. The default is to compute the
            variance of the flattened array.
        dtype : data-type, optional
            Type to use in computing the variance. For arrays of integer type the default
            is float32; for arrays of float types it is the same as the array type.
        out : `ndarray`, optional
            Alternate output array in which to place the result. It must have the same
            shape as the expected output, but the type is cast if necessary.
        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof,
            where N represents the number of non-NaN elements. By default, ddof is zero.
        keepdims : bool, optional
            If this is set to True, the axis which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the original a.

    Returns:
        variance : `ndarray`, see dtype parameter above
            If out is None, return a new array containing the variance, otherwise return
            a reference to the output array. If ddof is >= the number of non-NaN elements
            in a slice or the slice contains only NaNs, then the result for that slice is
            NaN.

    Raises:
        axis is neither a scalar nor None : NotImplementedError occurs.
        For complex numbers, NotImplementedError occurs.

    Note:
        The variance is the average of the squared deviations from the mean, i.e., var =
        mean(abs(x - x.mean())**2). The mean is normally calculated as x.sum() / N, where
        N = len(x). If, however, ddof is specified, the divisor N - ddof is used instead.
        In standard statistical practice, ddof=1 provides an unbiased estimator of the
        variance of a hypothetical infinite population. ddof=0 provides a maximum
        likelihood estimate of the variance for normally distributed variables. For
        floating-point input, the variance is computed using the same precision the input
        has. Depending on the input data, this can cause the results to be inaccurate,
        especially for float32 (see example below). Specifying a higher-accuracy
        accumulator using the dtype keyword can alleviate this issue.

    See Also:
        std : Standard deviation
        mean : Average
        var : Variance while not ignoring NaNs
        nanstd : Computes the standard deviation along the specified axis, while ignoring
            NaNs.
        nanmean : Computes  the arithmetic mean along the specified axis, ignoring NaNs.

    Examples:
        >>> import nlcpy as vp
        >>> a = vp.array([[1, vp.nan], [3, 4]])
        >>> vp.nanvar(a)
        array(1.55555556)
        >>> vp.nanvar(a, axis=0)
        array([1.,  0.])
        >>> vp.nanvar(a, axis=1)
        array([0.,  0.25])

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

    if a._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError('nanvar on VH is not yet implemented.')

    if out is not None:
        if out._memloc in {on_VH, on_VE_VH}:
            raise NotImplementedError('nanvar on VH is not yet implemented.')

    if axis is not None:
        keep_shape = list(a.shape)
        keep_shape[axis] = 1
    else:
        keep_shape = (1,) * a.ndim

    if dtype is None:
        if issubclass(a.dtype.type, (ny.integer, ny.bool_)):
            result_dtype = np.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    a_in = a

    cnt = ny.where(ny.isnan(a_in), 1, 0)
    if ny.sum(cnt) == 0:
        if axis is None:
            ret = ny.var(a_in, axis=None, dtype=result_dtype,
                         out=out, ddof=ddof, keepdims=keepdims)
            var_data = ret
        else:
            ret = ny.var(a_in, axis=axis, dtype=result_dtype,
                         out=out, ddof=ddof, keepdims=keepdims)
            var_data = ret

        if keepdims is True:
            var_data = var.reshape(keep_shape)
        else:
            var_data = ny.squeeze(var_data)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                var_data = ny.squeeze(var_data)
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
            return ny.var(arr, axis=axis, dtype=result_dtype, out=out,
                          ddof=ddof, keepdims=keepdims)

        if out is not None and not issubclass(out.dtype.type, ny.inexact):
            raise TypeError("If a is inexact, then out must be inexact")

        _keepdims = True

        h_mask = nlcpy_hatmask(mask)
        cnt = ny.add.reduce(h_mask, axis=axis, dtype=ny.intp, keepdims=_keepdims)
        avg = ny.add.reduce(arr, axis=axis, dtype=arr.dtype, keepdims=_keepdims)
        avg = nlcpy_divide_by_count(avg, cnt)

        arr = ny.subtract(arr, avg)
        arr = nlcpy_copyto(arr, 0, mask)
        if issubclass(arr.dtype.type, ny.complexfloating):
            sqr = ny.multiply(arr, arr.conj()).real
        else:
            sqr = ny.multiply(arr, arr)

        var_data = ny.add.reduce(sqr, axis=axis, dtype=sqr.dtype,
                                 out=None, keepdims=keepdims)
        if var_data.ndim < cnt.ndim:
            cnt = nlcpy_wrapit(cnt, var_data.shape)

        dof = cnt - ddof
        var_data = nlcpy_divide_by_count(var_data, dof)

        is_bad = (dof <= 0)
        if ny.any(is_bad):
            warnings.warn("Degrees of freedom <= 0 for slice.",
                          RuntimeWarning, stacklevel=3)
            var_data = nlcpy_copyto(var_data, ny.nan, is_bad)

        if axis is not None:
            keep_shape = list(a.shape)
            keep_shape[axis] = 1
        else:
            keep_shape = (1,) * a.ndim

        if keepdims is True:
            var_data = var_data.reshape(keep_shape)
        else:
            var_data = ny.squeeze(var_data)

        if out is not None:
            if isinstance(out, nlcpy.core.core.ndarray) is True:
                var_data = ny.squeeze(var_data)
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

    Args:
        a : array_like
            Calculate the standard deviation of these values.
        axis : None or int, optional
            Axis along which the standard deviation is computed. The default is to
            compute the standard deviation of the flattened array.
        dtype : dtype, optional
            Type to use in computing the standard deviation. For arrays of integer type
            the default is float64, for arrays of float types it is the same as the array
            type.
        out : `ndarray`, optional
            Alternative output array in which to place the result. It must have the same
            shape as the expected output but the type (of the calculated values) will be
            cast if necessary.
        ddof : int, optional
            Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements. By default, ddof is zero.
        keepdims : bool, optional
            If this is set to True, the axis which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the input array.

    Returns:
        standard_deviation : `ndarray`, see dtype parameter above.
            If out is None, return a new array containing the standard deviation,
            otherwise return a reference to the output array.

    Raises:
        axis is neither a scalar nor None : NotImplementedError occurs.
        For complex numbers, NotImplementedError occurs.

    Note:
        The standard deviation is the square root of the average of the squared
        deviations from the mean, i.e., std = sqrt(mean(abs(x - x.mean())**2)).
        The average squared deviation is normally calculated as x.sum() / N, where N =
        len(x). If, however, ddof=1 provides an unbiased estimator of the variance of the
        infinite population. ddof=0 provides a maximum likelihood estimate of the
        variance for normally distributed variables. The standard deviation computed in
        this function is the square root of the estimated variance, so even with ddof=1,
        it will not be an unbiased estimate of the standard deviation per se.
        For floating-point input, the std is computed using the same precision the input
        has. Depending on the input data, this can cause the results to be inaccurate,
        especially for float32 (see example below). Specifying a higher-accuracy
        accumulator using the dtype keyword can alleviate this issue.

    See Also:
        var : Computes the variance along the specified axis.
        mean : Computes the arithmetic mean along the specified axis.
        nanmean : Computes the arithmetic mean along the specified axis, ignoring NaNs.
        nanstd : Computes the standard deviation along the specified axis, while ignoring
            NaNs.
        nanvar : Computes the variance along the specified axis, while ignoring NaNs.

    Examples:
        >>> import nlcpy as vp
        >>> a = vp.array([[1, 2], [3, 4]])
        >>> vp.std(a)
        array(1.11803399) # may vary
        >>> vp.std(a, axis=0)
        array([1.,  1.])
        >>> vp.std(a, axis=1)
        array([0.5,  0.5])
        In single precision, std() can be inaccurate:
        >>> a = vp.zeros((2, 512*512), dtype=vp.float32)
        >>> a[0, :] = 1.0
        >>> a[1, :] = 0.1
        >>> vp.std(a)
        array(0.45002034, dtype=float32)
        Computing the standard deviation in float64 is more accurate:
        >>> vp.std(a, dtype=vp.float64)
        array(0.45) # may vary

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

    if a._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError('std on VH is not yet implemented.')

    if out is not None:
        if out._memloc in {on_VH, on_VE_VH}:
            raise NotImplementedError(
                'std on VH is not yet implemented.')
    axis_save = axis

    if dtype is None:
        if issubclass(a.dtype.type, (ny.integer, ny.bool_)):
            result_dtype = np.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    var_s = ny.var(a, axis=axis, dtype=result_dtype, out=out, ddof=ddof, keepdims=False)
    ans = ny.sqrt(var_s, out=out)

    if axis is not None:
        keep_shape = list(a.shape)
        keep_shape[axis] = 1
    else:
        keep_shape = (1,) * a.ndim

    if keepdims is True:
        ans = ans.reshape(keep_shape)
    else:
        ans = ny.squeeze(ans)

    if out is not None:
        if isinstance(out, nlcpy.core.core.ndarray) is True:
            ans = ny.squeeze(ans)
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

    Args:
        a : array_like
            Array containing numbers whose variance is desired. If a is not an array, a
            conversion is attempted.
        axis : None or int, optional
            Axis  along which the variance is computed. The default is to compute the
            variance of the flattened array.
        dtype : data-type, optional
            Type to use in computing the variance. For arrays of integer type the default
            is float32; for arrays of float types it is the same as the array type.
        out : `ndarray`, optional
            Alternate output array in which to place the result. It must have the same
            shape as the expected output, but the type is cast if necessary.
        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof,
            where N represents the number of elements. By default, ddof is zero. The
            array or list to be shuffled.
        keepdims : bool, optional
            If this is set to True, the axis which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the input array.

    Returns:
        variance : `ndarray`, see dtype parameter above
            If out=None, returns a new array containing the variance; otherwise, a
            reference to the output array is returned.

    Raises:
        axis is neither a scalar nor None : NotImplementedError occurs.
        For complex numbers, NotImplementedError occurs.

    Note:
        The variance is the average of the squared deviations from the mean, i.e., var =
        mean(abs(x - x.mean())**2) . The mean is normally calculated as x.sum() / N,
        where N = len(x). If, however, ddof is specified, the divisor N - ddof is used
        instead. In standard statistical practice, ddof=1 provides an unbiased estimator
        of the variance of a hypothetical infinite population. ddof=0 provides a maximum
        likelihood estimate of the variance for normally distributed variables. For
        floating-point input, the variance is computed using the same precision the input
        has. Depending on the input data, this can cause the results to be inaccurate,
        especially for float32 (see example below). Specifying a higher-accuracy
        accumulator using the dtype keyword can alleviate this issue.

    See Also:
        std : Computes the standard deviation along the specified axis.
        mean : Computes the arithmetic mean along the specified axis.
        nanmean : Computes the arithmetic mean along the specified axis, ignoring NaNs.
        nanstd : Computes the standard deviation along the specified axis, while ignoring
            NaNs.
        nanvar : Computes the variance along the specified axis, while ignoring NaNs.

    Examples:
        >>> import nlcpy as vp
        >>> a = vp.array([[1, 2], [3, 4]])
        >>> vp.var(a)
        array(1.25)
        >>> vp.var(a, axis=0)
        array([1.,  1.])
        >>> vp.var(a, axis=1)
        array([0.25,  0.25])
        In single precision, var() can be inaccurate:
        >>> a = vp.zeros((2, 512*512), dtype=vp.float32)
        >>> a[0, :] = 1.0
        >>> a[1, :] = 0.1
        >>> vp.var(a)
        array(0.20251831, dtype=float32)
        Computing the variance in float64 is more accurate:
        >>> vp.var(a, dtype=vp.float64)
        array(0.2025) # may vary
        >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
        0.2025

    """
    if a is None:
        if out is not None:
            return None

    if not isinstance(a, nlcpy.core.core.ndarray):
        a = ny.array(a)

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
        if issubclass(a.dtype.type, (ny.integer, ny.bool_)):
            result_dtype = np.result_type(a.dtype, a.dtype, 'f8')
        else:
            result_dtype = a.dtype
    else:
        result_dtype = dtype

    if axis is None:
        data_mean = ny.mean(a, dtype=result_dtype)
        total = a.size
    else:
        data_mean = ny.mean(a, axis=axis, dtype=result_dtype)
        total = a.shape[axis]

    if not isinstance(data_mean, nlcpy.core.core.ndarray):
        data_mean = ny.array(data_mean)

    ans = ny.mean(a * a, axis=axis, dtype=result_dtype,
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
        ans = ny.squeeze(ans)

    if out is not None:
        if isinstance(out, nlcpy.core.core.ndarray) is True:
            ans = ny.squeeze(ans)
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

# @}
#

cpdef corrcoef(a, y=None, rowvar=True, bias=nlcpy._NoValue, ddof=nlcpy._NoValue):
    """Returns Pearson product-moment correlation coefficients.

    Please refer to the documentation for `cov` for more detail. The relationship between
    the correlation coefficient matrix, R, and the covariance matrix, C, is ..math::
    R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }  The values of R are between
    -1 and 1, inclusive.

    Args:
        x : array_like
            A 1-D or 2-D array containing multiple variables and observations. Each row
            of x represents a variable, and each column a single observation of all those
            variables. Also see rowvar below.
        y : array_like, optional
            An additional set of variables and observations. y has the same shape as x.
        rowvar : bool, optional
            If rowvar is True (default), then each row represents a variable, with
            observations in the columns. Otherwise, the relationship is transposed: each
            column represents a variable, while the rows contain observations.
        bias : _NoValue, optional
            Has no effect, do not use.
        ddof : _NoValue, optional
            Has no effect, do not use.

    Returns:
        R : `ndarray`
            The correlation coefficient matrix of the variables.

    Raises:
        For complex numbers, NotImplementedError occurs.

    Note:
        Due to floating point rounding the resulting array may not be Hermitian, the
        diagonal elements may not be 1, and the elements may not satisfy the inequality
        abs(a) ddof.

    See Also:
        cov : Covariance matrix

    Examples:
        >>> import nlcpy as vp
        x = vp.array([[1,2,1,9,10,3,2,6,7],[2,1,8,3,7,5,10,7,2]])
        >>> vp.corrcoef(x)
        array([[ 1.        , -0.05640533],
               [-0.05640533,  1.        ]])
        >>> y = vp.array([2,1,1,8,9,4,3,5,7])
        >>> vp.corrcoef(x,y)
        array([[ 1.        , -0.05640533,  0.97094584],
               [-0.05640533,  1.        , -0.01315587],
               [ 0.97094584, -0.01315587,  1.        ]])

    """
    if bias is not nlcpy._NoValue or ddof is not nlcpy._NoValue:
        warnings.warn('bias and ddof have no effect and are deprecated',
                      DeprecationWarning)

    if not isinstance(a, nlcpy.core.core.ndarray):
        a = core.argument_conversion(a)

    if y is not None and not isinstance(y, nlcpy.core.core.ndarray):
        y = core.argument_conversion(y)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    cv_data = ny.cov(a, y, rowvar=rowvar)

    if not isinstance(cv_data, nlcpy.core.core.ndarray):
        cv_data = core.argument_conversion(cv_data)

    nlcpy_chk_type(a)
    nlcpy_chk_type(y)

    try:
        diag_data = ny.diag(cv_data)
    except ValueError:
        return cv_data / cv_data

    stddev = ny.sqrt(diag_data)
    tmp_c1 = cv_data / stddev[:, None]
    tmp_c2 = tmp_c1 / stddev[None, :]

    ans = tmp_c2

    if ans.size == 1:
        ans_cnv1 = ans.ravel()
        ans_cnv2 = ny.where(ans_cnv1 >= 1, 1, ans_cnv1)
        ans_cnv3 = ny.where(ans_cnv2 <= -1, -1, ans_cnv2)
        ret = ans_cnv3[0]

    else:
        ret = ny.fmin(1.0, ny.fmax(-1.0, ans))

    ret = ny.squeeze(ret)

    return ret


cpdef cov(a, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    """Estimates a covariance matrix, given data and weights.

    Covariance indicates the level to which two variables vary together. If we examine
    N-dimensional samples, :math:` X = [x_1, x_2, ... x_N]^T `, then the covariance
    matrix element :math:` C_{ij} ` is the covariance of :math:` x_i ` and :math:` x_j `.
    The element :math:` C_{ii} ` is the variance of :math:` x_i `. See the notes for an
    outline of the algorithm.

    Args:
        m : array_like
            A 1-D or 2-D array containing multiple variables and observations. Each row
            of m represents a variable, and each column a single observation of all those
            variables. Also see rowvar below.
        y : array_like, optional
            An additional set of variables and observations. y has the same form as that
            of m.
        rowvar : bool, optional
            If rowvar is True (default), then each row represents a variable, with
            observations in the columns. Otherwise, the relationship is transposed: each
            column represents a variable, while the rows contain observations.
        bias : bool, optional
            Default normalization (False) is by (N - 1), where N is the number of
            observations given (unbiased estimate). These arguments had no effect on the
            return values of the function and can be safely ignored in this and previous
            versions of nlcpy.
            If bias is True, then normalization is by N.
        ddof : int, optional
            If not None the default value implied by bias is overridden. Note that ddof=1
            will return the unbiased estimate, even if both fweights and aweights are
            specified, and ddof=0 will return the simple average. See the notes for the
            details. The default value is None.
        fweights : array_like, int, optional
            1-D array of integer frequency weights; the number of times each observation
            vector should be repeated.
        aweights : array_like, optional
            1-D array of observation vector weights. These relative weights are typically
            large for observations considered "important" and smaller for observations
            considered less "important". If ddof=0 the array of weights can be used to
            assign probabilities to observation vectors.

    Returns:
        out : `ndarray`
            The covariance matrix of the variables.

    Note:
        Assume that the observations are in the columns of the observation array m and
        let f = fweights and a = aweights for brevity. The steps to compute the weighted
        covariance are as follows:
        >>> import nlcpy as vp
        >>> m = vp.arange(10, dtype=vp.float64)
        >>> f = vp.arange(10) * 2
        >>> a = vp.arange(10) ** 2.
        >>> ddof = 9 # N - 1
        >>> w = f * a
        >>> v1 = vp.sum(w)
        >>> v2 = vp.sum(w * a)
        >>> m -= vp.sum(m * w, axis=None, keepdims=True) / v1
        >>> cov = vp.dot(m * w, m.T) * v1 / (v1**2 - ddof * v2)
        Note that when a == 1, the normalization factor v1 / (v1**2 - ddof * v2) goes
        over to 1 / (vp.sum(f) - ddof) as it should.

    See Also:
        corrcoef : Normalized covariance matrix

    Examples:
        Consider two variables, :math:` x_0 ` and :math:` x_1 `,
        which correlate perfectly,
        but in opposite directions:
        >>> import nlcpy as vp
        >>> x = vp.array([[0, 2], [1, 1], [2, 0]]).T
        >>> x
        array([[0, 1, 2],
              [2, 1, 0]])
        Note how :math:` x_0 ` increases while :math:` x_1 ` decreases.
        The covariance matrix shows this clearly:
        >>> vp.cov(x)
        array([[ 1., -1.],
              [-1.,  1.]])
        Note that element :math:` C_{0,1} `, which shows the correlation betweeni
        Further, note how x and y are combined:
        >>> x = [-2.1, -1,  4.3]
        >>> y = [3,  1.1,  0.12]
        >>> vp.cov(x, y)
        array([[11.71      , -4.286     ], # may vary
               [-4.286     ,  2.14413333]])
        >>> vp.cov(x)
        array(11.71)

    """

    if ddof is not None and isinstance(ddof, int) is False:
        raise ValueError("cov ddof({}) must be integer({})".format(ddof, type(ddof)))

    nlcpy_chk_type(a)

    if not isinstance(a, nlcpy.core.core.ndarray):
        a = core.argument_conversion(a)

    if y is not None and not isinstance(y, nlcpy.core.core.ndarray):
        y = core.argument_conversion(y)

    in_a = a
    if in_a.ndim > 2:
        raise ValueError("a has more than 2 dimensions")

    if y is not None:
        y = ny.asarray(y)
        y = core.argument_conversion(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")

    X = ny.array(in_a, ndmin=2, dtype=np.float64)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return ny.array([]).reshape(0, 0)

    if y is not None:
        y = ny.array(y, copy=False, ndmin=2, dtype=np.float64)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = ny.concatenate((X, y), axis=0)

    if ddof is None:
        if bias is False:
            ddof = 1
        else:
            ddof = 0

    w = None
    if fweights is not None:
        fweights = ny.asarray(fweights, dtype=np.float64)
        if fweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError("fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = ny.asarray(aweights, dtype=np.float64)
        if aweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError("aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = ny.average(X, axis=1, weights=w, returned=True)
    if w_sum.size > 1:
        w_sum = w_sum[0]

    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * ny.sum(w * aweights) / w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=3)
        fact = 0.0

    if isinstance(avg, nlcpy.core.core.ndarray) is False:
        avg = core.argument_conversion(avg)

    if isinstance(w_sum, nlcpy.core.core.ndarray) is False:
        w_sum = core.argument_conversion(w_sum)

    if avg.ndim > 0:
        X -= avg[:, None]
    else:
        X -= avg

    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T

    cnv_data = ny.dot(X, ny.conjugate(X_T))
    ret = cnv_data * ny.true_divide(1, fact)

    if ret.size == 1:
        ans_cnv1 = ret.ravel()
        ans_cnv2 = ans_cnv1[0]
    else:
        ans_cnv2 = ret

    ans_cnv2 = ny.squeeze(ans_cnv2)

    return ans_cnv2

# @}
#

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


def nlcpy_wrapit(obj, shape):
    a = obj.ravel()
    result = a.reshape(shape)

    return result

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


def nlcpy_hatmask(var):
    b = ny.where(var, False, True)
    return b

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


def nlcpy_replace_nan(a, val):
    a = ny.asanyarray(a)

    if a.dtype == np.object_:
        mask = ny.not_equal(a, a, dtype=bool)
    elif issubclass(a.dtype.type, ny.inexact):
        mask = ny.isnan(a)
    else:
        mask = None

    if mask is not None:
        a = ny.array(a)
        nlcpy_copyto(a, val, mask)

    return a, mask

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


def nlcpy_copyto(a, val, mask):
    if isinstance(a, nlcpy.core.core.ndarray):
        a[mask] = val
    else:
        a = a.dtype.type(val)
    return a

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


def nlcpy_nan_mask(a, out=None):
    if a.dtype.kind not in 'fc':
        return True

    y = ny.isnan(a, out=out)
    y = ny.invert(y, out=y)
    return y

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


def nlcpy_divide_by_count(a, b, out=None):
    try:
        if isinstance(a, nlcpy.core.core.ndarray):
            if out is None:
                return ny.divide(a, b, out=a, casting='unsafe')
            else:
                return ny.divide(a, b, out=out, casting='unsafe')
        else:
            if out is None:
                return a.dtype.type(a / b)
            else:
                return ny.divide(a, b, out=out, casting='unsafe')
    except Exception as e:
        pass


# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


def nlcpy_sq(axis, a):
    if axis is not None:
        ns = []
        for x in range(len(a.shape)):
            if x != axis:
                ns.append(a.shape[x])
        ans = ny.reshape(a, tuple(ns))
    else:
        ans = a

    return ans

# ----------------------------------------------------------------------------
# local function
# ---------------------------------------------------------------------------


def nlcpy_chk_type(a):
    if not isinstance(a, nlcpy.core.core.ndarray):
        return True

    if a is not None and a.dtype.kind in ('b', 'c', 'v'):
        raise NotImplementedError("dtype={} not supported".format(a.dtype))

    return True

# ----------------------------------------------------------------------------
# local function
# ---------------------------------------------------------------------------


def nlcpy_countnan(a):
    a = ny.asanyarray(a)
    ans = a[ny.isnan(a)]
    return ans.size


# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


def inner_convmeandata(data, shape, axis=None):
    save_shape = shape

    if not isinstance(data, nlcpy.core.core.ndarray):
        data = ny.array(data)

    if data.ndim == 0:
        b_d = data.reshape(1, -1)
        ans = ny.tile(b_d, save_shape)
    else:
        l_in = list(save_shape)
        l_x = list(save_shape)

        for x in range(len(l_in)):
            if x != axis:
                l_in[x] = 1

        for x in range(len(l_x)):
            if x == axis:
                l_x[x] = -1

        t = tuple(l_in)
        t_l = tuple(l_x)
        b_d = data.reshape(t_l)
        ans = ny.tile(b_d, t)

    return ans

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


def nlcpy_median_nancheck(data, result, axis):
    if data.size == 0:
        return result

    n = ny.isnan(data[..., -1])
    b = n.ravel()
    if n is True:
        result = data.dtype.type(np.nan)
    elif np.count_nonzero(b.get()) > 0:
        result[n] = ny.nan

    return result


# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


def nlcpy_chk_axis(a, axis=None):
    ret = False

    if isinstance(a, nlcpy.core.core.ndarray) is True:
        if axis is None:
            ret = True
        else:
            if type(axis) is tuple:
                raise ValueError('tuple axis is not supported')
            elif axis >= a.ndim:
                raise ValueError("Nlcpy AxisError: axis {} is out of bounds"
                                 " for array of dimension {}".format(axis, a.ndim))
            else:
                ret = True
    else:
        a = core.argument_conversion(a)
        if axis is None:
            ret = True
        elif type(axis) is tuple:
            raise ValueError('tuple axis is not supported')
        else:
            if axis >= a.ndim:
                raise ValueError("Nlcpy AxisError: axis {} is out of bounds"
                                 " for array of dimension {}".format(axis, a.ndim))
            else:
                ret = True

    return ret

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
from nlcpy.wrapper.numpy_wrap import numpy_wrap


def amax(a, axis=None, out=None, keepdims=nlcpy._NoValue,
         initial=nlcpy._NoValue, where=nlcpy._NoValue):
    """Returns the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose maximum is desired. If *a* is not an array, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the maximum is selected over multiple axes.
    out : ndarray, optional
        Alternative output array in which to place the result. Must be of the same shape
        and buffer length as the expected output.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
    initial : scalar, optional
        The maximum value of an output element. Must be present to allow computation on
        empty slice. See :func:`nlcpy.ufunc.reduce` for details.
    where : array_like of bool, optional
        Elements to compare for the maximum. See :func:`nlcpy.ufunc.reduce` for details.

    Returns
    -------
    amax : ndarray
        Maximum of *a*. An array with the same shape as *a*, with the specified axis
        removed.
        If *a* is a scalar, or if axis is None, this function returns the result as a
        0-dimention array. The same dtype as *a* is returned.

    Note
    ----
    NaN values are propagated, that is if at least one item is NaN, the corresponding max
    value will be NaN as well. To ignore NaN values, please use nanmax.

    Don't use :func:`amax` for element-wise comparison of 2 arrays; when ``a.shape[0]``
    is 2, ``maximum(a[0], a[1])`` is faster than ``amax(a, axis=0)``.

    Restriction
    -----------
    - If an ndarray is passed to ``where`` and ``where.shape != a.shape``,
      *NotImplementedError* occurs.
    - If an ndarray is passed to ``out`` and ``out.shape != amax.shape``,
      *NotImplementedError* occurs.

    See Also
    --------
    amin : Returns the minimum of an array or minimum along an axis.
    nanmax : Returns maximum of an array or maximum along an axis, ignoring any NaNs.
    maximum : Element-wise maximum of array elements.
    fmax : Element-wise maximum of array elements.
    argmax : Returns the indices of the maximum values along an axis.
    nanmin : Returns the minimum of an array or minimum along an axis, ignoring any NaNs.
    minimum : Element-wise minimum of array elements.
    fmin : Element-wise minimum of array elements.

    Examples
    --------
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
    >>> b = vp.arange(5, dtype=float)
    >>> b[2] = vp.NaN
    >>> vp.amax(b)
    array(nan)
    >>> vp.amax(b, where=~vp.isnan(b), initial=-1)
    array(4.)
    >>> vp.nanmax(b)
    array(4.)

    """
    args = dict()
    if keepdims is not nlcpy._NoValue:
        args["keepdims"] = keepdims
    if initial is not nlcpy._NoValue:
        args["initial"] = initial
    if where is not nlcpy._NoValue:
        args["where"] = where
    return nlcpy.maximum.reduce(a, axis=axis, out=out, **args)


max = amax


def amin(a, axis=None, out=None, keepdims=nlcpy._NoValue,
         initial=nlcpy._NoValue, where=nlcpy._NoValue):
    """Returns the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose minimum is desired. If *a* is not an array, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the minimum is selected over multiple axes.
    out : ndarray, optional
        Alternative output array in which to place the result. Must be of the same shape
        and buffer length as the expected output.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
    initial : scalar, optional
        The maximum value of an output element. Must be present to allow computation on
        empty slice. See :func:`nlcpy.ufunc.reduce` for details.
    where : array_like of bool, optional
        Elements to compare for the minimum. See :func:`nlcpy.ufunc.reduce` for details.

    Returns
    -------
    amin : ndarray
        Minimum of *a*. An array with the same shape as *a*, with the specified axis
        removed.
        If *a* is a scalar, or if axis is None, this function returns the result as a
        0-dimention array. The same dtype as *a* is returned.

    Note
    ----
    NaN values are propagated, that is if at least one item is NaN, the corresponding min
    value will be NaN as well. To ignore NaN values, please use nanmin.

    Don't use :func:`amin` for element-wise comparison of 2 arrays; when ``a.shape[0]``
    is 2, ``minimum(a[0], a[1])`` is faster than ``amin(a, axis=0)``.

    Restriction
    -----------
    - If an ndarray is passed to ``where`` and ``where.shape != a.shape``,
      *NotImplementedError* occurs.
    - If an ndarray is passed to ``out`` and ``out.shape != amin.shape``,
      *NotImplementedError* occurs.

    See Also
    --------
    amax : Returns the maximum of an array or maximum along an axis.
    nanmin : Returns minimum of an array or minimum along an axis, ignoring any NaNs.
    minimum : Element-wise minimum of array elements.
    fmin : Element-wise minimum of array elements.
    argmin : Returns the indices of the minimum values along an axis.
    nanmax : Returns the maximum of an array or maximum along an axis, ignoring any NaNs.
    maximum : Element-wise maximum of array elements.
    fmax : Element-wise maximum of array elements.

    Examples
    --------
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


min = amin


def nanmax(a, axis=None, out=None, keepdims=nlcpy._NoValue):
    """Returns maximum of an array or maximum along an axis, ignoring any NaNs.

    When all-NaN slices are encountered a ``RuntimeWarning`` is raised and Nan is
    returned for that slice.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose maximum is desired. If *a* is not an array, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the maximum is selected over multiple axes.
    out : ndarray, optional
        Alternative output array in which to place the result. Must be of the same shape
        and buffer length as the expected output.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.

    Returns
    -------
    nanmax : ndarray
        Maximum of *a*. An array with the same shape as *a*, with the specified axis
        removed.
        If *a* is a scalar, or if axis is None, this function returns the result as a
        0-dimention array. The same dtype as *a* is returned.

    Note
    ----
    NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
    This means that Not a Number is not equivalent to infinity. Positive infinity is
    treated as a very large number and negative infinity is treated as a very small (i.e.
    negative) number.

    If the input has a integer type the function is equivalent to :func:`amax`.

    See Also
    --------
    nanmin : Returns the minimum of an array or minimum along an axis, ignoring any NaNs.
    amax : Returns the maximum of an array or maximum along an axis.
    fmax : Element-wise maximum of array elements.
    maximum : Element-wise maximum of array elements.
    isnan : Tests element-wise for NaN and return result as a boolean array.
    isfinite : Tests element-wise for finiteness (not infinity or not Not a Number).
    amin : Returns the minimum of an array or maximum along an axis.
    fmin : Element-wise minimum of array elements.
    minimum : Element-wise minimum of array elements.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1, 2], [3, vp.nan]])
    >>> vp.nanmax(a)
    array(3.)
    >>> vp.nanmax(a, axis=0)
    array([3., 2.])
    >>> vp.nanmax(a, axis=1)
    array([2., 3.])

    When positive infinity and negative infinity are present:

    >>> vp.nanmax([1, 2, vp.nan, vp.inf])
    array(inf)
    >>> vp.nanmax([1, 2, vp.nan, vp.NINF])
    array(2.)

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

    When all-NaN slices are encountered a ``RuntimeWarning`` is raised and Nan is
    returned for that slice.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose minimum is desired. If a is not an array, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the minimum is selected over multiple axes.
    out : ndarray, optional
        Alternative output array in which to place the result. Must be of the same shape
        and buffer length as the expected output.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.

    Returns
    -------
    nanmin : ndarray
        Minimum of *a*. An array with the same shape as *a*, with the specified axis
        removed.
        If *a* is a scalar, or if axis is None, this function returns the result as a
        0-dimention array. The same dtype as *a* is returned.

    Note
    ----
    NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
    This means that Not a Number is not equivalent to infinity. Positive infinity is
    treated as a very large number and negative infinity is treated as a very small (i.e.
    negative) number.

    If the input has a integer type the function is equivalent to :func:`amin`.

    See Also
    --------
    nanmax : Returns the maximum of an array or maximum along an axis, ignoring any NaNs.
    amin : Returns the minimum of an array or maximum along an axis.
    fmin : Element-wise minimum of array elements.
    minimum : Element-wise minimum of array elements.
    isnan : Tests element-wise for NaN and return result as a boolean array.
    isfinite : Tests element-wise for finiteness (not infinity or not Not a Number).
    amax : Returns the maximum of an array or maximum along an axis.
    fmax : Element-wise maximum of array elements.
    maximum : Element-wise maximum of array elements.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1, 2], [3, vp.nan]])
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


@numpy_wrap
def ptp(a, axis=None, out=None, keepdims=nlcpy._NoValue):
    """Range of values (maximum - minimum) along an axis.

    The name of the function comes from the acronym for 'peak to peak'.

    Parameters
    ----------
    a : array_like
        Input values.
    axis : None or int or tuple of ints, optional
        Axis along which to find the peaks. By default, flatten the array. axis may be
        negative, in which case it counts from the last to the first axis. If this is a
        tuple of ints, a reduction is performed on multiple axes, instead of a single
        axis or all the axes as before.
    out : array_like
        Alternative output array in which to place the result. Must be of the same shape
        and buffer length as the expected output.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one.
        With this option, the result will broadcast correctly against the input array. If
        the default value is passed, then keepdims will not be passed through to the ptp
        method of sub-classes of  n-dimensional_array "ndarray", however any non-default
        value will be. If the sub-class' method does not implement keepdims any
        exceptions will be raised.

    Returns
    -------
    ptp : ndarray
        A new array holding the result, unless *out* was specified, in which case a
        reference to *out* is returned.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.ptp`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    Note
    ----
    :func:`ptp` preserves the data type of the array. This means the return value for an
    input of signed integers with n bits (e.g. *nlcpy.int32*, *nlcpy.int64*, etc) is also
    a signed integer with n bits.
    In that case, peak-to-peak values greater than 2**(n-1)-1 will be returned as
    negative values. An example with a work-around is shown below.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.array([[4, 9, 2, 10],
    ...             [6, 9, 7, 12]])
    >>> vp.ptp(x, axis=1)
    array([8, 6])
    >>> vp.ptp(x, axis=0)
    array([2, 0, 5, 2])
    >>> vp.ptp(x)
    array(10)

    This example shows that a negative value can be returned when the input is an array
    of signed integers.

    >>> y = vp.array([[1, 127], [0, 127], [-1, 127], [-2, 127]], dtype=vp.int32)
    >>> vp.ptp(y, axis=1)
    array([126, 127, 128, 129], dtype=int32)

    A work-around is to use the view() method to view the result as unsigned integers
    with the same bit width:

    >>> vp.ptp(y, axis=1).view(vp.uint32)
    array([126, 127, 128, 129], dtype=uint32)

    """
    raise NotImplementedError('ptp is not implemented yet.')


@numpy_wrap
def percentile(a, q, axis=None, out=None, overwrite_input=False,
               interpolation='linear', keepdims=False):
    """Computes the *q*-th percentile of the data along the specified axis.

    Returns the *q*-th percentile(s) of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be between 0 and 100
        inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The default is to compute
        the percentile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape and buffer length as the expected output, but the type (of the output) will
        be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array a to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input a after
        this function completes is undefined.
    interpolation  :   {'linear','lower','higher','midpoint','nearest'}
        This optional parameter specifies the interpolation method to use when the
        desired percentile lies between two data points i:
        'linear': ``i + (j - i) * fraction``, where `` fraction`` is the fractional part
        of the index surrounded by ``i`` and ``j`` 'lower': ``i``. 'higher': ``j``.
        'nearest': ``i`` or ``j``, whichever is nearest. 'midpoint': ``(i + j)/2``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the original array a.

    Returns
    -------
    percentile : scaler or ndarray
        If *q* is a single percentile and *axis*=None, then the result is a scalar. If
        multiple percentiles are given, first axis of the result corresponds to the
        percentiles. The other axes are the axes that remain after the reduction of *a*.
        If the input contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the same as that of
        the input. If *out* is specified, that array is returned instead.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.percentile`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    Note
    ----
    Given a vector ``V`` of length ``N``, the *q*-th percentile of ``V`` is the value
    ``q/100`` of the way from the minimum to the maximum in a sorted copy of ``V``. The
    values and distances of the two nearest neighbors as well as the interpolation
    parameter will determine the percentile if the normalized ranking does not match the
    location of ``q`` exactly. This function is the same as the median if ``q=50``, the
    same as the minimum if ``q=0`` and the same as the maximum if ``q=100``.

    See Also
    --------
    mean : Computes the arithmetic mean along the specified axis.
    percentile :
    median : Computes the median along the specified axis.
    nanpercentile : Computes the *q*-th percentile of the data along the specified axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> vp.percentile(a, 50)
    array(3.5)
    >>> vp.percentile(a, 50, axis=0)
    array([6.5, 4.5, 2.5])
    >>> vp.percentile(a, 50, axis=1)
    array([7., 2.])
    >>> vp.percentile(a, 50, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = vp.percentile(a, 50, axis=0)
    >>> out = vp.zeros_like(m)
    >>> vp.percentile(a, 50, axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])
    >>> b = a.copy()
    >>> vp.percentile(b, 50, axis=1, overwrite_input=True)
    array([7., 2.])

    """
    raise NotImplementedError('percentile is not implemented yet.')


@numpy_wrap
def nanpercentile(a, q, axis=None, out=None, overwrite_input=False,
                  interpolation='linear', keepdims=False):
    """Computes the *q*-th percentile of the data along the specified axis, while ignoring
    nan values.

    Returns the *q*-th percentile(s) of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array, containing nan values to
        be ignored.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be between 0 and 100
        inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The default is to compute
        the percentile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape and buffer length as the expected output, but the type (of the output) will
        be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array a to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input a after
        this function completes is undefined.
    interpolation  :   {'linear','lower','higher','midpoint','nearest'}
        This optional parameter specifies the interpolation method to use when the
        desired percentile lies between two data points i:
        'linear': ``i + (j - i) * fraction``, where `` fraction`` is the fractional part
        of the index surrounded by`` i`` and ``j``. 'lower': ``i``. 'higher': ``j``.
        'nearest': ``i`` or ``j``, whichever is nearest. 'midpoint': ``(i + j)/2``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the original array a. If this is anything but the default value it will
        be passed through (in the special case of an empty array) to the mean function of
        the underlying array. If the array is a sub-class and mean does not have the
        kwarg keepdims this will raise a RuntimeError.

    Returns
    -------
    percentile :  scalar or ndarray
        If *q* is a single percentile and *axis*=None, then the result is a scalar. If
        multiple percentiles are given, first axis of the result corresponds to the
        percentiles. The other axes are the axes that remain after the reduction of *a*.
        If the input contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the same as that of
        the input. If *out* is specified, that array is returned instead.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.nanpercentile`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    Note
    ----
    Given a vector ``V`` of length ``N``, the ``q``-th percentile of ``V`` is the value
    ``q/100`` of the way from the minimum to the maximum in a sorted copy of V. The
    values and distances of the two nearest neighbors as well as the interpolation
    parameter will determine the percentile if the normalized ranking does not match the
    location of ``q`` exactly. This function is the same as the median if ``q=50``, the
    same as the minimum if ``q=0`` and the same as the maximum if ``q=100``.

    See Also
    --------
    nanmean : Computes the arithmetic mean along the specified axis, ignoring NaNs.
    nanmedian : Computes the median along the specified axis, while ignoring NaNs.
    percentile : Compute the *q*-th percentile of the data along the specified axis.
    median : Computes the median along the specified axis.
    mean : Computes the arithmetic mean along the specified axis.
    nanpercentile :

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[10., 7., 4.], [3., 2., 1.]])
    >>> a[0][1] = vp.nan
    >>> a
    array([[10., nan,  4.],
           [ 3.,  2.,  1.]])
    >>> vp.percentile(a, 50)
    array(nan)
    >>> vp.nanpercentile(a, 50)
    array(3.)
    >>> vp.nanpercentile(a, 50, axis=0)
    array([6.5, 2. , 2.5])
    >>> vp.nanpercentile(a, 50, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = vp.nanpercentile(a, 50, axis=0)
    >>> out = vp.zeros_like(m)
    >>> vp.nanpercentile(a, 50, axis=0, out=out)
    array([6.5, 2. , 2.5])
    >>> m
    array([6.5, 2. , 2.5])
    >>> b = a.copy()
    >>> vp.nanpercentile(b, 50, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not vp.all(a==b)

    """
    raise NotImplementedError('nanpercentile is not implemented yet.')


@numpy_wrap
def quantile(a, q, axis=None, out=None, overwrite_input=False,
             interpolation='linear', keepdims=False):
    """Computes the *q*-th quantile of the data along the specified axis.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between 0 and 1
        inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is to compute
        the quantile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape and buffer length as the expected output, but the type (of the output) will
        be cast if necessary.
    overwrite_input : bool, optional
        if True, then allow the input array a to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input a after
        this function completes is undefined.
    interpolation  :   {'linear','lower','higher','midpoint','nearest'}
        This optional parameter specifies the interpolation method
        to use when the desired percentile lies between two data points ``i < j``:

        * 'linear': ``i + (j - i) * fraction``, where `` fraction`` is the fractional
          part of the index surrounded by `` i`` and ``j``.
        * 'lower': ``i``.
        * 'higher': ``j``.
        * 'nearest': ``i`` or ``j``, whichever is nearest.
        * 'midpoint': ``(i + j)/2``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the original array *a*.

    Returns
    -------
    quantile : scalar or ndarray
        If *q* is a single quantile and axis=None, then the result is a scalar. If
        multiple quantiles are given, first axis of the result corresponds to the
        quantiles. The other axes are the axes that remain after the reduction of *a*.
        If the input contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the same as that of
        the input. If *out* is specified, that array is returned instead.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.quantile`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    Note
    ----
    Given a vector ``V`` of length ``N``, the *q*-th quantile of ``V`` is the value ``q``
    of the way from the minimum to the maximum in a sorted copy of ``V``. The values and
    distances of the two nearest neighbors as well as the interpolation parameter will
    determine the quantile if the normalized ranking does not match the location of ``q``
    exactly. This function is the same as the median if ``q=0.5``, the same as the
    minimum if ``q=0.0`` and the same as the maximum if ``q=1.0``.

    See Also
    --------
    mean : Computes the arithmetic mean along the specified axis.
    percentile : Computes the *q*-th percentile of the data along the specified axis.
    median : Computes the median along the specified axis.
    nanquantile : Computes the *q*-th quantile of the data along the specified axis,
        while ignoring nan values.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> vp.quantile(a, 0.5)
    array(3.5)
    >>> vp.quantile(a, 0.5, axis=0)
    array([6.5, 4.5, 2.5])
    >>> vp.quantile(a, 0.5, axis=1)
    array([7., 2.])
    >>> vp.quantile(a, 0.5, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = vp.quantile(a, 0.5, axis=0)
    >>> out = vp.zeros_like(m)
    >>> vp.quantile(a, 0.5, axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])
    >>> b = a.copy()
    >>> vp.quantile(b, 0.5, axis=1, overwrite_input=True)
    array([7., 2.])

    """
    raise NotImplementedError('quantile is not implemented yet.')


@numpy_wrap
def nanquantile(a, q, axis=None, out=None, overwrite_input=False,
                interpolation='linear', keepdims=False):
    """Computes the *q*-th quantile of the data along the specified axis, while ignoring
    nan values. Returns the *q*-th quantile(s) of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array, containing nan values to
        be ignored
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between 0 and 1
        inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is to compute
        the quantile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape and buffer length as the expected output, but the type (of the output) will
        be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array a to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input a after
        this function completes is undefined.
    interpolation :  {'linear','lower','higher','midpoint','nearest'}
        This optional parameter specifies the interpolation method to use when the
        desired percentile lies between two data points ``i < j`` :

        * 'linear': ``i + (j - i) * fraction``, where ``fraction`` is the fractional
          part of the index surrounded by ``i`` and ``j``.
        * 'lower': ``i``.
        * 'higher': ``j``.
        * 'nearest': ``i`` or ``j``, whichever is nearest.
        * 'midpoint': ``(i + j)/2``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the original array *a*.

    Returns
    -------
    quantile : scalar or ndarray
        If *q* is a single percentile and *axis*=None, then the result is a scalar. If
        multiple quantiles are given, first axis of the result corresponds to the
        quantiles. The other axes are the axes that remain after the reduction of *a*.
        If the input contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the same as that of
        the input. If *out* is specified, that array is returned instead.

    Restriction
    -----------
    This function is the wrapper function to utilize :func:`numpy.nanquantile`.
    Calculations during this function perform on only Vector Host(Linux/x86).

    See Also
    --------
    mean : Computes the arithmetic mean along the specified axis.
    percentile : Computes the *q*-th percentile of the data along the specified axis.
    median : Computes the median along the specified axis.
    quantile : Computes the *q*-th quantile of the data along the specified axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[10., 7., 4.], [3., 2., 1.]])
    >>> a[0][1] = vp.nan
    >>> a
    array([[10., nan,  4.],
           [ 3.,  2.,  1.]])
    >>> vp.quantile(a, 0.5)
    array(nan)
    >>> vp.nanquantile(a, 0.5)
    array(3.)
    >>> vp.nanquantile(a, 0.5, axis=0)
    array([6.5, 2. , 2.5])
    >>> vp.nanquantile(a, 0.5, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = vp.nanquantile(a, 0.5, axis=0)
    >>> out = vp.zeros_like(m)
    >>> vp.nanquantile(a, 0.5, axis=0, out=out)
    array([6.5, 2. , 2.5])
    >>> m
    array([6.5, 2. , 2.5])
    >>> b = a.copy()
    >>> vp.nanquantile(b, 0.5, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not vp.all(a==b)

    """
    raise NotImplementedError('nanquantile is not implemented yet.')

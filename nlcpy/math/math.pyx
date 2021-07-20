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

# distutils: language = c++

import nlcpy
import numpy

from nlcpy import veo
from nlcpy.core cimport vememory
from nlcpy.core cimport core
from nlcpy.core.core import ndarray
from nlcpy.ufuncs import operations as ufunc_op
from nlcpy.core cimport broadcast
from nlcpy.core.core cimport *
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.core.dtype cimport get_dtype_number, get_dtype
from nlcpy.request cimport request

# ----------------------------------------------------------------------------
# Mathematical functions (except for universel functions)
# see: https://docs.scipy.org/doc/numpy/reference/routines.math.html
# ----------------------------------------------------------------------------


def sum(a, axis=None, dtype=None, out=None, keepdims=nlcpy._NoValue,
        initial=nlcpy._NoValue, where=nlcpy._NoValue):
    """Sum of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed. The default, axis=None, will sum all
        of the elements of the input array. If axis is negative it counts from the last
        to the first axis. If axis is a tuple of ints, a sum is performed on all of the
        axes specified in the tuple instead of a single axis or all the axes as before.
    dtype : dtype, optional
        The type used to represent the intermediate results. Defaults to the dtype of the
        output array if this is provided. If out is not provided, the dtype of *a* is
        used unless *a* has nlcpy.int32 or nlcpy.uint32. In that case, nlcpy.int64 or
        nlcpy.uint64 is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape as the expected output, but the type of the output values will be cast if
        necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the original array.
    initial : scalar, optional
        The value with which to start the reduction. Defaults to 0. If None is given, the
        first element of the reduction is used, and an error is thrown if the reduction
        is empty.
    where : array_like of bool, optional
        A boolean array which is broadcasted to match the dimensions of *a*, and selects
        elements to include in the reduction.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as *a*, with the specified axis removed. If *a* is a
        0-d array, or if *axis* is None, this function returns the result as a
        0-dimention array. If an output array is specified, a reference to *out* is
        returned.

    Note
    ----

    - `nlcpy.reduce.add` is called in this function.


    - Arithmetic is modular when using integer types, and no error is raised on overflow.


    - The sum of an empty array is the neutral element 0:

    >>> import nlcpy as vp
    >>> vp.sum([])
    array(0.)

    Restriction
    -----------
    - If an ndarray is passed to ``where`` and ``where.shape != a.shape``,
      *NotImplementedError* occurs.
    - If an ndarray is passed to ``out`` and ``out.shape != sum_along_axis.shape``,
      *NotImplementedError* occurs.

    See Also
    --------
    cumsum : Returns the cumulative sum of the elements along a given axis.
    mean : Computes the arithmetic mean along the specified axis.
    average : Computes the weighted average along the specified axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.sum([0.5, 1.5])
    array(2.)
    >>> vp.sum([0.5, 0.7, 0.2, 1.5], dtype=vp.int32)
    array(1, dtype=int32)
    >>> vp.sum([[0, 1], [0, 5]])
    array(6)
    >>> vp.sum([[0, 1], [0, 5]], axis=0)
    array([0, 6])
    >>> vp.sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])

    You can also start the sum with a value other than zero:

    >>> vp.sum([10], initial=5)
    array(15)

    """
    if where is nlcpy._NoValue:
        where = True
    if keepdims is nlcpy._NoValue:
        keepdims = False
    return ufunc_op.add.reduce(a, axis=axis, dtype=dtype, out=out,
                               initial=initial, keepdims=keepdims,
                               where=where)


def cumsum(a, axis=None, dtype=None, out=None):
    """Returns the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default (None) is to compute
        the cumsum over the flattened array.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the elements are
        summed. If dtype is not specified, it defaults to the dtype of *a*, unless dtype
        is nlcpy.int32. unless dtype is nlcpy.int32.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape and buffer length as the expected output but the type will be cast if
        necessary.

    Returns
    -------
    cumsum_along_axis : ndarray
        A new array holding the result is returned unless *out* is specified, in which
        case a reference to *out* is returned. The result has the same size as *a*, and
        the same shape as *a* if *axis* is not None or *a* is a 1-d array.

    Note
    ----

    Arithmetic is modular when using integer types, and no error is raised on overflow.

    See Also
    --------
    sum : Sum of array elements over a given axis.
    diff : Calculates the n-th discrete difference along the given axis.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1,2,3], [4,5,6]])
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> vp.cumsum(a)
    array([ 1,  3,  6, 10, 15, 21])
    >>> vp.cumsum(a, dtype=float)
    array([ 1.,  3.,  6., 10., 15., 21.])
    >>> vp.cumsum(a, axis=0)
    array([[1, 2, 3],
           [5, 7, 9]])
    >>> vp.cumsum(a, axis=1)
    array([[ 1,  3,  6],
           [ 4,  9, 15]])

    """
    a = core.argument_conversion(a)
    if a.ndim == 0:
        a = a.reshape(1)

    ########################################################################
    # TODO: VE-VH collaboration
    if a._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError("cumsum on VH is not yet impremented")

    ########################################################################
    # check order
    if a._f_contiguous and not a._c_contiguous:
        order_out = 'F'
    else:
        order_out = 'C'

    ########################################################################
    # axis check
    if axis is None:
        if a.ndim > 1:
            a = a.reshape(a.size)

        axis = 0
    elif isinstance(axis, nlcpy.ndarray) and axis.ndim > 0:
        raise TypeError(
            "only integer scalar arrays can be converted to a scalar index")
    elif not isinstance(axis, int):
        raise TypeError("'%s' object cannot be interpreted as an integer"
                        % (type(axis).__name__))
    else:
        if axis < 0:
            axis = a.ndim + axis
        if axis < 0 or axis > a.ndim-1:
            raise AxisError('axis ' + str(axis)
                            + ' is out of bounds for array of dimension ' + str(a.ndim))

    ########################################################################
    # dtype check
    if dtype is not None:
        if type(dtype) == str and dtype.find(',') > 0:
            raise TypeError('cannot perform cumsum with flexible type')
        dt = dtype
    elif a.dtype in ('bool', 'int32'):
        dt = 'int64'
    elif a.dtype == 'uint32':
        dt = 'uint64'
    else:
        dt = a.dtype

    ########################################################################
    # error check for "out"
    if out is not None:
        if type(out) != nlcpy.ndarray:
            raise TypeError("output must be an array")

        else:
            if a.ndim == 1:
                if a.shape[0] != out.shape[0]:
                    raise ValueError("provided out is the wrong size for the reduction")
                else:
                    y = out
            else:
                if a.ndim <= out.ndim:
                    for i in range(a.ndim):
                        if a.shape[i] != out.shape[i]:
                            raise ValueError("operands could not be broadcast together"
                                             + " with remapped shapes "
                                             + "[original->remapped]: "
                                             + str(out.shape).replace(" ", "") + "->"
                                             + str(out.shape[0:a.ndim]).replace(" ", "")
                                             + " " + str(a.shape).replace(" ", "")
                                             + "->" + str(a.shape).replace(" ", "")
                                             + " ")

                        # TODO: VE-VH collaboration
                        if out._memloc in {on_VH, on_VE_VH}:
                            raise NotImplementedError(
                                "cumsum on VH is not yet implemented.")
                    y = out
                else:
                    raise ValueError("Iterator input op_axes[0]["
                                     + str(a.ndim-out.ndim-1)+"] "
                                     + "(==" + str(out.ndim)
                                     + ") is not a valid axis of op[0], "
                                     + "which has " + str(out.ndim) + " dimensions ")
    else:
        y = core.ndarray(shape=a.shape, dtype=dt, order=order_out)

    if out is None or out.dtype == dt:
        w = y
    else:
        w = core.ndarray(shape=a.shape, dtype=dt, order=order_out)

    ########################################################################

    request._push_request(
        "nlcpy_add_accumulate",
        "accumulate_op",
        (a, y, w, axis, get_dtype_number(get_dtype(dt))),)
    return y


def diff(a, n=1, axis=-1, prepend=nlcpy._NoValue, append=nlcpy._NoValue):
    """Calculates the n-th discrete difference along the given axis.

    The first difference is given by ``out[i] = a[i+1] - a[i]`` along the given axis,
    higher differences are calculated by using diff recursively.

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.
    prepend, append : array_like, optional
        Values to prepend or append to *a* along axis prior to performing the difference.
        Scalar values are expanded to arrays with length 1 in the direction of axis and
        the shape of the input array in along all other axes. Otherwise the dimension and
        shape must match *a* except along axis.

    Returns
    -------
    diff : ndarray
        The n-th differences. The shape of the output is the same as *a* except along
        *axis* where the dimension is smaller by *n*. The type of the output is the same
        as the type of the difference between any two elements of *a*.

    Note
    ----

    Type is preserved for boolean arrays, so the result will contain `False` when
    consecutive elements are the same and `True` when they differ.


    For unsigned integer arrays, the results will also be unsigned. This should not be
    surprising, as the result is consistent with calculating the difference directly:

    >>> import nlcpy as vp
    >>> a = vp.array([1, 0], dtype='uint32')
    >>> vp.diff(a)
    array([4294967295], dtype=uint32)
    >>> vp.array([0], dtype='uint32') - vp.array([1], dtype='uint32')
    array([4294967295], dtype=uint32)

    if this is not desirable, then the array should be cast to a larger integer type
    first:

    >>> b = a.astype(vp.int64)
    >>> vp.diff(b)
    array([-1])

    See Also
    --------
    cumsum : Returns the cumulative sum of the elements along a given axis.

    Examples
    --------
    >>> x = vp.array([1, 2, 4, 7, 0])
    >>> vp.diff(x)
    array([ 1,  2,  3, -7])
    >>> vp.diff(x, n=2)
    array([  1,   1, -10])
    >>> x = vp.array([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> vp.diff(x)
    array([[2, 3, 4],
           [5, 1, 2]])
    >>> vp.diff(x, axis=0)
    array([[-1,  2,  0, -2]])

    """
    if n == 0:
        return a
    if n < 0:
        raise ValueError("order must be non-negative but got " + repr(n))
    a = nlcpy.asanyarray(a)
    nd = a.ndim
    if nd == 0:
        raise ValueError('diff requires input that is at least one dimensional')

    if axis < -nd or nd <= axis:
        raise AxisError('axis {} is out of bounds for array of dimension {}'
                        .format(axis, nd))
    axis = (nd + axis) if axis < 0 else axis

    combined = []
    if prepend is not nlcpy._NoValue:
        prepend = nlcpy.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = nlcpy.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not nlcpy._NoValue:
        append = nlcpy.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = nlcpy.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = nlcpy.concatenate(combined, axis)

    shape = list(a.shape)
    shape[axis] = max(0, shape[axis] - n)
    dtype_out = a.dtype
    order_out = 'C' if a._c_contiguous else 'F'
    out = nlcpy.ndarray(shape=shape, dtype=dtype_out, order=order_out)

    if out.size > 0:
        w = nlcpy.array(a)
        request._push_request(
            'nlcpy_diff',
            'math_op',
            (a, n, axis, out, w),
        )
    return out


def angle(z, deg=False):
    """Returns the angle of the complex argument.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    deg : bool, optional
        Returns angle in degrees if True, radians if False (default).

    Returns
    -------
    angle : ndarray
        The counterclockwise angle from the positive real axis on the complex plane in
        the range ``(-pi, pi]``, with dtype as nlcpy.float64. If *z* is a scalar, this
        function returns the result as a 0-dimention array.

    See Also
    --------
    arctan2 : Computes an element-wise inverse tangent
        of *x1*/*x2*.
    absolute : Computes an element-wise absolute value.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.angle([1.0, 1.0j, 1+1j])               # in radians
    array([0.        , 1.57079633, 0.78539816])
    >>> vp.angle(1+1j, deg=True)                  # in degrees
    array(45.)

    """
    x = nlcpy.asanyarray(z)

    if x.dtype in [numpy.dtype('complex64'), numpy.dtype('float32')]:
        dtype_out = numpy.dtype('float32')
    elif x.dtype in [numpy.dtype('complex128'), numpy.dtype('float64')]:
        dtype_out = numpy.dtype('float64')
    elif x.dtype in [numpy.dtype('int32'), numpy.dtype('int64'), numpy.dtype('uint32'),
                     numpy.dtype('uint64'), numpy.dtype('bool')]:
        dtype_out = numpy.dtype('float64')
    else:
        raise TypeError("Unknown datatype is specified")

    if x._c_contiguous:
        order_out = 'C'
    else:
        order_out = 'F'

    out = ndarray(x.shape, dtype=dtype_out, order=order_out)

    fpe_flags = request._get_fpe_flag()
    args = (
        x._ve_array,
        out._ve_array,
        veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),)

    request._push_and_flush_request(
        'nlcpy_angle',
        args,
    )

    if deg:
        return nlcpy.rad2deg(out)
    else:
        return out


def real(val):
    """Returns the real part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input arrays or scalars.

    Returns
    -------
    out : ndarray
        The real component of the complex argument. If *val* is real, the type of *val*
        is used for the output. If *val* has complex elements, the returned type is
        float. If *val* is a scalar, this function returns the result as a 0-dimention
        array.

    See Also
    --------
    imag : Returns the imaginary part of the complex argument.
    angle : Returns the angle of the complex argument.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([1+2j, 3+4j, 5+6j])
    >>> a.real
    array([1., 3., 5.])
    >>> a.real = 9
    >>> a
    array([9.+2.j, 9.+4.j, 9.+6.j])
    >>> a.real = vp.array([9, 8, 7])
    >>> a
    array([9.+2.j, 8.+4.j, 7.+6.j])
    >>> vp.real(1 + 1j)
    array(1.)

    """
    val = nlcpy.asarray(val)
    return val.real


def imag(val):
    """Returns the imaginary part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input arrays or scalars.

    Returns
    -------
    out : ndarray
        The imaginary component of the complex argument. If *val* is real, the type of
        *val* is used for the output. If *val* has complex elements, the returned type is
        float. If *val* is a scalar, this function returns the result as a 0-dimention
        array.

    See Also
    --------
    real : Returns the real part of the complex argument.
    angle : Returns the angle of the complex argument.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([1+2j, 3+4j, 5+6j])
    >>> a.imag
    array([2., 4., 6.])
    >>> a.imag = vp.array([8, 10, 12])
    >>> a
    array([1. +8.j, 3.+10.j, 5.+12.j])
    >>> vp.imag(1 + 1j)
    array(1.)

    """
    val = nlcpy.asarray(val)
    return val.imag


def prod(a, axis=None, dtype=None, out=None, keepdims=False,
         initial=nlcpy._NoValue, where=True):
    """Returns the product of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed. The default, *axis* = None, will
        calculate the product of all the elements in the input array. If *axis* is
        negative it counts from the last to the first axis.
        If *axis* is a tuple of ints, a product is performed on all of the axes specified
        in the tuple instead of a single axis or all the axes as before.
    dtype : dtype, optional
        The type of the returned array, as well as of the accumulator in which the
        elements are multiplied. The dtype of *a* is used by default unless *a* has an
        integer dtype of less precision than the default platform integer. In that case,
        if *a* is signed then the platform integer is used while if *a* is unsigned then
        an unsigned integer of the same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape as the expected output, but the type of the output values will be cast if
        necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
        If the default value is passed, then *keepdims* will not be passed through to the
        prod method of sub-classes of :func:`nlcpy.ndarray`, however any non-default
        value will be. If the sub-class' method does not implement *keepdims* any
        exceptions will be raised.
    initial : scalar, optional
        The starting value for this product. See :func:`nlcpy.ufunc.reduce` for details.
    where : array_like of bool, optional
        Elements to include in the product. See :func:`nlcpy.ufunc.reduce` for details.

    Returns
    -------
    product_along_axis : ndarray, see dtype parameter above.
        An array shaped as *a* but with the specified axis removed. If *axis* is None or
        *a* is a scalar value, this function returns the result as a 0-dimention array.
        Returns a reference to *out* if specified.

    Restriction
    -----------
    - If an ndarray is passed to ``where`` and ``where.shape != a.shape``,
      *NotImplementedError* occurs.
    - If an ndarray is passed to ``out`` and ``out.shape != product_along_axis.shape``,
      *NotImplementedError* occurs.

    See Also
    --------
    nlcpy.ndarray.prod : Equivalent method.

    Note
    ----
    - Arithmetic is modular when using integer types, and no error is raised on overflow:

    >>> import nlcpy as vp
    >>> x = vp.array([536870910, 536870910, 536870910, 536870910])
    >>> vp.prod(x)
    array(6917529010461212688)

    - The product of an empty array is the neutral element 1:

    >>> vp.prod([])
    array(1.)

    Examples
    --------
    By default, calculate the product of all elements:

    >>> import nlcpy as vp
    >>> vp.prod([1.,2.])
    array(2.)

    Even when the input array is two-dimensional:

    >>> vp.prod([[1.,2.],[3.,4.]])
    array(24.)

    But we can also specify the axis over which to multiply:

    >>> vp.prod([[1.,2.],[3.,4.]], axis=1)
    array([ 2., 12.])

    Or select specific elements to include:

    >>> vp.prod([1., vp.nan, 3.], where=[True, False, True])
    array(3.)

    If the type of x is unsigned, then the output type is the unsigned platform integer:

    >>> x = vp.array([1, 2, 3], dtype=vp.uint32)
    >>> vp.prod(x).dtype == vp.uint
    True

    If x is of a signed integer type, then the output type is the default platform
    integer:

    >>> x = vp.array([1, 2, 3], dtype=vp.int32)
    >>> vp.prod(x).dtype == int
    True

    You can also start the product with a value other than one:

    >>> vp.prod([1, 2], initial=5)
    array(10)

    """
    ret = nlcpy.multiply.reduce(a, axis=axis, dtype=dtype,
                                out=out, keepdims=keepdims, initial=initial, where=where)

    return ret


def clip(a, a_min, a_max, out=None, **kwargs):
    """Clips (limits) the values in an array.

    Given an interval, values outside the interval are clipped to the interval edges.
    For example, if an interval of ``[0, 1]`` is specified, values smaller than 0 become
    0, and values larger than 1 become 1.

    Equivalent to but faster than ``vp.maximum(a_min, vp.minimum(a, a_max))``.
    No check is performed to ensure ``a_min < a_max``.

    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min : scalar or array_like or None
        Minimum value. If *None*, clipping is not performed on the lower interval edge.
        Not more than one of *a_min* and *a_max* may be *None*.
    a_max : scalar or array_like or None
        Maximum value. If *None*, clipping is not performed on the upper interval edge.
        Not more than one of *a_min* and *a_max* may be *None*.
        If *a_min* or *a_max* are array_like, then *a*, *a_min*, and *a_max* will be
        broadcasted to match their shapes.
    out : ndarray, optional
        The results will be placed in this array. It may be the input array for
        in-place clipping. *out* must be of the right shape to hold the output.
        Its type is preserved.
    **kwargs
        For other keyword-only arguments,
        see the :ref:`ufunc docs <optional_keyword_arg>`.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of *a*, but where values < *a_min* are replaced with
        *a_min*, and those > *a_max* with *a_max*.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.arange(10)
    >>> vp.clip(a, 1, 8)
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> vp.clip(a, 3, 6, out=a)
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a = vp.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> vp.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
    array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])
    """
    a = nlcpy.asanyarray(a)
    return a.clip(a_min, a_max, out, **kwargs)

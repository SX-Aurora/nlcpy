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
import nlcpy
import warnings
from nlcpy import core
from nlcpy.request import request
from nlcpy.core.error import _AxisError as AxisError
from numpy.core._exceptions import UFuncTypeError

# ----------------------------------------------------------------------------
# adding and removing elements
# see: https://docs.scipy.org/doc/numpy/reference/
#             routines.array-manipulation.html#adding-and-removing-elements
# ----------------------------------------------------------------------------


def append(arr, values, axis=None):
    """Appends values to the end of an array.

    Parameters
    ----------
    arr : array_like
        Values are appended to a copy of this array.
    values : array_like
        These values are appended to a copy of arr.  It must be of the correct shape
        (the same shape as arr, excluding axis).  If axis is not specified, values
        can be any shape and will be flattened before use.
    axis : int, optional
        The axis along which values are appended.  If axis is not given, both arr and
        values are flattened before use.

    Returns
    -------
    append : ndarray
        A copy of arr with values appended to axis.  Note that append does not occur
        in-place: a new array is allocated and filled.  If axis is None, out is a
        flattened array.

    See Also
    --------
    insert : Inserts values along the given axis before the given indices.
    delete : Returns a new array with sub-arrays along an axis deleted.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    When axis is specified, values must have the correct shape.

    >>> vp.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> vp.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0) # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: all the input arrays must have same number of dimensions

    """

    arr = nlcpy.asanyarray(arr)
    if axis is None:
        if arr.ndim != 1:
            arr = arr.ravel()
        values = nlcpy.ravel(values)
        axis = arr.ndim - 1
    return nlcpy.concatenate((arr, values), axis=axis)


def delete(arr, obj, axis=None):
    """Returns a new array with sub-arrays along an axis deleted.

    For a one dimensional array, this returns those entries not returned by arr[obj].

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : slice, int or array of ints
        Indicate indices of sub-arrays to remove along the specified axis.
    axis : int, optional
        The axis along which to delete the subarray defined by obj.
        If axis is None, obj is applied to the flattened array.

    Returns
    -------
    out : ndarray
        A copy of arr with the elements specified by obj removed.
        Note that delete does not occur in-place. If axis is None, out is a flattened
        array.

    Note
    ----
    Often it is preferable to use a boolean mask. For example:

    >>> import nlcpy as vp
    >>> arr = vp.arange(12) + 1
    >>> mask = vp.ones(len(arr), dtype=bool)
    >>> mask[[0,2,4]] = False
    >>> result = arr[mask,...]

    Is equivalent to vp.delete(arr, [0,2,4], axis=0), but allows further use of mask.

    See Also
    --------
    insert : Inserts values along the given axis before the given indices.
    append : Appends values to the end of an array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> arr = vp.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    >>> arr
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> vp.delete(arr, 1, 0)
    array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]])
    >>> vp.delete(arr, slice(None, None, 2), 1)
    array([[ 2,  4],
           [ 6,  8],
           [10, 12]])
    >>> vp.delete(arr, [1,3,5], None)
    array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])

    """

    input_arr = nlcpy.asarray(arr)
    ndim = input_arr.ndim

    if input_arr._f_contiguous and not input_arr._c_contiguous:
        order_out = 'F'
    else:
        order_out = 'C'

    if axis is None:
        if ndim != 1:
            input_arr = input_arr.ravel()
        ndim = input_arr.ndim
        axis = ndim - 1

    if ndim == 0:
        warnings.warn(
            "in the future the special handling of scalars will be removed "
            "from delete and raise an error", DeprecationWarning, stacklevel=3)
        return input_arr.copy(order=order_out)

    if isinstance(axis, numpy.ndarray) or isinstance(axis, nlcpy.ndarray):
        axis = int(axis)
    elif not isinstance(axis, int):
        raise TypeError("an integer is required (got type "
                        + str(type(axis).__name__) + ")")

    if axis < -ndim or axis > ndim - 1:
        raise AxisError(
            "axis {} is out of bounds for array of dimension {}".format(axis, ndim))
    if axis < 0:
        axis += ndim

    N = input_arr.shape[axis]
    if isinstance(obj, slice):
        start, stop, step = obj.indices(N)
        xr = range(start, stop, step)
        if len(xr) == 0:
            return input_arr.copy(order=order_out)
        else:
            del_obj = nlcpy.arange(start, stop, step)
    else:
        del_obj = nlcpy.asarray(obj)
        if del_obj.ndim != 1:
            del_obj = del_obj.ravel()

        if del_obj.dtype == bool:
            warnings.warn("in the future insert will treat boolean arrays and "
                          "array-likes as boolean index instead of casting it "
                          "to integer", FutureWarning, stacklevel=3)
            del_obj = del_obj.astype(nlcpy.intp)

        if isinstance(obj, (int, nlcpy.integer)):
            if (obj < -N or obj >= N):
                raise IndexError(
                    "index %i is out of bounds for axis %i with "
                    "size %i" % (obj, axis, N))
            if (obj < 0):
                del_obj += N
        else:
            if del_obj.dtype != int:
                warnings.warn(
                    "using a non-integer array as obj in delete will result in an "
                    "error in the future", DeprecationWarning, stacklevel=3)
                del_obj = del_obj.astype(nlcpy.intp)

    if del_obj.size == 0:
        new = nlcpy.array(input_arr)
        return new
    else:
        new = nlcpy.empty(input_arr.shape, input_arr.dtype, order_out)
        idx = nlcpy.ones(input_arr.shape[axis], dtype=del_obj.dtype)
        obj_count = nlcpy.zeros([3], dtype='l')
        request._push_request(
            'nlcpy_delete',
            'manipulation_op',
            (input_arr, del_obj, axis, idx, new, obj_count)
        )
        count = obj_count.get()
        if count[1] != 0:
            warnings.warn(
                "in the future out of bounds indices will raise an error "
                "instead of being ignored by `numpy.delete`.",
                DeprecationWarning, stacklevel=3)
        if count[2] != 0:
            warnings.warn(
                "in the future negative indices will not be ignored by "
                "`numpy.delete`.", FutureWarning, stacklevel=3)
        sl = [slice(N - count[0]) if i == axis
              else slice(None) for i in range(new.ndim)]
        return new[sl].copy()


def insert(arr, obj, values, axis=None):
    """Inserts values along the given axis before the given indices.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : int, slice or sequence of ints
        Object that defines the index or indices before which values is inserted.
        Support for multiple insertions when obj is a single scalar or a sequence
        with one element (similar to calling insert multiple times).
    values : array_like
        Values to insert into arr. If the type of values is different from that of
        arr, values is converted to the type of arr. values should be shaped so that
        arr[...,obj,...] = values is legal.
    axis : int, optional
        Axis along which to insert values. If axis is None then arr is flattened
        first.

    Returns
    -------
    out : ndarray
        A copy of arr with values inserted. Note that insert does not occur in-place:
        a new array is returned. If axis is None, out is a flattened array.

    Note:
        Note that for higher dimensional inserts obj=0 behaves very different from
        obj=[0] just like arr[:,0,:] = values is different from arr[:,[0],:] = values.

    See Also
    --------
    append : Appends values to the end of an array.
    concatenate : Joins a sequence of arrays along an existing axis.
    delete : Returns a new array with sub-arrays along an axis deleted.

    Examples
    --------
    >>> import nlcpy as vp
    >>> from nlcpy import testing
    >>> a = vp.array([[1, 1], [2, 2], [3, 3]])
    >>> a
    array([[1, 1],
           [2, 2],
           [3, 3]])
    >>> vp.insert(a, 1, 5)
    array([1, 5, 1, 2, 2, 3, 3])
    >>> vp.insert(a, 1, 5, axis=1)
    array([[1, 5, 1],
           [2, 5, 2],
           [3, 5, 3]])

    Difference between sequence and scalars:

    >>> vp.insert(a, [1], [[1],[2],[3]], axis=1)
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])
    >>> vp.testing.assert_array_equal(
    ...                vp.insert(a, 1, [1, 2, 3], axis=1),
    ...                vp.insert(a, [1], [[1],[2],[3]], axis=1))
    >>> b = a.flatten()
    >>> b
    array([1, 1, 2, 2, 3, 3])
    >>> vp.insert(b, [2, 2], [5, 6])
    array([1, 1, 5, 6, 2, 2, 3, 3])
    >>> vp.insert(b, slice(2, 4), [5, 6])
    array([1, 1, 5, 2, 6, 2, 3, 3])
    >>> vp.insert(b, [2, 2], [7.13, False]) # type casting
    array([1, 1, 7, 0, 2, 2, 3, 3])
    >>> x = vp.arange(8).reshape(2, 4)
    >>> idx = (1, 3)
    >>> vp.insert(x, idx, 999, axis=1)
    array([[  0, 999,   1,   2, 999,   3],
           [  4, 999,   5,   6, 999,   7]])

    """
    a = nlcpy.asarray(arr)
    if axis is None:
        if a.ndim != 1:
            a = a.ravel()
        axis = 0
    elif a.ndim == 0:
        warnings.warn(
            "in the future the special handling of scalars will be removed "
            "from insert and raise an error", DeprecationWarning, stacklevel=3)
        a = a.copy(order='F' if a.flags.f_contiguous else 'C')
        a[...] = values
        return a
    elif isinstance(axis, nlcpy.ndarray) or isinstance(axis, numpy.ndarray):
        axis = int(axis)
    elif not isinstance(axis, int):
        raise TypeError("an integer is required "
                        "(got type {0})".format(type(axis).__name__))

    if axis < -a.ndim or axis >= a.ndim:
        raise nlcpy.AxisError(
            "axis {0} is out of bounds for array of dimension {1}".format(axis, a.ndim))

    if axis < 0:
        axis += a.ndim

    if type(obj) is slice:
        start, stop, step = obj.indices(a.shape[axis])
        obj = nlcpy.arange(start, stop, step)
    else:
        obj = nlcpy.array(obj)
        if obj.dtype.char == '?':
            warnings.warn(
                "in the future insert will treat boolean arrays and "
                "array-likes as a boolean index instead of casting it to "
                "integer", FutureWarning, stacklevel=3)
        elif obj.dtype.char in 'fdFD':
            if obj.size == 1:
                raise TypeError(
                    "slice indices must be integers or "
                    "None or have an __index__ method")
            else:
                warnings.warn(
                    "using a non-integer array as obj in insert will result in an "
                    "error in the future", DeprecationWarning, stacklevel=3)
        elif obj.dtype.char in 'IL':
            if obj.size == 1:
                objval = obj[()] if obj.ndim == 0 else obj[0]
                if objval > a.shape[axis]:
                    raise IndexError(
                        "index {0} is out of bounds for axis {1} with size {2}".format(
                            objval, axis, a.shape[axis]))
            else:
                tmp = 'float64' if obj.dtype.char == 'L' else 'int64'
                raise UFuncTypeError(
                    "Cannot cast ufunc 'add' output from dtype('{0}') to "
                    "dtype('{1}') with casting rule 'same_kind'".format(tmp, obj.dtype))
        obj = obj.astype('l')
        if obj.ndim > 1:
            raise ValueError(
                "index array argument obj to insert must be one dimensional or scalar")

    if obj.ndim == 0:
        if obj > a.shape[axis] or obj < -a.shape[axis]:
            raise IndexError(
                "index {0} is out of bounds for axis {1} with size {2}".format(
                    obj[()] if obj > 0 else obj[()] + a.shape[axis],
                    axis, a.shape[axis]))

    newshape = list(a.shape)
    if obj.size == 1:
        values = nlcpy.array(values, copy=False, ndmin=a.ndim, dtype=a.dtype)
        if obj.ndim == 0:
            values = nlcpy.moveaxis(values, 0, axis)
        newshape[axis] += values.shape[axis]
        obj = nlcpy.array(nlcpy.broadcast_to(obj, values.shape[axis]))
        val_shape = list(a.shape)
        val_shape[axis] = values.shape[axis]
        values = nlcpy.broadcast_to(values, val_shape)
    else:
        newshape[axis] += obj.size
        values = nlcpy.array(values, copy=False, ndmin=a.ndim, dtype=a.dtype)
        val_shape = list(a.shape)
        val_shape[axis] = obj.size
        values = nlcpy.broadcast_to(values, val_shape)

    out = nlcpy.empty(newshape, dtype=a.dtype)
    work = nlcpy.zeros(obj.size + out.shape[axis] + 2, dtype='l')
    work[-1] = -1
    request._push_request(
        'nlcpy_insert',
        'manipulation_op',
        (a, obj, values, out, axis, work)
    )
    if work[-1] != -1:
        raise IndexError(
            "index {0} is out of bounds for axis {1} with size {2}"
            .format(obj[work[-1]], axis, out.shape[axis]))
    return out


def resize(a, new_shape):
    """Returns a new array with the specified shape.

    If the new array is larger than the original array, then the new array
    is filled with repeated copies of *a*.
    Note that this behavior is different from a.resize(new_shape) which fills
    with zeros instead of repeated copies of *a*.

    Parameters
    ----------
    a : array_like
        Array to be resized.
    new_shape : int or sequence of ints
        Shape of resized array.

    Returns
    -------
    reshaped_array : ndarray
        The new array is formed from the data in the old array, repeated if necessary to
        fill out the required number of elements. The data are repeated in the order that
        they are stored in memory.

    Note
    ----

    Warning: This functionality does **not** consider axes separately, i.e. it does
    not apply interpolation/extrapolation.
    It fills the return array with the required number of elements, taken from
    `a` as they are laid out in memory, disregarding strides and
    axes.
    (This is in case the new shape is smaller. For larger, see above.)
    This functionality is therefore not suitable to resize images, or data where
    each axis represents a separate and distinct entity.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a=vp.array([[0,1],[2,3]])
    >>> vp.resize(a,(2,3))
    array([[0, 1, 2],
           [3, 0, 1]])
    >>> vp.resize(a,(1,4))
    array([[0, 1, 2, 3]])
    >>> vp.resize(a,(2,4))
    array([[0, 1, 2, 3],
           [0, 1, 2, 3]])
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    a = nlcpy.ravel(a)
    Na = a.size
    total_size = core.internal.prod(new_shape)
    if Na == 0 or total_size == 0:
        return nlcpy.zeros(new_shape, a.dtype)

    n_copies = int(total_size / Na)
    extra = total_size % Na
    if extra != 0:
        n_copies = n_copies + 1
        extra = Na - extra

    a = nlcpy.concatenate((a,) * n_copies)
    if extra > 0:
        a = a[:-extra]

    return nlcpy.reshape(a, new_shape)

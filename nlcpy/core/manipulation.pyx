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

# distutils: language = c++

from libcpp.vector cimport vector

import sys
import numpy
import warnings
from nlcpy import veo

from nlcpy.core.core cimport ndarray
from nlcpy.core cimport core
from nlcpy.core cimport internal
from nlcpy.core.core cimport MemoryLocation
from nlcpy.core cimport broadcast
from nlcpy.core cimport dtype as _dtype
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.request cimport request

from functools import reduce

import nlcpy
cimport cython
cimport cpython

cdef ndarray _ndarray_reshape(ndarray self, tuple shape, order):
    cdef int order_char = internal._normalize_order(order, False)

    if len(shape) == 1 and cpython.PySequence_Check(shape[0]):
        shape = tuple(shape[0])

    if order_char == b'A':
        if self._f_contiguous and not self._c_contiguous:
            order_char = b'F'
        else:
            order_char = b'C'
    if order_char == b'C':
        return _reshape(self, shape)
    else:
        # The Fortran-ordered case is equivalent to:
        #     1.) reverse the axes via transpose
        #     2.) C-ordered reshape using reversed shape
        #     3.) reverse the axes via transpose
        return _T(_reshape(_T(self), shape[::-1]))


cdef _ndarray_shape_setter(ndarray self, newshape):
    cdef vector.vector[Py_ssize_t] shape, strides
    if not cpython.PySequence_Check(newshape):
        newshape = (newshape,)
    shape = internal.infer_unknown_dimension(newshape, self.size)
    _get_strides_for_nocopy_reshape(self, shape, strides)
    if strides.size() != shape.size():
        raise AttributeError('incompatible shape')
    self._shape = shape
    self._strides = strides
    self._update_f_contiguity()


cdef ndarray _ndarray_ravel(ndarray self, order):
    cdef int order_char
    cdef vector[Py_ssize_t] shape
    shape.push_back(self.size)

    order_char = internal._normalize_order(order, True)
    if order_char == b'A':
        if self._f_contiguous and not self._c_contiguous:
            order_char = b'F'
        else:
            order_char = b'C'
    if order_char == b'C':
        return _reshape(self, shape)
    elif order_char == b'F':
        return _reshape(_T(self), shape)
    elif order_char == b'K':
        raise NotImplementedError(
            'ravel with order=\'K\' not yet implemented.')


cdef ndarray _ndarray_resize(ndarray self, tuple shape, refcheck):
    if len(shape) == 1 and cpython.PySequence_Check(shape[0]):
        shape = tuple(shape[0])
    _resize(self, shape, refcheck)


cpdef ndarray _reshape(ndarray self,
                       const vector[Py_ssize_t] &shape_spec):
    cdef vector[Py_ssize_t] shape, strides
    cdef ndarray newarray
    vh_view = None
    # infer unknown shape, when detected negative value from shape
    # e.g.)
    #   >>> x = nlcpy.arange(10)
    #   >>> x.reshape(5,-1)
    # in above case, shape (5,-1) is convert to (5,2)
    shape = internal.infer_unknown_dimension(shape_spec, self.size)
    if internal.vector_equal(shape, self._shape):
        return self.view()
    # estimate newarray's strides
    _get_strides_for_nocopy_reshape(self, shape, strides)
    if strides.size() == shape.size():
        if self._memloc in {MemoryLocation.on_VH, MemoryLocation.on_VE_VH}:
            vh_view = self.vh_data.reshape(shape)
        return self._view(shape, strides, False, True, True, vh_view=vh_view)

    #  reshape from not contiguous strides array, need to create copy
    newarray = self.copy()
    _get_strides_for_nocopy_reshape(newarray, shape, strides)

    if strides.size() != shape.size():
        raise ValueError('cannot reshape array of size {} into shape {}'
                         .format(self.size, tuple(shape)))
    newarray._set_shape_and_strides(shape, strides, False, True)
    return newarray

cpdef _resize(ndarray self,
              const vector[Py_ssize_t] shape,
              const int refcheck):
    if not (self.ndim==0 or self._c_contiguous or self._f_contiguous):
        raise ValueError('resize only works on single-segment arrays')

    # /* Compute total size of old and new arrays. The new size might overflow */
    oldsize = self.size
    newsize = internal.prod(shape)
    if newsize < 0:
        raise ValueError('negative dimensions not allowed')

    # /* Convert to number of bytes. The new count might overflow */
    itemsize = self.itemsize
    oldnbytes = oldsize * itemsize
    newnbytes = newsize * itemsize

    if oldnbytes != newnbytes:
        if False:  # TODO !(PyArray_FLAGS(self) & NPY_ARRAY_OWNDATA)
            raise ValueError('cannot resize this array: it does not own its data')

        if refcheck is True:
            if self.base is not None:  # or if self.weakreflist != NULL
                raise ValueError('cannot resize an array that '
                                 'references or is referenced\n'
                                 'by another array in this way. '
                                 'Use the np.resize function.')

            refcnt = sys.getrefcount(self)
            if refcnt > 2:
                raise ValueError('cannot resize an array that '
                                 'references or is referenced\n'
                                 'by another array in this way.\n'
                                 'Use the np.resize function or '
                                 'refcheck=False.')

        # /* Reallocate space if needed - allocating 0 is forbidden */
        new_data = _realloc_kernel(self, newnbytes if newnbytes!=0 else itemsize)
        if new_data == 0:
            raise MemoryError('cannot allocate memory for array')
        self.ve_adr = new_data

    new_nd = shape.size()
    if new_nd > 0:
        # if self.ndim != new_nd:
        # /* Different number of dimensions. */
        # /* Need new dimensions and strides arrays */
        self._shape = shape
        self._set_contiguous_strides(self.itemsize, True)
        # ## make new_strides variable
    else:
        self._shape = []
        self._set_contiguous_strides(self.itemsize, True)

    if newnbytes > oldnbytes:
        _fill_kernel(self.ravel()[oldsize:], 0)

    return


cdef _get_strides_for_nocopy_reshape(
        ndarray a,
        const vector[Py_ssize_t] &newshape,
        vector[Py_ssize_t] &newstrides):
    cdef Py_ssize_t size, itemsize, ndim, dim, last_stride
    size = a.size
    newstrides.clear()
    if size != internal.prod(newshape):
        return

    itemsize = a.itemsize
    if size == 1:
        newstrides.assign(<Py_ssize_t>newshape.size(), itemsize)
        return

    cdef vector[Py_ssize_t] shape, strides
    # get shape and strides as 1-d array
    # e.g.)
    #       a._shape (3,3,3) is converted to shape (27,)
    #       a._strides (72,24,8) is converted to strides (8,)
    internal.get_reduced_dims(a._shape, a._strides, itemsize, shape, strides)

    # set newarray's strides by newshape and 1-d shape and strides
    ndim = shape.size()
    dim = 0
    sh = shape[0]
    st = strides[0]
    last_stride = shape[0] * strides[0]
    for i in range(newshape.size()):
        size = newshape[i]
        if size <= 1:
            newstrides.push_back(last_stride)
            continue
        if dim >= ndim or shape[dim] % size != 0:
            newstrides.clear()
            break
        shape[dim] //= size
        last_stride = shape[dim] * strides[dim]
        newstrides.push_back(last_stride)
        if shape[dim] == 1:
            dim += 1


cpdef ndarray _T(ndarray self):
    cdef vector[Py_ssize_t] newshape, newstrides
    newshape.reserve(len(self._shape))
    newstrides.reserve(len(self._strides))
    newshape.assign(self._shape.rbegin(), self._shape.rend())
    newstrides.assign(self._strides.rbegin(), self._strides.rend())
    ret = self._view(newshape, newstrides, True, True)
    return ret

cdef ndarray _ndarray_transpose(ndarray self, axes):
    if axes is None:
        return _T(self)
    if isinstance(axes, int):  # _transpose() uses axes[i] loop
        axes = (axes,)
    return _transpose(self, axes)

cpdef ndarray _rollaxis(ndarray a, Py_ssize_t axis, Py_ssize_t start=0):
    cdef Py_ssize_t i, ndim = a.ndim
    cdef vector.vector[Py_ssize_t] axes
    if axis < 0:
        axis += ndim
    if start < 0:
        start += ndim
    if not (0 <= axis < ndim and 0 <= start <= ndim):
        raise AxisError('Axis out of range')
    if axis < start:
        start -= 1
    if axis == start:
        return a

    for i in range(ndim):
        axes.push_back(i)
    axes.erase(axes.begin() + axis)
    axes.insert(axes.begin() + start, axis)
    return _transpose(a, axes)

cpdef ndarray _transpose(ndarray self, const vector[Py_ssize_t] &axes):
    cdef vector.vector[Py_ssize_t] a_axes
    cdef vector.vector[char] axis_flags
    cdef Py_ssize_t i, ndim, axis
    cdef bint is_normal = True, is_trans = True

    ndim = self._shape.size()
    if <Py_ssize_t>axes.size() != ndim:
        raise ValueError('axes don\'t match array')

    axis_flags.resize(ndim, 0)
    for i in range(len(axes)):
        axis = axes[i]
        if axis < -ndim or axis >= ndim:
            raise AxisError('axis {} is out of bounds for array of dimension {}'
                            .format(axis, ndim))
        axis %= ndim
        a_axes.push_back(axis)
        if axis_flags[axis]:
            raise ValueError('repeated axis in transpose')
        axis_flags[axis] = 1
        is_normal &= i == axis
        is_trans &= ndim - 1 - i == axis

    if is_normal:
        return self.view()
    if is_trans:
        return _T(self)

    ret = self.view()
    ret._shape.clear()
    ret._strides.clear()
    for axis in a_axes:
        ret._shape.push_back(self._shape[axis])
        ret._strides.push_back(self._strides[axis])
    ret._update_contiguity()
    return ret


cpdef ndarray _reduced_view(ndarray self):
    cdef vector[Py_ssize_t] shape, strides
    cdef Py_ssize_t ndim
    ndim = self._shape.size()
    if ndim <= 1:
        return self
    internal.get_reduced_dims(
        self._shape, self._strides, self.itemsize, shape, strides)
    if ndim == <Py_ssize_t>shape.size():
        return self
    view = self._view(shape, strides, True, True)
    return view


cdef _fill_kernel(ndarray dst, src):
    typ = numpy.dtype(dst.dtype).type
    value_dst = nlcpy.array(typ(src))  # cast dtype
    src = broadcast.broadcast_to(value_dst, dst.shape)
    request._push_request(
        "nlcpy_copy",
        "creation_op",
        (src, dst),
    )


cpdef ndarray _ndarray_concatenate(op, axis, ret):

    if not cpython.PySequence_Check(op):
        raise TypeError("The first input argument needs to be a sequence")

    n = len(op)
    if n < 0:
        return None
    if n == 0:
        raise ValueError("need at least one array to concatenate")
    arrays = []
    for i in range(n):
        item = op[i]
        arrays.append(core.array(item))

    if axis is not None and isinstance(axis, int) is False:
        raise TypeError("'%s' object cannot be interpreted as an integer"
                        % type(axis).__name__)

    if axis is None or axis >= 32:  # NPY_MAXDIMS=32
        return _concatenate_flattened_arrays(n, arrays, 'C', ret)
    else:
        return _concatenate_arrays(n, arrays, axis, ret)

cdef ndarray _concatenate_arrays(n, arrays, int32_t axis, ret):

    ndim = arrays[0].ndim
    if ndim == 0:
        raise ValueError("zero-dimensional arrays cannot be concatenated")
    if -ndim <= axis < ndim:
        if -ndim <= axis < 0:
            axis = ndim + axis
    else:
        raise AxisError("axis %d is out of bounds for array of dimension %d"
                        % (axis, ndim))

    # /* Figure out the final clncatenated shape starting from the
    #    first array's shape. */
    shape = list(arrays[0].shape)
    for i in range(1, n):
        if arrays[i].ndim != ndim:
            raise ValueError(
                "all the input arrays must have same number of "
                "dimensions, but the array at index {} has {} "
                "dimension(s) and the array at index {} has {} "
                "dimension(s)".format(0, ndim, i, arrays[i].ndim))
        arr_shape = arrays[i].shape

        for idim in range(ndim):
            if idim == axis:
                shape[idim] += arr_shape[idim]
            elif shape[idim] != arr_shape[idim]:
                raise ValueError(
                    "all the input array dimensions for the "
                    "concatenation axis must match exactly, but "
                    "along dimension {}, the array at index {} has "
                    "size {} and the array at index {} has size {}".format(
                        idim, 0, shape[idim], i, arr_shape[idim]))

    if ret is not None:
        if ret.ndim != ndim:
            raise ValueError("Output array has wrong dimensionality")
        if list(ret.shape) != shape:
            raise ValueError("Output array is the wrong shape")
    else:
        dtype = numpy.result_type(*arrays)
        ret = ndarray(shape, dtype=dtype)

    view = ret.view()
    dsts = [None] * n
    s = 0
    for i in range(n):
        if not numpy.can_cast(arrays[i].dtype, view.dtype, casting="same_kind"):
            raise TypeError("Cannot cast scalar from dtype('{}') to dtype('{}') "
                            "according to the rule 'same_kind'"
                            .format(arrays[i].dtype, view.dtype))
        view._set_shape_and_strides(arrays[i].shape, ret.strides, True, True)
        dsts[i] = view
        s += reduce(lambda x, y: x * y, arrays[i].shape[axis:])
        view = ret.ravel()[s:]

    for i in range(n):
        src = arrays[i]
        dst = dsts[i]
        request._push_request(
            "nlcpy_copy",
            "creation_op",
            (src, dst),
        )

    return ret

cdef ndarray _concatenate_flattened_arrays(n, arrays, order, ret):

    shape = 0
    for i in range(n):
        shape += arrays[i].size

    if ret is not None:
        if ret.ndim != 1:
            raise ValueError('Output array must be 1D')
        if shape != ret.size:
            raise ValueError('Output array is the wrong size')
    else:
        dtype = numpy.result_type(*arrays)
        if dtype is None:
            return None
        ret = ndarray((shape,), dtype=dtype)
        if ret is None:
            return None

    view = ret.view()
    dsts = [None] * n
    for i in range(n):
        dsts[i] = view[:arrays[i].size].ravel()
        view = view[arrays[i].size:]

    for i in range(n):
        src = arrays[i]
        dst = dsts[i]
        request._push_request(
            "nlcpy_copy",
            "creation_op",
            (src.ravel(), dst),
        )

    return ret


cdef _realloc_kernel(ndarray src, nbytes):
    v = veo.VeoAlloc()
    dst = v.proc.realloc_mem(
        veo.VEMemPtr(v.proc, src.ve_adr, nbytes), nbytes)
    return dst.addr


cdef ndarray _ndarray_flatten(ndarray self, order):
    cdef int order_char
    cdef vector[Py_ssize_t] shape
    shape.push_back(self.size)

    order_char = internal._normalize_order(order, True)
    if order_char == b'A' or order_char == b'K':
        if self._f_contiguous and not self._c_contiguous:
            order_char = b'F'
        else:
            order_char = b'C'
    if order_char == b'C':
        return _reshape(self, shape).copy()
    elif order_char == b'F':
        raise NotImplementedError(
            'ravel with order=\'F\' not yet implemented.')


cpdef ndarray _moveaxis(ndarray a, source, destination):
    cdef vector.vector[Py_ssize_t] src, dest
    _normalize_axis_tuple(source, a.ndim, src)
    _normalize_axis_tuple(destination, a.ndim, dest)

    if src.size() != dest.size():
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    cdef vector.vector[Py_ssize_t] order
    cdef Py_ssize_t n = 0
    for i in range(a.ndim):
        n = <Py_ssize_t>i
        if not _has_element(src, n):
            order.push_back(n)

    cdef Py_ssize_t d, s
    for d, s in sorted(zip(dest, src)):
        order.insert(order.begin() + d, s)

    return _transpose(a, order)


cdef _normalize_axis_tuple(axis, Py_ssize_t ndim,
                           vector.vector[Py_ssize_t] &ret):
    """Normalizes an axis argument into a tuple of non-negative integer axes.

    Arguments `allow_duplicate` and `axis_name` are not supported.

    """
    if numpy.isscalar(axis):
        axis = (axis,)

    for ax in axis:
        if ax >= ndim or ax < -ndim:
            raise AxisError(
                'axis {} is out of bounds for array of '
                'dimension {}'.format(ax, ndim))
        if _has_element(ret, ax):
            raise AxisError('repeated axis')
        ret.push_back(ax % ndim)

cdef bint _has_element(const vector.vector[Py_ssize_t] &source, Py_ssize_t n):
    for i in range(source.size()):
        if source[i] == n:
            return True
    return False

cpdef ndarray _expand_dims(ndarray a, axis):
    if axis > a.ndim or axis < -a.ndim - 1:
        warnings.warn("Both axis > a.ndim and axis < -a.ndim - 1 are"
                      " deprecated and will raise an AxisError"
                      " in the future.", DeprecationWarning, stacklevel=2)
    lst = list(a.shape)
    if axis < 0:
        axis = a.ndim + axis + 1
    ll = lst[:axis] + [1, ] + lst[axis:]
    shape_out = tuple(ll)
    return a.reshape(shape_out)


cpdef ndarray _ndarray_squeeze(ndarray self, axis):
    a = self
    lst = list(a.shape)
    if axis is not None:
        if isinstance(axis, int):
            axis = [axis, ]
        ax = []
        msk = numpy.zeros(a.ndim, dtype="int32")
        for (i, axis_i) in enumerate(axis):
            if a.ndim > 0:
                if axis_i > a.ndim - 1 or axis_i < -a.ndim:
                    raise numpy.AxisError("axis_i %d is out of bounds for array"
                                          " of dimension %d" % (axis_i, a.ndim))
                if msk[axis_i] != 0:
                    raise ValueError("duplicate value in 'axis'")
                msk[axis_i] = 1
            elif a.ndim == 0:
                if axis_i != 0 and axis_i != -1:
                    raise numpy.AxisError("axis_i %d is out of bounds for array"
                                          " of dimension %d" % (axis_i, a.ndim))
            if axis_i < 0:
                axis_i = a.ndim + axis_i
            if a.ndim != 0 and lst[axis_i] != 1:
                raise ValueError("cannot select an axis_i to squeeze out"
                                 " which has size not equal to one")
            ax.append(axis_i)
        ll = []
        for (i, l_i) in enumerate(lst):
            if i not in ax:
                ll.append(l_i)
    else:
        ll = []
        for i in lst:
            if i != 1:
                ll.append(i)

    return a.reshape(tuple(ll))

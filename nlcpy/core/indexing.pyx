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

from libcpp.vector cimport vector
from libc.stdint cimport *

from nlcpy.core.core cimport ndarray
from nlcpy.core cimport internal
from nlcpy.core.core cimport MemoryLocation
from nlcpy.core cimport core
from nlcpy.core cimport manipulation
from nlcpy.core cimport broadcast
from nlcpy.core cimport dtype as _dtype
from nlcpy.request cimport request

import nlcpy
from nlcpy import veo
from nlcpy.core import error

import numpy
cimport cython
cimport cpython

cdef ndarray _ndarray_getitem(ndarray self, slices):
    # supports basic indexing (by slices, ints or Ellipsis) and
    # some parts of advanced indexing by integer or boolean arrays.
    cdef Py_ssize_t mask_i
    cdef list slice_list, adv_mask, adv_slices
    cdef bint advanced, mask_exists

    # parse and complement input indices.
    # if input index is list, tuple, boolean, numpy array, the index is transfered to VE.
    # slice_list: unpacked indices.
    # advanced: advanced slicing flag.
    # mask_exists: boolean slicing flag.
    slice_list, advanced, mask_exists = _prepare_slice_list(
        slices, self._shape.size())
    if mask_exists:
        mask_i = _get_mask_index(slice_list)
        return _getitem_mask_single(self, slice_list[mask_i], mask_i)
    if advanced:
        a, adv_slices, adv_mask = _prepare_advanced_indexing(
            self, slice_list)
        if sum(adv_mask) == 1:
            axis = adv_mask.index(True)
            return a.take(adv_slices[axis], axis)
        return _getitem_multiple(a, adv_slices)
    return _simple_getitem(self, slice_list)


cdef ndarray _getitem_multiple(ndarray a, list slices):
    a, reduced_idx, li, ri = _prepare_multiple_array_indexing(a, slices)
    return _take(a, reduced_idx, li, ri)


cdef _ndarray_setitem(ndarray self, slices, value):
    _setitem_core(self, slices, value)


cdef ndarray _ndarray_take(ndarray self, indices, axis, out):
    if axis is None:
        return _take(self, indices, 0, self._shape.size() - 1, out)
    else:
        return _take(self, indices, axis, axis, out)


cdef ndarray _take(ndarray a, indices, int li, int ri, ndarray out=None):
    # When li + 1 == ri this function behaves similarly to np.take
    cdef tuple out_shape, ind_shape, indices_shape
    cdef int i, ndim = a._shape.size()
    cdef Py_ssize_t ldim, cdim, rdim, index_range
    if ndim == 0:
        a = a.ravel()
        ndim = 1

    if not (-ndim <= li < ndim and -ndim <= ri < ndim):
        raise error._AxisError('axis %s is out of bounds for array of dimension %s'
                               % (li, ndim))

    if ndim == 1:
        li = ri = 0
    else:
        li %= ndim
        ri %= ndim
        assert 0 <= li <= ri

    if numpy.isscalar(indices):
        if type(indices) is not int:
            indices = int(indices)
        indices_shape = ()
        cdim = 1
    else:
        if not isinstance(indices, ndarray):
            indices = core.array(indices, dtype=int)
        indices = indices.astype(dtype=int, copy=False)
        indices_shape = indices.shape
        cdim = indices.size

    ldim = rdim = 1
    if ndim == 1:
        out_shape = indices_shape
        index_range = a.size
    else:
        a_shape = a.shape
        out_shape = a_shape[:li] + indices_shape + a_shape[ri + 1:]
        if len(indices_shape) != 0:
            indices = manipulation._reshape(
                indices,
                (1,) * li + indices_shape + (1,) * (ndim - (ri + 1)))
        for i in range(li):
            ldim *= a._shape[i]
        for i in range(ri + 1, ndim):
            rdim *= a._shape[i]
        index_range = a.size // (ldim * rdim)

    if out is None:
        out = ndarray(out_shape, dtype=a.dtype)
    else:
        if out.dtype != a.dtype:
            raise TypeError('Output dtype mismatch')
        if out.shape != out_shape:
            raise ValueError('Output shape mismatch')
    if a.size == 0 and out.size != 0:
        raise IndexError('cannot do a non-empty take from an empty axes.')

    # indices %= index_range

    # reduced_view = manipulation._reduced_view(a)
    # if reduced_view.base is not None:
    #     reduced = reduced_view.copy()
    # else:
    #     reduced = reduced_view
    reduced = a.ravel()
    if isinstance(indices, ndarray):
        return _take_kernel(
            reduced, indices, ldim, cdim, rdim, index_range, out)
    else:
        return _take_kernel(
            reduced, nlcpy.asarray(indices), ldim, cdim, rdim, index_range, out)

# TODO: boolean mask
cdef _setitem_core(ndarray a, slices, value):
    cdef Py_ssize_t i, li, ri
    cdef ndarray v, x, y, a_interm, reduced_idx
    cdef list slice_list, adv_mask, adv_slices
    cdef bint advanced, mask_exists

    slice_list, advanced, mask_exists = _prepare_slice_list(
        slices, a._shape.size())

    if mask_exists:
        mask_i = _get_mask_index(slice_list)
        _setitem_mask_single(a, slice_list[mask_i], value, mask_i)
        return

    if advanced:
        a, adv_slices, adv_mask = _prepare_advanced_indexing(a, slice_list)
        # scatter_op with single integer arrays
        if sum(adv_mask) == 1:
            axis = adv_mask.index(True)
            _scatter_op_single(a, adv_slices[axis], value, axis, axis, 'update')
            return

        # scatter_op with multiple integer arrays
        a_interm, reduced_idx, li, ri =\
            _prepare_multiple_array_indexing(a, adv_slices)
        _scatter_op_single(a_interm, reduced_idx, value, li, ri, 'update')
        return

    y = _simple_getitem(a, slice_list)
    if not isinstance(value, ndarray):
        value = nlcpy.array(value, dtype=y.dtype)
    x = value
    x = broadcast.broadcast_to(x, y.shape)
    if (internal.vector_equal(y._shape, x._shape) and
            internal.vector_equal(y._strides, x._strides)):
        if y.ve_adr == x.ve_adr:
            return  # Skip since x and y are the same array
    _copy_without_alloc(x, y)
    return


cpdef tuple _prepare_slice_list(slices, Py_ssize_t ndim):
    cdef Py_ssize_t i, n_newaxes, axis
    cdef list slice_list
    cdef char kind
    cdef bint advanced, mask_exists

    if isinstance(slices, tuple):
        slice_list = list(slices)
    elif isinstance(slices, list):
        slice_list = list(slices)  # copy list
        for s in slice_list:
            if not isinstance(s, int):
                break
        else:
            slice_list = [slice_list]
    else:
        slice_list = [slices]

    slice_list, n_newaxes = internal.complete_slice_list(slice_list, ndim)

    # Check if advanced is true,
    # and convert list/NumPy arrays to nlcpy.ndarray
    advanced = False
    mask_exists = False
    for i, s in enumerate(slice_list):
        to_ve = True
        if isinstance(s, list):
            # handle the case when s is an empty list
            # s = numpy.array(s, dtype='i8')
            s = numpy.array(s)
            if s.size == 0:
                s = s.astype(numpy.int32)
        elif isinstance(s, bool):
            s = numpy.array(s)
        elif isinstance(s, ndarray):
            to_ve = False
        elif not isinstance(s, numpy.ndarray):
            if isinstance(s, int) or isinstance(s, slice) or s is None:
                continue  # int, slice, ellipsis or None
            raise IndexError(
                'only integers, slices (\':\'), ellipsis (\'...\'), '
                'nlcpy.newaxis (\'None\') and integer or boolean arrays '
                'are valid indices')
        # case s is numpy.ndarray
        kind = ord(s.dtype.kind)
        if kind == b'i' or kind == b'u':
            advanced = True
        elif kind == b'b':
            mask_exists = True
        else:
            raise IndexError(
                'only integers, slices (\':\'), ellipsis (\'...\'), '
                'nlcpy.newaxis (\'None\') and integer or boolean arrays '
                'are valid indices')
        if to_ve:
            # send to ve
            slice_list[i] = core.array(s)

    if not mask_exists and len(slice_list) > ndim + n_newaxes:
        raise IndexError('too many indices for array')
    return slice_list, advanced, mask_exists


cdef tuple _prepare_advanced_indexing(ndarray a, list slice_list):
    cdef slice none_slice = slice(None)
    # split slices that can be handled by basic-indexing
    cdef list basic_slices = []
    cdef list adv_slices = []
    cdef list adv_mask = []
    cdef bint use_basic_indexing = False
    for i, s in enumerate(slice_list):
        if s is None:
            basic_slices.append(None)
            adv_slices.append(none_slice)
            adv_mask.append(False)
            use_basic_indexing = True
        elif isinstance(s, slice):
            basic_slices.append(s)
            adv_slices.append(none_slice)
            adv_mask.append(False)
            use_basic_indexing |= s != none_slice
        elif isinstance(s, ndarray):
            kind = s.dtype.kind
            assert kind == 'i' or kind == 'u'
            basic_slices.append(none_slice)
            adv_slices.append(s)
            adv_mask.append(True)
        elif isinstance(s, int):
            basic_slices.append(none_slice)
            scalar_array = ndarray((), dtype=numpy.int64)
            scalar_array.fill(s)
            adv_slices.append(scalar_array)
            adv_mask.append(True)
        else:
            raise IndexError(
                'only integers, slices (`:`), ellipsis (`...`),'
                'numpy.newaxis (`None`) and integer or '
                'boolean arrays are valid indices')

    # check if this is a combination of basic and advanced indexing
    if use_basic_indexing:
        a = _simple_getitem(a, basic_slices)
    return a, adv_slices, adv_mask


cdef tuple _prepare_multiple_array_indexing(ndarray a, list slices):
    # slices consist of either slice(None) or ndarray
    cdef Py_ssize_t i, p, li, ri, stride, prev_arr_i
    cdef ndarray reduced_idx
    cdef bint do_transpose

    values, shape = broadcast._broadcast_core(slices)
    slices = list(values)

    # check if transpose is necessasry
    # li:  index of the leftmost array in slices
    # ri:  index of the rightmost array in slices
    do_transpose = False
    prev_arr_i = -1
    li = 0
    ri = 0
    for i, s in enumerate(slices):
        if isinstance(s, ndarray):
            if prev_arr_i == -1:
                prev_arr_i = i
                li = i
            elif i - prev_arr_i > 1:
                do_transpose = True
            else:
                prev_arr_i = i
                ri = i

    if do_transpose:
        transp_a = []
        transp_b = []
        slices_a = []
        slices_b = []

        for i, s in enumerate(slices):
            if isinstance(s, ndarray):
                transp_a.append(i)
                slices_a.append(s)
            else:
                transp_b.append(i)
                slices_b.append(s)
        a = manipulation._transpose(a, transp_a + transp_b)
        slices = slices_a + slices_b
        li = 0
        ri = len(transp_a) - 1

    reduced_idx = ndarray(shape, dtype=numpy.int64)
    reduced_idx.fill(0)
    stride = 1
    for i in range(ri, li - 1, -1):
        s = slices[i]
        a_shape_i = a._shape[i]
        # wrap all out-of-bound indices
        if a_shape_i != 0:
            # TODO: modify VE kernel bug
            # _prepare_array_indexing(s, reduced_idx, a_shape_i, stride)
            reduced_idx += stride * \
                (s - nlcpy.floor_divide(s, a_shape_i) * a_shape_i)
        stride *= a_shape_i
    return a, reduced_idx, li, ri


cdef ndarray _simple_getitem(ndarray a, list slice_list):
    cdef vector[Py_ssize_t] shape, strides
    cdef ndarray v
    cdef Py_ssize_t i, j, offset, ndim
    cdef Py_ssize_t s_start, s_stop, s_step, dim, ind
    cdef slice ss
    # Create new shape and stride
    j = 0
    offset = 0
    ndim = a._shape.size()
    for i, s in enumerate(slice_list):
        # index is Newaxis (None)
        if s is None:
            shape.push_back(1)
            if j < ndim:
                strides.push_back(a._strides[j])
            elif ndim > 0:
                strides.push_back(a._strides[ndim - 1])
            else:
                strides.push_back(a.itemsize)
        elif ndim <= j:
            raise IndexError('too many indices for array')
        # index is Slice object
        elif isinstance(s, slice):
            ss = internal.complete_slice(s, a._shape[j])
            s_start = ss.start
            s_stop = ss.stop
            s_step = ss.step
            if s_step > 0:
                dim = (s_stop - s_start - 1) // s_step + 1
            else:
                dim = (s_stop - s_start + 1) // s_step + 1

            if dim == 0:
                strides.push_back(a._strides[j])
            else:
                strides.push_back(a._strides[j] * s_step)

            if s_start > 0:
                offset += a._strides[j] * s_start
            shape.push_back(dim)
            j += 1
        # index is integer
        elif numpy.isscalar(s):
            ind = int(s)
            if ind < 0:
                ind += a._shape[j]
            if not (0 <= ind < a._shape[j]):
                msg = ('Index %s is out of bounds for axis %s with '
                       'size %s' % (s, j, a._shape[j]))
                raise IndexError(msg)
            offset += ind * a._strides[j]
            j += 1
        else:
            raise TypeError('Invalid index type: %s' % type(slice_list[i]))

    v = a._view(shape, strides, True, True, True,
                vh_view=None, dtype=None, type=None, offset=offset)
    return v

cdef Py_ssize_t _get_mask_index(list slice_list) except *:
    cdef Py_ssize_t i, n_not_slice_none, mask_i
    cdef slice none_slice = slice(None)
    n_not_slice_none = 0
    mask_i = -1
    for i, s in enumerate(slice_list):
        if not isinstance(s, slice) or s != none_slice:
            n_not_slice_none += 1
            if isinstance(s, ndarray) and s.dtype == numpy.bool_:
                mask_i = i
    if n_not_slice_none != 1 or mask_i == -1:
        raise ValueError('currently, NlcPy only supports slices that '
                         'consist of one boolean array.')
    return mask_i


cpdef _prepare_mask_indexing_single(ndarray a, ndarray mask, Py_ssize_t axis):
    cdef ndarray mask_scanned, mask_br, mask_br_scanned
    cdef int n_true
    cdef tuple lshape, rshape, out_shape

    lshape = a.shape[:axis]
    rshape = a.shape[axis + mask._shape.size():]

    if mask.size == 0:
        masked_shape = lshape + (0,) + rshape
        mask_br = manipulation._reshape(mask, masked_shape)
        return mask_br, masked_shape

    for i, s in enumerate(mask._shape):
        if axis + i >= a.ndim:
            continue
        if a.shape[axis + i] != s:
            raise IndexError(
                'boolean index did not match indexed array along dimension {}; '
                'dimension is {} but corresponding boolean dimension is {}'
                .format(axis + i, a.shape[axis + i], s))

    # Get number of True in the mask to determine the shape of the array
    # after masking.
    n_true = _count_n_true_kernel(mask.ravel())
    masked_shape = lshape + (n_true,) + rshape

    # When mask covers the entire array, broadcasting is not necessary.
    if mask._shape.size() == a._shape.size() and axis == 0:
        return (
            mask,
            masked_shape)

    # The scan of the broadcasted array is used to index on kernel.
    mask = manipulation._reshape(
        mask,
        axis * (1,) + mask.shape + (a.ndim - axis - mask.ndim) * (1,))
    if mask._shape.size() > a._shape.size():
        raise IndexError('too many indices for array')

    mask = broadcast.broadcast_to(mask, a.shape)
    return mask, masked_shape


cpdef ndarray _getitem_mask_single(ndarray a, ndarray mask, int axis):
    cdef tuple masked_shape
    # broadcast mask data if necessary, and create masked shape
    mask, masked_shape = _prepare_mask_indexing_single(
        a, mask, axis)
    out = ndarray(masked_shape, dtype=a.dtype)
    if out.size == 0:
        return out
    _getitem_mask_kernel(a, mask, out)
    return out

cpdef _setitem_mask_single(ndarray a, ndarray mask, v, int axis):
    cdef tuple masked_shape
    cdef ndarray src
    # broadcast mask data if necessary, and create masked shape
    mask, masked_shape = _prepare_mask_indexing_single(
        a, mask, axis)
    if internal.prod(masked_shape) == 0:
        return
    if not isinstance(v, ndarray):
        src = core.array(v, dtype=a.dtype)
    else:
        src = v.astype(dtype=a.dtype, copy=False)

    src = broadcast.broadcast_to(src, masked_shape)
    _setitem_mask_kernel(a, mask, src)


cdef _scatter_op_single(
        ndarray a, ndarray indices, v, Py_ssize_t li=0, Py_ssize_t ri=0,
        op=''):
    cdef Py_ssize_t ndim, adim, cdim, rdim
    cdef tuple a_shape, indices_shape, lshape, rshape, v_shape

    ndim = a._shape.size()

    if ndim == 0:
        raise ValueError('requires a.ndim >= 1')
    if not (-ndim <= li < ndim and -ndim <= ri < ndim):
        raise ValueError('Axis overrun')

    if not isinstance(v, ndarray):
        v = core.array(v, dtype=a.dtype)
    else:
        v = v.astype(a.dtype, copy=False)

    a_shape = a.shape
    li %= ndim
    ri %= ndim

    lshape = a_shape[:li]
    rshape = a_shape[ri + 1:]
    adim = internal.prod(a_shape[li:ri + 1])

    indices_shape = indices.shape
    v_shape = lshape + indices_shape + rshape
    v = broadcast.broadcast_to(v, v_shape)

    cdim = indices.size
    rdim = internal.prod(rshape)
    indices = manipulation._reshape(
        indices,
        (1,) * len(lshape) + indices_shape + (1,) * len(rshape))
    indices = broadcast.broadcast_to(indices, v_shape)

    if op == 'update':
        reduced_view = manipulation._reduced_view(a)
        if reduced_view.base is not None:
            # TODO: improve performance
            reduced_copy = reduced_view.copy()
            _scatter_update_kernel(
                v, indices, cdim, rdim, adim, reduced_copy)
            _copy_without_alloc(reduced_copy, reduced_view)
            return
        else:
            reduced = reduced_view
            _scatter_update_kernel(
                v, indices, cdim, rdim, adim, reduced)
            return

    elif op == 'add':
        raise NotImplementedError
    else:
        raise ValueError('provided op is not supported')


cdef int64_t _count_n_true_kernel(ndarray mask):
    args = (mask._ve_array,)
    n_true = request._push_and_flush_request(
        'nlcpy_count_n_true',
        args,
        callback=None,
        sync=True
    )
    return n_true

cdef _getitem_mask_kernel(ndarray a, ndarray mask, ndarray out):
    request._push_request(
        "nlcpy_getitem_from_mask",
        "indexing_op",
        (a, mask, out),
    )

cdef _setitem_mask_kernel(ndarray a, ndarray mask, ndarray value):
    request._push_request(
        "nlcpy_setitem_from_mask",
        "indexing_op",
        (a, mask, value),
    )

cdef ndarray _take_kernel(ndarray reduced, ndarray indices,
                          Py_ssize_t ldim, Py_ssize_t cdim, Py_ssize_t rdim,
                          Py_ssize_t index_range, ndarray out):
    values, shape = broadcast._broadcast_core((indices, out))
    request._push_request(
        "nlcpy_take",
        "indexing_op",
        (reduced, values[0], values[1],
         int(ldim), int(cdim), int(rdim),
         int(index_range)),
    )
    return values[1]


cdef ndarray _scatter_update_kernel(ndarray val, ndarray indices,
                                    Py_ssize_t cdim, Py_ssize_t rdim,
                                    Py_ssize_t adim, ndarray reduced):
    request._push_request(
        "nlcpy_scatter_update",
        "indexing_op",
        (reduced, indices, val, cdim, rdim, adim),
    )

cdef _copy_without_alloc(ndarray src, ndarray dst):
    request._push_request(
        "nlcpy_copy",
        "creation_op",
        (src, dst),
    )

# cdef _prepare_array_indexing(
#                 ndarray s, ndarray reduced_idx, int a_shape_i, int stride):
#     request._push_request(
#         "nlcpy_prepare_indexing",
#         "indexing_op",
#         (s, reduced_idx, a_shape_i, stride),
#     )

cdef ndarray _ndarray_diagonal(ndarray self, offset, axis1, axis2):
    return _diagonal(self, offset, axis1, axis2)

cdef ndarray _diagonal(
        ndarray a, Py_ssize_t offset=0, Py_ssize_t axis1=0,
        Py_ssize_t axis2=1):
    cdef Py_ssize_t ndim = a.ndim
    if ndim < 2:
        raise ValueError('diag requires an array of at least two dimensions')
    if not (-ndim <= axis1 < ndim and -ndim <= axis2 < ndim):
        raise error._AxisError(
            'axis1(={0}) and axis2(={1}) must be within range '
            '(ndim={2})'.format(axis1, axis2, ndim))

    axis1 %= ndim
    axis2 %= ndim
    if axis1 < axis2:
        min_axis, max_axis = axis1, axis2
    else:
        min_axis, max_axis = axis2, axis1

    tr = list(range(ndim))
    del tr[max_axis]
    del tr[min_axis]
    if offset >= 0:
        a = manipulation._transpose(a, tr + [axis1, axis2])
    else:
        a = manipulation._transpose(a, tr + [axis2, axis1])
        offset = -offset

    diag_size = max(0, min(a.shape[-2], a.shape[-1] - offset))
    ret_shape = a.shape[:-2] + (diag_size,)
    if diag_size == 0:
        return nlcpy.ndarray(ret_shape, dtype=a.dtype)

    a = a[..., :diag_size, offset:offset + diag_size]

    ret = a.view()
    ret._set_shape_and_strides(
        a.shape[:-2] + (diag_size,),
        a.strides[:-2] + (a.strides[-1] + a.strides[-2],),
        True, True)
    return ret

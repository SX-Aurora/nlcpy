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

cimport cpython

cpdef inline Py_ssize_t prod(args):
    cdef Py_ssize_t n = 1
    for x in args:
        n *= x
    # for i in range(args.size()):
    #     n *= args[i]
    return n

cpdef inline tuple get_size(object size):
    if size is None:
        return ()
    if cpython.PySequence_Check(size):
        return tuple(size)
    if isinstance(size, int):
        return size,
    raise ValueError('size should be None, collections.abc.Sequence, or int')


cpdef inline Py_ssize_t set_contiguous_strides(
        const vector.vector[Py_ssize_t]& shape,
        vector.vector[Py_ssize_t]& strides,
        Py_ssize_t itemsize, bint is_c_contiguous):
    cdef Py_ssize_t st, sh
    cdef Py_ssize_t is_nonzero_size = 1
    cdef int i, ndim = shape.size()
    cdef Py_ssize_t idx
    strides.resize(ndim, 0)
    st = 1

    for i in range(ndim):
        if is_c_contiguous:
            idx = ndim - 1 - i
        else:
            idx = i
        strides[idx] = st * itemsize
        sh = shape[idx]
        if sh > 1:
            st *= sh
        # elif sh == 1:
        #     strides[idx] = 0
        elif sh == 0:
            is_nonzero_size = 0
    return st * is_nonzero_size


cpdef inline bint get_c_contiguity(
        vector[Py_ssize_t]& shape,
        vector[Py_ssize_t]& strides,
        Py_ssize_t itemsize):
    cdef vector[Py_ssize_t] r_shape, r_strides
    cdef Py_ssize_t ndim
    ndim = strides.size()
    if ndim == 0 or (ndim == 1 and strides[0] == itemsize):
        return True
    get_reduced_dims(shape, strides, itemsize, r_shape, r_strides)
    ndim = r_strides.size()
    return ndim == 0 or (ndim == 1 and r_strides[0] == itemsize)


cpdef bint vector_equal(
        const vector[Py_ssize_t]& x,
        const vector[Py_ssize_t]& y):
    cdef Py_ssize_t n = x.size()
    if n != <Py_ssize_t>y.size():
        return False
    for i in range(n):
        if x[i] != y[i]:
            return False
    return True


cpdef tuple check_all_arrays_memloc(arrays):
    cdef:
        int64_t max_size
        int flag
    # find maximum size and memloc flag
    # if all array's memory exist in VE, flag = MemoryLocation.on_VE
    # if all array's memory exist in VH, flag = MemoryLocation.on_VH
    max_size = 0
    flag = 0b111
    for x in list(arrays):
        if not isinstance(x, ndarray):
            continue
        max_size = max(max_size, x.size)
        flag = flag & x._memloc
    return max_size, flag


cpdef vector[Py_ssize_t] infer_unknown_dimension(
        const vector[Py_ssize_t]& shape,
        Py_ssize_t size) except *:
    cdef vector[Py_ssize_t] ret = shape
    cdef Py_ssize_t cnt=0, index=-1, new_size=1
    for i in range(shape.size()):
        if shape[i] < 0:
            cnt += 1
            index = i
        else:
            new_size *= shape[i]
    if cnt == 0:
        return ret
    if cnt > 1:
        raise ValueError('can only specify one unknown dimension')
    if (size != 0 and new_size == 0) or size % new_size != 0:
        raise ValueError('cannot reshape array of size {} into shape {}'
                         .format(size, tuple(shape)))
    ret[index] = size // new_size
    return ret


cdef void get_reduced_dims(
        const vector[Py_ssize_t]& shape,
        const vector[Py_ssize_t]& strides,
        Py_ssize_t itemsize,
        vector[Py_ssize_t]& reduced_shape,
        vector[Py_ssize_t]& reduced_strides):
    cdef vector.vector[Py_ssize_t] tmp_shape, tmp_strides
    cdef Py_ssize_t i, ndim, sh, st, prev_st, index
    ndim = shape.size()
    reduced_shape.clear()
    reduced_strides.clear()
    if ndim == 0:
        return

    for i in range(ndim):
        sh = shape[i]
        if sh == 0:
            reduced_shape.push_back(0)
            reduced_strides.push_back(itemsize)
            return
        if sh != 1:
            tmp_shape.push_back(sh)
            tmp_strides.push_back(strides[i])
    if tmp_shape.size() == 0:
        return

    reduced_shape.push_back(tmp_shape[0])
    reduced_strides.push_back(tmp_strides[0])
    index = 0
    for i in range(<Py_ssize_t>tmp_shape.size() - 1):
        sh = tmp_shape[i + 1]
        st = tmp_strides[i + 1]
        if tmp_strides[i] == sh * st:
            reduced_shape[index] *= sh
            reduced_strides[index] = st
        else:
            reduced_shape.push_back(sh)
            reduced_strides.push_back(st)
            index += 1


cpdef inline Py_ssize_t _extract_slice_element(x) except? 0:
    try:
        return x.__index__()
    except AttributeError:
        return int(x)


cpdef slice complete_slice(slice slc, Py_ssize_t dim):
    cdef Py_ssize_t start=0, stop=0, step=0
    cdef bint start_none, stop_none
    if slc.step is None:
        step = 1
    else:
        try:
            step = _extract_slice_element(slc.step)
        except TypeError:
            raise TypeError(
                'slice.step must be int or None or have __index__ method: '
                '{}'.format(slc))
        if step == 0:
            raise ValueError('Slice step must be nonzero.')

    start_none = slc.start is None
    if not start_none:
        try:
            start = _extract_slice_element(slc.start)
        except TypeError:
            raise TypeError(
                'slice.start must be int or None or have __index__ method: '
                '{}'.format(slc))

        if start < 0:
            start += dim

    stop_none = slc.stop is None
    if not stop_none:
        try:
            stop = _extract_slice_element(slc.stop)
        except TypeError:
            raise TypeError(
                'slice.stop must be int or None or have __index__ method: '
                '{}'.format(slc))

        if stop < 0:
            stop += dim

    if step > 0:
        start = 0 if start_none else max(0, min(dim, start))
        stop = dim if stop_none else max(start, min(dim, stop))
    else:
        start = dim - 1 if start_none else max(-1, min(dim - 1, start))
        stop = -1 if stop_none else max(-1, min(start, stop))

    return slice(start, stop, step)


cpdef tuple complete_slice_list(list slice_list, Py_ssize_t ndim):
    cdef Py_ssize_t i, n_newaxes, n_ellipses, ellipsis, n
    slice_list = list(slice_list)  # copy list
    # Expand ellipsis into empty slices
    ellipsis = -1
    n_newaxes = n_ellipses = 0
    # Count Newaxis and Ellipsis
    for i, s in enumerate(slice_list):
        if s is None:
            n_newaxes += 1
        elif s is Ellipsis:
            n_ellipses += 1
            ellipsis = i
    if n_ellipses > 1:
        raise ValueError('Only one Ellipsis is allowed in index')

    n = ndim - <Py_ssize_t>len(slice_list) + n_newaxes
    # complement shortage with None
    if n_ellipses == 1:
        slice_list[ellipsis:ellipsis + 1] = [slice(None)] * (n + 1)
    elif n > 0:
        slice_list += [slice(None)] * n
    return slice_list, n_newaxes


cpdef inline int _normalize_order(order, bint allow_k=True) except? 0:
    cdef int order_char = 0
    if type(order) not in (int, float, complex):
        order_char = b'C' if len(order) == 0 else ord(order[0])
    if allow_k and (order_char == b'K' or order_char == b'k'):
        order_char = b'K'
    elif order_char == b'A' or order_char == b'a':
        order_char = b'A'
    elif order_char == b'C' or order_char == b'c':
        order_char = b'C'
    elif order_char == b'F' or order_char == b'f':
        order_char = b'F'
    else:
        raise TypeError('order not understood')
    return order_char

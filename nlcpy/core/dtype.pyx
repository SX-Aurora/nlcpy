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


from libc.stdint cimport *

from nlcpy import veo
from nlcpy.core.dtype cimport ve_dtype

import numpy
cimport cython

all_type_chars = '?bhilqBHILQefdFD'
# for c in '?bhilqBHILQefdFD':
#    print('#', c, '...', np.dtype(c).name)
# ? ... bool
# b ... int8
# h ... int16
# i ... int32
# l ... int64
# q ... int64
# B ... uint8
# H ... uint16
# I ... uint32
# L ... uint64
# Q ... uint64
# e ... float16
# f ... float32
# d ... float64
# F ... complex64
# D ... complex128

cdef dict _dtype_dict = {}


cdef _nlcpy_not_supported_type_set = (
    numpy.dtype('float16'),
    numpy.dtype('float128'),
    numpy.dtype('int16'),
    numpy.dtype('int8'),
    numpy.dtype('uint16'),
    numpy.dtype('uint8'),
    numpy.dtype('object'),
)

cpdef _check_dtype_is_valid(dtype):
    if dtype in _nlcpy_not_supported_type_set:
        return False
    return True

cdef _init_dtype_dict():
    for i in (int, float, bool, complex, None):
        dtype = numpy.dtype(i)
        if dtype is numpy.dtype('bool'):
            _dtype_dict[i] = (dtype, numpy.dtype('i4').itemsize)
        else:
            _dtype_dict[i] = (dtype, dtype.itemsize)
    for i in all_type_chars:
        dtype = numpy.dtype(i)
        if dtype is numpy.dtype('bool'):
            item = (dtype, numpy.dtype('i4').itemsize)
        else:
            item = (dtype, dtype.itemsize)
        _dtype_dict[i] = item
        _dtype_dict[dtype.type] = item
    for i in {str(numpy.dtype(i)) for i in all_type_chars}:
        dtype = numpy.dtype(i)
        if dtype is numpy.dtype('bool'):
            _dtype_dict[i] = (dtype, numpy.dtype('i4').itemsize)
        else:
            _dtype_dict[i] = (dtype, dtype.itemsize)


_init_dtype_dict()

cdef DT_BOOL = numpy.dtype('bool')
cdef DT_I32 = numpy.dtype('i4')
cdef DT_I64 = numpy.dtype('i8')
cdef DT_U32 = numpy.dtype('u4')
cdef DT_U64 = numpy.dtype('u8')
cdef DT_F32 = numpy.dtype('f4')
cdef DT_F64 = numpy.dtype('f8')
cdef DT_C64 = numpy.dtype('c8')
cdef DT_C128 = numpy.dtype('c16')


@cython.profile(False)
cpdef get_dtype(t):
    if isinstance(t, numpy.dtype):  # Exact type check
        return t
    ret = _dtype_dict.get(t, None)
    if ret is None:
        return numpy.dtype(t)
    return ret[0]

cdef tuple _convert_dtype(t):
    if isinstance(t, numpy.dtype):
        if t is numpy.dtype('bool'):
            return t, numpy.dtype('i4').itemsize
        elif t.char == 'q':
            return numpy.dtype('l'), numpy.dtype('l').itemsize
        elif t.char == 'Q':
            return numpy.dtype('L'), numpy.dtype('L').itemsize
        else:
            return t, t.itemsize
    else:
        raise TypeError('unknown dtype was detected.')


cpdef tuple get_dtype_with_itemsize(t):
    if isinstance(t, numpy.dtype):  # Exact type check
        return _convert_dtype(t)
    ret = _dtype_dict.get(t, None)
    if ret is None:
        t = numpy.dtype(t)
        return _convert_dtype(t)
    return _convert_dtype(ret[0])


cpdef int get_dtype_number(numpy.dtype dtype):
    if dtype is numpy.dtype('bool'):
        return ve_dtype.ve_bool
    if dtype is numpy.dtype('int8'):
        return ve_dtype.ve_i8
    if dtype is numpy.dtype('int16'):
        return ve_dtype.ve_i16
    if dtype is numpy.dtype('int32'):
        return ve_dtype.ve_i32
    if dtype is numpy.dtype('int64'):
        return ve_dtype.ve_i64
    if dtype is numpy.dtype('uint8'):
        return ve_dtype.ve_u8
    if dtype is numpy.dtype('uint16'):
        return ve_dtype.ve_u16
    if dtype is numpy.dtype('uint32'):
        return ve_dtype.ve_u32
    if dtype is numpy.dtype('uint64'):
        return ve_dtype.ve_u64
    if dtype is numpy.dtype('float16'):
        # return ve_dtype.ve_f16
        pass
    if dtype is numpy.dtype('float32'):
        return ve_dtype.ve_f32
    if dtype is numpy.dtype('float64'):
        return ve_dtype.ve_f64
    if dtype is numpy.dtype('complex64'):
        return ve_dtype.ve_c64
    if dtype is numpy.dtype('complex128'):
        return ve_dtype.ve_c128

    raise ValueError('detected unknown dtype %s' % dtype)


cpdef numpy.dtype promote_dtype_to_supported(numpy.dtype dtype):
    if dtype in (numpy.dtype('i1'), numpy.dtype('i2')):
        return numpy.dtype('i4')
    elif dtype in (numpy.dtype('u1'), numpy.dtype('u2')):
        return numpy.dtype('u4')
    elif dtype == numpy.dtype('f2'):
        return numpy.dtype('f4')
    else:
        return dtype

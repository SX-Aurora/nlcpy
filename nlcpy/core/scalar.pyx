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
from cpython cimport mem
from libc.stdint cimport int8_t
from libc.stdint cimport int16_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t

import numpy

import nlcpy
from nlcpy.core cimport dtype as _dtype
from nlcpy.core import dtype as _dtype_module
from nlcpy.core cimport internal


cdef union Scalar:
    bint bool_
    int8_t int8_
    int16_t int16_
    int32_t int32_
    int64_t int64_
    uint8_t uint8_
    uint16_t uint16_
    uint32_t uint32_
    uint64_t uint64_
    float float32_
    double float64_


cdef dict _typenames_base = {
    numpy.dtype('float64'): 'double',
    numpy.dtype('float32'): 'float',
    numpy.dtype('float16'): 'float16',
    numpy.dtype('complex128'): 'complex<double>',
    numpy.dtype('complex64'): 'complex<float>',
    numpy.dtype('int64'): 'long long',
    numpy.dtype('int32'): 'int',
    numpy.dtype('int16'): 'short',
    numpy.dtype('int8'): 'signed char',
    numpy.dtype('uint64'): 'unsigned long long',
    numpy.dtype('uint32'): 'unsigned int',
    numpy.dtype('uint16'): 'unsigned short',
    numpy.dtype('uint8'): 'unsigned char',
    numpy.dtype('bool'): 'bool',
}


cpdef str get_typename(dtype):
    if dtype is None:
        raise ValueError('dtype is None')
    if dtype not in _typenames:
        dtype = _dtype.get_dtype(dtype).type
    return _typenames[dtype]


cdef dict _typenames = {}
cdef dict _dtype_kind_size_dict = {}


cdef _setup_type_dict():
    cdef char k
    for i in _dtype_module.all_type_chars:
        d = numpy.dtype(i)
        t = d.type
        _typenames[t] = _typenames_base[d]
        k = ord(d.kind)
        _dtype_kind_size_dict[t] = (k, d.itemsize)


_setup_type_dict()


cdef set _python_scalar_type_set = set(
    (int, float, bool, complex))
cdef set _numpy_scalar_type_set = set(_typenames.keys())


_int_iinfo = numpy.iinfo(int)
cdef _int_min = _int_iinfo.min
cdef _int_max = _int_iinfo.max
cdef _int_type = _int_iinfo.dtype.type
cdef bint _use_int32 = _int_type != numpy.int64
del _int_iinfo


cpdef _python_scalar_to_numpy_scalar(x):
    typ = type(x)
    if typ is bool:
        numpy_type = numpy.bool_
    elif typ is float:
        numpy_type = numpy.float_
    elif typ is complex:
        numpy_type = numpy.complex_
    else:
        if 0x8000000000000000 <= x:
            numpy_type = numpy.uint64
        elif _use_int32 and (x < _int_min or _int_max < x):
            numpy_type = numpy.int64
        else:
            # Generally `_int_type` is `numpy.int64`.
            # On Windows, it is `numpy.int32`.
            numpy_type = _int_type
    return numpy_type(x)


cpdef convert_scalar(x):
    typ = type(x)
    if typ in _python_scalar_type_set:
        return _python_scalar_to_numpy_scalar(x)
    elif typ in _numpy_scalar_type_set:
        return x
    elif typ is numpy.ndarray:
        return nlcpy.asarray(x)
    return None


cdef dict _mst_unsigned_to_signed = {
    i: (numpy.iinfo(j).max, (i, j))
    for i, j in [(numpy.dtype(i).type, numpy.dtype(i.lower()).type)
                 for i in "BHILQ"]}
cdef _numpy_min_scalar_type = numpy.min_scalar_type

cdef _min_scalar_type(x):
    # A non-negative integer may have two locally minimum scalar
    # types: signed/unsigned integer.
    # Return both for can_cast, while numpy.min_scalar_type only returns
    # the unsigned type.
    t = _numpy_min_scalar_type(x)
    dt = t.type
    if t.kind == 'u':
        m, dt2 = <tuple>_mst_unsigned_to_signed[dt]
        if x <= m:
            return dt2
    return dt

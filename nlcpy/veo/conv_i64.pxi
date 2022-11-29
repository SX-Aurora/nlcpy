#
# * The source code in this file is based on the soure code of PyVEO.
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
from libc.stdint cimport *


cdef union U64:
    uint64_t u64
    int64_t i64
    float f32[2]
    double d64


cdef class ConvToI64(object):
    @staticmethod
    def from_char(x):
        return <int64_t><char>x

    @staticmethod
    def from_uchar(x):
        return <int64_t><unsigned char>x

    @staticmethod
    def from_short(x):
        return <int64_t><short>x

    @staticmethod
    def from_ushort(x):
        return <int64_t><unsigned short>x

    @staticmethod
    def from_int(x):
        return <int64_t><int>x

    @staticmethod
    def from_uint(x):
        return <int64_t><unsigned int>x

    @staticmethod
    def from_long(x):
        return <long>x

    @staticmethod
    def from_ulong(x):
        cdef U64 u
        u.u64 = <unsigned long>x
        return u.i64

    @staticmethod
    def from_addr(addr):
        cdef U64 u
        u.u64 = <unsigned long>addr
        return u.i64

    @staticmethod
    def from_float(x):
        cdef U64 u
        u.f32[1] = <float>x
        u.f32[0] = <float>0
        return u.i64

    @staticmethod
    def from_double(x):
        cdef U64 u
        u.d64 = <double>x
        return u.i64

    @staticmethod
    def from_void(x):
        return x


cdef class ConvFromI64(object):
    @staticmethod
    cdef char to_char(int64_t x):
        return <char>x
        # return <char>(x & 0xff)

    @staticmethod
    cdef unsigned char to_uchar(int64_t x):
        return <unsigned char>x
        # return <unsigned char>(x & 0xff)

    @staticmethod
    cdef int16_t to_short(int64_t x):
        return <int16_t>x
        # return <short>(x & 0xffff)

    @staticmethod
    cdef uint16_t to_ushort(int64_t x):
        return <uint16_t>x
        # return <unsigned short>(x & 0xffff)

    @staticmethod
    cdef int32_t to_int(int64_t x):
        return <int32_t>x
        # return <int>(x & 0xffffffff)

    @staticmethod
    cdef uint32_t to_uint(int64_t x):
        return <uint32_t>x
        # return <unsigned int><int64_t>(x & 0xffffffff)

    @staticmethod
    cdef int64_t to_long(int64_t x):
        return x

    @staticmethod
    cdef uint64_t to_ulong(int64_t x):
        cdef U64 u
        u.i64 = x
        return u.u64

    @staticmethod
    cdef float to_float(int64_t x):
        cdef U64 u
        u.i64 = x
        return u.f32[1]

    @staticmethod
    cdef double to_double(int64_t x):
        cdef U64 u
        u.i64 = x
        return u.d64

    @staticmethod
    cdef to_void(int64_t x):
        return None


cdef conv_to_i64_func(proc, t):
    if t == "char":
        return ConvToI64.from_char
    elif t == "short":
        return ConvToI64.from_short
    elif t == "int":
        return ConvToI64.from_int
    elif t == "int32_t":
        return ConvToI64.from_int
    elif t == "long":
        return ConvToI64.from_long
    elif t == "int64_t":
        return ConvToI64.from_long
    elif t == "unsigned char":
        return ConvToI64.from_uchar
    elif t == "unsigned short":
        return ConvToI64.from_ushort
    elif t == "unsigned int":
        return ConvToI64.from_uint
    elif t == "uint32_t":
        return ConvToI64.from_uint
    elif t == "unsigned long":
        return ConvToI64.from_ulong
    elif t == "uint64_t":
        return ConvToI64.from_ulong
    elif t == "float":
        return ConvToI64.from_float
    elif t == "double":
        return ConvToI64.from_double
    elif t == "void":
        return ConvToI64.from_void
    elif type(t) is str and t.endswith("*"):
        return ConvToI64.from_addr
    elif type(t) is bytes and t.endswith(b"*"):
        return ConvToI64.from_addr
    else:
        raise TypeError("Don't know how to convert '%s' to I64" % t)

cdef conv_from_i64_func(proc, t):
    if t == "char":
        return ConvFromI64.to_char
    elif t == "short":
        return ConvFromI64.to_short
    elif t == "int":
        return ConvFromI64.to_int
    elif t == "int32_t":
        return ConvFromI64.to_int
    elif t == "long":
        return ConvFromI64.to_long
    elif t == "int64_t":
        return ConvFromI64.to_long
    elif t == "unsigned char":
        return ConvFromI64.to_uchar
    elif t == "unsigned short":
        return ConvFromI64.to_ushort
    elif t == "unsigned int":
        return ConvFromI64.to_uint
    elif t == "uint32_t":
        return ConvFromI64.to_uint
    elif t == "unsigned long":
        return ConvFromI64.to_ulong
    elif t == "uint64_t":
        return ConvFromI64.to_ulong
    elif t == "float":
        return ConvFromI64.to_float
    elif t == "double":
        return ConvFromI64.to_double
    elif t == "void":
        return ConvFromI64.to_void
    elif type(t) is str and t.endswith("*"):
        return proc.i64_to_addr
    elif type(t) is bytes and t.endswith(b"*"):
        return proc.i64_to_addr
    else:
        raise TypeError("Don't know how to convert from I64 to '%s'" % t)

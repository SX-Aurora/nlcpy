/*
#
# * The source code in this file is developed independently by NEC Corporation.
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
*/
#ifndef VE_ARRAY_H_INCLUDED
#define VE_ARRAY_H_INCLUDED

#define NLCPY_MAXNDIM 27

#include <stdint.h>

typedef struct ve_array_tag {
    uint64_t ve_adr;
    uint64_t ndim;
    uint64_t size;
    uint64_t shape[NLCPY_MAXNDIM];
    uint64_t strides[NLCPY_MAXNDIM];
    uint64_t dtype;  // equivalent to numpy.dtype.num
    uint64_t itemsize;
    uint64_t is_c_contiguous;
    uint64_t is_f_contiguous;
    union {
        int32_t          bint;
        int32_t          i4;
        int64_t          i8;
        uint32_t         u4;
        uint64_t         u8;
        float            f4;
        double           f8;
        float  _Complex  c8;
        double _Complex  c16;
    } scalar;
} ve_array;

#define SIZEOF_VE_ARRAY (int)sizeof(ve_array)
#define N_VE_ARRAY_ELEMENTS (int)sizeof(ve_array) / sizeof(uint64_t)

#define VE_ADR_OFFSET 0
#define NDIM_OFFSET VE_ADR_OFFSET + 1
#define SIZE_OFFSET NDIM_OFFSET + 1
#define SHAPE_OFFSET SIZE_OFFSET + 1
#define STRIDES_OFFSET SHAPE_OFFSET + NLCPY_MAXNDIM
#define DTYPE_OFFSET STRIDES_OFFSET + NLCPY_MAXNDIM
#define ITEMSIZE_OFFSET DTYPE_OFFSET + 1
#define C_CONTIGUOUS_OFFSET ITEMSIZE_OFFSET + 1
#define F_CONTIGUOUS_OFFSET C_CONTIGUOUS_OFFSET + 1
#define SCALAR_OFFSET F_CONTIGUOUS_OFFSET + 1

#endif /* VE_ARRAY_H_INCLUDED */

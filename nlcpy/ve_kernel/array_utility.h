/*
#
# * The source code in this file is developed independently by NEC Corporation.
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
*/
#ifndef ARRAY_UTILITY_H_INCLUDED
#define ARRAY_UTILITY_H_INCLUDED

#include "ve_array.h"

#ifdef __cplusplus
extern "C" {
#endif


uint64_t nlcpy__get_scalar(ve_array *val);
uint64_t nlcpy__get_ptr(ve_array *a);
void nlcpy__reset_coords(int64_t *coords, int64_t size);
uint64_t nlcpy__array_next(ve_array *a, uint64_t curr_adr, int64_t *coords);
uint64_t nlcpy__array_rnext(ve_array *a, uint64_t curr_adr, int64_t *coords);
uint64_t nlcpy__array_reduce_next(ve_array *a, uint64_t curr_adr, int64_t *coords, ve_array *b);
uint64_t nlcpy__array_reduce_rnext(ve_array *a, uint64_t curr_adr, int64_t *coords, ve_array *b);

void nlcpy__rearrange_axis(ve_array *a, int64_t *idx);
void nlcpy__exchange_shape_and_strides(ve_array *a);
void nlcpy__argnsort(ve_array *a, int64_t *idx, int64_t n);

#define MINVL_THRESHOLD 64
#define NG_STRIDE       0x80   // 128 Byte

#ifdef __cplusplus
}
#endif


#endif /* ARRAY_UTILITY_H_INCLUDED */

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
#ifndef ARRAY_UTILITY_H_INCLUDED
#define ARRAY_UTILITY_H_INCLUDED

#include <stdint.h>
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

typedef int32_t Bint;

typedef union bf_f64_tag{
    int64_t bf[1];
    double x;
} bf_f64_t;

typedef union bf_f32_tag{
    int32_t bf[1];
    float x;
} bf_f32_t;

inline Bint f64_to_Bint(double x) {
    bf_f64_t y;
    y.x = x;
    return (y.bf[0]&0x7fffffffffffffff) ? (Bint)1 : (Bint)0;
}

inline Bint f32_to_Bint(float x) {
    bf_f32_t y;
    y.x = x;
    return (y.bf[0]&0x7fffffff) ? (Bint)1 : (Bint)0;
}

inline Bint u64_to_Bint(uint64_t x) {
    return (x!=(uint64_t)0) ? (Bint)1 : (Bint)0;
}

inline Bint u32_to_Bint(uint32_t x) {
    return (x!=(uint32_t)0) ? (Bint)1 : (Bint)0;
}

inline Bint i64_to_Bint(int64_t x) {
    return (x!=(int64_t)0) ? (Bint)1 : (Bint)0;
}

inline Bint i32_to_Bint(int32_t x) {
    return (x!=(int32_t)0) ? (Bint)1 : (Bint)0;
}

// isinf
inline Bint isinf_f64(double x) {
    const int64_t E  = 0x7ff0000000000000;
    const int64_t M  = 0x000fffffffffffff;
    bf_f64_t y;
    y.x = x;
    const int64_t Isinf =  ((y.bf[0] & E) == E) && ((y.bf[0] & M) == (int64_t)0);
    return ( Isinf ) ? (Bint)1 : (Bint)0;
}

inline Bint isinf_f32(float x) {
    const int32_t E  = 0x7f800000;
    const int32_t M  = 0x007fffff;
    bf_f32_t y;
    y.x = x;
    const int32_t Isinf =  ((y.bf[0] & E) == E) && ((y.bf[0] & M) == (int32_t)0);
    return ( Isinf ) ? (Bint)1 : (Bint)0;
}

// isnan
inline Bint isnan_f64(double x) {
    const int64_t E  = 0x7ff0000000000000;
    const int64_t M  = 0x000fffffffffffff;
    bf_f64_t y;
    y.x = x;
    const int64_t Isnan =  ((y.bf[0] & E) == E) && ((y.bf[0] & M) != (int64_t)0);
    return ( Isnan ) ? (Bint)1 : (Bint)0;
}

inline Bint isnan_f32(float x) {
    const int32_t E  = 0x7f800000;
    const int32_t M  = 0x007fffff;
    bf_f32_t y;
    y.x = x;
    const int32_t Isnan =  ((y.bf[0] & E) == E) && ((y.bf[0] & M) != (int32_t)0);
    return ( Isnan ) ? (Bint)1 : (Bint)0;
}

float  nlcpy_g_nanf;
double nlcpy_g_nan;

#ifdef __cplusplus
}
#endif

#endif /* ARRAY_UTILITY_H_INCLUDED */

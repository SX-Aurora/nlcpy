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
@#include <stdio.h>
@#include <stdint.h>
@#include <stdbool.h>
@#include <stdlib.h>
@#include <limits.h>
@#include <alloca.h>
@#include <assert.h>
@#include <complex.h>

@#include "nlcpy.h"

include(macros.m4)dnl
#define_switch (x->dtype)

/****************************
 *
 *       @OPERATOR_NAME@
 *
 * **************************/

define(<--@macro_unary_operator@-->,<--@
uint64_t FILENAME_$1(ve_array *x, ve_array *y, int32_t *psw)
{
#begin_switch
    $2 *py = ($2 *)nlcpy__get_ptr(y);
    if (py == NULL) return NLCPY_ERROR_MEMORY;
    @TYPE1@ *px = (@TYPE1@ *)nlcpy__get_ptr(x);
    if (px == NULL) return NLCPY_ERROR_MEMORY;

/////////
// 0-d //
/////////
    if (y->ndim == 0) {
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
        @UNARY_OPERATOR@(*px,*py,$1)
} /* omp single */

////////////////
// contiguous //
////////////////
    } else if ((x->is_c_contiguous & y->is_c_contiguous) ||
               (x->is_f_contiguous & y->is_f_contiguous) )
    {
        int64_t i;
@#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
@#else
        const int nt = 1;
        const int it = 0;
@#endif /* _OPENMP */
        const int64_t is = y->size * it / nt;
        const int64_t ie = y->size * (it + 1) / nt;
        if (x->size == 1){
            @TYPE1@ px_s = px[0];
            for (i = is; i < ie; i++) {
                @UNARY_OPERATOR@(px_s,py[i],$1)
            }
        } else {
            for (i = is; i < ie; i++) {
                @UNARY_OPERATOR@(px[i],py[i],$1)
            }
        }

/////////
// 1-d //
/////////
    } else if (y->ndim == 1) {
        int64_t i;
        const int64_t n_inner = y->ndim - 1;
        const uint64_t ix0 = x->strides[n_inner] / x->itemsize;
        const uint64_t iy0 = y->strides[n_inner] / y->itemsize;
@#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
@#else
        const int nt = 1;
        const int it = 0;
@#endif /* _OPENMP */
        const int64_t is = y->size * it / nt;
        const int64_t ie = y->size * (it + 1) / nt;
        if (x->size == 1){
            @TYPE1@ px_s = px[0];
            for (i = is; i < ie; i++) {
                @UNARY_OPERATOR@(px_s,py[i*iy0],$1)
            }
        } else {
            for (i = is; i < ie; i++) {
                @UNARY_OPERATOR@(px[i*ix0],py[i*iy0],$1)
            }
        }

/////////
// N-d //
/////////
    } else if (y->ndim > 1 && y->ndim <= NLCPY_MAXNDIM){
@#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
@#else
        const int nt = 1;
        const int it = 0;
@#endif /* _OPENMP */
        int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * y->ndim);
        nlcpy__rearrange_axis(y, idx);
        int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t)*y->ndim);
        int64_t i, j, k;
        int64_t n_inner = y->ndim - 1;
        int64_t n_outer = 0;
        const int64_t n_inner2 = idx[n_inner];
        const int64_t n_outer2 = idx[n_outer];
        nlcpy__reset_coords(cnt_y, y->ndim);

        uint64_t ix = 0;
        uint64_t iy = 0;
        uint64_t ix0 = x->strides[n_inner2] / x->itemsize;
        uint64_t iy0 = y->strides[n_inner2] / y->itemsize;
        const int64_t len = y->shape[n_outer2];
        const int64_t cnt_s = len * it / nt;
        const int64_t cnt_e = len * (it + 1) / nt;
        for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
            ix = cnt * x->strides[n_outer2] / x->itemsize;
            iy = cnt * y->strides[n_outer2] / y->itemsize;
            for (;;) {
                // most inner loop for vectorize
                if (x->size == 1){
                    @TYPE1@ px_s = px[0];
                    for (i = 0; i < y->shape[n_inner2]; i++) {
                        @UNARY_OPERATOR@(px_s,py[i*iy0+iy],$1)
                    }
                } else {
                    for (i = 0; i < y->shape[n_inner2]; i++) {
                        @UNARY_OPERATOR@(px[i*ix0+ix],py[i*iy0+iy],$1)
                    }
                }
                // set next index
                for (k = n_inner-1; k >= 1; k--) {
                    int64_t kk = idx[k];
                    if (++cnt_y[kk] < y->shape[kk]) {
                        ix += x->strides[kk] / x->itemsize;
                        iy += y->strides[kk] / y->itemsize;
                        break;
                    }
                    cnt_y[kk] = 0;
                    ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                    iy -= (y->strides[kk] / y->itemsize) * (y->shape[kk] - 1);
                }
                if (k < 1) break;
            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
#end_switch
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


@-->)dnl
#if defined(DTAG_OUT_i32)
macro_unary_operator(i32,int32_t)dnl
#endif
#if defined(DTAG_OUT_i64)
macro_unary_operator(i64,int64_t)dnl
#endif
#if defined(DTAG_OUT_u32)
macro_unary_operator(u32,uint32_t)dnl
#endif
#if defined(DTAG_OUT_u64)
macro_unary_operator(u64,uint64_t)dnl
#endif
#if defined(DTAG_OUT_f32)
macro_unary_operator(f32,float)dnl
#endif
#if defined(DTAG_OUT_f64)
macro_unary_operator(f64,double)dnl
#endif
#if defined(DTAG_OUT_c64)
macro_unary_operator(c64,float _Complex)dnl
#endif
#if defined(DTAG_OUT_c128)
macro_unary_operator(c128,double _Complex)dnl
#endif
#if defined(DTAG_OUT_bool)
macro_unary_operator(bool,int32_t)dnl
#endif


//uint64_t FILENAME(ve_array *x, ve_array *y, int32_t *psw)
uint64_t FILENAME(ve_arguments *args, int32_t *psw)
{
    ve_array *x = &(args->copy.x);
    ve_array *y = &(args->copy.y);
    if (x->size == 0 || y->size == 0) return (uint64_t)NLCPY_ERROR_OK;
    uint64_t err = NLCPY_ERROR_OK;
    switch (y->dtype) {
#if defined(DTAG_OUT_i32)
    case ve_i32:  err = FILENAME_i32 (x, y, psw); break;
#endif
#if defined(DTAG_OUT_i64)
    case ve_i64:  err = FILENAME_i64 (x, y, psw); break;
#endif
#if defined(DTAG_OUT_u32)
    case ve_u32:  err = FILENAME_u32 (x, y, psw); break;
#endif
#if defined(DTAG_OUT_u64)
    case ve_u64:  err = FILENAME_u64 (x, y, psw); break;
#endif
#if defined(DTAG_OUT_f32)
    case ve_f32:  err = FILENAME_f32 (x, y, psw); break;
#endif
#if defined(DTAG_OUT_f64)
    case ve_f64:  err = FILENAME_f64 (x, y, psw); break;
#endif
#if defined(DTAG_OUT_c64)
    case ve_c64:  err = FILENAME_c64 (x, y, psw); break;
#endif
#if defined(DTAG_OUT_c128)
    case ve_c128: err = FILENAME_c128(x, y, psw); break;
#endif
#if defined(DTAG_OUT_bool)
    case ve_bool: err = FILENAME_bool(x, y, psw); break;
#endif
    default: err = NLCPY_ERROR_DTYPE;
    }

    return (uint64_t)err;
}

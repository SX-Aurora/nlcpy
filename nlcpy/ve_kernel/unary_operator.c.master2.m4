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

@#include "nlcpy.h"

include(macros.m4)dnl

/****************************
 *
 *       @OPERATOR_NAME@
 *
 * **************************/

define(<--@macro_unary_operator@-->,<--@
uint64_t FILENAME_@TYPE1_DTAG@_$1(ve_array *x, ve_array *y, int32_t where_flag, ve_array *where, int32_t *psw)
{
    @TYPE1@ *px = (@TYPE1@ *)nlcpy__get_ptr(x);
    if (px == NULL) return NLCPY_ERROR_MEMORY;
    $2 *py = ($2 *)nlcpy__get_ptr(y);
    if (py == NULL) return NLCPY_ERROR_MEMORY;
    ve_array *w = NULL;
    Bint *pw = NULL;
    if (where_flag) {
        w = where;
        pw = (Bint *)w->ve_adr;
    }
 
@#ifdef _OPENMP
    const int nt = omp_get_num_threads();
    const int it = omp_get_thread_num();
@#else
    const int nt = 1;
    const int it = 0;
@#endif /* _OPENMP */

   
/////////
// 0-d //
/////////
    if (y->ndim == 0) {
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
        if (!where_flag) {
            @UNARY_OPERATOR@(*px,*py,$1)
        } else {
            if (*pw) {
                @UNARY_OPERATOR@(*px,*py,$1)
            }
        }
} /* omp single */ 
   
/////////
// 1-d //
/////////
    } else if (y->ndim == 1) {
        int64_t i;
        const int64_t n_inner = y->ndim - 1;
        const uint64_t ix0 = x->strides[n_inner] / x->itemsize;
        const uint64_t iy0 = y->strides[n_inner] / y->itemsize;
        const int64_t len = y->size;
        const int64_t cnt_s = len * it / nt;
        const int64_t cnt_e = len * (it + 1) / nt;
        if (!where_flag) {
@#pragma _NEC ivdep
            for (i = cnt_s; i < cnt_e; i++) {
                @UNARY_OPERATOR@(px[i*ix0],py[i*iy0],$1)
            }
        } else {
            const uint64_t iw0 = w->strides[n_inner]/w->itemsize;
@#pragma _NEC ivdep
            for (i = cnt_s; i < cnt_e; i++) {
                if (pw[i*iw0]) {
                    @UNARY_OPERATOR@(px[i*ix0],py[i*iy0],$1)
                }
            }
        }

/////////
// N-d //
/////////
    } else if (y->ndim > 1 && y->ndim <= NLCPY_MAXNDIM){
        int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * y->ndim);
        nlcpy__rearrange_axis(y, idx);
        int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
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
        if (!where_flag) {
            for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
                ix = cnt * x->strides[n_outer2] / x->itemsize;
                iy = cnt * y->strides[n_outer2] / y->itemsize;
                for (;;) {
                    // most inner loop for vectorize
@#pragma _NEC ivdep
                    for (i = 0; i < y->shape[n_inner2]; i++) {
                        @UNARY_OPERATOR@(px[i*ix0+ix],py[i*iy0+iy],$1)
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
            uint64_t iw = 0;
            uint64_t iw0 = where->strides[n_inner2] / where->itemsize;
            for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
                ix = cnt * x->strides[n_outer2] / x->itemsize;
                iy = cnt * y->strides[n_outer2] / y->itemsize;
                iw = cnt * w->strides[n_outer2] / w->itemsize;
                for (;;) {
                    // most inner loop for vectorize
@#pragma _NEC ivdep
                    for (i = 0; i < y->shape[n_inner2]; i++) {
                        if (pw[i*iw0+iw]) {
                            @UNARY_OPERATOR@(px[i*ix0+ix],py[i*iy0+iy],$1)
                        }
                    }
                    // set next index
                    for (k = n_inner-1; k >= 1; k--) {
                        int64_t kk = idx[k];
                        if (++cnt_y[kk] < y->shape[kk]) {
                            ix += x->strides[kk] / x->itemsize;
                            iy += y->strides[kk] / y->itemsize;
                            iw += w->strides[kk] / w->itemsize;
                            break;
                        }
                        cnt_y[kk] = 0;
                        ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                        iy -= (y->strides[kk] / y->itemsize) * (y->shape[kk] - 1);
                        iw -= (w->strides[kk] / w->itemsize) * (w->shape[kk] - 1);
                    }
                    if (k < 1) break;
                }
            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


@-->)dnl
#if defined(DTYPE_i32)
macro_unary_operator(i32,int32_t)dnl
#endif
#if defined(DTYPE_i64)
macro_unary_operator(i64,int64_t)dnl
#endif
#if defined(DTYPE_u32)
macro_unary_operator(u32,uint32_t)dnl
#endif
#if defined(DTYPE_u64)
macro_unary_operator(u64,uint64_t)dnl
#endif
#if defined(DTYPE_f32)
macro_unary_operator(f32,float)dnl
#endif
#if defined(DTYPE_f64)
macro_unary_operator(f64,double)dnl
#endif
#if defined(DTYPE_c64)
macro_unary_operator(c64,float _Complex)dnl
#endif
#if defined(DTYPE_c128)
macro_unary_operator(c128,double _Complex)dnl
#endif
#if defined(DTYPE_bool)
macro_unary_operator(bool,int32_t)dnl
#endif

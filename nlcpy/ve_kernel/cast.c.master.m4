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

@#include "nlcpy.h"

include(macros.m4)dnl
#define_switch (x->dtype)

/****************************
 *
 *       @OPERATOR_NAME@
 *
 * **************************/

define(<--@macro_cast_operator@-->,<--@
uint64_t FILENAME_$1(ve_array *x, ve_array *y, int32_t where_flag, ve_array *where, int32_t *psw)
{
#begin_switch
    @TYPE1@ *px = (@TYPE1@ *)x->ve_adr;
    $2 *py = ($2 *)y->ve_adr;
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
    if (x->ndim == 0) {
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
        if (!where_flag) {
            *py = @CAST_OPERATOR@(*px,@DTAG1@,$1)
        } else {
            if (*pw) {
                *py = @CAST_OPERATOR@(*px,@DTAG1@,$1)
            }
        }
} /* omp single */

////////////////
// contiguous //
////////////////
    } else if (!where_flag &&
               ( (x->is_c_contiguous & y->is_c_contiguous) ||
                 (x->is_f_contiguous & y->is_f_contiguous) ))
    {
        int64_t i0;
        const int64_t lenm = y->size;
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;
        if (x->size == 1) {
            @TYPE1@ px_s = px[0];
ifelse(<--@$1@-->,<--@bool@-->,<--@dnl
// TODO: If you use ncc 3.0.1 or later, replace "novector" to "ivdep".
#pragma _NEC novector
@-->,<--@dnl
#pragma _NEC ivdep
@-->)
            for (i0 = cntm_s; i0 < cntm_e; i0++) {
                py[i0] = @CAST_OPERATOR@(px_s,@DTAG1@,$1)
            }
        } else {
ifelse(<--@$1@-->,<--@bool@-->,<--@dnl
// TODO: If you use ncc 3.0.1 or later, replace "novector" to "ivdep".
#pragma _NEC novector
@-->,<--@dnl
#pragma _NEC ivdep
@-->)
            for (i0 = cntm_s; i0 < cntm_e; i0++) {
                py[i0] = @CAST_OPERATOR@(px[i0],@DTAG1@,$1)
            }
        }

/////////
// 1-d //
/////////
    } else if (x->ndim == 1) {
        int64_t i0;
        int64_t n_inner = 0;
        uint64_t ix0 = x->strides[n_inner] / x->itemsize;
        uint64_t iy0 = y->strides[n_inner] / y->itemsize;
        const int64_t lenm = y->size;
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;
        if (!where_flag) {
ifelse(<--@$1@-->,<--@bool@-->,<--@dnl
// TODO: If you use ncc 3.0.1 or later, replace "novector" to "ivdep".
#pragma _NEC novector
@-->,<--@dnl
#pragma _NEC ivdep
@-->)
            for (i0 = cntm_s; i0 < cntm_e; i0++) {
                py[i0*iy0] = @CAST_OPERATOR@(px[i0*ix0],@DTAG1@,$1)
            }
        } else {
            Bint *pw = (Bint *)w->ve_adr;
            const uint64_t iw0 = w->strides[0]/w->itemsize;
ifelse(<--@$1@-->,<--@bool@-->,<--@dnl
// TODO: If you use ncc 3.0.1 or later, replace "novector" to "ivdep".
#pragma _NEC novector
@-->,<--@dnl
#pragma _NEC ivdep
@-->)
            for (i0 = cntm_s; i0 < cntm_e; i0++) {
                if (pw[i0*iw0]) {
                    py[i0*iy0] = @CAST_OPERATOR@(px[i0*ix0],@DTAG1@,$1)
                }
            }
        }

/////////
// N-d //
/////////
    } else if (x->ndim > 1 && x->ndim <= NLCPY_MAXNDIM){
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
        const int64_t lenm = y->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;
        if (!where_flag) {
            for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                ix = cntm * x->strides[n_outer2] / x->itemsize;
                iy = cntm * y->strides[n_outer2] / y->itemsize;
                for (;;) {
                    // most inner loop for vectorize
ifelse(<--@$1@-->,<--@bool@-->,<--@dnl
// TODO: If you use ncc 3.0.1 or later, replace "novector" to "ivdep".
#pragma _NEC novector
@-->,<--@dnl
#pragma _NEC ivdep
@-->)
                    for (i = 0; i < x->shape[n_inner2]; i++) {
                        py[i*iy0+iy] = @CAST_OPERATOR@(px[i*ix0+ix],@DTAG1@,$1)
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
            for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                ix = cntm * x->strides[n_outer2] / x->itemsize;
                iy = cntm * y->strides[n_outer2] / y->itemsize;
                iw = cntm * w->strides[n_outer2] / w->itemsize;
                for (;;) {
                    // most inner loop for vectorize
ifelse(<--@$1@-->,<--@bool@-->,<--@dnl
// TODO: If you use ncc 3.0.1 or later, replace "novector" to "ivdep".
#pragma _NEC novector
@-->,<--@dnl
#pragma _NEC ivdep
@-->)
                    for (i = 0; i < x->shape[n_inner2]; i++) {
                        if (pw[i*iw0+iw]) {
                            py[i*iy0+iy] = @CAST_OPERATOR@(px[i*ix0+ix],@DTAG1@,$1)
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
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
#end_switch
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


@-->)dnl
#if defined(DTAG_OUT_i32)
macro_cast_operator(i32,int32_t)dnl
#endif
#if defined(DTAG_OUT_i64)
macro_cast_operator(i64,int64_t)dnl
#endif
#if defined(DTAG_OUT_u32)
macro_cast_operator(u32,uint32_t)dnl
#endif
#if defined(DTAG_OUT_u64)
macro_cast_operator(u64,uint64_t)dnl
#endif
#if defined(DTAG_OUT_f32)
macro_cast_operator(f32,float)dnl
#endif
#if defined(DTAG_OUT_f64)
macro_cast_operator(f64,double)dnl
#endif
#if defined(DTAG_OUT_c64)
macro_cast_operator(c64,float _Complex)dnl
#endif
#if defined(DTAG_OUT_c128)
macro_cast_operator(c128,double _Complex)dnl
#endif
#if defined(DTAG_OUT_bool)
macro_cast_operator(bool,int32_t)dnl
#endif

uint64_t FILENAME(ve_array *x, ve_array *y, int32_t where_flag, ve_array *where, int32_t *psw)
{
    uint64_t err = NLCPY_ERROR_OK;
    switch (y->dtype) {
#if defined(DTAG_OUT_i32)
    case ve_i32:  err = FILENAME_i32 (x, y, where_flag, where, psw); break;
#endif
#if defined(DTAG_OUT_i64)
    case ve_i64:  err = FILENAME_i64 (x, y, where_flag, where, psw); break;
#endif
#if defined(DTAG_OUT_u32)
    case ve_u32:  err = FILENAME_u32 (x, y, where_flag, where, psw); break;
#endif
#if defined(DTAG_OUT_u64)
    case ve_u64:  err = FILENAME_u64 (x, y, where_flag, where, psw); break;
#endif
#if defined(DTAG_OUT_f32)
    case ve_f32:  err = FILENAME_f32 (x, y, where_flag, where, psw); break;
#endif
#if defined(DTAG_OUT_f64)
    case ve_f64:  err = FILENAME_f64 (x, y, where_flag, where, psw); break;
#endif
#if defined(DTAG_OUT_c64)
    case ve_c64:  err = FILENAME_c64 (x, y, where_flag, where, psw); break;
#endif
#if defined(DTAG_OUT_c128)
    case ve_c128: err = FILENAME_c128(x, y, where_flag, where, psw); break;
#endif
#if defined(DTAG_OUT_bool)
    case ve_bool: err = FILENAME_bool(x, y, where_flag, where, psw); break;
#endif
    default: err = NLCPY_ERROR_DTYPE;
    }

    return (uint64_t)err;
}

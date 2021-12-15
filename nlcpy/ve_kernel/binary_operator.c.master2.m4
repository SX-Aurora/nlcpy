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
@#include <float.h>
@#include <math.h>
@#include <complex.h>

@#include "nlcpy.h"

include(macros.m4)dnl

/****************************
 *
 *       @OPERATOR_NAME@
 *
 * **************************/

define(<--@macro_binary_operator@-->,<--@
uint64_t FILENAME_@DTAG1@_@DTAG2@_$1(
    ve_array *x,
    @TYPE1@ *px,
    ve_array *y,
    @TYPE2@ *py,
    ve_array *z,
    $2 *pz,
    ve_array *w,
    Bint *pw,
    const int64_t cnt_s,
    const int64_t cnt_e,
    int64_t *cnt_z,
    int64_t *idx,
    const int64_t n_inner,
    const int64_t n_outer,
    const uint64_t ix0,
    const uint64_t iy0,
    const uint64_t iz0,
    const uint64_t iw0,
    int32_t *psw
){

/////////
// 0-d //
/////////
    if (z->ndim == 0) {
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
        if (w == NULL) {
            @BINARY_OPERATOR@(*px,*py,*pz,$1)
        } else {
            if (*pw) {
                @BINARY_OPERATOR@(*px,*py,*pz,$1)
            }
        }
} /* omp single */

////////////////
// contiguous //
////////////////
    } else if (w == NULL &&
               ( (x->is_c_contiguous & y->is_c_contiguous & z->is_c_contiguous) ||
                 (x->is_f_contiguous & y->is_f_contiguous & z->is_f_contiguous) ) ) {
        int64_t i;
        if (x->size == 1 && y->size != 1) {
            @TYPE1@ px_s = px[0];
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
            for (i = cnt_s; i < cnt_e; i++) {
                @BINARY_OPERATOR@(px_s,py[i],pz[i],$1)
            }
        } else if (x->size != 1 && y->size == 1) {
                @TYPE2@ py_s = py[0];
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
            for (i = cnt_s; i < cnt_e; i++) {
                @BINARY_OPERATOR@(px[i],py_s,pz[i],$1)
            }
        } else {
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
            for (i = cnt_s; i < cnt_e; i++) {
                @BINARY_OPERATOR@(px[i],py[i],pz[i],$1)
            }
        }

/////////
// 1-d //
/////////
    } else if (z->ndim == 1) {
        int64_t i;
        if (w == NULL) {
            if (x->size == 1 && y->size != 1) {
                @TYPE1@ px_s = px[0];
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                for (i = cnt_s; i < cnt_e; i++) {
                    @BINARY_OPERATOR@(px_s,py[i*iy0],pz[i*iz0],$1)
                }
            } else if (x->size != 1 && y->size == 1) {
                @TYPE2@ py_s = py[0];
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                for (i = cnt_s; i < cnt_e; i++) {
                    @BINARY_OPERATOR@(px[i*ix0],py_s,pz[i*iz0],$1)
                }
            } else {
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                for (i = cnt_s; i < cnt_e; i++) {
                    @BINARY_OPERATOR@(px[i*ix0],py[i*iy0],pz[i*iz0],$1)
                }
            }
        } else {
            if (x->size == 1 && y->size != 1) {
                @TYPE1@ px_s = px[0];
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                for (i = cnt_s; i < cnt_e; i++) {
                    if (pw[i*iw0]) {
                        @BINARY_OPERATOR@(px_s,py[i*iy0],pz[i*iz0],$1)
                    }
                }
            } else if (x->size != 1 && y->size == 1) {
                @TYPE2@ py_s = py[0];
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                for (i = cnt_s; i < cnt_e; i++) {
                    if (pw[i*iw0]) {
                        @BINARY_OPERATOR@(px[i*ix0],py_s,pz[i*iz0],$1)
                    }
                }
            } else {
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                for (i = cnt_s; i < cnt_e; i++) {
                    if (pw[i*iw0]) {
                        @BINARY_OPERATOR@(px[i*ix0],py[i*iy0],pz[i*iz0],$1)
                    }
                }
            }
        }

/////////
// N-d //
/////////
    } else {
        int64_t i, j, k;
        uint64_t ix = 0;
        uint64_t iy = 0;
        uint64_t iz = 0;
        if (w == NULL) {
            for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
                ix = cnt * x->strides[n_outer] / x->itemsize;
                iy = cnt * y->strides[n_outer] / y->itemsize;
                iz = cnt * z->strides[n_outer] / z->itemsize;
                for (;;) {
                    // most inner loop for vectorize
                    if (x->size == 1 && y->size != 1) {
                        @TYPE1@ px_s = px[0];
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                        for (i = 0; i < z->shape[n_inner]; i++) {
                            @BINARY_OPERATOR@(px_s,py[i*iy0+iy],pz[i*iz0+iz],$1)
                        }
                    } else if (x->size != 1 && y->size == 1) {
                        @TYPE2@ py_s = py[0];
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                        for (i = 0; i < z->shape[n_inner]; i++) {
                            @BINARY_OPERATOR@(px[i*ix0+ix],py_s,pz[i*iz0+iz],$1)
                        }
                    } else {
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                        for (i = 0; i < z->shape[n_inner]; i++) {
                            @BINARY_OPERATOR@(px[i*ix0+ix],py[i*iy0+iy],pz[i*iz0+iz],$1)
                        }
                    }
                    // set next index
                    for (k = z->ndim - 2; k >= 1; k--) {
                        int64_t kk = idx[k];
                        if (++cnt_z[kk] < z->shape[kk]) {
                            ix += x->strides[kk] / x->itemsize;
                            iy += y->strides[kk] / y->itemsize;
                            iz += z->strides[kk] / z->itemsize;
                            break;
                        }
                        cnt_z[kk] = 0;
                        ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                        iy -= (y->strides[kk] / y->itemsize) * (y->shape[kk] - 1);
                        iz -= (z->strides[kk] / z->itemsize) * (z->shape[kk] - 1);
                    }
                    if (k < 1) break;
                }
            }
        } else {
            uint64_t iw = 0;
            for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
                ix = cnt * x->strides[n_outer] / x->itemsize;
                iy = cnt * y->strides[n_outer] / y->itemsize;
                iz = cnt * z->strides[n_outer] / z->itemsize;
                iw = cnt * w->strides[n_outer] / w->itemsize;
                for (;;) {
                    // most inner loop for vectorize
                    if (x->size == 1 && y->size != 1) {
                        @TYPE1@ px_s = px[0];
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                        for (i = 0; i < z->shape[n_inner]; i++) {
                            if (pw[i*iw0+iw]) {
                                @BINARY_OPERATOR@(px_s,py[i*iy0+iy],pz[i*iz0+iz],$1)
                            }
                        }
                    } else if (x->size != 1 && y->size == 1) {
                        @TYPE2@ py_s = py[0];
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                        for (i = 0; i < z->shape[n_inner]; i++) {
                            if (pw[i*iw0+iw]) {
                                @BINARY_OPERATOR@(px[i*ix0+ix],py_s,pz[i*iz0+iz],$1)
                            }
                        }
                    } else {
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                        for (i = 0; i < z->shape[n_inner]; i++) {
                            if (pw[i*iw0+iw]) {
                                @BINARY_OPERATOR@(px[i*ix0+ix],py[i*iy0+iy],pz[i*iz0+iz],$1)
                            }
                        }
                    }
                    // set next index
                    for (k = z->ndim - 2; k >= 1; k--) {
                        int64_t kk = idx[k];
                        if (++cnt_z[kk] < z->shape[kk]) {
                            ix += x->strides[kk] / x->itemsize;
                            iy += y->strides[kk] / y->itemsize;
                            iz += z->strides[kk] / z->itemsize;
                            iw += w->strides[kk] / w->itemsize;
                            break;
                        }
                        cnt_z[kk] = 0;
                        ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                        iy -= (y->strides[kk] / y->itemsize) * (y->shape[kk] - 1);
                        iz -= (z->strides[kk] / z->itemsize) * (z->shape[kk] - 1);
                        iw -= (w->strides[kk] / w->itemsize) * (w->shape[kk] - 1);
                    }
                    if (k < 1) break;
                }
            }
        }
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


@-->)dnl
#if defined(DTAG_i32)
macro_binary_operator(i32,int32_t)dnl
#endif
#if defined(DTAG_i64)
macro_binary_operator(i64,int64_t)dnl
#endif
#if defined(DTAG_u32)
macro_binary_operator(u32,uint32_t)dnl
#endif
#if defined(DTAG_u64)
macro_binary_operator(u64,uint64_t)dnl
#endif
#if defined(DTAG_f32)
macro_binary_operator(f32,float)dnl
#endif
#if defined(DTAG_f64)
macro_binary_operator(f64,double)dnl
#endif
#if defined(DTAG_c64)
macro_binary_operator(c64,float _Complex)dnl
#endif
#if defined(DTAG_c128)
macro_binary_operator(c128,double _Complex)dnl
#endif
#if defined(DTAG_bool)
macro_binary_operator(bool,int32_t)dnl
#endif

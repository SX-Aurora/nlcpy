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

#ifndef NLCPY_REDUCE_GLOBAL_VARIABLE
#define NLCPY_REDUCE_GLOBAL_VARIABLE
Bint nlcpy__global_bool;
uint32_t nlcpy__global_u32;
uint64_t nlcpy__global_u64;
int32_t nlcpy__global_i32;
int64_t nlcpy__global_i64;
float nlcpy__global_f32;
double nlcpy__global_f64;
float _Complex nlcpy__global_c64;
double _Complex nlcpy__global_c128;
#endif

/****************************
 *
 *       REDUCE OPERATOR
 *
 * **************************/

define(<--@macro_reduce_operator@-->,<--@
uint64_t FILENAME_@DTAG1@_$1(ve_array *x, ve_array *y, int32_t axis, int32_t init_flag,
                     ve_array *initial, int32_t where_flag, ve_array *where, int32_t *psw)
{
    @TYPE1@ *px = (@TYPE1@ *)nlcpy__get_ptr(x);
    if (px == NULL) return NLCPY_ERROR_MEMORY;
    $2 *py = ($2 *)nlcpy__get_ptr(y);
    if (py == NULL) return NLCPY_ERROR_MEMORY;
    $2 *pi = ($2 *)nlcpy__get_ptr(initial);
    if (pi == NULL) return NLCPY_ERROR_MEMORY;
    ve_array *w = where;
    $2 init_val = ($2)(*pi);

    // initialize
    if (init_flag) {
@#ifdef _OPENMP
@#pragma omp for
@#endif /* _OPENMP */
        for (uint64_t i=0; i<y->size; i++) py[i] = init_val;
    }
    else {
@#ifdef _OPENMP
@#pragma omp for
@#endif /* _OPENMP */
        for (uint64_t i=0; i<y->size; i++) py[i] = 0;
    }

/////////
// 0-d //
/////////
    if (x->ndim == 0){
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
        if (!where_flag) {
            if (!init_flag) {
                *py = *px;
            }
            else{
                @UNARY_OPERATOR@(*px,*py,$1)
            }
        } else {
            Bint *pw = (Bint *)w->ve_adr;
            if (*pw) {
                if (!init_flag) {
                    *py = *px;
                }
                else{
                    @UNARY_OPERATOR@(*px,*py,$1)
                }
            }
        }
} /* omp single */

/////////
// 1-d //
/////////
    } else if (x->ndim == 1){
        uint64_t i;
        uint64_t iw0 = 0;
        const uint64_t ix = x->strides[0]/x->itemsize;
        if (!where_flag) {
#if defined(add_reduce) || defined(multiply_reduce)
            $2 tmp;
#if defined(add_reduce)
            nlcpy__global_$1 = 0;
@#pragma omp barrier
            tmp = 0;
#elif defined(multiply_reduce)
            nlcpy__global_$1 = 1;
@#pragma omp barrier
            tmp = 1;
#else
#error add_reduce or minimum_reduce must be defined.
#endif
            uint64_t ii;
            if (!init_flag) {
                *py = *px;
                ii = 1;
            } else {
                ii = 0;
            }
            const int it = omp_get_thread_num();
            const int nt = omp_get_max_threads();
            const uint64_t ist = (x->shape[0]-ii)*it/nt;
            const uint64_t ien = (x->shape[0]-ii)*(it+1)/nt;
            if (it==0) tmp = ($2)(*py);
@#pragma _NEC ivdep
            for (i=ii+ist; i < ii+ien; i++) {
                @UNARY_OPERATOR@(px[i*ix],tmp,$1)
            }
@#pragma _NEC novector
            for (i=0; i < nt; i++) {
                if (i==it) {
                    @UNARY_OPERATOR@(tmp,nlcpy__global_$1,$1)
                }
@#pragma omp barrier
            }
            *py = nlcpy__global_$1;
#elif ( defined(maximum_reduce) || defined(minimum_reduce) ) && ( defined(DTAG1_i32) || defined(DTAG1_i64) || defined(DTAG1_f32) || defined(DTAG1_f64) || defined(DTAG1_bool)) && ( "$1" eq "i32" || "$1" eq "i64" || "$1" eq "f32" || "$1" eq "f64" || "$1" eq "bool")
            // Note that rational operations for complex and unsigned numbers occurs an error or warnning.
            uint64_t ii;
            if (!init_flag) {
                *py = *px;
                ii = 1;
            } else {
                ii = 0;
            }
            nlcpy__global_$1 = ($2)(*py);
@#pragma omp barrier
            const int it = omp_get_thread_num();
            const int nt = omp_get_max_threads();
            const uint64_t ist = (x->shape[0]-ii)*it/nt;
            const uint64_t ien = (x->shape[0]-ii)*(it+1)/nt;
            $2 tmp = ($2)(*py);
#if "$1" eq "f32" || "$1" eq "f64"
            double is_there_nan = (isnan_$1(tmp)) ? 1.0 : 0.0;
#elif "$1" eq "i32" || "$1" eq "i64" || "$1" eq "bool"
            double is_there_nan = 0.0;
#else
#error Not Impletended
#endif
@#pragma _NEC ivdep
            for (i=ii+ist; i < ii+ien; i++) {
#if defined(maximum_reduce)
                tmp = (tmp > px[i*ix]) ? tmp : ($2)px[i*ix];
#elif defined(minimum_reduce)
                tmp = (tmp < px[i*ix]) ? tmp : ($2)px[i*ix];
#endif
#if defined(DTAG1_f32) || defined(DTAG1_f64)
                // NaN check:
                // is_there_nan += (float)isnan_@DTAG1@(px[i*ix]);
                @TYPE1@ xx;
                *(&xx) = px[i*ix];
                // The following line checks for NaN and it is in an unnatural way.
                // Here, NaN == NaN becomes False.
                // If there is one or more qNaN, signaling NaN (sNaN) occurs.
                // qNaN might not be detected qNaN by compiler optimizations.
                // However, we prioritize the performance.
                is_there_nan += (! (xx == px[i*ix]) ) ? 1.0 : 0.0;
#endif
            }
#if defined(DTAG1_f32) || defined(DTAG1_f64)
            if (is_there_nan != (float)0) {
                tmp = NAN;
                // In the following function call, PSW flags are manipulated to skip the intentional sNaN above.
                retrieve_fpe_flags(psw);
                *psw &= 0x0000003d;
                set_fpe_flags(*psw);
            }
#endif
@#pragma omp critical
            {
                @UNARY_OPERATOR@(tmp,nlcpy__global_$1,$1)
            }
@#pragma omp barrier
            *py = nlcpy__global_$1;
#else
        // Unvectorizable case
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
            {
                if (!init_flag) {
                    *py = *px;
                    i = 1;
                } else {
                    i = 0;
                }
@#pragma _NEC novector
                for (; i < x->shape[0]; i++) {
                    @UNARY_OPERATOR@(px[i*ix],*py,$1)
                }
            } /* omp single */
#endif
        } else {
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
            {
                Bint *pw = (Bint *)w->ve_adr;
                const uint64_t iw = w->strides[0]/w->itemsize;
                if (!init_flag) {
                    for (i = 0; i < x->shape[0]; i++) {
                        if (pw[i*iw]) {
                            *py = *px;
                            i++;
                            break;
                        }
                    }
                } else {
                    i = 0;
                }
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                for (; i < x->shape[0]; i++) {
                    if (pw[i*iw]) {
                        @UNARY_OPERATOR@(px[i*ix],*py,$1)
                    }
                }
            } /* omp single */
        }

/////////
// N-d //
/////////
    } else if (x->ndim <= NLCPY_MAXNDIM) {
        int64_t n_inner = x->ndim - 1;
        int64_t n_outer = 0;

        int64_t *idx = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
        for (uint64_t i = 0; i < x->ndim; i++) {
            idx[i] = i;
        }
        if (x->ndim > 2) {
            int64_t max_idx[3], tmp;
            nlcpy__argnsort(x, max_idx, 3);
#ifndef add_reduce
            if (idx[max_idx[0]] == axis) {
                max_idx[0] = max_idx[1];
            }
#endif
            if (max_idx[0] != n_inner) {
                tmp = idx[n_inner];
                idx[n_inner] = idx[max_idx[0]];
                idx[max_idx[0]] = tmp;
            }
            if (idx[max_idx[1]] == axis) {
                max_idx[1] = max_idx[2];
            }
            if (idx[max_idx[1]] != n_outer) {
                tmp = idx[n_outer];
                idx[n_outer] = idx[max_idx[1]];
                idx[max_idx[1]] = tmp;
            }
        } else if (n_outer == axis) {
            idx[0] = 1;
            idx[1] = 0;
        }
        int64_t n_inner2 = idx[n_inner];
        int64_t n_outer2 = idx[n_outer];

@#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
@#else
        const int nt = 1;
        const int it = 0;
@#endif
        int64_t i;
        uint64_t adr_x = (uint64_t)x->ve_adr;
        uint64_t adr_y = (uint64_t)y->ve_adr;

        uint64_t ix = 0;
        uint64_t iy = 0;
        uint64_t iw = 0;
        uint64_t ix0 = x->strides[n_inner2] / x->itemsize;
        uint64_t iy0 = y->strides[n_inner2] / y->itemsize;
        uint64_t iw0 = 0;
        int64_t k = 0;
        int64_t *cnt_x = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
        nlcpy__reset_coords(cnt_x, x->ndim);
        const int64_t lenm = x->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;
        if (!where_flag) {
            for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                cnt_x[n_outer2] = cntm;
                ix = cntm * x->strides[n_outer2] / x->itemsize;
                iy = (n_outer2 == axis) ? 0 : cntm * y->strides[n_outer2] / y->itemsize;

                do {
                    if (n_inner2 == axis) {
                        // Note that a rational operation for complex and unsigned numbers occurs an error or warnning.
                        uint64_t ii;
                        if (!init_flag) {
                           py[iy] = px[ix];
                           ii = 1;
                        } else {
                           ii = 0;
                        }
#if ( defined(maximum_reduce) || defined(minimum_reduce) ) && ( defined(DTAG1_i32) || defined(DTAG1_i64) || defined(DTAG1_f32) || defined(DTAG1_f64) || defined(DTAG1_bool)) && ( "$1" eq "i32" || "$1" eq "i64" || "$1" eq "f32" || "$1" eq "f64" || "$1" eq "bool")
                        $2 tmp = py[iy];
#if "$1" eq "f32" || "$1" eq "f64"
                        float is_there_nan = (isnan_$1(py[iy])) ? 1.0 : 0.0;
#elif "$1" eq "i32" || "$1" eq "i64" || "$1" eq "bool"
                        float is_there_nan = 0.0;
#else
#error Not Impletended
#endif
@#pragma _NEC ivdep
                        for (i = ii; i < x->shape[n_inner2]; i++) {
#if defined(maximum_reduce)
                            tmp = (tmp > px[i*ix0+ix]) ? tmp : ($2)px[i*ix0+ix];
#elif defined(minimum_reduce)
                            tmp = (tmp < px[i*ix0+ix]) ? tmp : ($2)px[i*ix0+ix];
#endif
#if defined(DTAG1_f32) || defined(DTAG1_f64)
                            // NaN check:
                            //   is_there_nan += (float)isnan_@DTAG1@(px[i*ix0+ix]);
                            @TYPE1@ xx;
                            *(&xx) = px[i*ix0+ix];
                            // The following line checks for NaN and it is in an unnatural way.
                            // Here, NaN == NaN becomes False.
                            // If there is one or more qNaN, signaling NaN (sNaN) occurs.
                            // qNaN might not be detected qNaN by compiler optimizations.
                            // However, we prioritize the performance.
                            is_there_nan += (! (xx == px[i*ix0+ix]) ) ? 1.0 : 0.0;
#endif
                        }
#if defined(DTAG1_f32) || defined(DTAG1_f64)
                        if (is_there_nan != (float)0) {
                            tmp = NAN;
                            // In the following function call, PSW flags are manipulated to skip the intentional sNaN above.
                            retrieve_fpe_flags(psw);
                            *psw &= 0x0000003d;
                            set_fpe_flags(*psw);
                        }
#endif
                        py[iy] = tmp;
#else
                        // General case
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                        for (i = ii; i < x->shape[n_inner2]; i++) {
                            @UNARY_OPERATOR@(px[i*ix0+ix],py[iy],$1)
                        }
#endif
                    } else {
                        if (init_flag || cnt_x[axis] > 0) {
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                            for (i = 0; i < x->shape[n_inner2]; i++) {
                                @UNARY_OPERATOR@(px[i*ix0+ix],py[i*iy0+iy],$1)
                             }
                        } else {
                            for (i = 0; i < x->shape[n_inner2]; i++) {
                                py[i*iy0+iy] = @CAST_OPERATOR@(px[i*ix0+ix],@DTAG1@,$1)
                            }
                        }
                    }
                    for (k = n_inner - 1; k >= 1; k--) {
                        int64_t kk = idx[k];
                        if (kk == axis) {
                            if (++cnt_x[kk] < x->shape[kk]) {
                                ix += x->strides[kk] / x->itemsize;
                                break;
                            }
                            ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                        } else {
                            if (++cnt_x[kk] < x->shape[kk]) {
                                ix += x->strides[kk] / x->itemsize;
                                iy += y->strides[kk] / y->itemsize;
                                break;
                            }
                            ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                            iy -= (y->strides[kk] / y->itemsize) * (y->shape[kk] - 1);
                        }
                        cnt_x[kk] = 0;
                    }
                } while (k >= 1);
            }
        } else {
            Bint *pw = (Bint *)w->ve_adr;
            iw0 = w->strides[n_inner2] / w->itemsize;

            for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                ix = cntm * x->strides[n_outer2] / x->itemsize;
                iy = (n_outer2 == axis) ? 0 : cntm * y->strides[n_outer2] / y->itemsize;
                iw = cntm * w->strides[n_outer2] / w->itemsize;

                do {
                    if (n_inner2 == axis) {
                        if (!init_flag) {
                            for (i = 0; i < w->shape[n_inner2]; i++) {
                                if (pw[i*iw0+iw]) {
                                    py[iy] = @CAST_OPERATOR@(px[i*ix0+ix],@DTAG1@,$1)
                                    i++;
                                    break;
                                }
                            }
                        } else {
                            i = 0;
                        }
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                        for (; i < x->shape[n_inner2]; i++) {
                            if (pw[i*iw0+iw]) {
                                @UNARY_OPERATOR@(px[i*ix0+ix],py[iy],$1)
                            }
                        }
                    } else {
                        if (init_flag || cnt_x[axis] > 0) {
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                            for (i = 0; i < x->shape[n_inner2]; i++) {
                                if (pw[i*iw0+iw]) {
                                    @UNARY_OPERATOR@(px[i*ix0+ix],py[i*iy0+iy],$1)
                                }
                            }
                        } else {
                            for (i = 0; i < x->shape[n_inner2]; i++) {
                                if (pw[i*iw0+iw]) {
                                    py[i*iy0+iy] = @CAST_OPERATOR@(px[i*ix0+ix],@DTAG1@,$1)
                                }
                            }
                        }
                    }
                    for (k = n_inner - 1; k >= 1; k--) {
                        int64_t kk = idx[k];
                        if (kk == axis) {
                            if (++cnt_x[kk] < x->shape[kk]) {
                                ix += x->strides[kk] / x->itemsize;
                                iw += w->strides[kk] / w->itemsize;
                                break;
                            }
                            ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                            iw -= (w->strides[kk] / w->itemsize) * (x->shape[kk] - 1);
                        } else {
                            if (++cnt_x[kk] < x->shape[kk]) {
                                ix += x->strides[kk] / x->itemsize;
                                iy += y->strides[kk] / y->itemsize;
                                iw += w->strides[kk] / w->itemsize;
                                break;
                            }
                            ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                            iy -= (y->strides[kk] / y->itemsize) * (x->shape[kk] - 1);
                            iw -= (w->strides[kk] / w->itemsize) * (x->shape[kk] - 1);
                        }
                        cnt_x[kk] = 0;
                    }
                } while (k >= 1);
            }
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


@-->)dnl
#if defined(DTAG_i32)
macro_reduce_operator(i32,int32_t)dnl
#endif
#if defined(DTAG_i64)
macro_reduce_operator(i64,int64_t)dnl
#endif
#if defined(DTAG_u32)
macro_reduce_operator(u32,uint32_t)dnl
#endif
#if defined(DTAG_u64)
macro_reduce_operator(u64,uint64_t)dnl
#endif
#if defined(DTAG_f32)
macro_reduce_operator(f32,float)dnl
#endif
#if defined(DTAG_f64)
macro_reduce_operator(f64,double)dnl
#endif
#if defined(DTAG_c64)
macro_reduce_operator(c64,float _Complex)dnl
#endif
#if defined(DTAG_c128)
macro_reduce_operator(c128,double _Complex)dnl
#endif
#if defined(DTAG_bool)
macro_reduce_operator(bool,int32_t)dnl
#endif

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
 *       REDUCEAT OPERATOR
 *
 * **************************/

define(<--@macro_reduceat_operator@-->,<--@
uint64_t FILENAME_@DTAG1@_$1(ve_array *x, ve_array *indices, ve_array *y, int32_t axis, int32_t *bad_index, int32_t *psw)
{
    int64_t *pind = (int64_t *)indices->ve_adr;
    *bad_index = -1;
    for(uint64_t i = 0; i < indices->size; i++) {
        if (*(pind + i) < 0 || *(pind + i) > x->shape[axis] - 1) {
            *bad_index = (int32_t)i;
            return NLCPY_ERROR_INDEX;
        }
    }

    uint64_t *start = (uint64_t*)alloca(sizeof(uint64_t) * indices->size);
    uint64_t *end = (uint64_t*)alloca(sizeof(uint64_t) * indices->size);
    for (uint64_t i = 0; i < indices->size - 1; i++) {
        if (*(pind + i) >= *(pind + i + 1)) {
            start[i] = *(pind + i);
            end[i] = *(pind + i) + 1;
        } else {
            start[i] = *(pind + i);
            end[i] = *(pind + i + 1);
        }
    }
    start[indices->size - 1] = *(pind + indices->size - 1);
    end[indices->size - 1] = x->shape[axis];

    int64_t n_inner = x->ndim - 1;
    int64_t *idx = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
    int64_t max_idx;
    nlcpy__argnsort(x, &max_idx, 1);
    for (int64_t i = 0; i < x->ndim; i++) idx[i] = i;
    if(n_inner != max_idx) {
        int64_t tmp = idx[n_inner];
        idx[n_inner] = idx[max_idx];
        idx[max_idx] = tmp;
    }

    $2 *py = ($2 *)y->ve_adr;
    @TYPE1@ *px = (@TYPE1@ *)x->ve_adr;
@#ifdef _OPENMP
@#pragma omp parallel
@#endif /* _OPENMP */
{
@#ifdef _OPENMP
    const int nt = omp_get_num_threads();
    const int it = omp_get_thread_num();
@#else
    const int nt = 1;
    const int it = 0;
@#endif
    int64_t k;
    uint64_t ix0 = x->strides[idx[n_inner]] / x->itemsize;
    uint64_t iy0 = y->strides[idx[n_inner]] / y->itemsize;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
    const int64_t cntm_s = indices->size * it / nt;
    const int64_t cntm_e = indices->size * (it + 1) / nt;
    for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
        nlcpy__reset_coords(cnt, x->ndim);
        uint64_t ix = (idx[n_inner] == axis) ? 0 : start[cntm] * x->strides[axis] / x->itemsize;
        uint64_t iy = cntm * y->strides[axis] / y->itemsize;

        do {
            if (idx[n_inner] == axis) {
                py[iy] = @CAST_OPERATOR@(px[start[cntm]*ix0+ix], @DTAG1@, $1);
#ifdef add_reduceat
@#pragma _NEC ivdep
#else
@#pragma _NEC novector
#endif
                for (int64_t i = start[cntm] + 1; i < end[cntm]; i++) {
                    @UNARY_OPERATOR@(px[i*ix0+ix], py[iy],$1)
                }
            } else {
                if (cnt[axis] > 0) {
#ifdef left_shift_reduceat
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                    for (int64_t i = 0; i < x->shape[idx[n_inner]]; i++) {
                        @UNARY_OPERATOR@(px[i*ix0+ix], py[i*iy0+iy],$1)
                    }
                } else {
#ifdef left_shift_reduceat
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                    for (int64_t i = 0; i < x->shape[idx[n_inner]]; i++) {
                        py[i*iy0+iy] = @CAST_OPERATOR@(px[i*ix0+ix], @DTAG1@, $1);
                    }
                }
            }
            for (k = n_inner - 1; k >= 0; k--) {
                int64_t kk = idx[k];
                if (kk == axis) {
                    if (!cnt[kk]) {
                        cnt[kk] = start[cntm];
                    }
                    if (++cnt[kk] < end[cntm]) {
                        ix += x->strides[kk] / x->itemsize;
                        break;
                    }
                    ix -= (x->strides[kk] / x->itemsize) * (end[cntm] - start[cntm] - 1);
                } else {
                    if (++cnt[kk] < x->shape[kk]) {
                        ix += x->strides[kk] / x->itemsize;
                        iy += y->strides[kk] / y->itemsize;
                        break;
                    }
                    ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                    iy -= (y->strides[kk] / y->itemsize) * (x->shape[kk] - 1);
                }
                cnt[kk] = 0;
            }
        } while (k >= 0);
    }
} /* omp parallel */
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


@-->)dnl
#if defined(DTAG_i32)
macro_reduceat_operator(i32,int32_t)dnl
#endif
#if defined(DTAG_i64)
macro_reduceat_operator(i64,int64_t)dnl
#endif
#if defined(DTAG_u32)
macro_reduceat_operator(u32,uint32_t)dnl
#endif
#if defined(DTAG_u64)
macro_reduceat_operator(u64,uint64_t)dnl
#endif
#if defined(DTAG_f32)
macro_reduceat_operator(f32,float)dnl
#endif
#if defined(DTAG_f64)
macro_reduceat_operator(f64,double)dnl
#endif
#if defined(DTAG_c64)
macro_reduceat_operator(c64,float _Complex)dnl
#endif
#if defined(DTAG_c128)
macro_reduceat_operator(c128,double _Complex)dnl
#endif
#if defined(DTAG_bool)
macro_reduceat_operator(bool,int32_t)dnl
#endif

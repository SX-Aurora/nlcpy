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

/***************************
 *
 *       ACCUMULATE OPERATOR
 *
 * *************************/

define(<--@macro_accumulate_operator@-->,<--@
uint64_t FILENAME_@DTAG1@_$1(ve_array *x, ve_array *y, int32_t axis, int32_t *psw)
{
    $2 *py = ($2 *)y->ve_adr;

    int64_t n_inner = x->ndim - 1;
    int64_t n_outer = 0;
    int64_t idx[x->ndim], shp_x[x->ndim];
    int64_t i, idx_tmp, idx_max, shp_tmp, shp_max;

    for (i = 0; i < x->ndim; i++) {
        shp_x[i] = x->shape[i];
        idx[i] = i;
    }
    idx_max = 0;
    idx_tmp = 0;
    shp_max = 0;
    shp_tmp = 0;
    for (i = 0; i < x->ndim; i++) {
        shp_tmp = shp_x[i];
        if (shp_max <= shp_tmp) {
            shp_max = shp_x[i];
            idx_max = i;
        }
    }

    if (x->ndim > 1) {
        if (idx_max != n_inner) {
            idx_tmp = idx[idx_max];
            shp_tmp = shp_x[idx_max];
            idx[idx_max] = idx[n_inner];
            shp_x[idx_max] = shp_x[n_inner];
            idx[n_inner] = idx_tmp;
            shp_x[n_inner] = shp_tmp;

            if (axis == n_inner) {
                axis = idx_max;
            } else if (axis == idx_max) {
                axis = n_inner;
            }
        }

        if ((axis != n_inner) && (axis != n_inner-1)) {
            idx_tmp = idx[n_inner-1];
            shp_tmp = shp_x[n_inner-1];
            idx[n_inner-1] = idx[axis];
            shp_x[n_inner-1] = shp_x[axis];
            idx[axis] = idx_tmp;
            shp_x[axis] = shp_tmp;

            axis = n_inner - 1;
        }

        if (x->ndim > 2) {
            idx_max = 0;
            idx_tmp = 0;
            shp_max = 0;
            shp_tmp = 0;
            if (axis == n_inner) {
                for (int64_t i = 0; i < x->ndim-1; i++) {
                    shp_tmp = shp_x[i];
                    if (shp_max < shp_tmp) {
                        shp_max = shp_tmp;
                        idx_max = i;
                    }
                }
                if (idx_max != n_outer) {
                    idx_tmp = idx[n_outer];
                    idx[n_outer] = idx[idx_max];
                    idx[idx_max] = idx_tmp;
                }
            } else if (axis == n_inner-1) {
                for (int64_t i = 0; i < x->ndim-2; i++) {
                    shp_tmp = shp_x[i];
                    if (shp_max < shp_tmp) {
                        shp_max = shp_tmp;
                        idx_max = i;
                    }
                }
                if (idx_max != n_outer) {
                    idx_tmp = idx[n_outer];
                    idx[n_outer] = idx[idx_max];
                    idx[idx_max] = idx_tmp;
                }
            }
        }
    }
    @TYPE1@ *px = (@TYPE1@ *)x->ve_adr;
    if (x->ndim == 1) {
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
        int64_t i;
        int64_t ix0 = 1;
        int64_t iy0 = 1;
        $2 tmp;
        if ( x->shape[axis]>0 ) {
            tmp = @CAST_OPERATOR@(*px, @DTAG1@, $1);
            *py = tmp;
        }
        for (i = 1; i < x->shape[axis]; i++) {
            @TYPE1@ *px0 = px + i*ix0;
            $2 *py0 = py + i*iy0;
            @UNARY_OPERATOR@(*px0,tmp,$1)
            *py0 = tmp;
        }
} /* omp single */

    } else if (x->ndim == 2) {
        if (axis == n_inner) {
            idx_tmp = idx[n_inner];
            idx[n_inner] = idx[n_inner-1];
            idx[n_inner-1] = idx_tmp;
            axis = n_inner-1;
        }

        const int64_t outer = x->shape[idx[n_outer]];
        const int64_t inner = x->shape[idx[n_inner]];
        const int64_t ix0 = x->strides[idx[n_outer]] / x->itemsize;
        const int64_t ix1 = x->strides[idx[n_inner]] / x->itemsize;
        const int64_t iy0 = y->strides[idx[n_outer]] / y->itemsize;
        const int64_t iy1 = y->strides[idx[n_inner]] / y->itemsize;

@#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
@#else
        const int nt = 1;
        const int it = 0;
@#endif /* _OPENMP */
        $2 *tmp = ($2 *)alloca(sizeof($2)*inner);

        const int64_t is = inner * it / nt;
        const int64_t ie = inner * (it + 1) / nt;

        uint64_t i0;
        if (outer>0){
            for (uint64_t i1 = is; i1 < ie; i1++) {
                @TYPE1@ *px1 = px + i1*ix1;
                $2      *py1 = py + i1*iy1;
                tmp[i1] = @CAST_OPERATOR@(*px1, @DTAG1@, $1);
                *py1 = tmp[i1];
            }
        }
@#ifdef _OPENMP
@#pragma omp barrier
@#endif /* _OPENMP */
        for (uint64_t i0 = 1; i0 < outer; i0++) {
            @TYPE1@ *px0 = px + i0*ix0;
            $2      *py0 = py + i0*iy0;
            for (uint64_t i1 = is; i1 < ie; i1++) {
                @TYPE1@ *px1 = px0 + i1*ix1;
                $2      *py1 = py0 + i1*iy1;
                @UNARY_OPERATOR@(*px1,tmp[i1],$1);
                *py1 = tmp[i1];
            }
        }

    } else if (x->ndim > 2 && x->ndim <= NLCPY_MAXNDIM) {

        if (axis == n_inner) {
            idx_tmp = idx[n_inner];
            idx[n_inner] = idx[n_inner-1];
            idx[n_inner-1] = idx_tmp;
            axis = n_inner-1;
        }

        int64_t n_inner2 = idx[n_inner];
        int64_t n_outer2 = idx[n_outer];

@#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
@#else
        const int nt = 1;
        const int it = 0;
@#endif /* _OPENMP */
        int64_t i = 0;
        int64_t k = 0;
        int64_t kk = 0;

        int64_t ix = 0;
        int64_t iy = 0;
        int64_t ix0 = x->strides[n_inner2] / x->itemsize;
        int64_t iy0 = y->strides[n_inner2] / y->itemsize;
        int64_t *cnt_x = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
        nlcpy__reset_coords(cnt_x, x->ndim);

        const int64_t lenm = x->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;

        $2 *tmp = ($2 *)alloca(sizeof($2)*x->shape[n_inner2]);
        for (i = 0; i < x->shape[n_inner2]; i++) {
            tmp[i] = 0;
        }

        for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
            ix = cntm * x->strides[n_outer2] / x->itemsize;
            iy = cntm * y->strides[n_outer2] / y->itemsize;
            do {
                if (cnt_x[idx[axis]] > 0) {
                    for (i = 0; i < x->shape[n_inner2]; i++) {
                        @UNARY_OPERATOR@(px[i*ix0+ix],tmp[i],$1);
                        py[i*iy0+iy] = tmp[i];
                    }
                } else {
                    for (i = 0; i < x->shape[n_inner2]; i++) {
                        tmp[i] = @CAST_OPERATOR@(px[i*ix0+ix], @DTAG1@, $1);
                        py[i*iy0+iy] = tmp[i];
                    }
                }
                for (k = n_inner - 1; k > 0; k--) {
                    kk = idx[k];
                    if (++cnt_x[kk] < x->shape[kk]) {
                        ix += x->strides[kk] / x->itemsize;
                        iy += y->strides[kk] / y->itemsize;
                        break;
                    }
                    ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                    iy -= (y->strides[kk] / y->itemsize) * (y->shape[kk] - 1);
                    for (i = 0; i < x->shape[n_inner2]; i++) {
                        tmp[i] = 0;
                    }
                    cnt_x[kk] = 0;
                }
            } while(k > 0);
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


@-->)dnl
#if defined(DTAG_i32)
macro_accumulate_operator(i32,int32_t)dnl
#endif
#if defined(DTAG_i64)
macro_accumulate_operator(i64,int64_t)dnl
#endif
#if defined(DTAG_u32)
macro_accumulate_operator(u32,uint32_t)dnl
#endif
#if defined(DTAG_u64)
macro_accumulate_operator(u64,uint64_t)dnl
#endif
#if defined(DTAG_f32)
macro_accumulate_operator(f32,float)dnl
#endif
#if defined(DTAG_f64)
macro_accumulate_operator(f64,double)dnl
#endif
#if defined(DTAG_c64)
macro_accumulate_operator(c64,float _Complex)dnl
#endif
#if defined(DTAG_c128)
macro_accumulate_operator(c128,double _Complex)dnl
#endif
#if defined(DTAG_bool)
macro_accumulate_operator(bool,int32_t)dnl
#endif

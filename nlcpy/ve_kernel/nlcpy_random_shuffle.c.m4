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
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <alloca.h>
#include <assert.h>

#include "nlcpy.h"

include(macros.m4)dnl

/****************************
 *
 *       @OPERATOR_NAME@
 *
 * **************************/

define(<--@macro_random_shuffle@-->,<--@
uint64_t nlcpy_random_shuffle_$1(ve_array *x, ve_array *idx, ve_array *work, int32_t axis, int32_t *psw)
{
    $2 *px = ($2 *)x->ve_adr;
    if (px == NULL) return 0LU;
    int64_t *pi = (int64_t *)idx->ve_adr;
    if (pi == NULL) return 0LU;
    $2 *pw = ($2 *)work->ve_adr;
    if (px == NULL) return 0LU;

/////////
// 0-d //
/////////
    if (x->ndim == 0) {
        /* nothing to do */

/////////
// 1-d //
/////////
    } else if (x->ndim == 1) {
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
{
        const uint64_t ix0 = x->strides[0] / x->itemsize;
        const uint64_t ii0 = idx->strides[0] / idx->itemsize;
        const uint64_t iw0 = work->strides[0] / work->itemsize;
        int64_t idx_tmp;
        int64_t i;
        for (i = 0; i < x->size; i++) {
            px[i*ix0] = pw[pi[i*ii0]];
        }
} /* omp critical */

/////////
// N-d //
/////////
    } else if (x->ndim > 1 && x->ndim <= NLCPY_MAXNDIM){
#ifdef _OPENMP
        const int nt = omp_get_max_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */
        int64_t *cnt_x = (int64_t*)alloca(sizeof(int64_t) * x->ndim);
        nlcpy__reset_coords(cnt_x, x->ndim);
        int64_t i, j, k;
        const int64_t n_inner = x->ndim - 1;
        const int64_t n_outer = 0;
        uint64_t ix = 0;
        uint64_t iw = 0;
        const uint64_t ix0 = x->strides[n_inner] / x->itemsize;
        const uint64_t iw0 = work->strides[n_inner] / work->itemsize;
        const uint64_t ii0 = idx->strides[0] / idx->itemsize;

        const int64_t lenm = x->shape[n_outer];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;
        for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
            ix = cntm * x->strides[n_outer] / x->itemsize;
            if (n_outer == axis){
                iw = pi[cntm*ii0] * work->strides[n_outer] / work->itemsize;
            } else {
                iw = cntm * work->strides[n_outer] / work->itemsize;
            }
            for (;;) {
                // most inner loop for vectorize
                if (n_inner == axis) {
                    for (i = 0; i < x->shape[n_inner]; i++) {
                        px[ix+i*ix0] = pw[iw+pi[i*ii0]*iw0];
                    }
                } else {
                    for (i = 0; i < x->shape[n_inner]; i++) {
                        px[ix+i*ix0] = pw[iw+i*iw0];
                    }
                }
                // set next index
                for (k = n_inner-1; k >= 1; k--) {
                    if (++cnt_x[k] < x->shape[k]) {
                        ix += x->strides[k] / x->itemsize;
                        if (k == axis){
                            iw += (pi[cnt_x[k]] - pi[cnt_x[k-1]]) * work->strides[k] / work->itemsize;
                        } else {
                            iw += work->strides[k] / work->itemsize;
                        }
                        break;
                    }
                    cnt_x[k] = 0;
                    ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                    if (k == axis) {
                        iw -= (work->strides[k] / work->itemsize) * (pi[work->shape[k] - 1] - pi[0]);
                    } else {
                        iw -= (work->strides[k] / work->itemsize) * (work->shape[k] - 1);
                    }
                }
                if (k < 1) break;
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
macro_random_shuffle(bool,bool)dnl
macro_random_shuffle(i32,int32_t)dnl
macro_random_shuffle(i64,int64_t)dnl
macro_random_shuffle(u32,uint32_t)dnl
macro_random_shuffle(u64,uint64_t)dnl
macro_random_shuffle(f32,float)dnl
macro_random_shuffle(f64,double)dnl
macro_random_shuffle(c64,float _Complex)dnl
macro_random_shuffle(c128,double _Complex)dnl


uint64_t nlcpy_random_shuffle(ve_arguments *args, int32_t *psw)
{
    ve_array *x = &(args->shuffle.x);
    ve_array *idx = &(args->shuffle.idx);
    ve_array *work = &(args->shuffle.work);
    int32_t axis = args->shuffle.axis;

    assert(x->dtype == work->dtype);
    assert(idx->ndim == 1);
    assert(x->ndim == work->ndim);
    assert(x->size == work->size);
    assert(idx->is_c_contiguous);
    assert(x->shape[axis] == idx->size);
    uint64_t err;

    switch (x->dtype) {
    case ve_bool:  err = nlcpy_random_shuffle_bool (x, idx, work, axis, psw); break;
    case ve_i32 :  err = nlcpy_random_shuffle_i32  (x, idx, work, axis, psw); break;
    case ve_i64 :  err = nlcpy_random_shuffle_i64  (x, idx, work, axis, psw); break;
    case ve_u32 :  err = nlcpy_random_shuffle_u32  (x, idx, work, axis, psw); break;
    case ve_u64 :  err = nlcpy_random_shuffle_u64  (x, idx, work, axis, psw); break;
    case ve_f32 :  err = nlcpy_random_shuffle_f32  (x, idx, work, axis, psw); break;
    case ve_f64 :  err = nlcpy_random_shuffle_f64  (x, idx, work, axis, psw); break;
    case ve_c64 :  err = nlcpy_random_shuffle_c64  (x, idx, work, axis, psw); break;
    case ve_c128:  err = nlcpy_random_shuffle_c128 (x, idx, work, axis, psw); break;
    default: err = NLCPY_ERROR_DTYPE;
    }
    return (uint64_t)err;
}

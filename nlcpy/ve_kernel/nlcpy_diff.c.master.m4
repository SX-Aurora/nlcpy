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

include(macros.m4)dnl
@#include <stdio.h>
@#include <stdint.h>
@#include <stdbool.h>
@#include <stdlib.h>
@#include <limits.h>
@#include <alloca.h>
@#include <assert.h>

@#include "nlcpy.h"

define(<--@macro_unary_operator@-->,<--@
uint64_t FILENAME_$1(ve_array *a, ve_array *b, ve_array *w, int64_t n, int64_t axis, int32_t *psw)
{
    int64_t i, j, k;
    $2 *pa = ($2 *)a->ve_adr;
    $2 *pb = ($2 *)b->ve_adr;
    $2 *wk = ($2 *)w->ve_adr;
    if (a->ndim == 1) {
        int64_t iw0 = w->strides[0] / w->itemsize;
        for (i = 0; i < n; i++) {
            for (j = 0; j < a->size - i - 1; j++) {
                ifelse($1,bool,wk[j*iw0] = !wk[(j+1)*iw0] && wk[j*iw0] || wk[(j+1)*iw0] && !wk[j*iw0];,wk[j*iw0] = wk[(j+1)*iw0] - wk[j*iw0];)
            }
        }
        for (i = 0; i < b->size; i++) pb[i] = wk[i*iw0];
    } else if (a->ndim > 1 && a->ndim <= NLCPY_MAXNDIM){
        int64_t n_inner = a->ndim - 1;
        int64_t *idx = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__rearrange_axis(a, idx);

        int64_t n_inner2 = idx[n_inner];
        int64_t *cnt = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        int64_t *shape_wk = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        int64_t *astep = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        for (i = 0; i < a->ndim; i++) {
            shape_wk[i] = a->shape[i];
            astep[i] = w->strides[i] / w->itemsize;
        }
        for (i = 0; i < n; i++) {
            nlcpy__reset_coords(cnt, a->ndim);
            shape_wk[axis] = a->shape[axis] - i - 1;
            int64_t ia1 = 0;
            int64_t ia2 = astep[axis];
            do {
                if (n_inner2 == axis) {
#pragma _NEC ivdep
                    for (j = 0; j < shape_wk[n_inner2]; j++) {
                        ifelse($1,bool,wk[j*astep[n_inner2]+ia1] = !wk[(j+1)*astep[n_inner2]+ia1] && wk[j * astep[n_inner2] + ia1] || wk[(j+1)*astep[n_inner2]+ia1] && !wk[j*astep[n_inner2]+ia1];,wk[j*astep[n_inner2]+ia1] = wk[(j+1)*astep[n_inner2]+ia1] - wk[j*astep[n_inner2]+ia1];)
                    }
                } else {
#pragma _NEC ivdep
                    for (j = 0; j < shape_wk[n_inner2]; j++) {
                        ifelse($1,bool,wk[j*astep[n_inner2]+ia1] = !wk[j*astep[n_inner2]+ia2] && wk[j * astep[n_inner2] + ia1] || wk[j*astep[n_inner2]+ia2] && !wk[j*astep[n_inner2]+ia1];,wk[j*astep[n_inner2]+ia1] = wk[j*astep[n_inner2]+ia2] - wk[j*astep[n_inner2]+ia1];)
                    }
                }
                for (k = n_inner - 1; k >= 0; k--) {
                    int64_t kk = idx[k];
                    if (++cnt[kk] < shape_wk[kk]) {
                        ia1 += astep[kk];
                        ia2 += astep[kk];
                        break;
                    }
                    ia1 -= astep[kk] * (shape_wk[kk] - 1);
                    ia2 -= astep[kk] * (shape_wk[kk] - 1);
                    cnt[kk] = 0;
                }
            } while(k >= 0);
        }

        int64_t *bstep = (int64_t*)alloca(sizeof(int64_t) * b->ndim);
        for (i = 0; i < b->ndim; i++) {
            bstep[i] = b->strides[i] / b->itemsize;
        }
        nlcpy__reset_coords(cnt, a->ndim);
        int64_t ia = 0;
        int64_t ib = 0;
        do {
            for (j = 0; j < b->shape[n_inner2]; j++) {
                pb[j * bstep[n_inner2] + ib] = wk[j * astep[n_inner2] + ia];
            }
            for (k = n_inner - 1; k >= 0; k--) {
                int64_t kk = idx[k];
                if (++cnt[kk] < b->shape[kk]) {
                    ia += astep[kk];
                    ib += bstep[kk];
                    break;
                }
                ia -= astep[kk] * (b->shape[kk] - 1);
                ib -= bstep[kk] * (b->shape[kk] - 1);
                cnt[kk] = 0;
            }
        } while(k >= 0);
    } else {
        return NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

@-->)dnl
macro_unary_operator(bool,int32_t)dnl
macro_unary_operator(i32,int32_t)dnl
macro_unary_operator(i64,int64_t)dnl
macro_unary_operator(u32,uint32_t)dnl
macro_unary_operator(u64,uint64_t)dnl
macro_unary_operator(f32,float)dnl
macro_unary_operator(f64,double)dnl
macro_unary_operator(c64,float _Complex)dnl
macro_unary_operator(c128,double _Complex)dnl

uint64_t nlcpy_diff(
            ve_arguments *args,
            int32_t *psw
) {
    uint64_t err = NLCPY_ERROR_OK;
@#ifdef _OPENMP
@#pragma omp single
@#endif
{
    ve_array *a = &(args->diff.a);
    ve_array *b = &(args->diff.b);
    ve_array *w = &(args->diff.w);
    int64_t n = args->diff.n;
    int64_t axis = args->diff.axis;

    switch (a->dtype) {
    case ve_bool: err = FILENAME_bool(a, b, w, n, axis, psw); break;
    case ve_i32:  err = FILENAME_i32(a, b, w, n, axis, psw); break;
    case ve_i64:  err = FILENAME_i64(a, b, w, n, axis, psw); break;
    case ve_u32:  err = FILENAME_u32(a, b, w, n, axis, psw); break;
    case ve_u64:  err = FILENAME_u64(a, b, w, n, axis, psw); break;
    case ve_f32:  err = FILENAME_f32(a, b, w, n, axis, psw); break;
    case ve_f64:  err = FILENAME_f64(a, b, w, n, axis, psw); break;
    case ve_c64:  err = FILENAME_c64(a, b, w, n, axis, psw); break;
    case ve_c128: err = FILENAME_c128(a, b, w, n, axis, psw); break;
    default: err = NLCPY_ERROR_DTYPE; break;
    }
} /* omp single */
    return (uint64_t)err;
}

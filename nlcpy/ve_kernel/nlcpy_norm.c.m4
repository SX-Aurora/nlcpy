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
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <alloca.h>
#include <assert.h>

#include "nlcpy.h"

define(<--@macro_norm_real@-->,<--@
extern $2 $1lange_(const char*, const int64_t*, const int64_t*, const $2*, const int64_t*, $2*);
uint64_t nlcpy_norm_$1(const char norm, ve_array *x, ve_array *y, ve_array *work, int32_t *psw)
{
    const int64_t m = x->shape[0];
    const int64_t n = x->shape[1];
    const int64_t lda = m;

    $2* const px = ($2*)x->ve_adr;
    $2* const py = ($2*)y->ve_adr;
    $2* const pw = ($2*)work->ve_adr;
    if (px == NULL || py == NULL || pw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ix = 0;
    int64_t iy = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
    for (i = 0; i < x->ndim; i++) cnt[i] = 0;
    do {
        py[iy] = $1lange_(&norm, &m, &n, px+ix, &lda, pw);
        for (i = 2; i < x->ndim; i++) {
            if (++cnt[i] < x->shape[i]) {
                ix += x->strides[i] / x->itemsize;
                iy += y->strides[i-2] / y->itemsize;
                break;
            }
            cnt[i] = 0;
            ix -= (x->strides[i] / x->itemsize) * (x->shape[i] - 1);
            iy -= (y->strides[i-2] / y->itemsize) * (x->shape[i] - 1);
        }
    } while(i < x->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
macro_norm_real(s, float)dnl
macro_norm_real(d, double)dnl

define(<--@macro_norm_complex@-->,<--@
extern $3 $1lange_(const char*, const int64_t*, const int64_t*, const $2*, const int64_t*, $3*);
uint64_t nlcpy_norm_$1(const char norm, ve_array *x, ve_array *y, ve_array *work, int32_t *psw)
{
    const int64_t m = x->shape[0];
    const int64_t n = x->shape[1];
    const int64_t lda = m;

    $2* const px = ($2*)x->ve_adr;
    $3* const py = ($3*)y->ve_adr;
    $3* const pw = ($3*)work->ve_adr;
    if (px == NULL || py == NULL || pw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ix = 0;
    int64_t iy = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
    for (i = 0; i < x->ndim; i++) cnt[i] = 0;
    do {
        py[iy] = $1lange_(&norm, &m, &n, px+ix, &lda, pw);
        for (i = 2; i < x->ndim; i++) {
            if (++cnt[i] < x->shape[i]) {
                ix += x->strides[i] / x->itemsize;
                iy += y->strides[i-2] / y->itemsize;
                break;
            }
            cnt[i] = 0;
            ix -= (x->strides[i] / x->itemsize) * (x->shape[i] - 1);
            iy -= (y->strides[i-2] / y->itemsize) * (x->shape[i] - 1);
        }
    } while(i < x->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
macro_norm_complex(c, float _Complex, float)dnl
macro_norm_complex(z, double _Complex, double)dnl

uint64_t nlcpy_norm(int64_t norm, ve_array *x, ve_array *y, ve_array *work, int32_t *psw) {
    uint64_t err = NLCPY_ERROR_OK;
    switch(x->dtype) {
    case ve_f32:
        err = nlcpy_norm_s(norm, x, y, work, psw);
        break;
    case ve_f64:
        err = nlcpy_norm_d(norm, x, y, work, psw);
        break;
    case ve_c64:
        err = nlcpy_norm_c(norm, x, y, work, psw);
        break;
    case ve_c128:
        err = nlcpy_norm_z(norm, x, y, work, psw);
        break;
    default:
        err = NLCPY_ERROR_DTYPE;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)err;
}

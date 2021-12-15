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


extern float slange_(const char*, const int64_t*, const int64_t*, const float*, const int64_t*, float*);
uint64_t nlcpy_norm_s(const char norm, ve_array *x, ve_array *y, ve_array *work, int32_t *psw)
{
    const int64_t m = x->shape[0];
    const int64_t n = x->shape[1];
    const int64_t lda = m;

    float* const px = (float*)x->ve_adr;
    float* const py = (float*)y->ve_adr;
    float* const pw = (float*)work->ve_adr;
    if (px == NULL || py == NULL || pw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ix = 0;
    int64_t iy = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
    for (i = 0; i < x->ndim; i++) cnt[i] = 0;
    do {
        py[iy] = slange_(&norm, &m, &n, px+ix, &lda, pw);
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

extern double dlange_(const char*, const int64_t*, const int64_t*, const double*, const int64_t*, double*);
uint64_t nlcpy_norm_d(const char norm, ve_array *x, ve_array *y, ve_array *work, int32_t *psw)
{
    const int64_t m = x->shape[0];
    const int64_t n = x->shape[1];
    const int64_t lda = m;

    double* const px = (double*)x->ve_adr;
    double* const py = (double*)y->ve_adr;
    double* const pw = (double*)work->ve_adr;
    if (px == NULL || py == NULL || pw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ix = 0;
    int64_t iy = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
    for (i = 0; i < x->ndim; i++) cnt[i] = 0;
    do {
        py[iy] = dlange_(&norm, &m, &n, px+ix, &lda, pw);
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


extern float clange_(const char*, const int64_t*, const int64_t*, const float _Complex*, const int64_t*, float*);
uint64_t nlcpy_norm_c(const char norm, ve_array *x, ve_array *y, ve_array *work, int32_t *psw)
{
    const int64_t m = x->shape[0];
    const int64_t n = x->shape[1];
    const int64_t lda = m;

    float _Complex* const px = (float _Complex*)x->ve_adr;
    float* const py = (float*)y->ve_adr;
    float* const pw = (float*)work->ve_adr;
    if (px == NULL || py == NULL || pw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ix = 0;
    int64_t iy = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
    for (i = 0; i < x->ndim; i++) cnt[i] = 0;
    do {
        py[iy] = clange_(&norm, &m, &n, px+ix, &lda, pw);
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

extern double zlange_(const char*, const int64_t*, const int64_t*, const double _Complex*, const int64_t*, double*);
uint64_t nlcpy_norm_z(const char norm, ve_array *x, ve_array *y, ve_array *work, int32_t *psw)
{
    const int64_t m = x->shape[0];
    const int64_t n = x->shape[1];
    const int64_t lda = m;

    double _Complex* const px = (double _Complex*)x->ve_adr;
    double* const py = (double*)y->ve_adr;
    double* const pw = (double*)work->ve_adr;
    if (px == NULL || py == NULL || pw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ix = 0;
    int64_t iy = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
    for (i = 0; i < x->ndim; i++) cnt[i] = 0;
    do {
        py[iy] = zlange_(&norm, &m, &n, px+ix, &lda, pw);
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
    return (uint64_t)err;
}

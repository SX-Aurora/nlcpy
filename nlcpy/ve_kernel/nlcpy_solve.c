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


extern void dgesv_(const int64_t *n, const int64_t *nrhs, double *pa, const int64_t *lda, int64_t *ipiv, double *pb, const int64_t *ldb, int64_t *info);
uint64_t nlcpy_solve_d(ve_array *a, ve_array *b, int64_t *info, int32_t *psw)
{
    const int64_t n = a->shape[0];
    const int64_t lda = n;
    const int64_t ldb = n;
    const int64_t nrhs = b->shape[1];
    int64_t *ipiv = (int64_t*)alloca(sizeof(int64_t)*n);

    double *pa = (double*)a->ve_adr;
    double *pb = (double*)b->ve_adr;
    if (pa == NULL || pb == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ia = 0;
    int64_t ib = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*a->ndim);
    for (i = 0; i < a->ndim; i++) cnt[i] = 0;
    do {
        dgesv_(&n, &nrhs, pa+ia, &lda, ipiv, pb+ib, &ldb, info);
        if (*info) return (uint64_t)NLCPY_ERROR_OK;
        for (i = 2; i < a->ndim; i++) {
            if (++cnt[i] < a->shape[i]) {
                ia += a->strides[i] / a->itemsize;
                ib += b->strides[i] / b->itemsize;
                break;
            }
            cnt[i] = 0;
            ia -= (a->strides[i] / a->itemsize) * (a->shape[i] - 1);
            ib -= (b->strides[i] / b->itemsize) * (a->shape[i] - 1);
        }
    } while(i < a->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

extern void zgesv_(const int64_t *n, const int64_t *nrhs, double _Complex *pa, const int64_t *lda, int64_t *ipiv, double _Complex *pb, const int64_t *ldb, int64_t *info);
uint64_t nlcpy_solve_z(ve_array *a, ve_array *b, int64_t *info, int32_t *psw)
{
    const int64_t n = a->shape[0];
    const int64_t lda = n;
    const int64_t ldb = n;
    const int64_t nrhs = b->shape[1];
    int64_t *ipiv = (int64_t*)alloca(sizeof(int64_t)*n);

    double _Complex *pa = (double _Complex*)a->ve_adr;
    double _Complex *pb = (double _Complex*)b->ve_adr;
    if (pa == NULL || pb == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ia = 0;
    int64_t ib = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*a->ndim);
    for (i = 0; i < a->ndim; i++) cnt[i] = 0;
    do {
        zgesv_(&n, &nrhs, pa+ia, &lda, ipiv, pb+ib, &ldb, info);
        if (*info) return (uint64_t)NLCPY_ERROR_OK;
        for (i = 2; i < a->ndim; i++) {
            if (++cnt[i] < a->shape[i]) {
                ia += a->strides[i] / a->itemsize;
                ib += b->strides[i] / b->itemsize;
                break;
            }
            cnt[i] = 0;
            ia -= (a->strides[i] / a->itemsize) * (a->shape[i] - 1);
            ib -= (b->strides[i] / b->itemsize) * (a->shape[i] - 1);
        }
    } while(i < a->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_solve(ve_array *a, ve_array *b, int64_t *info, int32_t *psw) {
    uint64_t err = NLCPY_ERROR_OK;
    switch(a->dtype) {
    case ve_f64:
        err = nlcpy_solve_d(a, b, info, psw);
        break;
    case ve_c128:
        err = nlcpy_solve_z(a, b, info, psw);
        break;
    default:
        err = NLCPY_ERROR_DTYPE;
    }
    return (uint64_t)err;
}

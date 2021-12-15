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


extern void dgeqrf_(const int64_t *m, const int64_t *n, double *pa, const int64_t *lda, double *ptau, double *pw, const int64_t *lwork, int64_t *info);
extern void dorgqr_(const int64_t *m, const int64_t *mc, const int64_t *k, double *pa, const int64_t *lda, double *ptau, double *pw, const int64_t *lwork, int64_t *info);
uint64_t nlcpy_qr_d(int64_t m, int64_t n, int64_t jobq, ve_array *a, ve_array *tau, ve_array *r, ve_array *work, int32_t *psw)
{
    int64_t lda = a->shape[0];
    const int64_t lwork = work->size;
    int64_t info;

    double *pa = (double*)a->ve_adr;
    double *ptau = (double*)tau->ve_adr;
    double *pr = (double*)r->ve_adr;
    double *pw = (double*)work->ve_adr;
    if (pa == NULL || ptau == NULL || pw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }
    dgeqrf_(&m, &n, pa, &lda, ptau, pw, &lwork, &info);
    int64_t a_col_stride = a->strides[0] / a->itemsize;
    int64_t a_row_stride = a->strides[1] / a->itemsize;
    int64_t r_col_stride = r->strides[0] / r->itemsize;
    int64_t r_row_stride = r->strides[1] / r->itemsize;
    if (r->ndim > 1) {
        for (int64_t i = 0; i < r->shape[0]; i++) {
            for (int64_t j = i; j < r->shape[1]; j++) {
                int64_t aind = j * a_row_stride + i * a_col_stride;
                int64_t rind = j * r_row_stride + i * r_col_stride;
                pr[rind] = pa[aind];
            }
        }
    }
    if (jobq) {
        const int64_t k = (a->shape[0] < a->shape[1]) ? a->shape[0] : a->shape[1];
        const int64_t mc = (a->shape[0] == a->shape[1]) ? a->shape[0] : k;
        dorgqr_(&m, &mc, &k, pa, &lda, ptau, pw, &lwork, &info);
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

extern void zgeqrf_(const int64_t *m, const int64_t *n, double _Complex *pa, const int64_t *lda, double _Complex *ptau, double _Complex *pw, const int64_t *lwork, int64_t *info);
extern void zungqr_(const int64_t *m, const int64_t *mc, const int64_t *k, double _Complex *pa, const int64_t *lda, double _Complex *ptau, double _Complex *pw, const int64_t *lwork, int64_t *info);
uint64_t nlcpy_qr_z(int64_t m, int64_t n, int64_t jobq, ve_array *a, ve_array *tau, ve_array *r, ve_array *work, int32_t *psw)
{
    int64_t lda = a->shape[0];
    const int64_t lwork = work->size;
    int64_t info;

    double _Complex *pa = (double _Complex*)a->ve_adr;
    double _Complex *ptau = (double _Complex*)tau->ve_adr;
    double _Complex *pr = (double _Complex*)r->ve_adr;
    double _Complex *pw = (double _Complex*)work->ve_adr;
    if (pa == NULL || ptau == NULL || pw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }
    zgeqrf_(&m, &n, pa, &lda, ptau, pw, &lwork, &info);
    int64_t a_col_stride = a->strides[0] / a->itemsize;
    int64_t a_row_stride = a->strides[1] / a->itemsize;
    int64_t r_col_stride = r->strides[0] / r->itemsize;
    int64_t r_row_stride = r->strides[1] / r->itemsize;
    if (r->ndim > 1) {
        for (int64_t i = 0; i < r->shape[0]; i++) {
            for (int64_t j = i; j < r->shape[1]; j++) {
                int64_t aind = j * a_row_stride + i * a_col_stride;
                int64_t rind = j * r_row_stride + i * r_col_stride;
                pr[rind] = pa[aind];
            }
        }
    }
    if (jobq) {
        const int64_t k = (a->shape[0] < a->shape[1]) ? a->shape[0] : a->shape[1];
        const int64_t mc = (a->shape[0] == a->shape[1]) ? a->shape[0] : k;
        zungqr_(&m, &mc, &k, pa, &lda, ptau, pw, &lwork, &info);
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_qr(int64_t m, int64_t n, int64_t jobq, ve_array *a, ve_array *tau, ve_array *r, ve_array *work, int32_t *psw) {
    uint64_t err = NLCPY_ERROR_OK;
    switch(a->dtype) {
    case ve_f64:
        err = nlcpy_qr_d(m, n, jobq, a, tau, r, work, psw);
        break;
    case ve_c128:
        err = nlcpy_qr_z(m, n, jobq, a, tau, r, work, psw);
        break;
    default:
        err = NLCPY_ERROR_DTYPE;
    }
    return (uint64_t)err;
}

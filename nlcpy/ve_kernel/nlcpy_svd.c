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


extern void sgesdd_(const char *job, const int64_t *m, const int64_t *n, float *pa, const int64_t *lda, float *ps, float *pu, const int64_t *ldu, float *pvt, const int64_t *ldvt, float *pw, const int64_t *lwork, int64_t *piw, int64_t *info);
uint64_t nlcpy_svd_s(const char job, ve_array *a, ve_array *s, ve_array *u, ve_array *vt, ve_array *work, ve_array *iwork, int64_t *info, int32_t *psw)
{
    const int64_t m = a->shape[0];
    const int64_t n = a->shape[1];
    const int64_t lda = m;
    const int64_t ldu = u->shape[0];
    const int64_t ldvt = vt->shape[0];
    const int64_t lwork = work->size;

    float *pa = (float*)a->ve_adr;
    float *ps = (float*)s->ve_adr;
    float *pu = (float*)u->ve_adr;
    float *pvt = (float*)vt->ve_adr;
    float *pw = (float*)work->ve_adr;
    int64_t *piw = (int64_t*)iwork->ve_adr;
    if (!pa || !ps || !pu || !pvt || !pw || !piw) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ia = 0;
    int64_t is = 0;
    int64_t iu = 0;
    int64_t ivt = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*a->ndim);
    for (i = 0; i < a->ndim; i++) cnt[i] = 0;
    do {
        sgesdd_(&job, &m, &n, pa+ia, &lda, ps+is, pu+iu, &ldu, pvt+ivt, &ldvt, pw, &lwork, piw, info);
        for (i = 2; i < a->ndim; i++) {
            if (++cnt[i] < a->shape[i]) {
                ia += a->strides[i] / a->itemsize;
                is += s->strides[i-1] / s->itemsize;
                iu += u->strides[i] / u->itemsize;
                ivt += vt->strides[i] / vt->itemsize;
                break;
            }
            cnt[i] = 0;
            ia -= (a->strides[i] / a->itemsize) * (a->shape[i] - 1);
            is -= (s->strides[i-1] / s->itemsize) * (a->shape[i] - 1);
            iu -= (u->strides[i] / u->itemsize) * (a->shape[i] - 1);
            ivt -= (vt->strides[i] / vt->itemsize) * (a->shape[i] - 1);
        }
    } while(i < a->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

extern void dgesdd_(const char *job, const int64_t *m, const int64_t *n, double *pa, const int64_t *lda, double *ps, double *pu, const int64_t *ldu, double *pvt, const int64_t *ldvt, double *pw, const int64_t *lwork, int64_t *piw, int64_t *info);
uint64_t nlcpy_svd_d(const char job, ve_array *a, ve_array *s, ve_array *u, ve_array *vt, ve_array *work, ve_array *iwork, int64_t *info, int32_t *psw)
{
    const int64_t m = a->shape[0];
    const int64_t n = a->shape[1];
    const int64_t lda = m;
    const int64_t ldu = u->shape[0];
    const int64_t ldvt = vt->shape[0];
    const int64_t lwork = work->size;

    double *pa = (double*)a->ve_adr;
    double *ps = (double*)s->ve_adr;
    double *pu = (double*)u->ve_adr;
    double *pvt = (double*)vt->ve_adr;
    double *pw = (double*)work->ve_adr;
    int64_t *piw = (int64_t*)iwork->ve_adr;
    if (!pa || !ps || !pu || !pvt || !pw || !piw) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ia = 0;
    int64_t is = 0;
    int64_t iu = 0;
    int64_t ivt = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*a->ndim);
    for (i = 0; i < a->ndim; i++) cnt[i] = 0;
    do {
        dgesdd_(&job, &m, &n, pa+ia, &lda, ps+is, pu+iu, &ldu, pvt+ivt, &ldvt, pw, &lwork, piw, info);
        for (i = 2; i < a->ndim; i++) {
            if (++cnt[i] < a->shape[i]) {
                ia += a->strides[i] / a->itemsize;
                is += s->strides[i-1] / s->itemsize;
                iu += u->strides[i] / u->itemsize;
                ivt += vt->strides[i] / vt->itemsize;
                break;
            }
            cnt[i] = 0;
            ia -= (a->strides[i] / a->itemsize) * (a->shape[i] - 1);
            is -= (s->strides[i-1] / s->itemsize) * (a->shape[i] - 1);
            iu -= (u->strides[i] / u->itemsize) * (a->shape[i] - 1);
            ivt -= (vt->strides[i] / vt->itemsize) * (a->shape[i] - 1);
        }
    } while(i < a->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


extern void cgesdd_(const char *job, const int64_t *m, const int64_t *n, float _Complex *pa, const int64_t *lda, float *ps, float _Complex *pu, const int64_t *ldu, float _Complex *pvt, const int64_t *ldvt, float _Complex *pw, const int64_t *lwork, float *prw, int64_t *piw, int64_t *info);
uint64_t nlcpy_svd_c(const char job, ve_array *a, ve_array *s, ve_array *u, ve_array *vt, ve_array *work, ve_array *rwork, ve_array *iwork, int64_t *info, int32_t *psw)
{
    const int64_t m = a->shape[0];
    const int64_t n = a->shape[1];
    const int64_t lda = m;
    const int64_t ldu = u->shape[0];
    const int64_t ldvt = vt->shape[0];
    const int64_t lwork = work->size;

    float _Complex *pa = (float _Complex*)a->ve_adr;
    float *ps = (float*)s->ve_adr;
    float _Complex *pu = (float _Complex*)u->ve_adr;
    float _Complex *pvt = (float _Complex*)vt->ve_adr;
    float _Complex *pw = (float _Complex*)work->ve_adr;
    float *prw = (float*)rwork->ve_adr;
    int64_t *piw = (int64_t*)iwork->ve_adr;
    if (!pa || !ps || !pu || !pvt || !pw || !prw || !piw) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ia = 0;
    int64_t is = 0;
    int64_t iu = 0;
    int64_t ivt = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*a->ndim);
    for (i = 0; i < a->ndim; i++) cnt[i] = 0;
    do {
        cgesdd_(&job, &m, &n, pa+ia, &lda, ps+is, pu+iu, &ldu, pvt+ivt, &ldvt, pw, &lwork, prw, piw, info);
        for (i = 2; i < a->ndim; i++) {
            if (++cnt[i] < a->shape[i]) {
                ia += a->strides[i] / a->itemsize;
                is += s->strides[i-1] / s->itemsize;
                iu += u->strides[i] / u->itemsize;
                ivt += vt->strides[i] / vt->itemsize;
                break;
            }
            cnt[i] = 0;
            ia -= (a->strides[i] / a->itemsize) * (a->shape[i] - 1);
            is -= (s->strides[i-1] / s->itemsize) * (a->shape[i] - 1);
            iu -= (u->strides[i] / u->itemsize) * (a->shape[i] - 1);
            ivt -= (vt->strides[i] / vt->itemsize) * (a->shape[i] - 1);
        }
    } while(i < a->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

extern void zgesdd_(const char *job, const int64_t *m, const int64_t *n, double _Complex *pa, const int64_t *lda, double *ps, double _Complex *pu, const int64_t *ldu, double _Complex *pvt, const int64_t *ldvt, double _Complex *pw, const int64_t *lwork, double *prw, int64_t *piw, int64_t *info);
uint64_t nlcpy_svd_z(const char job, ve_array *a, ve_array *s, ve_array *u, ve_array *vt, ve_array *work, ve_array *rwork, ve_array *iwork, int64_t *info, int32_t *psw)
{
    const int64_t m = a->shape[0];
    const int64_t n = a->shape[1];
    const int64_t lda = m;
    const int64_t ldu = u->shape[0];
    const int64_t ldvt = vt->shape[0];
    const int64_t lwork = work->size;

    double _Complex *pa = (double _Complex*)a->ve_adr;
    double *ps = (double*)s->ve_adr;
    double _Complex *pu = (double _Complex*)u->ve_adr;
    double _Complex *pvt = (double _Complex*)vt->ve_adr;
    double _Complex *pw = (double _Complex*)work->ve_adr;
    double *prw = (double*)rwork->ve_adr;
    int64_t *piw = (int64_t*)iwork->ve_adr;
    if (!pa || !ps || !pu || !pvt || !pw || !prw || !piw) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ia = 0;
    int64_t is = 0;
    int64_t iu = 0;
    int64_t ivt = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*a->ndim);
    for (i = 0; i < a->ndim; i++) cnt[i] = 0;
    do {
        zgesdd_(&job, &m, &n, pa+ia, &lda, ps+is, pu+iu, &ldu, pvt+ivt, &ldvt, pw, &lwork, prw, piw, info);
        for (i = 2; i < a->ndim; i++) {
            if (++cnt[i] < a->shape[i]) {
                ia += a->strides[i] / a->itemsize;
                is += s->strides[i-1] / s->itemsize;
                iu += u->strides[i] / u->itemsize;
                ivt += vt->strides[i] / vt->itemsize;
                break;
            }
            cnt[i] = 0;
            ia -= (a->strides[i] / a->itemsize) * (a->shape[i] - 1);
            is -= (s->strides[i-1] / s->itemsize) * (a->shape[i] - 1);
            iu -= (u->strides[i] / u->itemsize) * (a->shape[i] - 1);
            ivt -= (vt->strides[i] / vt->itemsize) * (a->shape[i] - 1);
        }
    } while(i < a->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_svd(int64_t job, ve_array *a, ve_array *s, ve_array *u, ve_array *vt, ve_array *work, ve_array *rwork, ve_array *iwork, int64_t *info, int32_t *psw) {
    uint64_t err = NLCPY_ERROR_OK;
    switch(a->dtype) {
    case ve_f32:
        err = nlcpy_svd_s(job, a, s, u, vt, work, iwork, info, psw);
        break;
    case ve_f64:
        err = nlcpy_svd_d(job, a, s, u, vt, work, iwork, info, psw);
        break;
    case ve_c64:
        err = nlcpy_svd_c(job, a, s, u, vt, work, rwork, iwork, info, psw);
        break;
    case ve_c128:
        err = nlcpy_svd_z(job, a, s, u, vt, work, rwork, iwork, info, psw);
        break;
    default:
        err = NLCPY_ERROR_DTYPE;
    }
    return (uint64_t)err;
}

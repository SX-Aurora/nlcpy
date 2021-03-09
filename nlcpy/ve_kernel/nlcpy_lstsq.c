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


#include "nlcpy.h"


extern void sgelsd_(const int32_t *m, const int32_t *n, const int32_t *nrhs, float *pa, const int32_t *lda, float *pb, const int32_t *ldb, float *ps, const float *rcond, int32_t *rank, float *pw, const int32_t *lwork, int32_t *piw, int32_t *info);
uint64_t nlcpy_lstsq_s(ve_array *a, ve_array *b, ve_array *s, ve_array *work, ve_array *iwork, ve_array *cond, int32_t *rank, int32_t *info, int32_t *psw)
{
    const int32_t m = a->shape[0];
    const int32_t n = a->shape[1];
    const int32_t lda = m;
    const int32_t ldb = m < n ? n : m;
    const int32_t nrhs = b->shape[1];
    const int32_t lwork = work->size;
    const float rcond = ((float*)cond->ve_adr)[0];

    float *pa = (float*)a->ve_adr;
    float *pb = (float*)b->ve_adr;
    float *pw = (float*)work->ve_adr;
    int32_t *piw = (int32_t*)iwork->ve_adr;
    float*ps = (float*)s->ve_adr;
    if (pa == NULL || pb == NULL || ps == NULL || pw == NULL || piw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    sgelsd_(&m, &n, &nrhs, pa, &lda, pb, &ldb, ps, &rcond, rank, pw, &lwork, piw, info);
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

extern void dgelsd_(const int32_t *m, const int32_t *n, const int32_t *nrhs, double *pa, const int32_t *lda, double *pb, const int32_t *ldb, double *ps, const double *rcond, int32_t *rank, double *pw, const int32_t *lwork, int32_t *piw, int32_t *info);
uint64_t nlcpy_lstsq_d(ve_array *a, ve_array *b, ve_array *s, ve_array *work, ve_array *iwork, ve_array *cond, int32_t *rank, int32_t *info, int32_t *psw)
{
    const int32_t m = a->shape[0];
    const int32_t n = a->shape[1];
    const int32_t lda = m;
    const int32_t ldb = m < n ? n : m;
    const int32_t nrhs = b->shape[1];
    const int32_t lwork = work->size;
    const double rcond = ((double*)cond->ve_adr)[0];

    double *pa = (double*)a->ve_adr;
    double *pb = (double*)b->ve_adr;
    double *pw = (double*)work->ve_adr;
    int32_t *piw = (int32_t*)iwork->ve_adr;
    double*ps = (double*)s->ve_adr;
    if (pa == NULL || pb == NULL || ps == NULL || pw == NULL || piw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    dgelsd_(&m, &n, &nrhs, pa, &lda, pb, &ldb, ps, &rcond, rank, pw, &lwork, piw, info);
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


extern void cgelsd_(const int32_t *m, const int32_t *n, const int32_t *nrhs, float _Complex *pa, const int32_t *lda, float _Complex *pb, const int32_t *ldb, float *ps, const float *rcond, int32_t *rank, float _Complex *pw, const int32_t *lwork, float *prw, int32_t *piw, int32_t *info);
uint64_t nlcpy_lstsq_c(ve_array *a, ve_array *b, ve_array *s, ve_array *work, ve_array *rwork, ve_array *iwork, ve_array *cond, int32_t *rank, int32_t *info, int32_t *psw)
{
    const int32_t m = a->shape[0];
    const int32_t n = a->shape[1];
    const int32_t lda = m;
    const int32_t ldb = m < n ? n : m;
    const int32_t nrhs = b->shape[1];
    const int32_t lwork = work->size;
    const float rcond = ((float*)cond->ve_adr)[0];

    float _Complex *pa = (float _Complex*)a->ve_adr;
    float _Complex *pb = (float _Complex*)b->ve_adr;
    float _Complex *pw = (float _Complex*)work->ve_adr;
    float *prw = (float*)rwork->ve_adr;
    int32_t* const piw = (int32_t*)iwork->ve_adr;
    float *ps = (float*)s->ve_adr;
    if (pa == NULL || pb == NULL || ps == NULL || pw == NULL || prw == NULL || piw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    cgelsd_(&m, &n, &nrhs, pa, &lda, pb, &ldb, ps, &rcond, rank, pw, &lwork, prw, piw, info);
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

extern void zgelsd_(const int32_t *m, const int32_t *n, const int32_t *nrhs, double _Complex *pa, const int32_t *lda, double _Complex *pb, const int32_t *ldb, double *ps, const double *rcond, int32_t *rank, double _Complex *pw, const int32_t *lwork, double *prw, int32_t *piw, int32_t *info);
uint64_t nlcpy_lstsq_z(ve_array *a, ve_array *b, ve_array *s, ve_array *work, ve_array *rwork, ve_array *iwork, ve_array *cond, int32_t *rank, int32_t *info, int32_t *psw)
{
    const int32_t m = a->shape[0];
    const int32_t n = a->shape[1];
    const int32_t lda = m;
    const int32_t ldb = m < n ? n : m;
    const int32_t nrhs = b->shape[1];
    const int32_t lwork = work->size;
    const double rcond = ((double*)cond->ve_adr)[0];

    double _Complex *pa = (double _Complex*)a->ve_adr;
    double _Complex *pb = (double _Complex*)b->ve_adr;
    double _Complex *pw = (double _Complex*)work->ve_adr;
    double *prw = (double*)rwork->ve_adr;
    int32_t* const piw = (int32_t*)iwork->ve_adr;
    double *ps = (double*)s->ve_adr;
    if (pa == NULL || pb == NULL || ps == NULL || pw == NULL || prw == NULL || piw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    zgelsd_(&m, &n, &nrhs, pa, &lda, pb, &ldb, ps, &rcond, rank, pw, &lwork, prw, piw, info);
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_lstsq(ve_array *a, ve_array *b, ve_array *s, ve_array *work, ve_array *rwork, ve_array *iwork, ve_array *rcond, int32_t *rank, int32_t *info, int32_t *psw) {
    uint64_t err = NLCPY_ERROR_OK;
    switch(a->dtype) {
    case ve_f32:
        err = nlcpy_lstsq_s(a, b, s, work, iwork, rcond, rank, info, psw);
        break;
    case ve_f64:
        err = nlcpy_lstsq_d(a, b, s, work, iwork, rcond, rank, info, psw);
        break;
    case ve_c64:
        err = nlcpy_lstsq_c(a, b, s, work, rwork, iwork, rcond, rank, info, psw);
        break;
    case ve_c128:
        err = nlcpy_lstsq_z(a, b, s, work, rwork, iwork, rcond, rank, info, psw);
        break;
    default:
        err = NLCPY_ERROR_DTYPE;
    }
    return (uint64_t)err;
}

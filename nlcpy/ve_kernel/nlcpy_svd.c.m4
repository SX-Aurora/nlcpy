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

include(macros.m4)dnl
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <alloca.h>
#include <assert.h>

#include "nlcpy.h"

define(<--@macro_svd_real@-->,<--@
extern void $1gesdd_(const char *job, const int64_t *m, const int64_t *n, $2 *pa, const int64_t *lda, $2 *ps, $2 *pu, const int64_t *ldu, $2 *pvt, const int64_t *ldvt, $2 *pw, const int64_t *lwork, int64_t *piw, int64_t *info);
uint64_t nlcpy_svd_$1(const char job, ve_array *a, ve_array *s, ve_array *u, ve_array *vt, ve_array *work, ve_array *iwork, int64_t *info, int32_t *psw)
{
    const int64_t m = a->shape[0];
    const int64_t n = a->shape[1];
    const int64_t lda = m;
    const int64_t ldu = u->shape[0];
    const int64_t ldvt = vt->shape[0];
    const int64_t lwork = work->size;

    $2 *pa = ($2*)a->ve_adr;
    $2 *ps = ($2*)s->ve_adr;
    $2 *pu = ($2*)u->ve_adr;
    $2 *pvt = ($2*)vt->ve_adr;
    $2 *pw = ($2*)work->ve_adr;
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
        $1gesdd_(&job, &m, &n, pa+ia, &lda, ps+is, pu+iu, &ldu, pvt+ivt, &ldvt, pw, &lwork, piw, info);
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
@-->)dnl
macro_svd_real(s, float)dnl
macro_svd_real(d, double)dnl

define(<--@macro_svd_complex@-->,<--@
extern void $1gesdd_(const char *job, const int64_t *m, const int64_t *n, $2 *pa, const int64_t *lda, $3 *ps, $2 *pu, const int64_t *ldu, $2 *pvt, const int64_t *ldvt, $2 *pw, const int64_t *lwork, $3 *prw, int64_t *piw, int64_t *info);
uint64_t nlcpy_svd_$1(const char job, ve_array *a, ve_array *s, ve_array *u, ve_array *vt, ve_array *work, ve_array *rwork, ve_array *iwork, int64_t *info, int32_t *psw)
{
    const int64_t m = a->shape[0];
    const int64_t n = a->shape[1];
    const int64_t lda = m;
    const int64_t ldu = u->shape[0];
    const int64_t ldvt = vt->shape[0];
    const int64_t lwork = work->size;

    $2 *pa = ($2*)a->ve_adr;
    $3 *ps = ($3*)s->ve_adr;
    $2 *pu = ($2*)u->ve_adr;
    $2 *pvt = ($2*)vt->ve_adr;
    $2 *pw = ($2*)work->ve_adr;
    $3 *prw = ($3*)rwork->ve_adr;
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
        $1gesdd_(&job, &m, &n, pa+ia, &lda, ps+is, pu+iu, &ldu, pvt+ivt, &ldvt, pw, &lwork, prw, piw, info);
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
@-->)dnl
macro_svd_complex(c, float _Complex, float)dnl
macro_svd_complex(z, double _Complex, double)dnl

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

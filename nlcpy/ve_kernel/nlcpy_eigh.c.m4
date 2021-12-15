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

define(<--@macro_eigh_real@-->,<--@
extern void $1syevd_(const char *jobz, const char *uplo, const int64_t *n, $2 *pa, const int64_t *lda, $2 *pw, $2 *pwork, const int64_t *lwork, int64_t *iwork, const int64_t *liwork, int64_t *info);
uint64_t nlcpy_eigh_$1(ve_array *a, ve_array *w, ve_array *work, ve_array *iwork, const char jobz, const char uplo, int64_t *info, int32_t *psw)
{
    const int64_t n = a->shape[0];
    const int64_t lda = n;
    const int64_t lwork = work->size;
    const int64_t liwork = iwork->size;

    $2 *pa = ($2*)a->ve_adr;
    $2 *pw = ($2*)w->ve_adr;
    $2 *pwork = ($2*)work->ve_adr;
    int64_t *piwork = (int64_t*)iwork->ve_adr;

    if (pa == NULL || pw == NULL || pwork == NULL || piwork == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ia = 0;
    int64_t iw = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*a->ndim);
    for (i = 0; i < a->ndim; i++) cnt[i] = 0;
    do {
        $1syevd_(&jobz, &uplo, &n, pa+ia, &lda, pw+iw, pwork, &lwork, piwork, &liwork, info);
        for (i = 2; i < a->ndim; i++) {
            if (++cnt[i] < a->shape[i]) {
                ia += a->strides[i] / a->itemsize;
                iw += w->strides[i-1] / w->itemsize;
                break;
            }
            cnt[i] = 0;
            ia -= (a->strides[i] / a->itemsize) * (a->shape[i] - 1);
            iw -= (w->strides[i-1] / w->itemsize) * (a->shape[i] - 1);
        }
    } while(i < a->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
macro_eigh_real(s, float)dnl
macro_eigh_real(d, double)dnl

define(<--@macro_eigh_complex@-->,<--@
extern void $1heevd_(const char *jobz, const char *uplo, const int64_t *n, $2 *pa, const int64_t *lda, $3 *pw, $2 *pwork, const int64_t *lwork, $3 *rwork, const int64_t *lrwork, int64_t *iwork, const int64_t *liwork, int64_t *info);
uint64_t nlcpy_eigh_$1(ve_array *a, ve_array *w, ve_array *work, ve_array *rwork, ve_array *iwork, const char jobz, const char uplo, int64_t *info, int32_t *psw)
{
    const int64_t n = a->shape[0];
    const int64_t lda = n;
    const int64_t lwork = work->size;
    const int64_t lrwork = rwork->size;
    const int64_t liwork = iwork->size;

    $2 *pa = ($2*)a->ve_adr;
    $3 *pw = ($3*)w->ve_adr;
    $2 *pwork = ($2*)work->ve_adr;
    $3 *prwork = ($3*)rwork->ve_adr;
    int64_t *piwork = (int64_t*)iwork->ve_adr;

    if (pa == NULL || pw == NULL || pwork == NULL || prwork == NULL || piwork == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ia = 0;
    int64_t iw = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*a->ndim);
    for (i = 0; i < a->ndim; i++) cnt[i] = 0;
    do {
        $1heevd_(&jobz, &uplo, &n, pa+ia, &lda, pw+iw, pwork, &lwork, prwork, &lrwork, piwork, &liwork, info);
        for (i = 2; i < a->ndim; i++) {
            if (++cnt[i] < a->shape[i]) {
                ia += a->strides[i] / a->itemsize;
                iw += w->strides[i-1] / w->itemsize;
                break;
            }
            cnt[i] = 0;
            ia -= (a->strides[i] / a->itemsize) * (a->shape[i] - 1);
            iw -= (w->strides[i-1] / w->itemsize) * (a->shape[i] - 1);
        }
    } while(i < a->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
macro_eigh_complex(c, float _Complex, float)dnl
macro_eigh_complex(z, double _Complex, double)dnl

uint64_t nlcpy_eigh(ve_array *a, ve_array *w, ve_array *work, ve_array *rwork, ve_array *iwork, int64_t jobz, int64_t uplo, int64_t *info, int32_t *psw) {
    uint64_t err = NLCPY_ERROR_OK;
    switch(a->dtype) {
    case ve_f32:
        err = nlcpy_eigh_s(a, w, work, iwork, jobz, uplo, info, psw);
        break;
    case ve_f64:
        err = nlcpy_eigh_d(a, w, work, iwork, jobz, uplo, info, psw);
        break;
    case ve_c64:
        err = nlcpy_eigh_c(a, w, work, rwork, iwork, jobz, uplo, info, psw);
        break;
    case ve_c128:
        err = nlcpy_eigh_z(a, w, work, rwork, iwork, jobz, uplo, info, psw);
        break;
    default:
        err = NLCPY_ERROR_DTYPE;
    }
    return (uint64_t)err;
}

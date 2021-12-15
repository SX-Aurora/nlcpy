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

define(<--@macro_lstsq_real@-->,<--@
extern void $1gelsd_(const int64_t *m, const int64_t *n, const int64_t *nrhs, $2 *pa, const int64_t *lda, $2 *pb, const int64_t *ldb, $2 *ps, const $2 *rcond, int64_t *rank, $2 *pw, const int64_t *lwork, int64_t *piw, int64_t *info);
uint64_t nlcpy_lstsq_$1(ve_array *a, ve_array *b, ve_array *s, ve_array *work, ve_array *iwork, ve_array *cond, int64_t *rank, int64_t *info, int32_t *psw)
{
    const int64_t m = a->shape[0];
    const int64_t n = a->shape[1];
    const int64_t lda = m;
    const int64_t ldb = m < n ? n : m;
    const int64_t nrhs = b->shape[1];
    const int64_t lwork = work->size;
    const $2 rcond = (($2*)cond->ve_adr)[0];

    $2 *pa = ($2*)a->ve_adr;
    $2 *pb = ($2*)b->ve_adr;
    $2 *pw = ($2*)work->ve_adr;
    int64_t *piw = (int64_t*)iwork->ve_adr;
    $2*ps = ($2*)s->ve_adr;
    if (pa == NULL || pb == NULL || ps == NULL || pw == NULL || piw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    $1gelsd_(&m, &n, &nrhs, pa, &lda, pb, &ldb, ps, &rcond, rank, pw, &lwork, piw, info);
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
macro_lstsq_real(s, float)dnl
macro_lstsq_real(d, double)dnl

define(<--@macro_lstsq_complex@-->,<--@
extern void $1gelsd_(const int64_t *m, const int64_t *n, const int64_t *nrhs, $2 *pa, const int64_t *lda, $2 *pb, const int64_t *ldb, $3 *ps, const $3 *rcond, int64_t *rank, $2 *pw, const int64_t *lwork, $3 *prw, int64_t *piw, int64_t *info);
uint64_t nlcpy_lstsq_$1(ve_array *a, ve_array *b, ve_array *s, ve_array *work, ve_array *rwork, ve_array *iwork, ve_array *cond, int64_t *rank, int64_t *info, int32_t *psw)
{
    const int64_t m = a->shape[0];
    const int64_t n = a->shape[1];
    const int64_t lda = m;
    const int64_t ldb = m < n ? n : m;
    const int64_t nrhs = b->shape[1];
    const int64_t lwork = work->size;
    const $3 rcond = (($3*)cond->ve_adr)[0];

    $2 *pa = ($2*)a->ve_adr;
    $2 *pb = ($2*)b->ve_adr;
    $2 *pw = ($2*)work->ve_adr;
    $3 *prw = ($3*)rwork->ve_adr;
    int64_t* const piw = (int64_t*)iwork->ve_adr;
    $3 *ps = ($3*)s->ve_adr;
    if (pa == NULL || pb == NULL || ps == NULL || pw == NULL || prw == NULL || piw == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    $1gelsd_(&m, &n, &nrhs, pa, &lda, pb, &ldb, ps, &rcond, rank, pw, &lwork, prw, piw, info);
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
macro_lstsq_complex(c, float _Complex, float)dnl
macro_lstsq_complex(z, double _Complex, double)dnl

uint64_t nlcpy_lstsq(ve_array *a, ve_array *b, ve_array *s, ve_array *work, ve_array *rwork, ve_array *iwork, ve_array *rcond, int64_t *rank, int64_t *info, int32_t *psw) {
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

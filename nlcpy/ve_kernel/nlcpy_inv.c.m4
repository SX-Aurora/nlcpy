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

define(<--@macro_inv@-->,<--@
extern void $1getrf_(const int64_t *m, const int64_t *n, $2 *pa, const int64_t *lda, int64_t *ipiv, int64_t *info);
extern void $1getri_(const int64_t *n, $2 *pa, const int64_t *lda, int64_t *ipiv, $2 *pw, const int64_t *lwork, int64_t *info);
uint64_t nlcpy_inv_$1(ve_array *a, ve_array *ipiv, ve_array *work, int64_t *info, int32_t *psw)
{
    const int64_t n = a->shape[0];
    const int64_t lda = n;
    const int64_t lwork = work->size;

    $2 *pa = ($2*)a->ve_adr;
    $2 *pw = ($2*)work->ve_adr;
    int64_t *pipiv = (int64_t*)ipiv->ve_adr;
    if (pa == NULL || pw == NULL || pipiv == NULL ) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ia = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*a->ndim);
    for (i = 0; i < a->ndim; i++) cnt[i] = 0;
    do {
        $1getrf_(&n, &n, pa+ia, &lda, pipiv, info);
        if (*info) return (uint64_t)NLCPY_ERROR_OK;
        $1getri_(&n, pa+ia, &lda, pipiv, pw, &lwork, info);
        if (*info) return (uint64_t)NLCPY_ERROR_OK;
        for (i = 2; i < a->ndim; i++) {
            if (++cnt[i] < a->shape[i]) {
                ia += a->strides[i] / a->itemsize;
                break;
            }
            cnt[i] = 0;
            ia -= (a->strides[i] / a->itemsize) * (a->shape[i] - 1);
        }
    } while(i < a->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
macro_inv(d, double)dnl
macro_inv(z, double _Complex)dnl

uint64_t nlcpy_inv(ve_array *a, ve_array *ipiv, ve_array *work, int64_t *info, int32_t *psw) {
    uint64_t err = NLCPY_ERROR_OK;
    switch(a->dtype) {
    case ve_f64:
        err = nlcpy_inv_d(a, ipiv, work, info, psw);
        break;
    case ve_c128:
        err = nlcpy_inv_z(a, ipiv, work, info, psw);
        break;
    default:
        err = NLCPY_ERROR_DTYPE;
    }
    return (uint64_t)err;
}

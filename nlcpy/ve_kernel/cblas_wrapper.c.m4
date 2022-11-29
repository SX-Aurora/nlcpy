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
#include <inc_i64/cblas.h>

define(<--@cblas_wrapper@-->,<--@
uint64_t wrapper_cblas_$1(
    ve_array *x,
    ve_array *y,
    ve_array *z,
    int32_t *psw
){
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    $2 *px = ($2 *)x->ve_adr;
    if (px == NULL) {
        px = ($2 *)nlcpy__get_scalar(x);
        if (px == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }
    $2 *py = ($2 *)y->ve_adr;
    if (py == NULL) {
        py = ($2 *)nlcpy__get_scalar(y);
        if (py == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }
    $2 *pz = ($2 *)z->ve_adr;
    if (pz == NULL) {
       return (uint64_t)NLCPY_ERROR_MEMORY;
    }
    assert(x->ndim <= 1);
    assert(y->ndim <= 1);
    assert(z->ndim <= 1);
    assert(x->size == y->size);

ifelse($3,sub,<--@
    cblas_$1(x->size, px, x->strides[0] / x->itemsize,
                        py, y->strides[0] / y->itemsize, pz);
@-->,<--@dnl
    *pz = cblas_$1(x->size, px, x->strides[0] / x->itemsize,
                            py, y->strides[0] / y->itemsize);
@-->)dnl
} /* omp single */
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
cblas_wrapper(sdot, float)dnl
cblas_wrapper(ddot, double)dnl
cblas_wrapper(cdotu_sub, float _Complex, sub)dnl
cblas_wrapper(zdotu_sub, double _Complex, sub)dnl



define(<--@cblas_wrapper@-->,<--@
uint64_t wrapper_cblas_$1(
    int64_t order,
    int64_t transA,
    int64_t transB,
    int64_t m,
    int64_t n,
    int64_t k,
    ve_array *a,
    int64_t lda,
    ve_array *b,
    int64_t ldb,
    ve_array *c,
    int64_t ldc,
    int32_t *psw
){
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    $2* const pa = ($2 *)a->ve_adr;
    $2* const pb = ($2 *)b->ve_adr;
    $2* const pc = ($2 *)c->ve_adr;
    if (pa == NULL || pb == NULL || pc == NULL) {
        return NLCPY_ERROR_MEMORY;
    }
ifelse($3,complex,<--@dnl
    const $2 _alpha = 1 + 0I;
    const $2 _beta = 0 + 0I;
    const $2 *alpha = ($2 *)&_alpha;
    const $2 *beta = ($2 *)&_beta;
@-->,<--@dnl
    const $2 alpha = ($2)1;
    const $2 beta = ($2)0;
@-->)dnl
    cblas_$1(order, transA, transB, m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);
    retrieve_fpe_flags(psw);
} /* omp single */
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
cblas_wrapper(sgemm,  float)dnl
cblas_wrapper(dgemm,  double)dnl
cblas_wrapper(cgemm,  float  _Complex, complex)dnl
cblas_wrapper(zgemm,  double _Complex, complex)dnl



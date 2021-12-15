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
#include <inc_i64/cblas.h>

define(<--@cblas_wrapper@-->,<--@
uint64_t wrapper_cblas_$1(ve_arguments *args, int32_t *psw)
{
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    ve_array *x = &(args->binary.x);
    ve_array *y = &(args->binary.y);
    ve_array *z = &(args->binary.z);

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
uint64_t wrapper_cblas_$1(ve_arguments *args, int32_t *psw)
{
    const int64_t order = args->gemm.order;
    const int64_t transA = args->gemm.transA;
    const int64_t transB = args->gemm.transB;
    const int64_t m = args->gemm.m;
    const int64_t n = args->gemm.n;
    const int64_t k = args->gemm.k;
ifelse($3,complex,<--@dnl
    const void *alpha = (void *)nlcpy__get_scalar(&(args->gemm.alpha));
    if (alpha == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
@-->,<--@dnl
    const $2 alpha = *(($2 *)nlcpy__get_scalar(&(args->gemm.alpha)));
@-->)dnl
    $2* const a = ($2 *)args->gemm.a.ve_adr;
    const int64_t lda = args->gemm.lda;
    $2* const b = ($2 *)args->gemm.b.ve_adr;
    const int64_t ldb = args->gemm.ldb;
ifelse($3,complex,<--@dnl
    const void *beta = (void *)nlcpy__get_scalar(&(args->gemm.beta));
    if (beta == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
@-->,<--@dnl
    const $2 beta = *(($2 *)nlcpy__get_scalar(&(args->gemm.beta)));
@-->)dnl
    $2* const c = ($2 *)args->gemm.c.ve_adr;
    const int64_t ldc = args->gemm.ldc;

    if (a == NULL || b == NULL || c == NULL) {
        return NLCPY_ERROR_MEMORY;
    }


#ifdef _OPENMP
    const int64_t nt = omp_get_num_threads();
    const int64_t it = omp_get_thread_num();
#else
    const int64_t nt = 1;
    const int64_t it = 0;
#endif /* _OPENMP */

    const int64_t m_s = m * it / nt;
    const int64_t m_e = m * (it + 1) / nt;
    const int64_t m_d = m_e - m_s;
    const int64_t n_s = n * it / nt;
    const int64_t n_e = n * (it + 1) / nt;
    const int64_t n_d = n_e - n_s;

    int64_t mode = 1;
    if ( n > nt ) {
        mode = 2;
    }
    int64_t iar, iac, ibr, ibc, icr, icc;
    if (transA == CblasNoTrans ) {
        iar = 1;
        iac = lda;
    } else {
        iar = lda;
        iac = 1;
    }
    if (transB == CblasNoTrans ) {
        ibr = 1;
        ibc = ldb;
    } else {
        ibr = ldb;
        ibc = 1;
    }
    if (order == CblasColMajor ) {
        icr = 1;
        icc = ldc;
    } else {
        icr = ldc;
        icc = 1;
    }

    if (order == CblasColMajor) {
        if ( mode == 1 ) {
            // split 'm'
            cblas_$1(order, transA, transB, m_d, n, k, alpha, a + m_s * iar, lda, b, ldb, beta, c + m_s * icr, ldc);
        } else {
            // split 'n'
            cblas_$1(order, transA, transB, m, n_d, k, alpha, a, lda, b + n_s * ibc, ldb, beta, c + n_s * icc, ldc);
        }
    } else {
        if ( mode == 1 ) {
            // split 'm'
            cblas_$1(order, transA, transB, m_d, n, k, alpha, a + m_s * iac, lda, b, ldb, beta, c + m_s * icr, ldc);
        } else {
            // split 'n'
            cblas_$1(order, transA, transB, m, n_d, k, alpha, a, lda, b + n_s * ibr, ldb, beta, c + n_s * icc, ldc);
        }
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
cblas_wrapper(sgemm,  float)dnl
cblas_wrapper(dgemm,  double)dnl
cblas_wrapper(cgemm,  float  _Complex, complex)dnl
cblas_wrapper(zgemm,  double _Complex, complex)dnl



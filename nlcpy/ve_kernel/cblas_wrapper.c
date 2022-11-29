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

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <alloca.h>
#include <assert.h>

#include "nlcpy.h"
#include <inc_i64/cblas.h>


uint64_t wrapper_cblas_sdot(
    ve_array *x,
    ve_array *y,
    ve_array *z,
    int32_t *psw
){
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    float *px = (float *)x->ve_adr;
    if (px == NULL) {
        px = (float *)nlcpy__get_scalar(x);
        if (px == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }
    float *py = (float *)y->ve_adr;
    if (py == NULL) {
        py = (float *)nlcpy__get_scalar(y);
        if (py == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }
    float *pz = (float *)z->ve_adr;
    if (pz == NULL) {
       return (uint64_t)NLCPY_ERROR_MEMORY;
    }
    assert(x->ndim <= 1);
    assert(y->ndim <= 1);
    assert(z->ndim <= 1);
    assert(x->size == y->size);

    *pz = cblas_sdot(x->size, px, x->strides[0] / x->itemsize,
                            py, y->strides[0] / y->itemsize);
} /* omp single */
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t wrapper_cblas_ddot(
    ve_array *x,
    ve_array *y,
    ve_array *z,
    int32_t *psw
){
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    double *px = (double *)x->ve_adr;
    if (px == NULL) {
        px = (double *)nlcpy__get_scalar(x);
        if (px == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }
    double *py = (double *)y->ve_adr;
    if (py == NULL) {
        py = (double *)nlcpy__get_scalar(y);
        if (py == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }
    double *pz = (double *)z->ve_adr;
    if (pz == NULL) {
       return (uint64_t)NLCPY_ERROR_MEMORY;
    }
    assert(x->ndim <= 1);
    assert(y->ndim <= 1);
    assert(z->ndim <= 1);
    assert(x->size == y->size);

    *pz = cblas_ddot(x->size, px, x->strides[0] / x->itemsize,
                            py, y->strides[0] / y->itemsize);
} /* omp single */
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t wrapper_cblas_cdotu_sub(
    ve_array *x,
    ve_array *y,
    ve_array *z,
    int32_t *psw
){
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    float _Complex *px = (float _Complex *)x->ve_adr;
    if (px == NULL) {
        px = (float _Complex *)nlcpy__get_scalar(x);
        if (px == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }
    float _Complex *py = (float _Complex *)y->ve_adr;
    if (py == NULL) {
        py = (float _Complex *)nlcpy__get_scalar(y);
        if (py == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }
    float _Complex *pz = (float _Complex *)z->ve_adr;
    if (pz == NULL) {
       return (uint64_t)NLCPY_ERROR_MEMORY;
    }
    assert(x->ndim <= 1);
    assert(y->ndim <= 1);
    assert(z->ndim <= 1);
    assert(x->size == y->size);


    cblas_cdotu_sub(x->size, px, x->strides[0] / x->itemsize,
                        py, y->strides[0] / y->itemsize, pz);
} /* omp single */
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t wrapper_cblas_zdotu_sub(
    ve_array *x,
    ve_array *y,
    ve_array *z,
    int32_t *psw
){
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    double _Complex *px = (double _Complex *)x->ve_adr;
    if (px == NULL) {
        px = (double _Complex *)nlcpy__get_scalar(x);
        if (px == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }
    double _Complex *py = (double _Complex *)y->ve_adr;
    if (py == NULL) {
        py = (double _Complex *)nlcpy__get_scalar(y);
        if (py == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }
    double _Complex *pz = (double _Complex *)z->ve_adr;
    if (pz == NULL) {
       return (uint64_t)NLCPY_ERROR_MEMORY;
    }
    assert(x->ndim <= 1);
    assert(y->ndim <= 1);
    assert(z->ndim <= 1);
    assert(x->size == y->size);


    cblas_zdotu_sub(x->size, px, x->strides[0] / x->itemsize,
                        py, y->strides[0] / y->itemsize, pz);
} /* omp single */
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}




uint64_t wrapper_cblas_sgemm(
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
    float* const pa = (float *)a->ve_adr;
    float* const pb = (float *)b->ve_adr;
    float* const pc = (float *)c->ve_adr;
    if (pa == NULL || pb == NULL || pc == NULL) {
        return NLCPY_ERROR_MEMORY;
    }
    const float alpha = (float)1;
    const float beta = (float)0;
    cblas_sgemm(order, transA, transB, m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);
    retrieve_fpe_flags(psw);
} /* omp single */
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t wrapper_cblas_dgemm(
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
    double* const pa = (double *)a->ve_adr;
    double* const pb = (double *)b->ve_adr;
    double* const pc = (double *)c->ve_adr;
    if (pa == NULL || pb == NULL || pc == NULL) {
        return NLCPY_ERROR_MEMORY;
    }
    const double alpha = (double)1;
    const double beta = (double)0;
    cblas_dgemm(order, transA, transB, m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);
    retrieve_fpe_flags(psw);
} /* omp single */
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t wrapper_cblas_cgemm(
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
    float  _Complex* const pa = (float  _Complex *)a->ve_adr;
    float  _Complex* const pb = (float  _Complex *)b->ve_adr;
    float  _Complex* const pc = (float  _Complex *)c->ve_adr;
    if (pa == NULL || pb == NULL || pc == NULL) {
        return NLCPY_ERROR_MEMORY;
    }
    const float  _Complex _alpha = 1 + 0I;
    const float  _Complex _beta = 0 + 0I;
    const float  _Complex *alpha = (float  _Complex *)&_alpha;
    const float  _Complex *beta = (float  _Complex *)&_beta;
    cblas_cgemm(order, transA, transB, m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);
    retrieve_fpe_flags(psw);
} /* omp single */
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t wrapper_cblas_zgemm(
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
    double _Complex* const pa = (double _Complex *)a->ve_adr;
    double _Complex* const pb = (double _Complex *)b->ve_adr;
    double _Complex* const pc = (double _Complex *)c->ve_adr;
    if (pa == NULL || pb == NULL || pc == NULL) {
        return NLCPY_ERROR_MEMORY;
    }
    const double _Complex _alpha = 1 + 0I;
    const double _Complex _beta = 0 + 0I;
    const double _Complex *alpha = (double _Complex *)&_alpha;
    const double _Complex *beta = (double _Complex *)&_beta;
    cblas_zgemm(order, transA, transB, m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);
    retrieve_fpe_flags(psw);
} /* omp single */
    return (uint64_t)NLCPY_ERROR_OK;
}



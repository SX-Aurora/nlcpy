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


#include <cblas.h>
#include "nlcpy.h"


uint64_t wrapper_cblas_sdot(ve_arguments *args, int32_t *psw)
{
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    ve_array *x = &(args->binary.x);
    ve_array *y = &(args->binary.y);
    ve_array *z = &(args->binary.z);

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

uint64_t wrapper_cblas_ddot(ve_arguments *args, int32_t *psw)
{
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    ve_array *x = &(args->binary.x);
    ve_array *y = &(args->binary.y);
    ve_array *z = &(args->binary.z);

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

uint64_t wrapper_cblas_cdotu_sub(ve_arguments *args, int32_t *psw)
{
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    ve_array *x = &(args->binary.x);
    ve_array *y = &(args->binary.y);
    ve_array *z = &(args->binary.z);

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

uint64_t wrapper_cblas_zdotu_sub(ve_arguments *args, int32_t *psw)
{
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    ve_array *x = &(args->binary.x);
    ve_array *y = &(args->binary.y);
    ve_array *z = &(args->binary.z);

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




uint64_t wrapper_cblas_sgemm(ve_arguments *args, int32_t *psw)
{
    const int32_t order = args->gemm.order;
    const int32_t transA = args->gemm.transA;
    const int32_t transB = args->gemm.transB;
    const int32_t m = args->gemm.m;
    const int32_t n = args->gemm.n;
    const int32_t k = args->gemm.k;
    const float alpha = *((float *)nlcpy__get_scalar(&(args->gemm.alpha)));
    float* const a = (float *)args->gemm.a.ve_adr;
    const int32_t lda = args->gemm.lda;
    float* const b = (float *)args->gemm.b.ve_adr;
    const int32_t ldb = args->gemm.ldb;
    const float beta = *((float *)nlcpy__get_scalar(&(args->gemm.beta)));
    float* const c = (float *)args->gemm.c.ve_adr;
    const int32_t ldc = args->gemm.ldc;

    if (a == NULL || b == NULL || c == NULL) {
        return NLCPY_ERROR_MEMORY;
    }


#ifdef _OPENMP
    const int32_t nt = omp_get_num_threads();
    const int32_t it = omp_get_thread_num();
#else
    const int32_t nt = 1;
    const int32_t it = 0;
#endif /* _OPENMP */

    const int32_t m_s = m * it / nt;
    const int32_t m_e = m * (it + 1) / nt;
    const int32_t m_d = m_e - m_s;
    const int32_t n_s = n * it / nt;
    const int32_t n_e = n * (it + 1) / nt;
    const int32_t n_d = n_e - n_s;
    
    int32_t mode = 1;
    if ( n > nt ) { 
        mode = 2;
    }
    int32_t iar, iac, ibr, ibc, icr, icc;
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
            cblas_sgemm(order, transA, transB, m_d, n, k, alpha, a + m_s * iar, lda, b, ldb, beta, c + m_s * icr, ldc);  
        } else {
            // split 'n'
            cblas_sgemm(order, transA, transB, m, n_d, k, alpha, a, lda, b + n_s * ibc, ldb, beta, c + n_s * icc, ldc);  
        }
    } else {
        if ( mode == 1 ) { 
            // split 'm'
            cblas_sgemm(order, transA, transB, m_d, n, k, alpha, a + m_s * iac, lda, b, ldb, beta, c + m_s * icr, ldc);  
        } else {
            // split 'n'
            cblas_sgemm(order, transA, transB, m, n_d, k, alpha, a, lda, b + n_s * ibr, ldb, beta, c + n_s * icc, ldc);  
        }
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t wrapper_cblas_dgemm(ve_arguments *args, int32_t *psw)
{
    const int32_t order = args->gemm.order;
    const int32_t transA = args->gemm.transA;
    const int32_t transB = args->gemm.transB;
    const int32_t m = args->gemm.m;
    const int32_t n = args->gemm.n;
    const int32_t k = args->gemm.k;
    const double alpha = *((double *)nlcpy__get_scalar(&(args->gemm.alpha)));
    double* const a = (double *)args->gemm.a.ve_adr;
    const int32_t lda = args->gemm.lda;
    double* const b = (double *)args->gemm.b.ve_adr;
    const int32_t ldb = args->gemm.ldb;
    const double beta = *((double *)nlcpy__get_scalar(&(args->gemm.beta)));
    double* const c = (double *)args->gemm.c.ve_adr;
    const int32_t ldc = args->gemm.ldc;

    if (a == NULL || b == NULL || c == NULL) {
        return NLCPY_ERROR_MEMORY;
    }


#ifdef _OPENMP
    const int32_t nt = omp_get_num_threads();
    const int32_t it = omp_get_thread_num();
#else
    const int32_t nt = 1;
    const int32_t it = 0;
#endif /* _OPENMP */

    const int32_t m_s = m * it / nt;
    const int32_t m_e = m * (it + 1) / nt;
    const int32_t m_d = m_e - m_s;
    const int32_t n_s = n * it / nt;
    const int32_t n_e = n * (it + 1) / nt;
    const int32_t n_d = n_e - n_s;
    
    int32_t mode = 1;
    if ( n > nt ) { 
        mode = 2;
    }
    int32_t iar, iac, ibr, ibc, icr, icc;
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
            cblas_dgemm(order, transA, transB, m_d, n, k, alpha, a + m_s * iar, lda, b, ldb, beta, c + m_s * icr, ldc);  
        } else {
            // split 'n'
            cblas_dgemm(order, transA, transB, m, n_d, k, alpha, a, lda, b + n_s * ibc, ldb, beta, c + n_s * icc, ldc);  
        }
    } else {
        if ( mode == 1 ) { 
            // split 'm'
            cblas_dgemm(order, transA, transB, m_d, n, k, alpha, a + m_s * iac, lda, b, ldb, beta, c + m_s * icr, ldc);  
        } else {
            // split 'n'
            cblas_dgemm(order, transA, transB, m, n_d, k, alpha, a, lda, b + n_s * ibr, ldb, beta, c + n_s * icc, ldc);  
        }
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t wrapper_cblas_cgemm(ve_arguments *args, int32_t *psw)
{
    const int32_t order = args->gemm.order;
    const int32_t transA = args->gemm.transA;
    const int32_t transB = args->gemm.transB;
    const int32_t m = args->gemm.m;
    const int32_t n = args->gemm.n;
    const int32_t k = args->gemm.k;
    const void *alpha = (void *)nlcpy__get_scalar(&(args->gemm.alpha));
    if (alpha == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float  _Complex* const a = (float  _Complex *)args->gemm.a.ve_adr;
    const int32_t lda = args->gemm.lda;
    float  _Complex* const b = (float  _Complex *)args->gemm.b.ve_adr;
    const int32_t ldb = args->gemm.ldb;
    const void *beta = (void *)nlcpy__get_scalar(&(args->gemm.beta));
    if (beta == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float  _Complex* const c = (float  _Complex *)args->gemm.c.ve_adr;
    const int32_t ldc = args->gemm.ldc;

    if (a == NULL || b == NULL || c == NULL) {
        return NLCPY_ERROR_MEMORY;
    }


#ifdef _OPENMP
    const int32_t nt = omp_get_num_threads();
    const int32_t it = omp_get_thread_num();
#else
    const int32_t nt = 1;
    const int32_t it = 0;
#endif /* _OPENMP */

    const int32_t m_s = m * it / nt;
    const int32_t m_e = m * (it + 1) / nt;
    const int32_t m_d = m_e - m_s;
    const int32_t n_s = n * it / nt;
    const int32_t n_e = n * (it + 1) / nt;
    const int32_t n_d = n_e - n_s;
    
    int32_t mode = 1;
    if ( n > nt ) { 
        mode = 2;
    }
    int32_t iar, iac, ibr, ibc, icr, icc;
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
            cblas_cgemm(order, transA, transB, m_d, n, k, alpha, a + m_s * iar, lda, b, ldb, beta, c + m_s * icr, ldc);  
        } else {
            // split 'n'
            cblas_cgemm(order, transA, transB, m, n_d, k, alpha, a, lda, b + n_s * ibc, ldb, beta, c + n_s * icc, ldc);  
        }
    } else {
        if ( mode == 1 ) { 
            // split 'm'
            cblas_cgemm(order, transA, transB, m_d, n, k, alpha, a + m_s * iac, lda, b, ldb, beta, c + m_s * icr, ldc);  
        } else {
            // split 'n'
            cblas_cgemm(order, transA, transB, m, n_d, k, alpha, a, lda, b + n_s * ibr, ldb, beta, c + n_s * icc, ldc);  
        }
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t wrapper_cblas_zgemm(ve_arguments *args, int32_t *psw)
{
    const int32_t order = args->gemm.order;
    const int32_t transA = args->gemm.transA;
    const int32_t transB = args->gemm.transB;
    const int32_t m = args->gemm.m;
    const int32_t n = args->gemm.n;
    const int32_t k = args->gemm.k;
    const void *alpha = (void *)nlcpy__get_scalar(&(args->gemm.alpha));
    if (alpha == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex* const a = (double _Complex *)args->gemm.a.ve_adr;
    const int32_t lda = args->gemm.lda;
    double _Complex* const b = (double _Complex *)args->gemm.b.ve_adr;
    const int32_t ldb = args->gemm.ldb;
    const void *beta = (void *)nlcpy__get_scalar(&(args->gemm.beta));
    if (beta == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex* const c = (double _Complex *)args->gemm.c.ve_adr;
    const int32_t ldc = args->gemm.ldc;

    if (a == NULL || b == NULL || c == NULL) {
        return NLCPY_ERROR_MEMORY;
    }


#ifdef _OPENMP
    const int32_t nt = omp_get_num_threads();
    const int32_t it = omp_get_thread_num();
#else
    const int32_t nt = 1;
    const int32_t it = 0;
#endif /* _OPENMP */

    const int32_t m_s = m * it / nt;
    const int32_t m_e = m * (it + 1) / nt;
    const int32_t m_d = m_e - m_s;
    const int32_t n_s = n * it / nt;
    const int32_t n_e = n * (it + 1) / nt;
    const int32_t n_d = n_e - n_s;
    
    int32_t mode = 1;
    if ( n > nt ) { 
        mode = 2;
    }
    int32_t iar, iac, ibr, ibc, icr, icc;
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
            cblas_zgemm(order, transA, transB, m_d, n, k, alpha, a + m_s * iar, lda, b, ldb, beta, c + m_s * icr, ldc);  
        } else {
            // split 'n'
            cblas_zgemm(order, transA, transB, m, n_d, k, alpha, a, lda, b + n_s * ibc, ldb, beta, c + n_s * icc, ldc);  
        }
    } else {
        if ( mode == 1 ) { 
            // split 'm'
            cblas_zgemm(order, transA, transB, m_d, n, k, alpha, a + m_s * iac, lda, b, ldb, beta, c + m_s * icr, ldc);  
        } else {
            // split 'n'
            cblas_zgemm(order, transA, transB, m, n_d, k, alpha, a, lda, b + n_s * ibr, ldb, beta, c + n_s * icc, ldc);  
        }
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}



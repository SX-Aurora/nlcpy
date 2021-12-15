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
#include <inc_i64/asl.h>



/****************************
 *
 *       @OPERATOR_NAME@
 *
 * **************************/


uint64_t nlcpy_argsort_bool(ve_array *val, ve_array *idx, int32_t *psw)
{
    asl_error_t asl_err;
    asl_sort_t sort;
    int32_t *pval = (int32_t *)nlcpy__get_ptr(val);
    if (pval == NULL) return NLCPY_ERROR_MEMORY;
    int64_t *pidx = (int64_t *)nlcpy__get_ptr(idx);
    if (pidx == NULL) return NLCPY_ERROR_MEMORY;

/////////
// 0-d //
/////////
    if (val->ndim == 0) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    pval[0] = 0;
} /* omp single */

/////////
// 1-d //
/////////
    } else if (val->ndim == 1) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        const uint64_t ival0 = val->strides[0] / val->itemsize;
        const uint64_t iidx0 = idx->strides[0] / idx->itemsize;
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* create sorter */
        asl_err = asl_sort_create_i32(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, val->size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* execute sort */
        asl_err = asl_sort_execute_i32(sort, val->size, pval, ASL_NULL, ASL_NULL, pidx);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */

/////////
// N-d //
/////////
    } else if (val->ndim > 1 && val->ndim <= NLCPY_MAXNDIM){
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */
        int64_t *cnt_val = (int64_t*)alloca(sizeof(int64_t) * val->ndim);
        nlcpy__reset_coords(cnt_val, val->ndim);
        int64_t i, j, k;
        int64_t n_inner = val->ndim - 1;
        int64_t n_outer = 0;
        uint64_t ival = 0;
        uint64_t iidx = 0;
        const uint64_t ival0 = val->strides[n_inner] / val->itemsize;
        const uint64_t iidx0 = idx->strides[n_inner] / idx->itemsize;
        const uint64_t sort_size = val->shape[n_inner];
        const int64_t len = val->shape[n_outer];
        const int64_t cnt_s = len * it / nt;
        const int64_t cnt_e = len * (it + 1) / nt;
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
{
        /* create sorter */
        asl_err = asl_sort_create_i32(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp critical */
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, sort_size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;

        for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
            ival = cnt * val->strides[n_outer] / val->itemsize;
            iidx = cnt * idx->strides[n_outer] / idx->itemsize;
            for (;;) {
                /* execute sort */
                asl_err = asl_sort_execute_i32(sort, sort_size, &(pval[ival]), ASL_NULL, ASL_NULL, &(pidx[iidx]));
                if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
                // set next index
                for (k = n_inner-1; k >= 1; k--) {
                    if (++cnt_val[k] < val->shape[k]) {
                        ival += val->strides[k] / val->itemsize;
                        iidx += idx->strides[k] / idx->itemsize;
                        break;
                    }
                    cnt_val[k] = 0;
                    ival -= (val->strides[k] / val->itemsize) * (val->shape[k] - 1);
                    iidx -= (idx->strides[k] / idx->itemsize) * (idx->shape[k] - 1);
                }
                if (k < 1) break;
            }
        }
#ifdef _OPENMP
#pragma omp barrier
#endif /* _OPENMP */
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif /* _OPENMP */
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    /* restore thread count */
    asl_err = asl_library_set_thread_count(nt);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
}

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_argsort_i32(ve_array *val, ve_array *idx, int32_t *psw)
{
    asl_error_t asl_err;
    asl_sort_t sort;
    int32_t *pval = (int32_t *)nlcpy__get_ptr(val);
    if (pval == NULL) return NLCPY_ERROR_MEMORY;
    int64_t *pidx = (int64_t *)nlcpy__get_ptr(idx);
    if (pidx == NULL) return NLCPY_ERROR_MEMORY;

/////////
// 0-d //
/////////
    if (val->ndim == 0) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    pval[0] = 0;
} /* omp single */

/////////
// 1-d //
/////////
    } else if (val->ndim == 1) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        const uint64_t ival0 = val->strides[0] / val->itemsize;
        const uint64_t iidx0 = idx->strides[0] / idx->itemsize;
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* create sorter */
        asl_err = asl_sort_create_i32(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, val->size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* execute sort */
        asl_err = asl_sort_execute_i32(sort, val->size, pval, ASL_NULL, ASL_NULL, pidx);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */

/////////
// N-d //
/////////
    } else if (val->ndim > 1 && val->ndim <= NLCPY_MAXNDIM){
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */
        int64_t *cnt_val = (int64_t*)alloca(sizeof(int64_t) * val->ndim);
        nlcpy__reset_coords(cnt_val, val->ndim);
        int64_t i, j, k;
        int64_t n_inner = val->ndim - 1;
        int64_t n_outer = 0;
        uint64_t ival = 0;
        uint64_t iidx = 0;
        const uint64_t ival0 = val->strides[n_inner] / val->itemsize;
        const uint64_t iidx0 = idx->strides[n_inner] / idx->itemsize;
        const uint64_t sort_size = val->shape[n_inner];
        const int64_t len = val->shape[n_outer];
        const int64_t cnt_s = len * it / nt;
        const int64_t cnt_e = len * (it + 1) / nt;
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
{
        /* create sorter */
        asl_err = asl_sort_create_i32(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp critical */
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, sort_size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;

        for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
            ival = cnt * val->strides[n_outer] / val->itemsize;
            iidx = cnt * idx->strides[n_outer] / idx->itemsize;
            for (;;) {
                /* execute sort */
                asl_err = asl_sort_execute_i32(sort, sort_size, &(pval[ival]), ASL_NULL, ASL_NULL, &(pidx[iidx]));
                if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
                // set next index
                for (k = n_inner-1; k >= 1; k--) {
                    if (++cnt_val[k] < val->shape[k]) {
                        ival += val->strides[k] / val->itemsize;
                        iidx += idx->strides[k] / idx->itemsize;
                        break;
                    }
                    cnt_val[k] = 0;
                    ival -= (val->strides[k] / val->itemsize) * (val->shape[k] - 1);
                    iidx -= (idx->strides[k] / idx->itemsize) * (idx->shape[k] - 1);
                }
                if (k < 1) break;
            }
        }
#ifdef _OPENMP
#pragma omp barrier
#endif /* _OPENMP */
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif /* _OPENMP */
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    /* restore thread count */
    asl_err = asl_library_set_thread_count(nt);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
}

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_argsort_i64(ve_array *val, ve_array *idx, int32_t *psw)
{
    asl_error_t asl_err;
    asl_sort_t sort;
    int64_t *pval = (int64_t *)nlcpy__get_ptr(val);
    if (pval == NULL) return NLCPY_ERROR_MEMORY;
    int64_t *pidx = (int64_t *)nlcpy__get_ptr(idx);
    if (pidx == NULL) return NLCPY_ERROR_MEMORY;

/////////
// 0-d //
/////////
    if (val->ndim == 0) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    pval[0] = 0;
} /* omp single */

/////////
// 1-d //
/////////
    } else if (val->ndim == 1) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        const uint64_t ival0 = val->strides[0] / val->itemsize;
        const uint64_t iidx0 = idx->strides[0] / idx->itemsize;
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* create sorter */
        asl_err = asl_sort_create_i64(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, val->size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* execute sort */
        asl_err = asl_sort_execute_i64(sort, val->size, pval, ASL_NULL, ASL_NULL, pidx);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */

/////////
// N-d //
/////////
    } else if (val->ndim > 1 && val->ndim <= NLCPY_MAXNDIM){
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */
        int64_t *cnt_val = (int64_t*)alloca(sizeof(int64_t) * val->ndim);
        nlcpy__reset_coords(cnt_val, val->ndim);
        int64_t i, j, k;
        int64_t n_inner = val->ndim - 1;
        int64_t n_outer = 0;
        uint64_t ival = 0;
        uint64_t iidx = 0;
        const uint64_t ival0 = val->strides[n_inner] / val->itemsize;
        const uint64_t iidx0 = idx->strides[n_inner] / idx->itemsize;
        const uint64_t sort_size = val->shape[n_inner];
        const int64_t len = val->shape[n_outer];
        const int64_t cnt_s = len * it / nt;
        const int64_t cnt_e = len * (it + 1) / nt;
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
{
        /* create sorter */
        asl_err = asl_sort_create_i64(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp critical */
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, sort_size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;

        for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
            ival = cnt * val->strides[n_outer] / val->itemsize;
            iidx = cnt * idx->strides[n_outer] / idx->itemsize;
            for (;;) {
                /* execute sort */
                asl_err = asl_sort_execute_i64(sort, sort_size, &(pval[ival]), ASL_NULL, ASL_NULL, &(pidx[iidx]));
                if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
                // set next index
                for (k = n_inner-1; k >= 1; k--) {
                    if (++cnt_val[k] < val->shape[k]) {
                        ival += val->strides[k] / val->itemsize;
                        iidx += idx->strides[k] / idx->itemsize;
                        break;
                    }
                    cnt_val[k] = 0;
                    ival -= (val->strides[k] / val->itemsize) * (val->shape[k] - 1);
                    iidx -= (idx->strides[k] / idx->itemsize) * (idx->shape[k] - 1);
                }
                if (k < 1) break;
            }
        }
#ifdef _OPENMP
#pragma omp barrier
#endif /* _OPENMP */
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif /* _OPENMP */
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    /* restore thread count */
    asl_err = asl_library_set_thread_count(nt);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
}

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_argsort_u32(ve_array *val, ve_array *idx, int32_t *psw)
{
    asl_error_t asl_err;
    asl_sort_t sort;
    uint32_t *pval = (uint32_t *)nlcpy__get_ptr(val);
    if (pval == NULL) return NLCPY_ERROR_MEMORY;
    int64_t *pidx = (int64_t *)nlcpy__get_ptr(idx);
    if (pidx == NULL) return NLCPY_ERROR_MEMORY;

/////////
// 0-d //
/////////
    if (val->ndim == 0) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    pval[0] = 0;
} /* omp single */

/////////
// 1-d //
/////////
    } else if (val->ndim == 1) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        const uint64_t ival0 = val->strides[0] / val->itemsize;
        const uint64_t iidx0 = idx->strides[0] / idx->itemsize;
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* create sorter */
        asl_err = asl_sort_create_u32(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, val->size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* execute sort */
        asl_err = asl_sort_execute_u32(sort, val->size, pval, ASL_NULL, ASL_NULL, pidx);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */

/////////
// N-d //
/////////
    } else if (val->ndim > 1 && val->ndim <= NLCPY_MAXNDIM){
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */
        int64_t *cnt_val = (int64_t*)alloca(sizeof(int64_t) * val->ndim);
        nlcpy__reset_coords(cnt_val, val->ndim);
        int64_t i, j, k;
        int64_t n_inner = val->ndim - 1;
        int64_t n_outer = 0;
        uint64_t ival = 0;
        uint64_t iidx = 0;
        const uint64_t ival0 = val->strides[n_inner] / val->itemsize;
        const uint64_t iidx0 = idx->strides[n_inner] / idx->itemsize;
        const uint64_t sort_size = val->shape[n_inner];
        const int64_t len = val->shape[n_outer];
        const int64_t cnt_s = len * it / nt;
        const int64_t cnt_e = len * (it + 1) / nt;
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
{
        /* create sorter */
        asl_err = asl_sort_create_u32(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp critical */
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, sort_size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;

        for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
            ival = cnt * val->strides[n_outer] / val->itemsize;
            iidx = cnt * idx->strides[n_outer] / idx->itemsize;
            for (;;) {
                /* execute sort */
                asl_err = asl_sort_execute_u32(sort, sort_size, &(pval[ival]), ASL_NULL, ASL_NULL, &(pidx[iidx]));
                if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
                // set next index
                for (k = n_inner-1; k >= 1; k--) {
                    if (++cnt_val[k] < val->shape[k]) {
                        ival += val->strides[k] / val->itemsize;
                        iidx += idx->strides[k] / idx->itemsize;
                        break;
                    }
                    cnt_val[k] = 0;
                    ival -= (val->strides[k] / val->itemsize) * (val->shape[k] - 1);
                    iidx -= (idx->strides[k] / idx->itemsize) * (idx->shape[k] - 1);
                }
                if (k < 1) break;
            }
        }
#ifdef _OPENMP
#pragma omp barrier
#endif /* _OPENMP */
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif /* _OPENMP */
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    /* restore thread count */
    asl_err = asl_library_set_thread_count(nt);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
}

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_argsort_u64(ve_array *val, ve_array *idx, int32_t *psw)
{
    asl_error_t asl_err;
    asl_sort_t sort;
    uint64_t *pval = (uint64_t *)nlcpy__get_ptr(val);
    if (pval == NULL) return NLCPY_ERROR_MEMORY;
    int64_t *pidx = (int64_t *)nlcpy__get_ptr(idx);
    if (pidx == NULL) return NLCPY_ERROR_MEMORY;

/////////
// 0-d //
/////////
    if (val->ndim == 0) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    pval[0] = 0;
} /* omp single */

/////////
// 1-d //
/////////
    } else if (val->ndim == 1) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        const uint64_t ival0 = val->strides[0] / val->itemsize;
        const uint64_t iidx0 = idx->strides[0] / idx->itemsize;
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* create sorter */
        asl_err = asl_sort_create_u64(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, val->size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* execute sort */
        asl_err = asl_sort_execute_u64(sort, val->size, pval, ASL_NULL, ASL_NULL, pidx);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */

/////////
// N-d //
/////////
    } else if (val->ndim > 1 && val->ndim <= NLCPY_MAXNDIM){
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */
        int64_t *cnt_val = (int64_t*)alloca(sizeof(int64_t) * val->ndim);
        nlcpy__reset_coords(cnt_val, val->ndim);
        int64_t i, j, k;
        int64_t n_inner = val->ndim - 1;
        int64_t n_outer = 0;
        uint64_t ival = 0;
        uint64_t iidx = 0;
        const uint64_t ival0 = val->strides[n_inner] / val->itemsize;
        const uint64_t iidx0 = idx->strides[n_inner] / idx->itemsize;
        const uint64_t sort_size = val->shape[n_inner];
        const int64_t len = val->shape[n_outer];
        const int64_t cnt_s = len * it / nt;
        const int64_t cnt_e = len * (it + 1) / nt;
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
{
        /* create sorter */
        asl_err = asl_sort_create_u64(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp critical */
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, sort_size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;

        for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
            ival = cnt * val->strides[n_outer] / val->itemsize;
            iidx = cnt * idx->strides[n_outer] / idx->itemsize;
            for (;;) {
                /* execute sort */
                asl_err = asl_sort_execute_u64(sort, sort_size, &(pval[ival]), ASL_NULL, ASL_NULL, &(pidx[iidx]));
                if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
                // set next index
                for (k = n_inner-1; k >= 1; k--) {
                    if (++cnt_val[k] < val->shape[k]) {
                        ival += val->strides[k] / val->itemsize;
                        iidx += idx->strides[k] / idx->itemsize;
                        break;
                    }
                    cnt_val[k] = 0;
                    ival -= (val->strides[k] / val->itemsize) * (val->shape[k] - 1);
                    iidx -= (idx->strides[k] / idx->itemsize) * (idx->shape[k] - 1);
                }
                if (k < 1) break;
            }
        }
#ifdef _OPENMP
#pragma omp barrier
#endif /* _OPENMP */
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif /* _OPENMP */
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    /* restore thread count */
    asl_err = asl_library_set_thread_count(nt);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
}

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_argsort_f32(ve_array *val, ve_array *idx, int32_t *psw)
{
    asl_error_t asl_err;
    asl_sort_t sort;
    float *pval = (float *)nlcpy__get_ptr(val);
    if (pval == NULL) return NLCPY_ERROR_MEMORY;
    int64_t *pidx = (int64_t *)nlcpy__get_ptr(idx);
    if (pidx == NULL) return NLCPY_ERROR_MEMORY;

/////////
// 0-d //
/////////
    if (val->ndim == 0) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    pval[0] = 0;
} /* omp single */

/////////
// 1-d //
/////////
    } else if (val->ndim == 1) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        const uint64_t ival0 = val->strides[0] / val->itemsize;
        const uint64_t iidx0 = idx->strides[0] / idx->itemsize;
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* create sorter */
        asl_err = asl_sort_create_s(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, val->size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* execute sort */
        asl_err = asl_sort_execute_s(sort, val->size, pval, ASL_NULL, ASL_NULL, pidx);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */

/////////
// N-d //
/////////
    } else if (val->ndim > 1 && val->ndim <= NLCPY_MAXNDIM){
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */
        int64_t *cnt_val = (int64_t*)alloca(sizeof(int64_t) * val->ndim);
        nlcpy__reset_coords(cnt_val, val->ndim);
        int64_t i, j, k;
        int64_t n_inner = val->ndim - 1;
        int64_t n_outer = 0;
        uint64_t ival = 0;
        uint64_t iidx = 0;
        const uint64_t ival0 = val->strides[n_inner] / val->itemsize;
        const uint64_t iidx0 = idx->strides[n_inner] / idx->itemsize;
        const uint64_t sort_size = val->shape[n_inner];
        const int64_t len = val->shape[n_outer];
        const int64_t cnt_s = len * it / nt;
        const int64_t cnt_e = len * (it + 1) / nt;
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
{
        /* create sorter */
        asl_err = asl_sort_create_s(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp critical */
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, sort_size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;

        for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
            ival = cnt * val->strides[n_outer] / val->itemsize;
            iidx = cnt * idx->strides[n_outer] / idx->itemsize;
            for (;;) {
                /* execute sort */
                asl_err = asl_sort_execute_s(sort, sort_size, &(pval[ival]), ASL_NULL, ASL_NULL, &(pidx[iidx]));
                if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
                // set next index
                for (k = n_inner-1; k >= 1; k--) {
                    if (++cnt_val[k] < val->shape[k]) {
                        ival += val->strides[k] / val->itemsize;
                        iidx += idx->strides[k] / idx->itemsize;
                        break;
                    }
                    cnt_val[k] = 0;
                    ival -= (val->strides[k] / val->itemsize) * (val->shape[k] - 1);
                    iidx -= (idx->strides[k] / idx->itemsize) * (idx->shape[k] - 1);
                }
                if (k < 1) break;
            }
        }
#ifdef _OPENMP
#pragma omp barrier
#endif /* _OPENMP */
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif /* _OPENMP */
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    /* restore thread count */
    asl_err = asl_library_set_thread_count(nt);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
}

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_argsort_f64(ve_array *val, ve_array *idx, int32_t *psw)
{
    asl_error_t asl_err;
    asl_sort_t sort;
    double *pval = (double *)nlcpy__get_ptr(val);
    if (pval == NULL) return NLCPY_ERROR_MEMORY;
    int64_t *pidx = (int64_t *)nlcpy__get_ptr(idx);
    if (pidx == NULL) return NLCPY_ERROR_MEMORY;

/////////
// 0-d //
/////////
    if (val->ndim == 0) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    pval[0] = 0;
} /* omp single */

/////////
// 1-d //
/////////
    } else if (val->ndim == 1) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        const uint64_t ival0 = val->strides[0] / val->itemsize;
        const uint64_t iidx0 = idx->strides[0] / idx->itemsize;
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* create sorter */
        asl_err = asl_sort_create_d(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, val->size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* execute sort */
        asl_err = asl_sort_execute_d(sort, val->size, pval, ASL_NULL, ASL_NULL, pidx);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */

/////////
// N-d //
/////////
    } else if (val->ndim > 1 && val->ndim <= NLCPY_MAXNDIM){
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */
        int64_t *cnt_val = (int64_t*)alloca(sizeof(int64_t) * val->ndim);
        nlcpy__reset_coords(cnt_val, val->ndim);
        int64_t i, j, k;
        int64_t n_inner = val->ndim - 1;
        int64_t n_outer = 0;
        uint64_t ival = 0;
        uint64_t iidx = 0;
        const uint64_t ival0 = val->strides[n_inner] / val->itemsize;
        const uint64_t iidx0 = idx->strides[n_inner] / idx->itemsize;
        const uint64_t sort_size = val->shape[n_inner];
        const int64_t len = val->shape[n_outer];
        const int64_t cnt_s = len * it / nt;
        const int64_t cnt_e = len * (it + 1) / nt;
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        /* set thread count */
        asl_err = asl_library_set_thread_count(1);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp single */
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
{
        /* create sorter */
        asl_err = asl_sort_create_d(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
} /* omp critical */
        /* preallocate */
        asl_err = asl_sort_preallocate(sort, sort_size);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        /* set strides */
        asl_err = asl_sort_set_input_key_long_stride(sort, ival0);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;

        for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
            ival = cnt * val->strides[n_outer] / val->itemsize;
            iidx = cnt * idx->strides[n_outer] / idx->itemsize;
            for (;;) {
                /* execute sort */
                asl_err = asl_sort_execute_d(sort, sort_size, &(pval[ival]), ASL_NULL, ASL_NULL, &(pidx[iidx]));
                if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
                // set next index
                for (k = n_inner-1; k >= 1; k--) {
                    if (++cnt_val[k] < val->shape[k]) {
                        ival += val->strides[k] / val->itemsize;
                        iidx += idx->strides[k] / idx->itemsize;
                        break;
                    }
                    cnt_val[k] = 0;
                    ival -= (val->strides[k] / val->itemsize) * (val->shape[k] - 1);
                    iidx -= (idx->strides[k] / idx->itemsize) * (idx->shape[k] - 1);
                }
                if (k < 1) break;
            }
        }
#ifdef _OPENMP
#pragma omp barrier
#endif /* _OPENMP */
        /* destroy sorter */
        asl_err = asl_sort_destroy(sort);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif /* _OPENMP */
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
    /* restore thread count */
    asl_err = asl_library_set_thread_count(nt);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
}

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}



uint64_t nlcpy_argsort(ve_arguments *args, int32_t *psw)
{
    ve_array *val = &(args->unary.x);
    ve_array *idx = &(args->unary.z);
    uint64_t err = NLCPY_ERROR_OK;

    switch (val->dtype) {
    case ve_bool: err = nlcpy_argsort_bool (val, idx, psw); break;
    case ve_i32:  err = nlcpy_argsort_i32 (val, idx, psw); break;
    case ve_i64:  err = nlcpy_argsort_i64 (val, idx, psw); break;
    case ve_u32:  err = nlcpy_argsort_u32 (val, idx, psw); break;
    case ve_u64:  err = nlcpy_argsort_u64 (val, idx, psw); break;
    case ve_f32:  err = nlcpy_argsort_f32 (val, idx, psw); break;
    case ve_f64:  err = nlcpy_argsort_f64 (val, idx, psw); break;
    default: err = NLCPY_ERROR_DTYPE;
    }

    return (uint64_t)err;
}

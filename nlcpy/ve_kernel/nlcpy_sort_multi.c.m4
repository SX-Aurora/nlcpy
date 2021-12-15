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

include(macros.m4)dnl

/****************************
 *
 *       @OPERATOR_NAME@
 *
 * **************************/

define(<--@macro_asl_sort_multi@-->,<--@
uint64_t nlcpy_sort_multi_$1(ve_array *x, ve_array *y, ve_array *w, int32_t stable, int32_t *psw)
{
    asl_error_t asl_err;
    asl_sort_t sort;
    asl_sortalgorithm_t alg = stable ? ASL_SORTALGORITHM_AUTO_STABLE : ASL_SORTALGORITHM_AUTO;
    $2 *px = ($2 *)nlcpy__get_ptr(x);
    $2 *py = ($2 *)nlcpy__get_ptr(y);
    int64_t *pw = (int64_t *)nlcpy__get_ptr(w);
    if (px == NULL || py == NULL || pw == NULL) return NLCPY_ERROR_MEMORY;

    int64_t i, j, k;
    uint64_t ix0 = x->strides[0] / x->itemsize;
    uint64_t ix1 = x->strides[1] / x->itemsize;
    uint64_t iy0 = y->strides[0] / y->itemsize;
    uint64_t iy1 = y->strides[1] / y->itemsize;
    uint64_t sort_size = x->shape[0];

    /* create sorter */
    asl_err = asl_sort_create_$3(&sort, ASL_SORTORDER_ASCENDING, alg);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;

    /* preallocate */
    asl_err = asl_sort_preallocate(sort, sort_size);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;

    asl_err = asl_sort_set_input_key_long_stride(sort, ix0);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;

    asl_int_t ix = (x->shape[1] - 1) * ix1;
    asl_int_t iy = (y->shape[1] - 2) * iy1;
    asl_err = asl_sort_execute_$3(sort, sort_size, px + ix, ASL_NULL, ASL_NULL, pw);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
    ix -= ix1;
    for (j = 0; j < sort_size; j++) {
        py[iy + j * iy0] = px[ix + pw[j] * ix0];
    }
    asl_err = asl_sort_set_input_key_long_stride(sort, iy0);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
    for (int64_t i = x->shape[1] - 2; i > 0; i--) {
        asl_err = asl_sort_execute_$3(sort, sort_size, py + iy, pw, ASL_NULL, pw);
        if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
        ix -= ix1;
        iy -= iy1;
        for (j = 0; j < sort_size; j++) {
            py[iy + j * iy0] = px[ix + pw[j] * ix0];
        }
    }
    asl_err = asl_sort_set_output_key_long_stride(sort, iy0);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
    asl_err = asl_sort_execute_$3(sort, sort_size, py + iy, pw, py + iy, pw);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;
    for (i = 1; i < x->shape[1]; i++) {
        ix += ix1;
        iy += iy1;
        for (j = 0; j < sort_size; j++) {
            py[iy + j * iy0] = px[ix + pw[j] * ix0];
        }
    }
    /* destroy sorter */
    asl_err = asl_sort_destroy(sort);
    if (asl_err != ASL_ERROR_OK) return NLCPY_ERROR_ASL;

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

@-->)dnl
macro_asl_sort_multi(bool,int32_t,i32)dnl
macro_asl_sort_multi(i32,int32_t,i32)dnl
macro_asl_sort_multi(i64,int64_t,i64)dnl
macro_asl_sort_multi(u32,uint32_t,u32)dnl
macro_asl_sort_multi(u64,uint64_t,u64)dnl
macro_asl_sort_multi(f32,float,s)dnl
macro_asl_sort_multi(f64,double,d)dnl


uint64_t nlcpy_sort_multi(ve_arguments *args, int32_t *psw)
{
    ve_array *x = &(args->sort_multi.x);
    ve_array *y = &(args->sort_multi.y);
    ve_array *w = &(args->sort_multi.w);
    int32_t stable = args->sort_multi.stable;
    uint64_t err = NLCPY_ERROR_OK;

#ifdef _OPENMP
#pragma omp single
#endif
{
    assert(x->ndim == 2 && y->ndim == 2 && w->ndim == 1);
    switch (x->dtype) {
    case ve_bool: err = nlcpy_sort_multi_bool (x, y, w, stable, psw); break;
    case ve_i32:  err = nlcpy_sort_multi_i32 (x, y, w, stable, psw); break;
    case ve_i64:  err = nlcpy_sort_multi_i64 (x, y, w, stable, psw); break;
    case ve_u32:  err = nlcpy_sort_multi_u32 (x, y, w, stable, psw); break;
    case ve_u64:  err = nlcpy_sort_multi_u64 (x, y, w, stable, psw); break;
    case ve_f32:  err = nlcpy_sort_multi_f32 (x, y, w, stable, psw); break;
    case ve_f64:  err = nlcpy_sort_multi_f64 (x, y, w, stable, psw); break;
    default: err = NLCPY_ERROR_DTYPE;
    }
}
    return (uint64_t)err;
}

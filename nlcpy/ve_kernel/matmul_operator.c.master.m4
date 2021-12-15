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
@#include <stdio.h>
@#include <stdint.h>
@#include <stdbool.h>
@#include <stdlib.h>
@#include <limits.h>
@#include <alloca.h>
@#include <assert.h>

@#include "nlcpy.h"

/****************************
 *
 *       Matrix-matrix multiplication
 *
 * **************************/

#define_switch (x->dtype:i32,i64,u32,u64,f32,f64,c64,c128@y->dtype:i32,i64,u32,u64,f32,f64,c64,c128)

define(<--@macro_binary_operator@-->,<--@
uint64_t FILENAME_$1(ve_array *x, ve_array *y, ve_array *z, int32_t *psw)
{
    int64_t i, j, k;
    int32_t corder = (z->is_c_contiguous) ? 1 : 0;
#begin_switch

    @TYPE1@ *px = (@TYPE1@*)(x->ve_adr);
    @TYPE2@ *py = (@TYPE2@*)(y->ve_adr);
    if (px == NULL || py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    int64_t i0x,i1x,i0y,i1y;
    int64_t m,kx,ky,n;
    if       (x->ndim==2){
        i0x = x->strides[1]/sizeof(@TYPE1@);
        i1x = x->strides[0]/sizeof(@TYPE1@);
        m   = x->shape[0];
        kx  = x->shape[1];
    } else if (x->ndim==1){
        i0x = x->strides[0]/sizeof(@TYPE1@);
        i1x = 0;
        m   = 1;
        kx  = x->shape[0];
    } else {
        assert(x->ndim==2||x->ndim==1);
    }
    if       (y->ndim==2){
        i0y = y->strides[1]/sizeof(@TYPE2@);
        i1y = y->strides[0]/sizeof(@TYPE2@);
        ky  = y->shape[0];
        n   = y->shape[1];
    }else if (y->ndim==1){
        i0y = 0;
        i1y = y->strides[0]/sizeof(@TYPE2@);
        ky= y->shape[0];
        n = 1;
    } else {
        assert(y->ndim==2||y->ndim==1);
    }
    assert(kx==ky);

    $2 *pz = ($2 *)z->ve_adr;

    int64_t i0z,i1z;
    i0z = (corder) ? 1 : m;
    i1z = (corder) ? n : 1;

@#ifdef _OPENMP
    const int nt = omp_get_num_threads();
    const int it = omp_get_thread_num();
@#else
    const int nt = 1;
    const int it = 0;
@#endif /* _OPENMP */
    const int64_t len1 = m * n;
    const int64_t cnt1_s = len1 * it / nt;
    const int64_t cnt1_e = len1 * (it + 1) / nt;
    const int64_t n_outer = 0;
    const int64_t len2 = n;
    const int64_t cnt2_s = len2 * it / nt;
    const int64_t cnt2_e = len2 * (it + 1) / nt;

    for (i = cnt1_s; i < cnt1_e; i++) {
        pz[i] = ($2)0;
    }
@#ifdef _OPENMP
@#pragma omp barrier
@#endif
    for (j = cnt2_s; j < cnt2_e; j++) {
        for (k = 0; k < kx; k++) {
            for (i = 0; i < m; i++) {
                pz[j*i0z+i*i1z] += ($2)(($2)px[k*i0x+i*i1x] * ($2)py[j*i0y+k*i1y]);
            }
        }
    }

#end_switch
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


@-->)dnl
macro_binary_operator(i32,  int32_t,        ve_i32)dnl
macro_binary_operator(i64,  int64_t,        ve_i64)dnl
macro_binary_operator(u32,  uint32_t,       ve_u32)dnl
macro_binary_operator(u64,  uint64_t,       ve_u64)dnl
macro_binary_operator(f32,  float,          ve_f32)dnl
macro_binary_operator(f64,  double,         ve_f64)dnl
macro_binary_operator(c64,  float _Complex, ve_c64)dnl
macro_binary_operator(c128, double _Complex,ve_c128)dnl


uint64_t FILENAME(ve_arguments *args, int32_t *psw)
//ve_array *x, ve_array *y, ve_array *z, int32_t corder, int32_t *psw)
{
    ve_array *x = &(args->binary.x);
    ve_array *y = &(args->binary.y);
    ve_array *z = &(args->binary.z);
    assert(x->ndim<=2&&y->ndim<=2);

    uint64_t err = NLCPY_ERROR_OK;
    switch (z->dtype) {
    case ve_i32:  err = FILENAME_i32 (x, y, z, psw); break;
    case ve_i64:  err = FILENAME_i64 (x, y, z, psw); break;
    case ve_u32:  err = FILENAME_u32 (x, y, z, psw); break;
    case ve_u64:  err = FILENAME_u64 (x, y, z, psw); break;
    case ve_f32:  err = FILENAME_f32 (x, y, z, psw); break;
    case ve_f64:  err = FILENAME_f64 (x, y, z, psw); break;
    case ve_c64:  err = FILENAME_c64 (x, y, z, psw); break;
    case ve_c128: err = FILENAME_c128(x, y, z, psw); break;
    default: err = NLCPY_ERROR_DTYPE;
    }

    return (uint64_t)err;
}

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
@#include "nlcpy.h"

#define_switch (x->dtype @ y->dtype)


define(<--@macro_binary_operator@-->,<--@
uint64_t FILENAME_$1(ve_array *x, ve_array *y, ve_array *z, int32_t *psw)
{
#begin_switch
    $2 *pz = ($2 *)z->ve_adr;
    if (pz == NULL) {
        return NLCPY_ERROR_MEMORY;
    }
    @TYPE1@ *px = (@TYPE1@ *)x->ve_adr;
    if  (px == NULL) {
        px = (@TYPE1@ *)nlcpy__get_scalar(x);
        if (px == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }
    @TYPE2@ *py = (@TYPE2@ *)y->ve_adr;
    if  (py == NULL) {
        py = (@TYPE2@ *)nlcpy__get_scalar(y);
        if (px == NULL) {
            return NLCPY_ERROR_MEMORY;
        }
    }

    pz[0] = 0;

/////////
// 0-d //
/////////
    if (x->ndim == 0) {
        @BINARY_OPERATOR@(*px,*py,*pz,$1)
    
/////////
// 1-d //
/////////
    } else if (x->ndim == 1) {
        int64_t i;
        const int64_t n_inner = 0;
        const uint64_t ix0 = x->strides[n_inner] / x->itemsize;
        const uint64_t iy0 = y->strides[n_inner] / y->itemsize;
 
        for (i = 0; i < x->size; i++) {
            @BINARY_OPERATOR@(px[i*ix0],py[i*iy0],*pz,$1)
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
#end_switch
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
//ve_array *x, ve_array *y, ve_array *z, int32_t *psw)
{
    uint64_t err = NLCPY_ERROR_OK;
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
    ve_array *x = &(args->binary.x);
    ve_array *y = &(args->binary.y);
    ve_array *z = &(args->binary.z);

    assert(x->size == y->size);
    assert(x->ndim == y->ndim);
    for (uint64_t i = 0; i < x->ndim; i++) assert(x->shape[i] == y->shape[i]);
    
    switch (z->dtype) {
    case ve_i32:  err = FILENAME_i32 (x, y, z, psw); break;
    case ve_i64:  err = FILENAME_i64 (x, y, z, psw); break;
    case ve_u32:  err = FILENAME_u32 (x, y, z, psw); break;
    case ve_u64:  err = FILENAME_u64 (x, y, z, psw); break;
    case ve_f32:  err = FILENAME_f32 (x, y, z, psw); break;
    case ve_f64:  err = FILENAME_f64 (x, y, z, psw); break;
    case ve_c64:  err = FILENAME_c64 (x, y, z, psw); break;
    case ve_c128: err = FILENAME_c128(x, y, z, psw); break;
    default: err = NLCPY_ERROR_DTYPE; break;
    }
} /* omp single */
    return (uint64_t)err;
}

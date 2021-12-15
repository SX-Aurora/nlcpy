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

/////////
// 1-d //
/////////
    if (x->ndim == 1 && y->ndim == 1) {
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
        int64_t i;
        const int64_t n_inner = 0;
        const uint64_t ix0 = x->strides[n_inner] / x->itemsize;
        const uint64_t iy0 = y->strides[n_inner] / y->itemsize;

        for (i = 0; i < x->size; i++) {
            @BINARY_OPERATOR@(px[i*ix0],py[i*iy0],*pz,$1)
        }
} /* omp single */
/////////
// N-d //
/////////
    } else {
@#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
@#else
        const int nt = 1;
        const int it = 0;
@#endif /* _OPENMP */

        int64_t i, j;
        int64_t len_idx = z->ndim + 1;
        int64_t *idx = (int64_t*)alloca(sizeof(int64_t) * len_idx);
        int64_t *shape = (int64_t*)alloca(sizeof(int64_t) * len_idx);
        for (i = 0; i < z->ndim; i++) {
            shape[i] = z->shape[i];
        }
        shape[len_idx - 1] = x->shape[x->ndim - 1];
        nlcpy__rearrange_axis(z, idx);
        if (z->ndim == 1) {
            idx[1] = 1;
        } else {
            if (shape[idx[z->ndim - 1]] <= shape[len_idx - 1]) {
                uint64_t temp = idx[0];
                idx[0] = idx[z->ndim - 1];
                idx[z->ndim - 1] = temp;
                idx[len_idx - 1] = len_idx - 1;
            } else {
                idx[len_idx - 1] = idx[z->ndim - 1];
                idx[z->ndim - 1] = len_idx - 1;
            }
        }
        int64_t *cnt_idx = (int64_t*)alloca(sizeof(int64_t) * len_idx);
        nlcpy__reset_coords(cnt_idx, len_idx);
        const int64_t n_inner = idx[len_idx - 1];
        const int64_t n_outer = idx[0];

        uint64_t ix = 0;
        uint64_t iy = 0;
        uint64_t iz = 0;
        const uint64_t axis_y = (y->ndim > 1) ? (y->ndim - 2) : 0;
        uint64_t ix0, iy0, iz0;
        if (n_inner > z->ndim - 1) {
            ix0 = x->strides[x->ndim - 1] / x->itemsize;
            iy0 = y->strides[axis_y] / y->itemsize;
            iz0 = 0;
        } else {
            if (n_inner < x->ndim - 1) {
                ix0 = x->strides[n_inner] / x->itemsize;
                iy0 = 0;
            } else {
                ix0 = 0;
                uint64_t idx_y = n_inner - x->ndim + 1;
                if (idx_y == axis_y) idx_y++;
                iy0 = y->strides[idx_y] / y->itemsize;
            }
            iz0 = z->strides[n_inner] / z->itemsize;
        }
        const int64_t lenm = z->shape[n_outer];
        const int64_t cnt_s = lenm * it / nt;
        const int64_t cnt_e = lenm * (it + 1) / nt;
        for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
            if (n_outer < x->ndim - 1) {
                ix = cnt * x->strides[n_outer] / x->itemsize;
                iy = 0;
            } else {
                ix = 0;
                uint64_t idx_y = n_outer - x->ndim + 1;
                if (idx_y == axis_y) idx_y++;
                iy = cnt * y->strides[idx_y] / y->itemsize;
            }
            iz = cnt * z->strides[n_outer] / z->itemsize;
            do {
                if (n_inner == len_idx - 1) {
                    for (i = 0; i < shape[n_inner]; i++) {
                        @BINARY_OPERATOR@(px[ix+i*ix0],py[iy+i*iy0],pz[iz],$1)
                    }
                } else {
                    for (i = 0; i < shape[n_inner]; i++) {
                        @BINARY_OPERATOR@(px[ix+i*ix0],py[iy+i*iy0],pz[iz+i*iz0],$1)
                    }
                }
                for (j = len_idx - 2; j >= 1; j--) {
                    int64_t jj = idx[j];
                    if (++cnt_idx[jj] < shape[jj]) {
                        if (jj < z->ndim) {
                            iz += z->strides[jj] / z->itemsize;
                            if (jj < x->ndim - 1) {
                                ix += x->strides[jj] / x->itemsize;
                            } else {
                                uint64_t idx_y = jj - x->ndim + 1;
                                if (idx_y == axis_y) idx_y++;
                                iy += y->strides[idx_y] / y->itemsize;
                            }
                        } else {
                            ix += x->strides[x->ndim - 1] / x->itemsize;
                            iy += y->strides[axis_y] / y->itemsize;
                        }
                        break;
                    }
                    cnt_idx[jj] = 0;
                    if (jj < z->ndim) {
                        iz -= (z->strides[jj] / z->itemsize) * (shape[jj] - 1);
                        if (jj < x->ndim - 1) {
                            ix -= (x->strides[jj] / x->itemsize) * (shape[jj] - 1);
                        } else {
                            uint64_t idx_y = jj - x->ndim + 1;
                            if (idx_y == axis_y) idx_y++;
                            iy -= (y->strides[idx_y] / y->itemsize) * (shape[jj] - 1);
                        }
                    } else {
                        ix -= (x->strides[x->ndim - 1] / x->itemsize) * (shape[jj] - 1);
                        iy -= (y->strides[axis_y] / y->itemsize) * (shape[jj] - 1);
                    }
                }
            } while (j >= 1);
        }
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
#end_switch
}


@-->)dnl
macro_binary_operator(bool, int32_t,        ve_i32)dnl
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
    ve_array *x = &(args->binary.x);
    ve_array *y = &(args->binary.y);
    ve_array *z = &(args->binary.z);

    switch (z->dtype) {
    case ve_bool:  err = FILENAME_bool (x, y, z, psw); break;
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
    return (uint64_t)err;
}

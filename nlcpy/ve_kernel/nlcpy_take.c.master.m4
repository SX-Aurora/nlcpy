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
@#include <stdio.h>
@#include <stdint.h>
@#include <stdbool.h>
@#include <stdlib.h>
@#include <limits.h>
@#include <alloca.h>
@#include <assert.h>

@#include "nlcpy.h"

include(macros.m4)dnl

#define_switch (a_src->dtype:bool,i32,i64,u32,u64,f32,f64,c64,c128 @ a_idx->dtype:i64)


define(<--@macro_take@-->,<--@
uint64_t nlcpy_take_$1(ve_array *a_src, ve_array *a_idx, ve_array *a_out,
                       int64_t ldim, int64_t cdim, int64_t rdim,
                       int64_t index_range, int32_t *psw) {
@#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
@#else
        const int nt = 1;
        const int it = 0;
@#endif /* _OPENMP */

#begin_switch
    @TYPE1@ *d_src = (@TYPE1@ *)a_src->ve_adr;
    @TYPE2@ *d_idx = (@TYPE2@ *)a_idx->ve_adr;
    $2 *d_out = ($2 *)a_out->ve_adr;
    if (d_src == NULL || d_idx == NULL || d_out == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    if (a_src->size == 0 || a_idx->size == 0 || a_out->size == 0) {
        return NLCPY_ERROR_OK;
    }

/////////
// 0-d //
/////////
    if (a_idx->ndim == 0) {
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
        int64_t idx_out = d_idx[0] % index_range;
        if (idx_out < 0) idx_out += index_range;
        d_out[0] = ($2)d_src[idx_out];
} /* omp single */

/////////
// 1-d //
/////////
    } else if (a_idx->ndim == 1) {
        int64_t i;
        int64_t n_inner_idx = a_idx->ndim - 1;
        uint64_t isrc0 = a_src->strides[0] / a_src->itemsize;
        uint64_t iidx0 = a_idx->strides[n_inner_idx] / a_idx->itemsize;
        int64_t len = a_out->size;
        const int64_t cnt_s = len * it / nt;
        const int64_t cnt_e = len * (it + 1) / nt;
        /*
        for (i = cnt_s; i < cnt_e; i++) {
            int64_t idx_out = d_idx[i*iidx0] % index_range;
            if (idx_out < 0) idx_out += index_range;
            if (ldim != 1) idx_out += (i / (cdim * rdim)) * index_range;
            if (rdim != 1) idx_out = idx_out * rdim + i % rdim;
            d_out[i] = d_src[idx_out*isrc0];
        }
        */
        if (ldim != 1 && rdim != 1) {
            for (i = cnt_s; i < cnt_e; i++) {
                int64_t idx_out = d_idx[i*iidx0] % index_range;
                if (idx_out < 0) idx_out += index_range;
                idx_out += (i / (cdim * rdim)) * index_range;
                idx_out = idx_out * rdim + i % rdim;
                d_out[i] = ($2)d_src[idx_out*isrc0];
            }
        } else if (ldim == 1 && rdim != 1) {
            for (i = cnt_s; i < cnt_e; i++) {
                int64_t idx_out = d_idx[i*iidx0] % index_range;
                if (idx_out < 0) idx_out += index_range;
                idx_out = idx_out * rdim + i % rdim;
                d_out[i] = ($2)d_src[idx_out*isrc0];
            }
        } else if (ldim != 1 && rdim == 1) {
            for (i = cnt_s; i < cnt_e; i++) {
                int64_t idx_out = d_idx[i*iidx0] % index_range;
                if (idx_out < 0) idx_out += index_range;
                idx_out += (i / (cdim * rdim)) * index_range;
                d_out[i] = ($2)d_src[idx_out*isrc0];
            }
        } else {
            for (i = cnt_s; i < cnt_e; i++) {
                int64_t idx_out = d_idx[i*iidx0] % index_range;
                if (idx_out < 0) idx_out += index_range;
                d_out[i] = ($2)d_src[idx_out*isrc0];
            }
        }

/////////
// N-d //
/////////
    } else if (a_idx->ndim > 1 && a_idx->ndim < NLCPY_MAXNDIM) {
        int64_t i, k;
        int64_t n_inner = a_idx->ndim - 1;
        int64_t n_outer = 0;

        int64_t *rel_idx = (int64_t *)alloca(sizeof(int64_t) * a_idx->ndim);
        nlcpy__rearrange_axis(a_idx, rel_idx);
        n_inner = rel_idx[n_inner];
        n_outer = rel_idx[n_outer];
        int64_t *cnt_idx = (int64_t *)alloca(sizeof(int64_t) * a_idx->ndim);
        nlcpy__reset_coords(cnt_idx, a_idx->ndim);
        int64_t len = a_idx->shape[n_outer];
        const int64_t cnt_s = len * it / nt;
        const int64_t cnt_e = len * (it + 1) / nt;

        uint64_t iidx = 0;
        uint64_t iout = 0;
        uint64_t isrc0 = a_src->strides[0] / a_src->itemsize;
        uint64_t iidx0 = a_idx->strides[n_inner] / a_idx->itemsize;
        uint64_t iout0 = a_out->strides[n_inner] / a_out->itemsize;
        int64_t totalcnt = 0;

        for (int64_t cnt = cnt_s; cnt < cnt_e; cnt++) {
            iidx = cnt * a_idx->strides[n_outer] / a_idx->itemsize;
            iout = cnt * a_out->strides[n_outer] / a_out->itemsize;
            for(;;) {
                /*
                for (i = 0; i < a_idx->shape[n_inner]; i++) {
                    int64_t idx_out = d_idx[i*iidx0+iidx] % index_range;
                    int64_t totalcnt = i*iout0 + iout;
                    if (idx_out < 0) idx_out += index_range;
                    if (ldim != 1) idx_out += (totalcnt / (cdim * rdim)) * index_range;
                    if (rdim != 1) idx_out = idx_out * rdim + totalcnt % rdim;
                    d_out[totalcnt] = d_src[idx_out*isrc0];
                }
                */

                if (ldim != 1 && rdim != 1) {
                    for (i = 0; i < a_idx->shape[n_inner]; i++) {
                        int64_t idx_out = d_idx[i*iidx0+iidx] % index_range;
                        int64_t totalcnt = i*iout0 + iout;
                        if (idx_out < 0) idx_out += index_range;
                        idx_out += (totalcnt / (cdim * rdim)) * index_range;
                        idx_out = idx_out * rdim + totalcnt % rdim;
                        d_out[totalcnt] = ($2)d_src[idx_out*isrc0];
                    }
                } else if (ldim == 1 && rdim != 1) {
                    for (i = 0; i < a_idx->shape[n_inner]; i++) {
                        int64_t idx_out = d_idx[i*iidx0+iidx] % index_range;
                        int64_t totalcnt = i*iout0 + iout;
                        if (idx_out < 0) idx_out += index_range;
                        idx_out = idx_out * rdim + totalcnt % rdim;
                        d_out[totalcnt] = ($2)d_src[idx_out*isrc0];
                    }
                } else if (ldim != 1 && rdim == 1) {
                    for (i = 0; i < a_idx->shape[n_inner]; i++) {
                        int64_t idx_out = d_idx[i*iidx0+iidx] % index_range;
                        int64_t totalcnt = i*iout0 + iout;
                        if (idx_out < 0) idx_out += index_range;
                        idx_out += (totalcnt / (cdim * rdim)) * index_range;
                        d_out[totalcnt] = ($2)d_src[idx_out*isrc0];
                    }
                } else {
                    for (i = 0; i < a_idx->shape[n_inner]; i++) {
                        int64_t idx_out = d_idx[i*iidx0+iidx] % index_range;
                        int64_t totalcnt = i*iout0 + iout;
                        if (idx_out < 0) idx_out += index_range;
                        d_out[totalcnt] = ($2)d_src[idx_out*isrc0];
                    }
                }

                for (k = a_idx->ndim - 2; k >= 1; k--) {
                    int64_t kk = rel_idx[k];
                    if (++cnt_idx[kk] < a_idx->shape[kk]) {
                        iidx += a_idx->strides[kk] / a_idx->itemsize;
                        iout += a_out->strides[kk] / a_out->itemsize;
                        break;
                    }
                    cnt_idx[kk] = 0;
                    iidx -= (a_idx->strides[kk] / a_idx->itemsize) * (a_idx->shape[kk] - 1);
                    iout -= (a_out->strides[kk] / a_out->itemsize) * (a_out->shape[kk] - 1);
                }
                if (k < 1) break;
            }
        }

    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
#end_switch
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
macro_take(bool,int32_t)dnl
macro_take(i32,int32_t)dnl
macro_take(i64,int64_t)dnl
macro_take(u32,uint32_t)dnl
macro_take(u64,uint64_t)dnl
macro_take(f32,float)dnl
macro_take(f64,double)dnl
macro_take(c64,float _Complex)dnl
macro_take(c128,double _Complex)dnl


uint64_t nlcpy_take(ve_arguments *args, int32_t *psw) {
    ve_array *a_src = &(args->take.src);
    ve_array *a_idx = &(args->take.idx);
    ve_array *a_out = &(args->take.out);
    int64_t ldim = args->take.ldim;
    int64_t cdim = args->take.cdim;
    int64_t rdim = args->take.rdim;
    int64_t index_range = args->take.index_range;
    assert(a_idx->size == a_out->size);
    assert(a_src->ndim == 1);
    assert(a_idx->ndim == a_out->ndim);

    uint64_t err = NLCPY_ERROR_OK;

    switch (a_out->dtype) {
        case ve_i32  : err |= nlcpy_take_i32  (a_src, a_idx, a_out, ldim, cdim, rdim, index_range, psw); break;
        case ve_i64  : err |= nlcpy_take_i64  (a_src, a_idx, a_out, ldim, cdim, rdim, index_range, psw); break;
        case ve_u32  : err |= nlcpy_take_u32  (a_src, a_idx, a_out, ldim, cdim, rdim, index_range, psw); break;
        case ve_u64  : err |= nlcpy_take_u64  (a_src, a_idx, a_out, ldim, cdim, rdim, index_range, psw); break;
        case ve_f32  : err |= nlcpy_take_f32  (a_src, a_idx, a_out, ldim, cdim, rdim, index_range, psw); break;
        case ve_f64  : err |= nlcpy_take_f64  (a_src, a_idx, a_out, ldim, cdim, rdim, index_range, psw); break;
        case ve_c64  : err |= nlcpy_take_c64  (a_src, a_idx, a_out, ldim, cdim, rdim, index_range, psw); break;
        case ve_c128 : err |= nlcpy_take_c128 (a_src, a_idx, a_out, ldim, cdim, rdim, index_range, psw); break;
        case ve_bool : err |= nlcpy_take_bool (a_src, a_idx, a_out, ldim, cdim, rdim, index_range, psw); break;
        default: return (uint64_t)NLCPY_ERROR_DTYPE;
    }

    return (uint64_t)err;
}

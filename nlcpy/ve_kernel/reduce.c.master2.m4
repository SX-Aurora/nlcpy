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

@#include "nlcpy.h"

include(macros.m4)dnl

/****************************
 *
 *       REDUCE OPERATOR
 *
 * **************************/

define(<--@macro_reduce_operator@-->,<--@
uint64_t FILENAME_@TYPE1_DTAG@_$1(ve_array *x, ve_array *y, int32_t axis, int32_t init_flag,
                     ve_array *initial, int32_t where_flag, ve_array *where, int32_t *psw)
{
    @TYPE1@ *px = (@TYPE1@ *)nlcpy__get_ptr(x);
    if (px == NULL) return NLCPY_ERROR_MEMORY;
    $2 *py = ($2 *)nlcpy__get_ptr(y);
    if (py == NULL) return NLCPY_ERROR_MEMORY;
    $2 *pi = ($2 *)nlcpy__get_ptr(initial);
    if (pi == NULL) return NLCPY_ERROR_MEMORY;
    ve_array *w = where;
    $2 init_val = ($2)(*pi);

@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
    // initialize
    if (init_flag) {
        for (uint64_t i=0; i<y->size; i++) py[i] = init_val;
    }
    else {
        for (uint64_t i=0; i<y->size; i++) py[i] = 0;
    }
} /* omp single */ 

/////////
// 0-d //
/////////
    if (x->ndim == 0){
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
        if (!where_flag) {
            if (!init_flag) {
                *py = *px;
            } 
            else{
                @UNARY_OPERATOR@(*px,*py,$1)
            }
        } else {
            Bint *pw = (Bint *)w->ve_adr;
            if (*pw) {
                if (!init_flag) {
                    *py = *px;
                }
                else{
                    @UNARY_OPERATOR@(*px,*py,$1)
                }
            }
        }
} /* omp single */ 

/////////
// 1-d //
/////////
    } else if (x->ndim == 1){
@#ifdef _OPENMP
@#pragma omp single
@#endif /* _OPENMP */
{
        uint64_t i;
        uint64_t iw0 = 0;
        const uint64_t ix = x->strides[0]/x->itemsize;
        if (!where_flag) {
            if (!init_flag) {
                *py = *px;
                i = 1;
            } else {
                i = 0;
            }
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
            for (; i < x->shape[0]; i++) {
                @UNARY_OPERATOR@(px[i*ix],*py,$1)
            }
        } else {
            Bint *pw = (Bint *)w->ve_adr;
            const uint64_t iw = w->strides[0]/w->itemsize;
            if (!init_flag) {
                for (i = 0; i < x->shape[0]; i++) {
                    if (pw[i*iw]) {
                        *py = *px;
                        i++;
                        break;
                    }
                }
            } else {
                i = 0;
            }
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
            for (; i < x->shape[0]; i++) {
                if (pw[i*iw]) {
                    @UNARY_OPERATOR@(px[i*ix],*py,$1)
                }
            }
        }
} /* omp single */ 

/////////
// N-d //
/////////
    } else if (x->ndim <= NLCPY_MAXNDIM) {
        int64_t n_inner = x->ndim - 1;
        int64_t n_outer = 0;

        int64_t *idx = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
        for (uint64_t i = 0; i < x->ndim; i++) {
            idx[i] = i;
        }
        if (x->ndim > 2) {
            int64_t max_idx[3], tmp;
            nlcpy__argnsort(x, max_idx, 3);
            if (max_idx[0] != n_inner) {
                tmp = idx[n_inner];
                idx[n_inner] = idx[max_idx[0]];
                idx[max_idx[0]] = tmp;
            }
            if (idx[max_idx[1]] == axis) {
                max_idx[1] = max_idx[2];
            }
            if (idx[max_idx[1]] != n_outer || idx[n_outer] == axis) {
                tmp = idx[n_outer];
                idx[n_outer] = idx[max_idx[1]];
                idx[max_idx[1]] = tmp;
            }
        } else if (n_outer == axis) {
            idx[0] = 1;
            idx[1] = 0;
        }    
        int64_t n_inner2 = idx[n_inner];
        int64_t n_outer2 = idx[n_outer];

@#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
@#else
        const int nt = 1;
        const int it = 0;
@#endif
        int64_t i;
        uint64_t adr_x = (uint64_t)x->ve_adr;
        uint64_t adr_y = (uint64_t)y->ve_adr;

        uint64_t ix = 0;
        uint64_t iy = 0;
        uint64_t iw = 0;
        uint64_t ix0 = x->strides[n_inner2] / x->itemsize;
        uint64_t iy0 = y->strides[n_inner2] / y->itemsize;
        uint64_t iw0 = 0;
        int64_t k = 0;
        int64_t *cnt_x = (int64_t*)alloca(sizeof(int64_t)*x->ndim);
        nlcpy__reset_coords(cnt_x, x->ndim);
        const int64_t lenm = x->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;
        if (!where_flag) {
            for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                cnt_x[n_outer2] = cntm;
                ix = cntm * x->strides[n_outer2] / x->itemsize;
                iy = (n_outer2 == axis) ? 0 : cntm * y->strides[n_outer2] / y->itemsize;

                do {
                    if (n_inner2 == axis) {
                            if (!init_flag) {
                                py[iy] = px[ix];
                                i = 1;
                            } else {
                                i = 0;
                            }
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                            for (; i < x->shape[n_inner2]; i++) {
                                @UNARY_OPERATOR@(px[i*ix0+ix],py[iy],$1)
                            }
                        } else {
                            if (init_flag || cnt_x[axis] > 0) {
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                                for (i = 0; i < x->shape[n_inner2]; i++) {
                                    @UNARY_OPERATOR@(px[i*ix0+ix],py[i*iy0+iy],$1)
                                }
                            } else {
                                for (i = 0; i < x->shape[n_inner2]; i++) {
                                    py[i*iy0+iy] = @CAST_OPERATOR@(px[i*ix0+ix],@TYPE1_DTAG@,$1)
                                }
                            }
                        }
                    for (k = n_inner - 1; k >= 1; k--) {
                        int64_t kk = idx[k];
                        if (kk == axis) {
                            if (++cnt_x[kk] < x->shape[kk]) {
                                ix += x->strides[kk] / x->itemsize;
                                break;
                            }
                            ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                        } else {
                            if (++cnt_x[kk] < x->shape[kk]) {
                                ix += x->strides[kk] / x->itemsize;
                                iy += y->strides[kk] / y->itemsize;
                                break;
                            }
                            ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                            iy -= (y->strides[kk] / y->itemsize) * (y->shape[kk] - 1);
                        }
                        cnt_x[kk] = 0;
                    }
                } while (k >= 1);
            }
        } else {
            Bint *pw = (Bint *)w->ve_adr;
            iw0 = w->strides[n_inner2] / w->itemsize;

            for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                ix = cntm * x->strides[n_outer2] / x->itemsize;
                iy = (n_outer2 == axis) ? 0 : cntm * y->strides[n_outer2] / y->itemsize;
                iw = cntm * w->strides[n_outer2] / w->itemsize;

                do {
                    if (n_inner2 == axis) {
                        if (!init_flag) {
                            for (i = 0; i < w->shape[n_inner2]; i++) {
                                if (pw[i*iw0+iw]) {
                                    py[iy] = @CAST_OPERATOR@(px[i*ix0+ix],@TYPE1_DTAG@,$1)
                                    i++;
                                    break;
                                }
                            }
                        } else {
                            i = 0;
                        }
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                        for (; i < x->shape[n_inner2]; i++) {
                            if (pw[i*iw0+iw]) {
                                @UNARY_OPERATOR@(px[i*ix0+ix],py[iy],$1)
                            }
                        }
                    } else {
                        if (init_flag || cnt_x[axis] > 0) {
#ifdef left_shift_reduce
@#pragma _NEC novector
#else
@#pragma _NEC ivdep
#endif
                            for (i = 0; i < x->shape[n_inner2]; i++) {
                                if (pw[i*iw0+iw]) {
                                    @UNARY_OPERATOR@(px[i*ix0+ix],py[i*iy0+iy],$1)
                                }
                            }
                        } else {
                            for (i = 0; i < x->shape[n_inner2]; i++) {
                                if (pw[i*iw0+iw]) {
                                    py[i*iy0+iy] = @CAST_OPERATOR@(px[i*ix0+ix],@TYPE1_DTAG@,$1)
                                }
                            }
                        }
                    }
                    for (k = n_inner - 1; k >= 1; k--) {
                        int64_t kk = idx[k];
                        if (kk == axis) {
                            if (++cnt_x[kk] < x->shape[kk]) {
                                ix += x->strides[kk] / x->itemsize;
                                iw += w->strides[kk] / w->itemsize;
                                break;
                            }
                            ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                            iw -= (w->strides[kk] / w->itemsize) * (x->shape[kk] - 1);
                        } else {
                            if (++cnt_x[kk] < x->shape[kk]) {
                                ix += x->strides[kk] / x->itemsize;
                                iy += y->strides[kk] / y->itemsize;
                                iw += w->strides[kk] / w->itemsize;
                                break;
                            }
                            ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                            iy -= (y->strides[kk] / y->itemsize) * (x->shape[kk] - 1);
                            iw -= (w->strides[kk] / w->itemsize) * (x->shape[kk] - 1);
                        }
                        cnt_x[kk] = 0;
                    }
                } while (k >= 1);
            }
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


@-->)dnl
#if defined(DTYPE_i32)
macro_reduce_operator(i32,int32_t)dnl
#endif
#if defined(DTYPE_i64)
macro_reduce_operator(i64,int64_t)dnl
#endif
#if defined(DTYPE_u32)
macro_reduce_operator(u32,uint32_t)dnl
#endif
#if defined(DTYPE_u64)
macro_reduce_operator(u64,uint64_t)dnl
#endif
#if defined(DTYPE_f32)
macro_reduce_operator(f32,float)dnl
#endif
#if defined(DTYPE_f64)
macro_reduce_operator(f64,double)dnl
#endif
#if defined(DTYPE_c64)
macro_reduce_operator(c64,float _Complex)dnl
#endif
#if defined(DTYPE_c128)
macro_reduce_operator(c128,double _Complex)dnl
#endif
#if defined(DTYPE_bool)
macro_reduce_operator(bool,int32_t)dnl
#endif

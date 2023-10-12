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

include(macros.m4)dnl
#include <stdint.h>
#include <alloca.h>
#include <math.h>
#include <complex.h>

#include "nlcpy.h"

define(<--@macro_fnorm@-->,<--@
uint64_t nlcpy_fnorm_$1(ve_array *x, ve_array *y, ve_array *w1, ve_array *w2, int64_t axis1, int64_t axis2, int32_t *psw)
{
    $2* const px = ($2*)x->ve_adr;
    $3* const py = ($3*)y->ve_adr;
    $3* const pw1 = ($3*)w1->ve_adr;
    $3* const pw2 = ($3*)w2->ve_adr;
    if (px == NULL || py == NULL || pw1 == NULL || pw2 == NULL) {
        return NLCPY_ERROR_MEMORY;
    }
#ifdef _OPENMP
    const int nt = omp_get_num_threads();
    const int it = omp_get_thread_num();
#else
    const int nt = 1;
    const int it = 0;
#endif
    uint64_t i, j, k;
    uint64_t ix, iw;
    uint64_t len, i_s, i_e;
    // initialize
    len = w2->size;
    i_s = len * it / nt;
    i_e = len * (it + 1) / nt;
    for (i = i_s; i < i_e; i++) pw2[i] = 0;
    len = y->size;
    i_s = len * it / nt;
    i_e = len * (it + 1) / nt;
    for (i = i_s; i < i_e; i++) py[i] = 0;
    uint64_t n_inner = x->ndim - 1;
    uint64_t n_outer = 0;
    int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * x->ndim);
    int64_t ndim = (x->ndim > 2) ? 3 : 2;
    int64_t *max_idx = (int64_t *)alloca(sizeof(int64_t) * ndim);
    for (i = 0; i < x->ndim; i++) idx[i] = i;
    nlcpy__argnsort(x, max_idx, ndim);
    if (ndim == 2) {
        if (max_idx[0] == axis1) {
            idx[n_outer] = max_idx[1];
            idx[n_inner] = max_idx[0];
        } else {
            idx[n_outer] = max_idx[0];
            idx[n_inner] = max_idx[1];
        }
    } else {
        idx[n_inner] = max_idx[0];
        idx[max_idx[0]] = n_inner;
        if (idx[max_idx[1]] == axis1) max_idx[1] = max_idx[2];
        int64_t tmp = idx[n_outer];
        idx[n_outer] = idx[max_idx[1]];
        idx[max_idx[1]] = tmp;
    }
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t) * x->ndim);
    nlcpy__reset_coords(cnt, x->ndim);
    int64_t n_inner2 = idx[n_inner];
    int64_t n_outer2 = idx[n_outer];
    uint64_t ix0 = x->strides[n_inner2] / x->itemsize;
    uint64_t iw0;
    uint64_t stride_outer2;
    if (n_outer2 < axis1) {
        stride_outer2 = w2->strides[n_outer2] / w2->itemsize;
    } else if (n_outer2 > axis1) {
        stride_outer2 = w2->strides[n_outer2 - 1] / w2->itemsize;
    } else {
        return NLCPY_ERROR_INTERNAL;
    }
    if (n_inner2 < axis1) {
        iw0 = w2->strides[n_inner2] / w2->itemsize;
    } else {
        iw0 = w2->strides[n_inner2 - 1] / w2->itemsize;
    }

    if (x->is_c_contiguous | x->is_f_contiguous) {
        // abs(x) * abs(x)
        len = x->size;
        i_s = len * it / nt;
        i_e = len * (it + 1) / nt;
        for (i = i_s; i < i_e; i++) {
            $3 tmp = $5(px[i]);
            pw1[i] = tmp * tmp;
        }
#ifdef _OPENMP
#pragma omp barrier
#endif
#ifdef DEBUG_BARRIER
        nlcpy__sleep_thread();
#endif /* DEBUG_BARRIER */
        // sum along first axis
        len = w1->shape[n_outer2];
        i_s = len * it / nt;
        i_e = len * (it + 1) / nt;
        for (i = i_s; i < i_e; i++) {
            ix = i * w1->strides[n_outer2] / w1->itemsize;
            iw = i * stride_outer2;
            do {
                if (n_inner2 == axis1) {
                    for (j = 0; j < w1->shape[n_inner2]; j++) {
                        pw2[iw] += pw1[j*ix0+ix];
                    }
                } else {
                    for (j = 0; j < w1->shape[n_inner2]; j++) {
                        pw2[j*iw0+iw] += pw1[j*ix0+ix];
                    }
                }
                for (k = n_inner - 1; k >= 1; k--) {
                    int64_t kk = idx[k];
                    if (++cnt[kk] < w1->shape[kk]) {
                        ix += w1->strides[kk] / w1->itemsize;
                        if (kk < axis1) {
                            iw += w2->strides[kk] / w2->itemsize;
                        } else if (kk > axis1) {
                            iw += w2->strides[kk - 1] / w2->itemsize;
                        }
                        break;
                    }
                    ix -= (w1->strides[kk] / w1->itemsize) * (w1->shape[kk] - 1);
                    if (kk < axis1) {
                        iw -= (w2->strides[kk] / w2->itemsize) * (w1->shape[kk] - 1);
                    } else if (kk > axis1) {
                        iw -= (w2->strides[kk - 1] / w2->itemsize) * (w1->shape[kk] - 1);
                    }
                    cnt[kk] = 0;
                }
            } while (k >= 1);
        }
    } else {
#ifdef _OPENMP
#pragma omp barrier
#endif
#ifdef DEBUG_BARRIER
        nlcpy__sleep_thread();
#endif /* DEBUG_BARRIER */
        // abs(x) * abs(x) and sum along the first axis
        len = x->shape[n_outer2];
        i_s = len * it / nt;
        i_e = len * (it + 1) / nt;
        for (i = i_s; i < i_e; i++) {
            ix = i * x->strides[n_outer2] / x->itemsize;
            iw = i * stride_outer2;
            do {
                if (n_inner2 == axis1) {
                    for (j = 0; j < x->shape[n_inner2]; j++) {
                        $3 tmp = $5(px[j*ix0+ix]);
                        pw2[iw] += tmp * tmp;
                    }
                } else {
                    for (j = 0; j < x->shape[n_inner2]; j++) {
                        $3 tmp = $5(px[j*ix0+ix]);
                        pw2[j*iw0+iw] += tmp * tmp;
                    }
                }
                for (k = n_inner - 1; k >= 1; k--) {
                    int64_t kk = idx[k];
                    if (++cnt[kk] < x->shape[kk]) {
                        ix += x->strides[kk] / x->itemsize;
                        if (kk < axis1) {
                            iw += w2->strides[kk] / w2->itemsize;
                        } else if (kk > axis1) {
                            iw += w2->strides[kk - 1] / w2->itemsize;
                        }
                        break;
                    }
                    ix -= (x->strides[kk] / x->itemsize) * (x->shape[kk] - 1);
                    if (kk < axis1) {
                        iw -= (w2->strides[kk] / w2->itemsize) * (x->shape[kk] - 1);
                    } else if (kk > axis1) {
                        iw -= (w2->strides[kk - 1] / w2->itemsize) * (x->shape[kk] - 1);
                    }
                    cnt[kk] = 0;
                }
            } while (k >= 1);
        }
    }

#ifdef _OPENMP
#pragma omp barrier
#endif
#ifdef DEBUG_BARRIER
        nlcpy__sleep_thread();
#endif /* DEBUG_BARRIER */
    // sum along second axis
    if (w2->ndim == 1) {
#ifdef _OPENMP
#pragma omp single
#endif
{
        for (i = 0; i < w2->size; i++) {
            *py += pw2[i];
        }
}
    } else {
        if (axis1 < axis2) {
            axis2--;
        }
        n_inner = w2->ndim - 1;
        ndim = (w2->ndim > 2) ? 3 : 2;
        for (i = 0; i < w2->ndim; i++) idx[i] = i;
        nlcpy__argnsort(w2, max_idx, ndim);
        if (ndim == 2) {
            if (max_idx[0] == axis2) {
                idx[n_outer] = max_idx[1];
                idx[n_inner] = max_idx[0];
            } else {
                idx[n_outer] = max_idx[0];
                idx[n_inner] = max_idx[1];
            }
        } else {
            idx[n_inner] = max_idx[0];
            idx[max_idx[0]] = n_inner;
            if (idx[max_idx[1]] == axis2) max_idx[1] = max_idx[2];
            int64_t tmp = idx[n_outer];
            idx[n_outer] = idx[max_idx[1]];
            idx[max_idx[1]] = tmp;
        }
        n_inner2 = idx[n_inner];
        n_outer2 = idx[n_outer];
        int64_t iy;
        int64_t iy0;
        if (n_outer2 < axis2) {
            stride_outer2 = y->strides[n_outer2] / y->itemsize;
        } else if (n_outer2 > axis2) {
            stride_outer2 = y->strides[n_outer2 - 1] / y->itemsize;
        } else {
            return NLCPY_ERROR_INTERNAL;
        }
        if (n_inner2 < axis2) {
            iy0 = y->strides[n_inner2] / y->itemsize;
        } else if (n_inner2 > axis2) {
            iy0 = y->strides[n_inner2 - 1] / y->itemsize;
        }
        iw0 = w2->strides[n_inner2] / w2->itemsize;
        nlcpy__reset_coords(cnt, w2->ndim);
        len = w2->shape[n_outer2];
        i_s = len * it / nt;
        i_e = len * (it + 1) / nt;
        for (i = i_s; i < i_e; i++) {
            iw = i * w2->strides[n_outer2] / w2->itemsize;
            iy = i * stride_outer2;

            do {
                if (n_inner2 == axis2) {
                    for (j = 0; j < w2->shape[n_inner2]; j++) {
                        py[iy] += pw2[j*iw0+iw];
                    }
                } else {
                    for (j = 0; j < w2->shape[n_inner2]; j++) {
                        py[j*iy0+iy] += pw2[j*iw0+iw];
                    }
                }
                for (k = n_inner - 1; k >= 1; k--) {
                    int64_t kk = idx[k];
                    if (++cnt[kk] < w2->shape[kk]) {
                        iw += w2->strides[kk] / w2->itemsize;
                        if (kk < axis2) {
                            iy += y->strides[kk] / y->itemsize;
                        } else if (kk > axis2) {
                            iy += y->strides[kk - 1] / y->itemsize;
                        }
                        break;
                    }
                    iw -= (w2->strides[kk] / w2->itemsize) * (w2->shape[kk] - 1);
                    if (kk < axis2) {
                        iy -= (y->strides[kk] / y->itemsize) * (w2->shape[kk] - 1);
                    } else if (kk > axis2) {
                        iy -= (y->strides[kk - 1] / y->itemsize) * (w2->shape[kk] - 1);
                    }
                    cnt[kk] = 0;
                }
            } while (k >= 1);
        }
    }
#ifdef _OPENMP
#pragma omp barrier
#endif
#ifdef DEBUG_BARRIER
        nlcpy__sleep_thread();
#endif /* DEBUG_BARRIER */
    i_s = y->size * it / nt;
    i_e = y->size * (it + 1) / nt;
    for (i = i_s; i < i_e; i++) {
        py[i] = $4(py[i]);
    }
    retrieve_fpe_flags(psw);
    return NLCPY_ERROR_OK;
}
@-->)dnl
macro_fnorm(s, float, float, sqrtf)dnl
macro_fnorm(d, double, double, sqrt)dnl
macro_fnorm(c, float _Complex, float, sqrtf, cabsf)dnl
macro_fnorm(z, double _Complex, double, sqrt, cabs)dnl


uint64_t nlcpy_fnorm(ve_arguments *args, int32_t *psw) {
    uint64_t err = NLCPY_ERROR_OK;
    ve_array *x = &(args->fnorm.x);
    ve_array *y = &(args->fnorm.y);
    ve_array *work1 = &(args->fnorm.work1);
    ve_array *work2 = &(args->fnorm.work2);
    int64_t axis1 = args->fnorm.axis1;
    int64_t axis2 = args->fnorm.axis2;
    switch(x->dtype) {
    case ve_f32:
        err = nlcpy_fnorm_s(x, y, work1, work2, axis1, axis2, psw);
        break;
    case ve_f64:
        err = nlcpy_fnorm_d(x, y, work1, work2, axis1, axis2, psw);
        break;
    case ve_c64:
        err = nlcpy_fnorm_c(x, y, work1, work2, axis1, axis2, psw);
        break;
    case ve_c128:
        err = nlcpy_fnorm_z(x, y, work1, work2, axis1, axis2, psw);
        break;
    default:
        err = NLCPY_ERROR_DTYPE;
    }
    return (uint64_t)err;
}

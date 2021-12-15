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
#include <complex.h>

#include "nlcpy.h"




inline void clip_compare_bool(int32_t a, int32_t amin, int32_t amax, int32_t *pout) {
    if (amax <= amin) {
        *pout = amax;
    } else if (a < amin) {
        *pout = amin;
    } else if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_i32(int32_t a, int32_t amin, int32_t amax, int32_t *pout) {
    if (amax <= amin) {
        *pout = amax;
    } else if (a < amin) {
        *pout = amin;
    } else if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_i64(int64_t a, int64_t amin, int64_t amax, int64_t *pout) {
    if (amax <= amin) {
        *pout = amax;
    } else if (a < amin) {
        *pout = amin;
    } else if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_u32(uint32_t a, uint32_t amin, uint32_t amax, uint32_t *pout) {
    if (amax <= amin) {
        *pout = amax;
    } else if (a < amin) {
        *pout = amin;
    } else if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_u64(uint64_t a, uint64_t amin, uint64_t amax, uint64_t *pout) {
    if (amax <= amin) {
        *pout = amax;
    } else if (a < amin) {
        *pout = amin;
    } else if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_f32(float a, float amin, float amax, float *pout) {
    if (amax <= amin) {
        *pout = amax;
    } else if (a < amin) {
        *pout = amin;
    } else if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_f64(double a, double amin, double amax, double *pout) {
    if (amax <= amin) {
        *pout = amax;
    } else if (a < amin) {
        *pout = amin;
    } else if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}


inline void clip_compare_maximum_bool(int32_t a, int32_t amax, int32_t *pout) {
    if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_maximum_i32(int32_t a, int32_t amax, int32_t *pout) {
    if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_maximum_i64(int64_t a, int64_t amax, int64_t *pout) {
    if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_maximum_u32(uint32_t a, uint32_t amax, uint32_t *pout) {
    if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_maximum_u64(uint64_t a, uint64_t amax, uint64_t *pout) {
    if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_maximum_f32(float a, float amax, float *pout) {
    if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_maximum_f64(double a, double amax, double *pout) {
    if (a > amax) {
        *pout = amax;
    } else {
        *pout = a;
    }
}


inline void clip_compare_minimum_bool(int32_t a, int32_t amin, int32_t *pout) {
    if (a < amin) {
        *pout = amin;
    } else {
        *pout = a;
    }
}

inline void clip_compare_minimum_i32(int32_t a, int32_t amin, int32_t *pout) {
    if (a < amin) {
        *pout = amin;
    } else {
        *pout = a;
    }
}

inline void clip_compare_minimum_i64(int64_t a, int64_t amin, int64_t *pout) {
    if (a < amin) {
        *pout = amin;
    } else {
        *pout = a;
    }
}

inline void clip_compare_minimum_u32(uint32_t a, uint32_t amin, uint32_t *pout) {
    if (a < amin) {
        *pout = amin;
    } else {
        *pout = a;
    }
}

inline void clip_compare_minimum_u64(uint64_t a, uint64_t amin, uint64_t *pout) {
    if (a < amin) {
        *pout = amin;
    } else {
        *pout = a;
    }
}

inline void clip_compare_minimum_f32(float a, float amin, float *pout) {
    if (a < amin) {
        *pout = amin;
    } else {
        *pout = a;
    }
}

inline void clip_compare_minimum_f64(double a, double amin, double *pout) {
    if (a < amin) {
        *pout = amin;
    } else {
        *pout = a;
    }
}


inline void clip_compare_c64(float _Complex a, float _Complex amin, float _Complex amax, float _Complex *pout) {
    if (crealf(amax) <= crealf(amin)) {
        *pout = amax;
    } else if (crealf(a) < crealf(amin) || crealf(a) == crealf(amin) && cimagf(a) < cimagf(amin)) {
        *pout = amin;
    } else if (crealf(a) > crealf(amax) || crealf(a) == crealf(amax) && cimagf(a) > cimagf(amax)) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_c128(double _Complex a, double _Complex amin, double _Complex amax, double _Complex *pout) {
    if (creal(amax) <= creal(amin)) {
        *pout = amax;
    } else if (creal(a) < creal(amin) || creal(a) == creal(amin) && cimag(a) < cimag(amin)) {
        *pout = amin;
    } else if (creal(a) > creal(amax) || creal(a) == creal(amax) && cimag(a) > cimag(amax)) {
        *pout = amax;
    } else {
        *pout = a;
    }
}


inline void clip_compare_maximum_c64(float _Complex a, float _Complex amax, float _Complex *pout) {
    if (crealf(a) > crealf(amax) || crealf(a) == crealf(amax) && cimagf(a) > cimagf(amax)) {
        *pout = amax;
    } else {
        *pout = a;
    }
}

inline void clip_compare_maximum_c128(double _Complex a, double _Complex amax, double _Complex *pout) {
    if (creal(a) > creal(amax) || creal(a) == creal(amax) && cimag(a) > cimag(amax)) {
        *pout = amax;
    } else {
        *pout = a;
    }
}


inline void clip_compare_minimum_c64(float _Complex a, float _Complex amin, float _Complex *pout) {
    if (crealf(a) < crealf(amin) || crealf(a) == crealf(amin) && cimagf(a) < cimagf(amin)) {
        *pout = amin;
    } else {
        *pout = a;
    }
}

inline void clip_compare_minimum_c128(double _Complex a, double _Complex amin, double _Complex *pout) {
    if (creal(a) < creal(amin) || creal(a) == creal(amin) && cimag(a) < cimag(amin)) {
        *pout = amin;
    } else {
        *pout = a;
    }
}



uint64_t clip_bool(ve_array *a, ve_array *out, ve_array *amin, ve_array *amax, ve_array *where, int32_t no_out, int32_t *psw)
{
    int32_t *pa = (int32_t *)nlcpy__get_ptr(a);
    int32_t *pout = (int32_t *)nlcpy__get_ptr(out);
    int32_t *pmin = (int32_t *)nlcpy__get_ptr(amin);
    int32_t *pmax = (int32_t *)nlcpy__get_ptr(amax);
    Bint *pw = (Bint *)nlcpy__get_ptr(where);
    if (!pa || !pout || !pmin || !pmax || !pw) return NLCPY_ERROR_MEMORY;

    if (a->ndim < 2) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        if (amin->size > 0 && amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_bool(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_bool(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                }
            }
        } else if (amin->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_minimum_bool(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_minimum_bool(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                }
            }
        } else if (amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_maximum_bool(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_maximum_bool(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                }
            }
        }
}
    } else if (a->ndim <= NLCPY_MAXNDIM) {
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */

        int64_t i, j;
        int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__rearrange_axis(a, idx);
        int64_t n_inner = a->ndim - 1;
        int64_t n_outer = 0;
        int64_t n_inner2 = idx[a->ndim - 1];
        int64_t n_outer2 = idx[0];
        int64_t ia = 0;
        int64_t iout = 0;
        int64_t iw = 0;
        int64_t ia0 = a->strides[n_inner2] / a->itemsize;
        int64_t iout0 = out->strides[n_inner2] / out->itemsize;
        int64_t *cnt_a = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__reset_coords(cnt_a, a->ndim);

        const int64_t lenm = a->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;

        if (amin->size > 0 && amax->size > 0) {
            int64_t imin = 0;
            int64_t imax = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_bool(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_bool(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amin->size > 0) {
            int64_t imin = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_minimum_bool(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_minimum_bool(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amax->size > 0) {
            int64_t imax = 0;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_maximum_bool(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_maximum_bool(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t clip_i32(ve_array *a, ve_array *out, ve_array *amin, ve_array *amax, ve_array *where, int32_t no_out, int32_t *psw)
{
    int32_t *pa = (int32_t *)nlcpy__get_ptr(a);
    int32_t *pout = (int32_t *)nlcpy__get_ptr(out);
    int32_t *pmin = (int32_t *)nlcpy__get_ptr(amin);
    int32_t *pmax = (int32_t *)nlcpy__get_ptr(amax);
    Bint *pw = (Bint *)nlcpy__get_ptr(where);
    if (!pa || !pout || !pmin || !pmax || !pw) return NLCPY_ERROR_MEMORY;

    if (a->ndim < 2) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        if (amin->size > 0 && amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_i32(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_i32(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                }
            }
        } else if (amin->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_minimum_i32(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_minimum_i32(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                }
            }
        } else if (amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_maximum_i32(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_maximum_i32(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                }
            }
        }
}
    } else if (a->ndim <= NLCPY_MAXNDIM) {
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */

        int64_t i, j;
        int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__rearrange_axis(a, idx);
        int64_t n_inner = a->ndim - 1;
        int64_t n_outer = 0;
        int64_t n_inner2 = idx[a->ndim - 1];
        int64_t n_outer2 = idx[0];
        int64_t ia = 0;
        int64_t iout = 0;
        int64_t iw = 0;
        int64_t ia0 = a->strides[n_inner2] / a->itemsize;
        int64_t iout0 = out->strides[n_inner2] / out->itemsize;
        int64_t *cnt_a = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__reset_coords(cnt_a, a->ndim);

        const int64_t lenm = a->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;

        if (amin->size > 0 && amax->size > 0) {
            int64_t imin = 0;
            int64_t imax = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_i32(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_i32(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amin->size > 0) {
            int64_t imin = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_minimum_i32(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_minimum_i32(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amax->size > 0) {
            int64_t imax = 0;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_maximum_i32(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_maximum_i32(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t clip_i64(ve_array *a, ve_array *out, ve_array *amin, ve_array *amax, ve_array *where, int32_t no_out, int32_t *psw)
{
    int64_t *pa = (int64_t *)nlcpy__get_ptr(a);
    int64_t *pout = (int64_t *)nlcpy__get_ptr(out);
    int64_t *pmin = (int64_t *)nlcpy__get_ptr(amin);
    int64_t *pmax = (int64_t *)nlcpy__get_ptr(amax);
    Bint *pw = (Bint *)nlcpy__get_ptr(where);
    if (!pa || !pout || !pmin || !pmax || !pw) return NLCPY_ERROR_MEMORY;

    if (a->ndim < 2) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        if (amin->size > 0 && amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_i64(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_i64(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                }
            }
        } else if (amin->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_minimum_i64(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_minimum_i64(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                }
            }
        } else if (amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_maximum_i64(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_maximum_i64(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                }
            }
        }
}
    } else if (a->ndim <= NLCPY_MAXNDIM) {
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */

        int64_t i, j;
        int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__rearrange_axis(a, idx);
        int64_t n_inner = a->ndim - 1;
        int64_t n_outer = 0;
        int64_t n_inner2 = idx[a->ndim - 1];
        int64_t n_outer2 = idx[0];
        int64_t ia = 0;
        int64_t iout = 0;
        int64_t iw = 0;
        int64_t ia0 = a->strides[n_inner2] / a->itemsize;
        int64_t iout0 = out->strides[n_inner2] / out->itemsize;
        int64_t *cnt_a = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__reset_coords(cnt_a, a->ndim);

        const int64_t lenm = a->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;

        if (amin->size > 0 && amax->size > 0) {
            int64_t imin = 0;
            int64_t imax = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_i64(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_i64(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amin->size > 0) {
            int64_t imin = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_minimum_i64(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_minimum_i64(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amax->size > 0) {
            int64_t imax = 0;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_maximum_i64(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_maximum_i64(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t clip_u32(ve_array *a, ve_array *out, ve_array *amin, ve_array *amax, ve_array *where, int32_t no_out, int32_t *psw)
{
    uint32_t *pa = (uint32_t *)nlcpy__get_ptr(a);
    uint32_t *pout = (uint32_t *)nlcpy__get_ptr(out);
    uint32_t *pmin = (uint32_t *)nlcpy__get_ptr(amin);
    uint32_t *pmax = (uint32_t *)nlcpy__get_ptr(amax);
    Bint *pw = (Bint *)nlcpy__get_ptr(where);
    if (!pa || !pout || !pmin || !pmax || !pw) return NLCPY_ERROR_MEMORY;

    if (a->ndim < 2) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        if (amin->size > 0 && amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_u32(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_u32(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                }
            }
        } else if (amin->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_minimum_u32(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_minimum_u32(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                }
            }
        } else if (amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_maximum_u32(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_maximum_u32(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                }
            }
        }
}
    } else if (a->ndim <= NLCPY_MAXNDIM) {
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */

        int64_t i, j;
        int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__rearrange_axis(a, idx);
        int64_t n_inner = a->ndim - 1;
        int64_t n_outer = 0;
        int64_t n_inner2 = idx[a->ndim - 1];
        int64_t n_outer2 = idx[0];
        int64_t ia = 0;
        int64_t iout = 0;
        int64_t iw = 0;
        int64_t ia0 = a->strides[n_inner2] / a->itemsize;
        int64_t iout0 = out->strides[n_inner2] / out->itemsize;
        int64_t *cnt_a = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__reset_coords(cnt_a, a->ndim);

        const int64_t lenm = a->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;

        if (amin->size > 0 && amax->size > 0) {
            int64_t imin = 0;
            int64_t imax = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_u32(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_u32(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amin->size > 0) {
            int64_t imin = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_minimum_u32(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_minimum_u32(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amax->size > 0) {
            int64_t imax = 0;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_maximum_u32(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_maximum_u32(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t clip_u64(ve_array *a, ve_array *out, ve_array *amin, ve_array *amax, ve_array *where, int32_t no_out, int32_t *psw)
{
    uint64_t *pa = (uint64_t *)nlcpy__get_ptr(a);
    uint64_t *pout = (uint64_t *)nlcpy__get_ptr(out);
    uint64_t *pmin = (uint64_t *)nlcpy__get_ptr(amin);
    uint64_t *pmax = (uint64_t *)nlcpy__get_ptr(amax);
    Bint *pw = (Bint *)nlcpy__get_ptr(where);
    if (!pa || !pout || !pmin || !pmax || !pw) return NLCPY_ERROR_MEMORY;

    if (a->ndim < 2) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        if (amin->size > 0 && amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_u64(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_u64(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                }
            }
        } else if (amin->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_minimum_u64(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_minimum_u64(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                }
            }
        } else if (amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_maximum_u64(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_maximum_u64(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                }
            }
        }
}
    } else if (a->ndim <= NLCPY_MAXNDIM) {
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */

        int64_t i, j;
        int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__rearrange_axis(a, idx);
        int64_t n_inner = a->ndim - 1;
        int64_t n_outer = 0;
        int64_t n_inner2 = idx[a->ndim - 1];
        int64_t n_outer2 = idx[0];
        int64_t ia = 0;
        int64_t iout = 0;
        int64_t iw = 0;
        int64_t ia0 = a->strides[n_inner2] / a->itemsize;
        int64_t iout0 = out->strides[n_inner2] / out->itemsize;
        int64_t *cnt_a = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__reset_coords(cnt_a, a->ndim);

        const int64_t lenm = a->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;

        if (amin->size > 0 && amax->size > 0) {
            int64_t imin = 0;
            int64_t imax = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_u64(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_u64(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amin->size > 0) {
            int64_t imin = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_minimum_u64(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_minimum_u64(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amax->size > 0) {
            int64_t imax = 0;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_maximum_u64(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_maximum_u64(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t clip_f32(ve_array *a, ve_array *out, ve_array *amin, ve_array *amax, ve_array *where, int32_t no_out, int32_t *psw)
{
    float *pa = (float *)nlcpy__get_ptr(a);
    float *pout = (float *)nlcpy__get_ptr(out);
    float *pmin = (float *)nlcpy__get_ptr(amin);
    float *pmax = (float *)nlcpy__get_ptr(amax);
    Bint *pw = (Bint *)nlcpy__get_ptr(where);
    if (!pa || !pout || !pmin || !pmax || !pw) return NLCPY_ERROR_MEMORY;

    if (a->ndim < 2) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        if (amin->size > 0 && amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_f32(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_f32(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                }
            }
        } else if (amin->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_minimum_f32(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_minimum_f32(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                }
            }
        } else if (amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_maximum_f32(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_maximum_f32(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                }
            }
        }
}
    } else if (a->ndim <= NLCPY_MAXNDIM) {
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */

        int64_t i, j;
        int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__rearrange_axis(a, idx);
        int64_t n_inner = a->ndim - 1;
        int64_t n_outer = 0;
        int64_t n_inner2 = idx[a->ndim - 1];
        int64_t n_outer2 = idx[0];
        int64_t ia = 0;
        int64_t iout = 0;
        int64_t iw = 0;
        int64_t ia0 = a->strides[n_inner2] / a->itemsize;
        int64_t iout0 = out->strides[n_inner2] / out->itemsize;
        int64_t *cnt_a = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__reset_coords(cnt_a, a->ndim);

        const int64_t lenm = a->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;

        if (amin->size > 0 && amax->size > 0) {
            int64_t imin = 0;
            int64_t imax = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_f32(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_f32(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amin->size > 0) {
            int64_t imin = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_minimum_f32(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_minimum_f32(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amax->size > 0) {
            int64_t imax = 0;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_maximum_f32(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_maximum_f32(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t clip_f64(ve_array *a, ve_array *out, ve_array *amin, ve_array *amax, ve_array *where, int32_t no_out, int32_t *psw)
{
    double *pa = (double *)nlcpy__get_ptr(a);
    double *pout = (double *)nlcpy__get_ptr(out);
    double *pmin = (double *)nlcpy__get_ptr(amin);
    double *pmax = (double *)nlcpy__get_ptr(amax);
    Bint *pw = (Bint *)nlcpy__get_ptr(where);
    if (!pa || !pout || !pmin || !pmax || !pw) return NLCPY_ERROR_MEMORY;

    if (a->ndim < 2) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        if (amin->size > 0 && amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_f64(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_f64(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                }
            }
        } else if (amin->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_minimum_f64(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_minimum_f64(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                }
            }
        } else if (amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_maximum_f64(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_maximum_f64(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                }
            }
        }
}
    } else if (a->ndim <= NLCPY_MAXNDIM) {
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */

        int64_t i, j;
        int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__rearrange_axis(a, idx);
        int64_t n_inner = a->ndim - 1;
        int64_t n_outer = 0;
        int64_t n_inner2 = idx[a->ndim - 1];
        int64_t n_outer2 = idx[0];
        int64_t ia = 0;
        int64_t iout = 0;
        int64_t iw = 0;
        int64_t ia0 = a->strides[n_inner2] / a->itemsize;
        int64_t iout0 = out->strides[n_inner2] / out->itemsize;
        int64_t *cnt_a = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__reset_coords(cnt_a, a->ndim);

        const int64_t lenm = a->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;

        if (amin->size > 0 && amax->size > 0) {
            int64_t imin = 0;
            int64_t imax = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_f64(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_f64(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amin->size > 0) {
            int64_t imin = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_minimum_f64(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_minimum_f64(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amax->size > 0) {
            int64_t imax = 0;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_maximum_f64(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_maximum_f64(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t clip_c64(ve_array *a, ve_array *out, ve_array *amin, ve_array *amax, ve_array *where, int32_t no_out, int32_t *psw)
{
    float _Complex *pa = (float _Complex *)nlcpy__get_ptr(a);
    float _Complex *pout = (float _Complex *)nlcpy__get_ptr(out);
    float _Complex *pmin = (float _Complex *)nlcpy__get_ptr(amin);
    float _Complex *pmax = (float _Complex *)nlcpy__get_ptr(amax);
    Bint *pw = (Bint *)nlcpy__get_ptr(where);
    if (!pa || !pout || !pmin || !pmax || !pw) return NLCPY_ERROR_MEMORY;

    if (a->ndim < 2) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        if (amin->size > 0 && amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_c64(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_c64(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                }
            }
        } else if (amin->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_minimum_c64(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_minimum_c64(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                }
            }
        } else if (amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_maximum_c64(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_maximum_c64(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                }
            }
        }
}
    } else if (a->ndim <= NLCPY_MAXNDIM) {
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */

        int64_t i, j;
        int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__rearrange_axis(a, idx);
        int64_t n_inner = a->ndim - 1;
        int64_t n_outer = 0;
        int64_t n_inner2 = idx[a->ndim - 1];
        int64_t n_outer2 = idx[0];
        int64_t ia = 0;
        int64_t iout = 0;
        int64_t iw = 0;
        int64_t ia0 = a->strides[n_inner2] / a->itemsize;
        int64_t iout0 = out->strides[n_inner2] / out->itemsize;
        int64_t *cnt_a = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__reset_coords(cnt_a, a->ndim);

        const int64_t lenm = a->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;

        if (amin->size > 0 && amax->size > 0) {
            int64_t imin = 0;
            int64_t imax = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_c64(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_c64(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amin->size > 0) {
            int64_t imin = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_minimum_c64(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_minimum_c64(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amax->size > 0) {
            int64_t imax = 0;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_maximum_c64(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_maximum_c64(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t clip_c128(ve_array *a, ve_array *out, ve_array *amin, ve_array *amax, ve_array *where, int32_t no_out, int32_t *psw)
{
    double _Complex *pa = (double _Complex *)nlcpy__get_ptr(a);
    double _Complex *pout = (double _Complex *)nlcpy__get_ptr(out);
    double _Complex *pmin = (double _Complex *)nlcpy__get_ptr(amin);
    double _Complex *pmax = (double _Complex *)nlcpy__get_ptr(amax);
    Bint *pw = (Bint *)nlcpy__get_ptr(where);
    if (!pa || !pout || !pmin || !pmax || !pw) return NLCPY_ERROR_MEMORY;

    if (a->ndim < 2) {
#ifdef _OPENMP
#pragma omp single
#endif /* _OPENMP */
{
        if (amin->size > 0 && amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_c128(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_c128(pa[i*ia0], pmin[i*imin0], pmax[i*imax0], pout+i*iout0);
                }
            }
        } else if (amin->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imin0 = amin->strides[0] / amin->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_minimum_c128(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_minimum_c128(pa[i*ia0], pmin[i*imin0], pout+i*iout0);
                }
            }
        } else if (amax->size > 0) {
            uint64_t ia0 = a->strides[0] / a->itemsize;
            uint64_t iout0 = out->strides[0] / out->itemsize;
            uint64_t imax0 = amax->strides[0] / amax->itemsize;
            if (where->size) {
                uint64_t iw0 = where->strides[0] / where->itemsize;
                for (int64_t i = 0; i < a->shape[0]; i++) {
                    if (pw[i*iw0]) {
                        clip_compare_maximum_c128(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                    } else if (no_out){
                        pout[i*iout0] = pa[i*ia0];
                    }
                }
            } else {
                for (int64_t i = 0; i < a->size; i++) {
                    clip_compare_maximum_c128(pa[i*ia0], pmax[i*imax0], pout+i*iout0);
                }
            }
        }
}
    } else if (a->ndim <= NLCPY_MAXNDIM) {
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int it = omp_get_thread_num();
#else
        const int nt = 1;
        const int it = 0;
#endif /* _OPENMP */

        int64_t i, j;
        int64_t *idx = (int64_t *)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__rearrange_axis(a, idx);
        int64_t n_inner = a->ndim - 1;
        int64_t n_outer = 0;
        int64_t n_inner2 = idx[a->ndim - 1];
        int64_t n_outer2 = idx[0];
        int64_t ia = 0;
        int64_t iout = 0;
        int64_t iw = 0;
        int64_t ia0 = a->strides[n_inner2] / a->itemsize;
        int64_t iout0 = out->strides[n_inner2] / out->itemsize;
        int64_t *cnt_a = (int64_t*)alloca(sizeof(int64_t) * a->ndim);
        nlcpy__reset_coords(cnt_a, a->ndim);

        const int64_t lenm = a->shape[n_outer2];
        const int64_t cntm_s = lenm * it / nt;
        const int64_t cntm_e = lenm * (it + 1) / nt;

        if (amin->size > 0 && amax->size > 0) {
            int64_t imin = 0;
            int64_t imax = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_c128(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_c128(pa[ia+i*ia0], pmin[imin+i*imin0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amin->size > 0) {
            int64_t imin = 0;
            int64_t imin0 = amin->strides[n_inner2] / amin->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_minimum_c128(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imin = cntm * amin->strides[n_outer2] / amin->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_minimum_c128(pa[ia+i*ia0], pmin[imin+i*imin0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imin += amin->strides[jj] / amin->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imin -= (amin->strides[jj] / amin->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        } else if (amax->size > 0) {
            int64_t imax = 0;
            int64_t imax0 = amax->strides[n_inner2] / amax->itemsize;

            if (where->size) {
                int64_t iw0 = where->strides[n_inner2] / where->itemsize;
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    iw = cntm * where->strides[n_outer2] / where->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            if (pw[iw+i*iw0]) {
                                clip_compare_maximum_c128(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                            } else if (no_out){
                                pout[iout+i*iout0] = pa[ia+i*ia0];
                            }
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                iw += where->strides[jj] / where->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            iw -= (where->strides[jj] / where->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            } else {
                for (int64_t cntm = cntm_s; cntm < cntm_e; cntm++) {
                    ia = cntm * a->strides[n_outer2] / a->itemsize;
                    iout = cntm * out->strides[n_outer2] / out->itemsize;
                    imax = cntm * amax->strides[n_outer2] / amax->itemsize;
                    do {
                        for (i = 0; i < a->shape[n_inner2]; i++) {
                            clip_compare_maximum_c128(pa[ia+i*ia0], pmax[imax+i*imax0], pout+iout+i*iout0);
                        }
                        for (j = n_inner - 1; j > 0; j--) {
                            uint64_t jj = idx[j];
                            if (++cnt_a[jj] < a->shape[jj]) {
                                ia += a->strides[jj] / a->itemsize;
                                iout += out->strides[jj] / out->itemsize;
                                imax += amax->strides[jj] / amax->itemsize;
                                break;
                            }
                            ia -= (a->strides[jj] / a->itemsize) * (a->shape[jj] - 1);
                            iout -= (out->strides[jj] / out->itemsize) * (a->shape[jj] - 1);
                            imax -= (amax->strides[jj] / amax->itemsize) * (a->shape[jj] - 1);
                            cnt_a[jj] = 0;
                        }
                    } while(j > 0);
                }
            }
        }
    } else {
        return (uint64_t)NLCPY_ERROR_NDIM;
    }
    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_clip(ve_arguments *args, int32_t *psw)
{
    ve_array *a = &(args->clip.a);
    ve_array *out = &(args->clip.out);
    ve_array *work = &(args->clip.work);
    ve_array *amin = &(args->clip.amin);
    ve_array *amax = &(args->clip.amax);
    ve_array *where = &(args->clip.where);
    int32_t no_out = (out->ve_adr == work->ve_adr);
    uint64_t err = NLCPY_ERROR_OK;
    if (out->dtype == work->dtype) {
        switch (out->dtype) {
        case ve_i32: err = clip_i32 (a, out, amin, amax, where, no_out, psw); break;
        case ve_i64: err = clip_i64 (a, out, amin, amax, where, no_out, psw); break;
        case ve_u32: err = clip_u32 (a, out, amin, amax, where, no_out, psw); break;
        case ve_u64: err = clip_u64 (a, out, amin, amax, where, no_out, psw); break;
        case ve_f32: err = clip_f32 (a, out, amin, amax, where, no_out, psw); break;
        case ve_f64: err = clip_f64 (a, out, amin, amax, where, no_out, psw); break;
        case ve_c64: err = clip_c64 (a, out, amin, amax, where, no_out, psw); break;
        case ve_c128: err = clip_c128 (a, out, amin, amax, where, no_out, psw); break;
        case ve_bool: err = clip_bool (a, out, amin, amax, where, no_out, psw); break;
        default: return (uint64_t)NLCPY_ERROR_DTYPE;
        }
    } else {
        switch (work->dtype) {
        case ve_i32: err |= clip_i32 (a, work, amin, amax, where, no_out, psw); break;
        case ve_i64: err |= clip_i64 (a, work, amin, amax, where, no_out, psw); break;
        case ve_u32: err |= clip_u32 (a, work, amin, amax, where, no_out, psw); break;
        case ve_u64: err |= clip_u64 (a, work, amin, amax, where, no_out, psw); break;
        case ve_f32: err |= clip_f32 (a, work, amin, amax, where, no_out, psw); break;
        case ve_f64: err |= clip_f64 (a, work, amin, amax, where, no_out, psw); break;
        case ve_c64: err |= clip_c64 (a, work, amin, amax, where, no_out, psw); break;
        case ve_c128: err |= clip_c128 (a, work, amin, amax, where, no_out, psw); break;
        case ve_bool: err |= clip_bool (a, work, amin, amax, where, no_out, psw); break;
        default: return (uint64_t)NLCPY_ERROR_DTYPE;
        }

#ifdef _OPENMP
#pragma omp barrier
#endif /* _OPENMP */

        int32_t pswc;
        switch (out->dtype) {
        case ve_i32: err |= nlcpy_cast_i32 (work, out, 0, where, &pswc); break;
        case ve_i64: err |= nlcpy_cast_i64 (work, out, 0, where, &pswc); break;
        case ve_u32: err |= nlcpy_cast_u32 (work, out, 0, where, &pswc); break;
        case ve_u64: err |= nlcpy_cast_u64 (work, out, 0, where, &pswc); break;
        case ve_f32: err |= nlcpy_cast_f32 (work, out, 0, where, &pswc); break;
        case ve_f64: err |= nlcpy_cast_f64 (work, out, 0, where, &pswc); break;
        case ve_c64: err |= nlcpy_cast_c64 (work, out, 0, where, &pswc); break;
        case ve_c128: err |= nlcpy_cast_c128 (work, out, 0, where, &pswc); break;
        case ve_bool: err |= nlcpy_cast_bool (work, out, 0, where, &pswc); break;
        default: return (uint64_t)NLCPY_ERROR_DTYPE;
        }
        *psw |= pswc;
    }
    return (uint64_t)err;
}

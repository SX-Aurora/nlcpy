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
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <alloca.h>
#include <assert.h>
#include <complex.h>

#include "nlcpy.h"

define(<--@macro_eig_real@-->,<--@
extern void $1geev_(const char *jobvl, const char *jobvr, const int64_t *n, $2 *pa, const int64_t *lda, $2 *pwr, $2 *pwi, $2 *pvl, const int64_t *ldvl, $2 *pvr, const int64_t *ldvr, $2 *pwork, const int64_t *lwork, int64_t *info);
uint64_t nlcpy_eig_$1(ve_array *a, ve_array *wr, ve_array *wi, ve_array *vr, ve_array *vc, ve_array *work, const char jobvr, int64_t *info, int32_t *psw)
{
    const int64_t n = a->shape[0];
    const int64_t lda = n;
    const int64_t ldvl = 1;
    const int64_t ldvr = n;
    const int64_t lwork = work->size;
    const char jobvl = 'N';

    $2 *pa = ($2*)a->ve_adr;
    $2 *pwr = ($2*)wr->ve_adr;
    $2 *pwi = ($2*)wi->ve_adr;
    $2 *pvr = ($2*)vr->ve_adr;
    double _Complex* const pvc = (double _Complex*)vc->ve_adr;
    $2 *pwork = ($2*)work->ve_adr;
    if (!pa || !pwr || !pwi || !pvr || !pvc || !pwork ) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ia = 0;
    int64_t iwr = 0;
    int64_t iwi = 0;
    int64_t ivr = 0;
    int64_t ivc = 0;
    int64_t stride_j_vr = vr->strides[0] / vr->itemsize;
    int64_t stride_j_vc = vc->strides[0] / vc->itemsize;
    int64_t stride_k_vr = vr->strides[1] / vr->itemsize;
    int64_t stride_k_vc = vc->strides[1] / vc->itemsize;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*a->ndim);
    for (i = 0; i < a->ndim; i++) cnt[i] = 0;
    do {
        $1geev_(&jobvl, &jobvr, &n, pa+ia, &lda, pwr+iwr, pwi+iwi, pvr, &ldvr, pvr+ivr, &ldvr, pwork, &lwork, info);
        if (jobvr == 'V') {
            for (int64_t k = 0; k < vr->shape[0]; k++) {
                int64_t iwi2 = iwi + k * (wi->strides[0] / wi->itemsize);
                int64_t ivr2 = ivr + k * (vr->strides[1] / vr->itemsize);
                int64_t ivc2 = ivc + k * (vc->strides[1] / vc->itemsize);
                if (pwi[iwi2]) {
#pragma _NEC ivdep
                    for (int64_t j = 0; j < vr->shape[1]; j++) {
                        pvc[ivc2] = pvr[ivr2] + I*pvr[ivr2+stride_k_vr];
                        pvc[ivc2+stride_k_vc] = pvr[ivr2] - I*pvr[ivr2+stride_k_vr];
                        ivr2 += stride_j_vr;
                        ivc2 += stride_j_vc;
                    }
                    k++;
                } else {
                    for (int64_t j = 0; j < vr->shape[1]; j++) {
                        pvc[ivc2] = pvr[ivr2];
                        ivr2 += stride_j_vr;
                        ivc2 += stride_j_vc;
                    }
                }
            }
        }
        for (i = 2; i < a->ndim; i++) {
            if (++cnt[i] < a->shape[i]) {
                ia += a->strides[i] / a->itemsize;
                iwr += wr->strides[i - 1] / wr->itemsize;
                iwi += wi->strides[i - 1] / wi->itemsize;
                ivr += vr->strides[i] / vr->itemsize;
                ivc += vc->strides[i] / vc->itemsize;
                break;
            }
            cnt[i] = 0;
            ia -= (a->strides[i] / a->itemsize) * (a->shape[i] - 1);
            iwr -= (wr->strides[i - 1] / wr->itemsize) * (a->shape[i] - 1);
            iwi -= (wi->strides[i - 1] / wi->itemsize) * (a->shape[i] - 1);
            ivr -= (vr->strides[i] / vr->itemsize) * (a->shape[i] - 1);
            ivc -= (vc->strides[i] / vc->itemsize) * (a->shape[i] - 1);
        }
    } while(i < a->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
macro_eig_real(s, float)dnl
macro_eig_real(d, double)dnl

define(<--@macro_eig_complex@-->,<--@
extern void $1geev_(const char *jobvl, const char *jobvr, const int64_t *n, $2 *pa, const int64_t *lda, $2 *pw, $2 *pvl, const int64_t *ldvl, $2 *pvr, const int64_t *ldvr, $2 *pwork, const int64_t *lwork, $3 *prwork, int64_t *info);
uint64_t nlcpy_eig_$1(ve_array *a, ve_array *w, ve_array *v, ve_array *work, ve_array *rwork, const char jobvr, int64_t *info, int32_t *psw)
{
    const int64_t n = a->shape[0];
    const int64_t lda = n;
    const int64_t ldvl = 1;
    const int64_t ldvr = n;
    const int64_t lwork = work->size;
    const char jobvl = 'N';

    $2 *pa = ($2*)a->ve_adr;
    $2 *pw = ($2*)w->ve_adr;
    $2 *pv = ($2*)v->ve_adr;
    $2 *pwork = ($2*)work->ve_adr;
    $3 *prwork = ($3*)rwork->ve_adr;
    if (!pa || !pw || !pv || !pwork || !prwork) {
        return NLCPY_ERROR_MEMORY;
    }

    int64_t i;
    int64_t ia = 0;
    int64_t iw = 0;
    int64_t iv = 0;
    int64_t *cnt = (int64_t*)alloca(sizeof(int64_t)*a->ndim);
    for (i = 0; i < a->ndim; i++) cnt[i] = 0;
    do {
        $1geev_(&jobvl, &jobvr, &n, pa+ia, &lda, pw+iw, pv, &ldvr, pv+iv, &ldvr, pwork, &lwork, prwork, info);
        for (i = 2; i < a->ndim; i++) {
            if (++cnt[i] < a->shape[i]) {
                ia += a->strides[i] / a->itemsize;
                iw += w->strides[i-1] / w->itemsize;
                iv += v->strides[i] / v->itemsize;
                break;
            }
            cnt[i] = 0;
            ia -= (a->strides[i] / a->itemsize) * (a->shape[i] - 1);
            iw -= (w->strides[i-1] / w->itemsize) * (a->shape[i] - 1);
            iv -= (v->strides[i] / v->itemsize) * (a->shape[i] - 1);
        }
    } while(i < a->ndim);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl
macro_eig_complex(c, float _Complex, float)dnl
macro_eig_complex(z, double _Complex, double)dnl

uint64_t nlcpy_eig(ve_array *a, ve_array *wr, ve_array *wi, ve_array *vr, ve_array *vc, ve_array *work, ve_array *rwork, int64_t jobvr, int64_t *info, int32_t *psw) {
    uint64_t err = NLCPY_ERROR_OK;
    switch(a->dtype) {
    case ve_f32:
        err = nlcpy_eig_s(a, wr, wi, vr, vc, work, jobvr, info, psw);
        break;
    case ve_f64:
        err = nlcpy_eig_d(a, wr, wi, vr, vc, work, jobvr, info, psw);
        break;
    case ve_c64:
        err = nlcpy_eig_c(a, wr, vr, work, rwork, jobvr, info, psw);
        break;
    case ve_c128:
        err = nlcpy_eig_z(a, wr, vr, work, rwork, jobvr, info, psw);
        break;
    default:
        err = NLCPY_ERROR_DTYPE;
    }
    return (uint64_t)err;
}

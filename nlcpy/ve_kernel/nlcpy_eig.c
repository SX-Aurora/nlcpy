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


extern void sgeev_(const char *jobvl, const char *jobvr, const int64_t *n, float *pa, const int64_t *lda, float *pwr, float *pwi, float *pvl, const int64_t *ldvl, float *pvr, const int64_t *ldvr, float *pwork, const int64_t *lwork, int64_t *info);
uint64_t nlcpy_eig_s(ve_array *a, ve_array *wr, ve_array *wi, ve_array *vr, ve_array *vc, ve_array *work, const char jobvr, int64_t *info, int32_t *psw)
{
    const int64_t n = a->shape[0];
    const int64_t lda = n;
    const int64_t ldvl = 1;
    const int64_t ldvr = n;
    const int64_t lwork = work->size;
    const char jobvl = 'N';

    float *pa = (float*)a->ve_adr;
    float *pwr = (float*)wr->ve_adr;
    float *pwi = (float*)wi->ve_adr;
    float *pvr = (float*)vr->ve_adr;
    double _Complex* const pvc = (double _Complex*)vc->ve_adr;
    float *pwork = (float*)work->ve_adr;
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
        sgeev_(&jobvl, &jobvr, &n, pa+ia, &lda, pwr+iwr, pwi+iwi, pvr, &ldvr, pvr+ivr, &ldvr, pwork, &lwork, info);
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

extern void dgeev_(const char *jobvl, const char *jobvr, const int64_t *n, double *pa, const int64_t *lda, double *pwr, double *pwi, double *pvl, const int64_t *ldvl, double *pvr, const int64_t *ldvr, double *pwork, const int64_t *lwork, int64_t *info);
uint64_t nlcpy_eig_d(ve_array *a, ve_array *wr, ve_array *wi, ve_array *vr, ve_array *vc, ve_array *work, const char jobvr, int64_t *info, int32_t *psw)
{
    const int64_t n = a->shape[0];
    const int64_t lda = n;
    const int64_t ldvl = 1;
    const int64_t ldvr = n;
    const int64_t lwork = work->size;
    const char jobvl = 'N';

    double *pa = (double*)a->ve_adr;
    double *pwr = (double*)wr->ve_adr;
    double *pwi = (double*)wi->ve_adr;
    double *pvr = (double*)vr->ve_adr;
    double _Complex* const pvc = (double _Complex*)vc->ve_adr;
    double *pwork = (double*)work->ve_adr;
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
        dgeev_(&jobvl, &jobvr, &n, pa+ia, &lda, pwr+iwr, pwi+iwi, pvr, &ldvr, pvr+ivr, &ldvr, pwork, &lwork, info);
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


extern void cgeev_(const char *jobvl, const char *jobvr, const int64_t *n, float _Complex *pa, const int64_t *lda, float _Complex *pw, float _Complex *pvl, const int64_t *ldvl, float _Complex *pvr, const int64_t *ldvr, float _Complex *pwork, const int64_t *lwork, float *prwork, int64_t *info);
uint64_t nlcpy_eig_c(ve_array *a, ve_array *w, ve_array *v, ve_array *work, ve_array *rwork, const char jobvr, int64_t *info, int32_t *psw)
{
    const int64_t n = a->shape[0];
    const int64_t lda = n;
    const int64_t ldvl = 1;
    const int64_t ldvr = n;
    const int64_t lwork = work->size;
    const char jobvl = 'N';

    float _Complex *pa = (float _Complex*)a->ve_adr;
    float _Complex *pw = (float _Complex*)w->ve_adr;
    float _Complex *pv = (float _Complex*)v->ve_adr;
    float _Complex *pwork = (float _Complex*)work->ve_adr;
    float *prwork = (float*)rwork->ve_adr;
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
        cgeev_(&jobvl, &jobvr, &n, pa+ia, &lda, pw+iw, pv, &ldvr, pv+iv, &ldvr, pwork, &lwork, prwork, info);
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

extern void zgeev_(const char *jobvl, const char *jobvr, const int64_t *n, double _Complex *pa, const int64_t *lda, double _Complex *pw, double _Complex *pvl, const int64_t *ldvl, double _Complex *pvr, const int64_t *ldvr, double _Complex *pwork, const int64_t *lwork, double *prwork, int64_t *info);
uint64_t nlcpy_eig_z(ve_array *a, ve_array *w, ve_array *v, ve_array *work, ve_array *rwork, const char jobvr, int64_t *info, int32_t *psw)
{
    const int64_t n = a->shape[0];
    const int64_t lda = n;
    const int64_t ldvl = 1;
    const int64_t ldvr = n;
    const int64_t lwork = work->size;
    const char jobvl = 'N';

    double _Complex *pa = (double _Complex*)a->ve_adr;
    double _Complex *pw = (double _Complex*)w->ve_adr;
    double _Complex *pv = (double _Complex*)v->ve_adr;
    double _Complex *pwork = (double _Complex*)work->ve_adr;
    double *prwork = (double*)rwork->ve_adr;
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
        zgeev_(&jobvl, &jobvr, &n, pa+ia, &lda, pw+iw, pv, &ldvr, pv+iv, &ldvr, pwork, &lwork, prwork, info);
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

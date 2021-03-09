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
#include "nlcpy.h"

uint64_t nlcpy__get_scalar(ve_array *val) {
    switch(val->dtype) {
    case ve_bool:
        return (uint64_t)(&val->scalar.bint);
    case ve_i32:
        return (uint64_t)(&val->scalar.i4);
    case ve_i64:
        return (uint64_t)(&val->scalar.i8);
    case ve_u32:
        return (uint64_t)(&val->scalar.u4);
    case ve_u64:
        return (uint64_t)(&val->scalar.u8);
    case ve_f32:
        return (uint64_t)(&val->scalar.f4);
    case ve_f64:
        return (uint64_t)(&val->scalar.f8);
    case ve_c64:
        return (uint64_t)(&val->scalar.c8);
    case ve_c128:
        return (uint64_t)(&val->scalar.c16);
    default:
        return 0LU;
    }
    return;
}

uint64_t nlcpy__get_ptr(ve_array *a) {
    if (a->ve_adr == 0LU) {
        return nlcpy__get_scalar(a);
    } else {
        int64_t strides = 0;
//#pragma _NEC novector
        for (int64_t i = 0; i < NLCPY_MAXNDIM; i++) {
            strides += a->strides[i];
        }
        if (strides == 0) a->size = 1;
        return a->ve_adr;
    }
}


void nlcpy__reset_coords(int64_t *coords, int64_t size) {
    int64_t i;
#pragma _NEC novector
    for (i = 0; i < size; i++) {
        coords[i] = 0;
    }
}

void nlcpy__argnsort(ve_array *a, int64_t *idx, int64_t n) {
    int64_t i, j, tmp;
    int64_t wk_a[a->ndim], wk_idx[a->ndim];
#pragma _NEC novector
    for (i = 0; i < a->ndim; i++) {
        wk_a[i] = a->shape[i];
        wk_idx[i] = i;
    }

    // find top n max shape
    for (i = 0; i < n; i++) {
#pragma _NEC novector
        for (j = i + 1; j < a->ndim; j++) {
            if (wk_a[i] < wk_a[j]) {
                tmp = wk_a[i];
                wk_a[i] = wk_a[j];
                wk_a[j] = tmp;
                tmp = wk_idx[i];
                wk_idx[i] = wk_idx[j];
                wk_idx[j] = tmp;
            }
        } 
    }
#pragma _NEC novector
    for (i = 0; i < n; i++) idx[i] = wk_idx[i];
}

void nlcpy__rearrange_axis(ve_array *a, int64_t *idx) {
    int64_t i, j, tmp;
    int64_t wk_a[a->ndim], wk_idx[a->ndim];
#pragma _NEC novector
    for (i = 0; i < a->ndim; i++) {
        wk_a[i] = a->shape[i];
        wk_idx[i] = i;
    }

    // find top 2 max shape
    i = a->ndim-1; //inner
#pragma _NEC novector
    for (j = i; j >= 0; j--) {
        if (wk_a[i] < wk_a[j]) {
            tmp = wk_a[i];
            wk_a[i] = wk_a[j];
            wk_a[j] = tmp;
            tmp = wk_idx[i];
            wk_idx[i] = wk_idx[j];
            wk_idx[j] = tmp;
        }
    }
    i = 0; //outer
#pragma _NEC novector
    for (j = i + 1; j < a->ndim-1; j++) {
        if (wk_a[i] < wk_a[j]) {
            tmp = wk_a[i];
            wk_a[i] = wk_a[j];
            wk_a[j] = tmp;
            tmp = wk_idx[i];
            wk_idx[i] = wk_idx[j];
            wk_idx[j] = tmp;
        }
    }
#pragma _NEC novector
    for (i = 0; i < a->ndim; i++) idx[i] = wk_idx[i];
}

void nlcpy__exchange_shape_and_strides(ve_array *a) {
    int64_t i, j, sh, st, smax1, smax2, max1_idx, max2_idx, n_inner, n_outer;
    
    smax1 = 0;
    smax2 = 0;
    n_inner = a->ndim - 1;
    n_outer = 0;
    max1_idx = n_inner;
    max2_idx = n_outer;
    // find max and 2nd max shape
#pragma _NEC novector
    for (i = 0; i < a->ndim; i++) {
        sh = a->shape[i];
        if (smax1 < sh) {
            smax2 = smax1;
            smax1 = sh;
            max2_idx = max1_idx;
            max1_idx = i;
        } else if (smax2 < sh) {
            smax2 = sh;
            max2_idx = i;
        } 
    }
    // exchange shape and strides
    if (max1_idx != n_inner) {
        sh = a->shape[max1_idx];
        st = a->strides[max1_idx];
        a->shape[max1_idx] = a->shape[n_inner];
        a->strides[max1_idx] = a->strides[n_inner];
        a->shape[n_inner] = sh;
        a->strides[n_inner] = st;

        // case : max2_idx == n_inner
        if (max2_idx == n_inner) {
           max2_idx = max1_idx;
        }
    }
    if (max2_idx != n_outer) {
        sh = a->shape[max2_idx];
        st = a->strides[max2_idx];
        a->shape[max2_idx] = a->shape[n_outer];
        a->strides[max2_idx] = a->strides[n_outer];
        a->shape[n_outer] = sh;
        a->strides[n_outer] = st;
    }

}


uint64_t nlcpy__array_next(ve_array *a, uint64_t curr_adr, int64_t *coords) {
    int64_t i, ndim_m1;
    ndim_m1 = a->ndim - 1;
#pragma _NEC novector
    for (i = ndim_m1; i >= 0; i--) {
        if (coords[i] < a->shape[i] - 1) {
            coords[i]++;
            return curr_adr + a->strides[i];
        } else {
            coords[i] = 0;
            curr_adr -= a->strides[i] * (a->shape[i] - 1);
        }
    }
    return curr_adr;
}


uint64_t nlcpy__array_rnext(ve_array *a, uint64_t curr_adr, int64_t *coords) {
    int64_t i, ndim_m1;
    ndim_m1 = a->ndim - 1;
#pragma _NEC novector
    for (i = 0; i <= ndim_m1; i++) { 
        if (coords[i] < a->shape[i] - 1) {
            coords[i]++;
            return curr_adr + a->strides[i];
        } else {
            coords[i] = 0;
            curr_adr -= a->strides[i] * (a->shape[i] - 1);
        }
    }
    return curr_adr;
}


uint64_t nlcpy__array_reduce_next(ve_array *a, uint64_t curr_adr, int64_t *coords, ve_array *b) {
    int64_t i, ndim_m1;
    ndim_m1 = b->ndim - 1;
#pragma _NEC novector
    for (i = ndim_m1; i >= 0; i--) {
        if (coords[i] < b->shape[i] - 1) {
            coords[i]++;
            // If array is reduced along the i-th axis, the address is not shifted.
            if (a->shape[i]==b->shape[i]) curr_adr += a->strides[i];
            return curr_adr;
        } else {
            coords[i] = 0;
            curr_adr -= a->strides[i] * (a->shape[i] - 1);
        }
    }
    return curr_adr;
}


uint64_t nlcpy__array_reduce_rnext(ve_array *a, uint64_t curr_adr, int64_t *coords, ve_array *b) {
    int64_t i, ndim_m1;
    ndim_m1 = b->ndim - 1;
#pragma _NEC novector
    for (i = 0; i <= ndim_m1; i++) { 
        if (coords[i] < b->shape[i] - 1) {
            coords[i]++;
            if (a->shape[i]==b->shape[i]) curr_adr += a->strides[i];
            return curr_adr;
        } else {
            coords[i] = 0;
            curr_adr -= a->strides[i] * (a->shape[i] - 1);
        }
    }
    return curr_adr;
}

void nlcpy_memcpy(uint64_t *dst, uint64_t *src, uint64_t nbytes) {
    memcpy( (void*)dst, (void*)src, nbytes );
    return;
}



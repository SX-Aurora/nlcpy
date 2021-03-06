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



/****************************
 *
 *       @OPERATOR_NAME@
 *
 * **************************/


#define DBG_PRT(...)
//#define ERR_PRT(...)
//#define DBG_PRT printf
#define ERR_PRT printf

#define C_CONTIGUOUS  0
#define F_CONTIGUOUS  1
#define KEEP          2
#define OTHER         3

#define FFT_DIM_2 2
#define FFT_DIM_3 3

asl_fft_t fft;


static inline uint64_t nlcpy_get_contiguous_status(ve_array *x){
    if (! x->is_c_contiguous && x->is_f_contiguous) {
        return F_CONTIGUOUS;
    } else if (! x->is_f_contiguous && x->is_c_contiguous)  {
        return C_CONTIGUOUS;
    } else if (x->is_f_contiguous && x->is_c_contiguous){
        return KEEP;
    } else {
        return OTHER;
    }
}


static inline uint64_t nlcpy_generate_asl_error(asl_error_t err){
    if (err == ASL_ERROR_MEMORY){
        return (uint64_t)(NLCPY_ERROR_MEMORY | NLCPY_ERROR_ASL);
    }
    return (uint64_t)NLCPY_ERROR_ASL;
}


static inline uint64_t nlcpy_destroy_handle() {
    asl_error_t err = ASL_ERROR_OK;
    if (asl_fft_is_valid(fft)) {
        err = asl_fft_destroy(fft);
    }
    return (uint64_t)err;
}


uint64_t nlcpy_fft_destroy_handle() {
    asl_error_t err = ASL_ERROR_OK;
    err = nlcpy_destroy_handle();
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_fft_1d_c128_c128(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int reuse, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err); 
        err = asl_fft_create_complex_1d_d(&fft, n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_complex_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[x->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);
            err = asl_fft_set_spatial_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_complex_forward_d(fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_ifft_1d_c128_c128(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int reuse, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err); 
        err = asl_fft_create_complex_1d_d(&fft, n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_complex_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[x->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);
            err = asl_fft_set_spatial_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_complex_backward_d(fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_fft_1d_c64_c64(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int reuse, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err); 
        err = asl_fft_create_complex_1d_s(&fft, n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_complex_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[x->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);
            err = asl_fft_set_spatial_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_complex_forward_s(fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_ifft_1d_c64_c64(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int reuse, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err); 
        err = asl_fft_create_complex_1d_s(&fft, n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_complex_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[x->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);
            err = asl_fft_set_spatial_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_complex_backward_s(fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_internal_fft_1d_c128_c128(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    asl_fft_t _fft;
    err = asl_fft_create_complex_1d_d(&_fft, n_in);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(_fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_complex_forward_d(_fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[x->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);
            err = asl_fft_set_spatial_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_complex_forward_d(_fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    err = asl_fft_destroy(_fft);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_internal_ifft_1d_c128_c128(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    asl_fft_t _fft;
    err = asl_fft_create_complex_1d_d(&_fft, n_in);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(_fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_complex_backward_d(_fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[x->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);
            err = asl_fft_set_spatial_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_complex_backward_d(_fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    err = asl_fft_destroy(_fft);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_internal_fft_1d_c64_c64(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    asl_fft_t _fft;
    err = asl_fft_create_complex_1d_s(&_fft, n_in);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(_fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_complex_forward_s(_fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[x->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);
            err = asl_fft_set_spatial_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_complex_forward_s(_fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    err = asl_fft_destroy(_fft);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_internal_ifft_1d_c64_c64(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    asl_fft_t _fft;
    err = asl_fft_create_complex_1d_s(&_fft, n_in);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(_fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_complex_backward_s(_fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[x->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);
            err = asl_fft_set_spatial_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_complex_backward_s(_fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    err = asl_fft_destroy(_fft);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}



uint64_t nlcpy_rfft_1d_f64_c128(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int reuse, int32_t *psw)
{
    double *px = (double *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_1d_d(&fft, n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t n = n_in;
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_real_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[y->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);

            err = asl_fft_set_spatial_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_real_forward_d(fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_irfft_1d_c128_f64(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int reuse, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double *py = (double *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_1d_d(&fft, n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t n = n_in;
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_real_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[y->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);

            err = asl_fft_set_spatial_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_real_backward_d(fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_rfft_1d_f32_c64(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int reuse, int32_t *psw)
{
    float *px = (float *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_1d_s(&fft, n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t n = n_in;
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_real_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[y->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);

            err = asl_fft_set_spatial_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_real_forward_s(fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_irfft_1d_c64_f32(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int reuse, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float *py = (float *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_1d_s(&fft, n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t n = n_in;
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_real_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[y->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);

            err = asl_fft_set_spatial_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_real_backward_s(fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}



uint64_t nlcpy_internal_rfft_1d_f64_c128(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int32_t *psw)
{
    double *px = (double *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    asl_fft_t _fft;
    err = asl_fft_create_real_1d_d(&_fft, n_in);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t n = n_in;
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(_fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_real_forward_d(_fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[y->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);

            err = asl_fft_set_spatial_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_real_forward_d(_fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    err = asl_fft_destroy(_fft);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_internal_irfft_1d_c128_f64(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double *py = (double *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    asl_fft_t _fft;
    err = asl_fft_create_real_1d_d(&_fft, n_in);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t n = n_in;
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(_fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_real_backward_d(_fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[y->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);

            err = asl_fft_set_spatial_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_real_backward_d(_fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    err = asl_fft_destroy(_fft);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_internal_rfft_1d_f32_c64(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int32_t *psw)
{
    float *px = (float *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    asl_fft_t _fft;
    err = asl_fft_create_real_1d_s(&_fft, n_in);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t n = n_in;
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(_fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_real_forward_s(_fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[y->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);

            err = asl_fft_set_spatial_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_real_forward_s(_fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    err = asl_fft_destroy(_fft);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_internal_irfft_1d_c64_f32(ve_array *x, ve_array *y, const int64_t axis, const int64_t n_in, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float *py = (float *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    int64_t order_f = nlcpy_get_contiguous_status(x);

    asl_fft_t _fft;
    err = asl_fft_create_real_1d_s(&_fft, n_in);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        if ((order_f==KEEP && x->shape[0]==1) || (order_f==C_CONTIGUOUS && axis==x->ndim-1) || (order_f==F_CONTIGUOUS && axis==0)) {
            asl_int_t n = n_in;
            asl_int_t m = x->size / x->shape[axis];
            asl_int64_t xs, ys;
            if ( order_f==KEEP ) {
                xs = 1;
                ys = 1;
            } else {
                xs = x->strides[axis] / x->itemsize;
                ys = y->strides[axis] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride_1d(_fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, x->shape[axis]*xs);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, y->shape[axis]*ys);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_execute_real_backward_s(_fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            int64_t *cnt_y = (int64_t*)alloca(sizeof(int64_t) * y->ndim);
            asl_int64_t x0, y0;
            if ( order_f==F_CONTIGUOUS ) {
                x0 = x->strides[0] / x->itemsize;
                y0 = y->strides[0] / y->itemsize;
            } else {
                x0 = x->strides[x->ndim-1] / x->itemsize;
                y0 = y->strides[y->ndim-1] / y->itemsize;
            }
            asl_int_t n = n_in;
            asl_int64_t m = x->strides[axis] / x->itemsize;
            nlcpy__reset_coords(cnt_y, y->ndim);

            err = asl_fft_set_spatial_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_1d(_fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_multiplicity(_fft, (asl_int_t)m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(_fft, x0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_multiplicity_long_stride(_fft, y0);

            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            uint64_t ix = 0;
            uint64_t iy = 0;
            for (;;) {
                // do FFT along axis
                err = asl_fft_execute_real_backward_s(_fft, px + ix, py + iy);
                if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

                int64_t k;
                // set next index
                if ( order_f==F_CONTIGUOUS ) {
                    for (k = y->ndim-1; k > axis; k--) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k - 1 > axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k <= axis) break;
                } else {
                    for (k = 0; k < axis; k++) {
                        if (++cnt_y[k] < y->shape[k]) {
                            ix += x->strides[k] / x->itemsize;
                            iy += y->strides[k] / y->itemsize;
                            break;
                        }
                        cnt_y[k] = 0;
                        if (k + 1 < axis) {
                            ix -= (x->strides[k] / x->itemsize) * (x->shape[k] - 1);
                            iy -= (y->strides[k] / y->itemsize) * (y->shape[k] - 1);
                        }
                    }
                    if (k >= axis) break;
                }

            }
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    err = asl_fft_destroy(_fft);
    if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_recursive_fft_1d_c128_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int32_t *psw)
{
    uint64_t err = NLCPY_ERROR_OK;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    if (axes->size <= 0) return (uint64_t)NLCPY_ERROR_NDIM;

    err = nlcpy_internal_fft_1d_c128_c128(x, y, _axes[0], _n_in[0], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    for(int idx=1; idx < axes->size; idx++){
        ve_array *work = y;
        err = nlcpy_internal_fft_1d_c128_c128(work, y, _axes[idx], _n_in[idx], psw);
        if (err != NLCPY_ERROR_OK ) return (uint64_t)err;
    }
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_recursive_ifft_1d_c128_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int32_t *psw)
{
    uint64_t err = NLCPY_ERROR_OK;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    if (axes->size <= 0) return (uint64_t)NLCPY_ERROR_NDIM;

    err = nlcpy_internal_ifft_1d_c128_c128(x, y, _axes[0], _n_in[0], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    for(int idx=1; idx < axes->size; idx++){
        ve_array *work = y;
        err = nlcpy_internal_ifft_1d_c128_c128(work, y, _axes[idx], _n_in[idx], psw);
        if (err != NLCPY_ERROR_OK ) return (uint64_t)err;
    }
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_recursive_fft_1d_c64_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int32_t *psw)
{
    uint64_t err = NLCPY_ERROR_OK;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    if (axes->size <= 0) return (uint64_t)NLCPY_ERROR_NDIM;

    err = nlcpy_internal_fft_1d_c64_c64(x, y, _axes[0], _n_in[0], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    for(int idx=1; idx < axes->size; idx++){
        ve_array *work = y;
        err = nlcpy_internal_fft_1d_c64_c64(work, y, _axes[idx], _n_in[idx], psw);
        if (err != NLCPY_ERROR_OK ) return (uint64_t)err;
    }
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_recursive_ifft_1d_c64_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int32_t *psw)
{
    uint64_t err = NLCPY_ERROR_OK;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    if (axes->size <= 0) return (uint64_t)NLCPY_ERROR_NDIM;

    err = nlcpy_internal_ifft_1d_c64_c64(x, y, _axes[0], _n_in[0], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    for(int idx=1; idx < axes->size; idx++){
        ve_array *work = y;
        err = nlcpy_internal_ifft_1d_c64_c64(work, y, _axes[idx], _n_in[idx], psw);
        if (err != NLCPY_ERROR_OK ) return (uint64_t)err;
    }
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_recursive_rfft_1d_f64_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int32_t *psw)
{
    uint64_t err = NLCPY_ERROR_OK;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    if (axes->size <= 1) return (uint64_t)NLCPY_ERROR_NDIM;

    uint64_t idx_end = axes->size-1;
    err = nlcpy_internal_rfft_1d_f64_c128(x, y, _axes[idx_end], _n_in[idx_end], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    ve_array *work = y;
    err = nlcpy_internal_fft_1d_c128_c128(work, y, _axes[0], _n_in[0], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    for(int idx=1; idx < idx_end; idx++){
        ve_array *work = y;
        err = nlcpy_internal_fft_1d_c128_c128(work, y, _axes[idx], _n_in[idx], psw);
        if (err != NLCPY_ERROR_OK ) return (uint64_t)err;
    }

    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_recursive_irfft_1d_c128_f64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int32_t *psw)
{
    uint64_t err = NLCPY_ERROR_OK;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    if (axes->size <= 1) return (uint64_t)NLCPY_ERROR_NDIM;

    uint64_t idx_end = axes->size-1;
    err = nlcpy_internal_ifft_1d_c128_c128(x, y, _axes[0], _n_in[0], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    for(int idx=1; idx < idx_end; idx++){
        ve_array *work = y;
        err = nlcpy_internal_ifft_1d_c128_c128(work, y, _axes[idx], _n_in[idx], psw);
        if (err != NLCPY_ERROR_OK ) return (uint64_t)err;
    }
    ve_array *work = y;
    err = nlcpy_internal_irfft_1d_c128_f64(work, y, _axes[idx_end], _n_in[idx_end], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_recursive_rfft_1d_f32_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int32_t *psw)
{
    uint64_t err = NLCPY_ERROR_OK;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    if (axes->size <= 1) return (uint64_t)NLCPY_ERROR_NDIM;

    uint64_t idx_end = axes->size-1;
    err = nlcpy_internal_rfft_1d_f32_c64(x, y, _axes[idx_end], _n_in[idx_end], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    ve_array *work = y;
    err = nlcpy_internal_fft_1d_c64_c64(work, y, _axes[0], _n_in[0], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    for(int idx=1; idx < idx_end; idx++){
        ve_array *work = y;
        err = nlcpy_internal_fft_1d_c64_c64(work, y, _axes[idx], _n_in[idx], psw);
        if (err != NLCPY_ERROR_OK ) return (uint64_t)err;
    }

    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_recursive_irfft_1d_c64_f32(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int32_t *psw)
{
    uint64_t err = NLCPY_ERROR_OK;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    if (axes->size <= 1) return (uint64_t)NLCPY_ERROR_NDIM;

    uint64_t idx_end = axes->size-1;
    err = nlcpy_internal_ifft_1d_c64_c64(x, y, _axes[0], _n_in[0], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    for(int idx=1; idx < idx_end; idx++){
        ve_array *work = y;
        err = nlcpy_internal_ifft_1d_c64_c64(work, y, _axes[idx], _n_in[idx], psw);
        if (err != NLCPY_ERROR_OK ) return (uint64_t)err;
    }
    ve_array *work = y;
    err = nlcpy_internal_irfft_1d_c64_f32(work, y, _axes[idx_end], _n_in[idx_end], psw);
    if (err != NLCPY_ERROR_OK ) return (uint64_t)err;

    return (uint64_t)NLCPY_ERROR_OK;
}

static inline uint64_t check_multiplicity_convertible_axes(int64_t *axes, uint64_t start, uint64_t end) 
{
    const uint64_t len = end - start + 1;
    uint64_t unique_check[len];
    uint64_t unique_cnt=0;

    for(int i=0; i < len; i++){
        if ( axes[i] < start || axes[i] > end ){
            return 0;
        }

#pragma _NEC novector
        for(int j=0; j < unique_cnt; j++){
            if (unique_check[j] == axes[i]){
                return 0;
            }
        }

        unique_check[unique_cnt] = axes[i];
        unique_cnt++;
    }

    return 1;
}


uint64_t nlcpy_fft_2d_c128_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_2d_d(&fft, n1, n2);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if((order_f==C_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==F_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else if((order_f==F_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==C_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){

            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        }else{
            err = nlcpy_recursive_fft_1d_c128_c128(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_ifft_2d_c128_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_2d_d(&fft, n1, n2);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if((order_f==C_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==F_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else if((order_f==F_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==C_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){

            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        }else{
            err = nlcpy_recursive_ifft_1d_c128_c128(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_fft_2d_c64_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_2d_s(&fft, n1, n2);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if((order_f==C_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==F_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else if((order_f==F_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==C_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){

            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        }else{
            err = nlcpy_recursive_fft_1d_c64_c64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_ifft_2d_c64_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_2d_s(&fft, n1, n2);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if((order_f==C_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==F_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else if((order_f==F_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==C_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){

            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];
            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        }else{
            err = nlcpy_recursive_ifft_1d_c64_c64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_rfft_2d_f64_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    double *px = (double *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_2d_d(&fft, n1, n2);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if((order_f==C_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==F_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){

            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if((order_f==F_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==C_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            err = nlcpy_recursive_rfft_1d_f64_c128(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_irfft_2d_c128_f64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double *py = (double *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_2d_d(&fft, n1, n2);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if((order_f==C_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==F_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){

            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if((order_f==F_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==C_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            err = nlcpy_recursive_irfft_1d_c128_f64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_rfft_2d_f32_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    float *px = (float *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_2d_s(&fft, n1, n2);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if((order_f==C_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==F_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){

            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if((order_f==F_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==C_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            err = nlcpy_recursive_rfft_1d_f32_c64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_irfft_2d_c64_f32(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float *py = (float *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_2d_s(&fft, n1, n2);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if((order_f==C_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==F_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){

            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_2; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if((order_f==F_CONTIGUOUS && ((axis_1 == x->ndim - 2 && axis_2 == x->ndim - 1) || (axis_1 == x->ndim - 1 && axis_2 == x->ndim - 2)))
        || (order_f==C_CONTIGUOUS && ((axis_1 == 0 && axis_2 == 1) || (axis_1 == 1 && axis_2 == 0))) ){
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_spatial_long_stride_2d(fft, ls_2[axis_1], ls_2[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_2d(fft, ls_1[axis_1], ls_1[axis_2]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            err = nlcpy_recursive_irfft_1d_c64_f32(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_fft_3d_c128_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const uint64_t axis_3 = _axes[2];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    const asl_int_t n3 = (asl_int_t)_n_in[2];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_3d_d(&fft, n1, n2, n3);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 2))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 2))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            err = nlcpy_recursive_fft_1d_c128_c128(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_ifft_3d_c128_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const uint64_t axis_3 = _axes[2];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    const asl_int_t n3 = (asl_int_t)_n_in[2];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_3d_d(&fft, n1, n2, n3);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 2))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 2))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            err = nlcpy_recursive_ifft_1d_c128_c128(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_fft_3d_c64_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const uint64_t axis_3 = _axes[2];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    const asl_int_t n3 = (asl_int_t)_n_in[2];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_3d_s(&fft, n1, n2, n3);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 2))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 2))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            err = nlcpy_recursive_fft_1d_c64_c64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_ifft_3d_c64_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const uint64_t axis_3 = _axes[2];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    const asl_int_t n3 = (asl_int_t)_n_in[2];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_3d_s(&fft, n1, n2, n3);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 2))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 2))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else {
            err = nlcpy_recursive_ifft_1d_c64_c64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_rfft_3d_f64_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    double *px = (double *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const uint64_t axis_3 = _axes[2];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    const asl_int_t n3 = (asl_int_t)_n_in[2];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_3d_d(&fft, n1, n2, n3);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);
        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 3))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_3);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 3))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_3);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        }else{
            err = nlcpy_recursive_rfft_1d_f64_c128(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_irfft_3d_c128_f64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double *py = (double *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const uint64_t axis_3 = _axes[2];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    const asl_int_t n3 = (asl_int_t)_n_in[2];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_3d_d(&fft, n1, n2, n3);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);
        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 3))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_3);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 3))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_3);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        }else{
            err = nlcpy_recursive_irfft_1d_c128_f64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_rfft_3d_f32_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    float *px = (float *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const uint64_t axis_3 = _axes[2];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    const asl_int_t n3 = (asl_int_t)_n_in[2];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_3d_s(&fft, n1, n2, n3);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);
        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 3))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_3);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 3))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_3);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        }else{
            err = nlcpy_recursive_rfft_1d_f32_c64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_irfft_3d_c64_f32(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{
    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float *py = (float *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t axis_1 = _axes[0];
    const uint64_t axis_2 = _axes[1];
    const uint64_t axis_3 = _axes[2];
    const asl_int_t n1 = (asl_int_t)_n_in[0];
    const asl_int_t n2 = (asl_int_t)_n_in[1];
    const asl_int_t n3 = (asl_int_t)_n_in[2];
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_3d_s(&fft, n1, n2, n3);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);
        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 3))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_3);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
                for(int i=0; i < FFT_DIM_3; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - 3, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, 3))) {
            m = (asl_int_t)(x->size / (x->shape[axis_1] * x->shape[axis_2] * x->shape[axis_3]));
            err = asl_fft_set_half_complex_axis(fft, ASL_AXIS_3);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[x->ndim], ls_2[x->ndim];
#pragma _NEC novector
            for(int i=0; i < x->ndim; i++){
                ls_1[i] = x->strides[i] / x->itemsize;
                ls_2[i] = y->strides[i] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride_3d(fft, ls_2[axis_1], ls_2[axis_2], ls_2[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride_3d(fft, ls_1[axis_1], ls_1[axis_2], ls_1[axis_3]);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = ls_1[idx];
            const asl_int64_t ys = ls_2[idx];

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        }else{
            err = nlcpy_recursive_irfft_1d_c64_f32(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_fft_nd_c128_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{

    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t dim_val = axes->size;
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_d(&fft, dim_val, _n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {
            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim - 1;
            if(order_f==F_CONTIGUOUS){
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {
            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = x->strides[idx] / x->itemsize;
            const asl_int64_t ys = y->strides[idx] / y->itemsize;

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else {
            err = nlcpy_recursive_fft_1d_c128_c128(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_ifft_nd_c128_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{

    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t dim_val = axes->size;
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_d(&fft, dim_val, _n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {
            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim - 1;
            if(order_f==F_CONTIGUOUS){
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {
            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = x->strides[idx] / x->itemsize;
            const asl_int64_t ys = y->strides[idx] / y->itemsize;

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else {
            err = nlcpy_recursive_ifft_1d_c128_c128(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_fft_nd_c64_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{

    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t dim_val = axes->size;
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_s(&fft, dim_val, _n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {
            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim - 1;
            if(order_f==F_CONTIGUOUS){
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {
            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = x->strides[idx] / x->itemsize;
            const asl_int64_t ys = y->strides[idx] / y->itemsize;

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else {
            err = nlcpy_recursive_fft_1d_c64_c64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_ifft_nd_c64_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{

    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t dim_val = axes->size;
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_complex_s(&fft, dim_val, _n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {
            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t xs=1, ys=1;
            int idx = x->ndim - 1;
            if(order_f==F_CONTIGUOUS){
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {
            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }

            err = asl_fft_set_spatial_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = x->strides[idx] / x->itemsize;
            const asl_int64_t ys = y->strides[idx] / y->itemsize;

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_complex_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else {
            err = nlcpy_recursive_ifft_1d_c64_c64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


uint64_t nlcpy_rfft_nd_f64_c128(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{

    double *px = (double *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double _Complex *py = (double _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t dim_val = axes->size;
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_d(&fft, dim_val, _n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {

            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < axes->size; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            err = asl_fft_set_half_complex_axis(fft, dim_val);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {

            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < axes->size; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            err = asl_fft_set_half_complex_axis(fft, dim_val);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = x->strides[idx] / x->itemsize;
            const asl_int64_t ys = y->strides[idx] / y->itemsize;

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else {
            err = nlcpy_recursive_rfft_1d_f64_c128(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_irfft_nd_c128_f64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{

    double _Complex *px = (double _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    double *py = (double *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t dim_val = axes->size;
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_d(&fft, dim_val, _n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {

            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < axes->size; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            err = asl_fft_set_half_complex_axis(fft, dim_val);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {

            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < axes->size; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            err = asl_fft_set_half_complex_axis(fft, dim_val);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = x->strides[idx] / x->itemsize;
            const asl_int64_t ys = y->strides[idx] / y->itemsize;

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_d(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else {
            err = nlcpy_recursive_irfft_1d_c128_f64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_rfft_nd_f32_c64(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{

    float *px = (float *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float _Complex *py = (float _Complex *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t dim_val = axes->size;
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_s(&fft, dim_val, _n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {

            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < axes->size; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            err = asl_fft_set_half_complex_axis(fft, dim_val);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {

            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < axes->size; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            err = asl_fft_set_half_complex_axis(fft, dim_val);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = x->strides[idx] / x->itemsize;
            const asl_int64_t ys = y->strides[idx] / y->itemsize;

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_forward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else {
            err = nlcpy_recursive_rfft_1d_f32_c64(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}

uint64_t nlcpy_irfft_nd_c64_f32(ve_array *x, ve_array *y, ve_array *axes, ve_array *n_in, int reuse, int32_t *psw)
{

    float _Complex *px = (float _Complex *)nlcpy__get_ptr(x);
    if (px == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    float *py = (float *)nlcpy__get_ptr(y);
    if (py == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_axes = (int64_t *)nlcpy__get_ptr(axes);
    if (_axes == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;
    int64_t *_n_in = (int64_t *)nlcpy__get_ptr(n_in);
    if (_n_in == NULL) return (uint64_t)NLCPY_ERROR_MEMORY;

    asl_error_t err = ASL_ERROR_OK;
    const uint64_t dim_val = axes->size;
    if ( !reuse ) {
        err = nlcpy_destroy_handle();
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        err = asl_fft_create_real_s(&fft, dim_val, _n_in);
        if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
    }

    if (y->ndim > 0 && y->ndim <= NLCPY_MAXNDIM){
        asl_int_t m;
        int64_t order_f = nlcpy_get_contiguous_status(x);

        if ((order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {

            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < axes->size; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            err = asl_fft_set_half_complex_axis(fft, dim_val);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            asl_int64_t xs=1, ys=1;
            int idx = x->ndim-1;
            if(order_f==F_CONTIGUOUS){
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[i];
                    ys = ys * y->shape[i];
                }
            }else{
#pragma _NEC novector
                for(int i=0; i < dim_val; i++){
                    xs = xs * x->shape[idx-i];
                    ys = ys * y->shape[idx-i];
                }
            }

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
        } else if ((order_f == F_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, x->ndim - dim_val, x->ndim - 1)) || (order_f == C_CONTIGUOUS && check_multiplicity_convertible_axes(_axes, 0, dim_val - 1))) {

            m = (asl_int_t)x->size;
#pragma _NEC novector
            for(int i=0; i < axes->size; i++){
                m = (asl_int_t)(m / x->shape[_axes[i]]);
            }

            err = asl_fft_set_half_complex_axis(fft, dim_val);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            asl_int64_t ls_1[dim_val], ls_2[dim_val];
#pragma _NEC novector
            for(int i=0; i < dim_val; i++){
                ls_1[i] = x->strides[_axes[i]] / x->itemsize;
                ls_2[i] = y->strides[_axes[i]] / y->itemsize;
            }
            
            err = asl_fft_set_multiplicity(fft, m);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_spatial_long_stride(fft, ls_2);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_long_stride(fft, ls_1);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);


            const uint64_t idx = (order_f == F_CONTIGUOUS) ? 0 : x->ndim - 1;
            const asl_int64_t xs = x->strides[idx] / x->itemsize;
            const asl_int64_t ys = y->strides[idx] / y->itemsize;

            err = asl_fft_set_spatial_multiplicity_long_stride(fft, ys);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);
            err = asl_fft_set_frequency_multiplicity_long_stride(fft, xs);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

            err = asl_fft_execute_real_backward_s(fft, px, py);
            if (err != ASL_ERROR_OK ) return nlcpy_generate_asl_error(err);

        } else {
            err = nlcpy_recursive_irfft_1d_c64_f32(x, y, axes, n_in, psw);
            return (uint64_t)err;
        }
    } else {
        // above NLCPY_MAXNDIM
        return (uint64_t)NLCPY_ERROR_NDIM;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)NLCPY_ERROR_OK;
}


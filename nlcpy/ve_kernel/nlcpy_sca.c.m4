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

#define NDEBUG
//#define SCA_LOGGING

#ifdef NDEBUG
#define DBG_PRT_FUNC_NAME
#define DBG_PRT_I(x)
#define DBG_PRT_IP(x)
#define DBG_PRT_FP(x)
#define DBG_PRT(...)
#define DBG_FOR(i, n)
#else  /* if not defined NDEBUG */
#define DBG_PRT_FUNC_NAME printf("--- %s ---\n", __func__)
#define DBG_PRT_I(x) printf("%s: %ld\n", #x, x)
#define DBG_PRT_IP(x) printf("%s: %ld\n", #x, *x)
#define DBG_PRT_FP(x) printf("%s: %lf\n", #x, *x)
#define DBG_PRT printf
#define DBG_FOR(i, n) for(int64_t i = 0; i < n; i++)
#endif /* end NDEBUG */

include(macros.m4)dnl
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <alloca.h>
#include <assert.h>

#include "nlcpy.h"
#include <inc_i64/sca.h>

define(<--@macro_sca_stencil_create@-->,<--@
uint64_t nlcpy_sca_stencil_create_$1(uint64_t *hnd_adr, int32_t *psw)
{
#ifdef SCA_LOGGING
    sca_library_set_logging_level(SCA_LOGLEVEL_INFO);
#endif
    sca_error_t err;
    sca_stencil_t *sten = (sca_stencil_t *)malloc(sizeof(sca_stencil_t));
    err = sca_stencil_create_$1(sten);
    if (err != SCA_ERROR_OK) {
       return NLCPY_ERROR_SCA;
    }
    *hnd_adr = (uint64_t)sten;
    retrieve_fpe_flags(psw);
    return NLCPY_ERROR_OK;
}
@-->)dnl
macro_sca_stencil_create(s)dnl
macro_sca_stencil_create(d)dnl


uint64_t nlcpy_sca_stencil_reset_elements(sca_stencil_t *sten, int32_t *psw)
{
    sca_error_t err;
    sca_int_t nelm;
    err = sca_stencil_get_element_count(*sten, &nelm);
    if (err != SCA_ERROR_OK) {
        return NLCPY_ERROR_SCA;
    }
    err = sca_stencil_remove_elements(*sten, 0, nelm);
    if (err != SCA_ERROR_OK) {
        return NLCPY_ERROR_SCA;
    }
    retrieve_fpe_flags(psw);
    return NLCPY_ERROR_OK;
}


uint64_t nlcpy_sca_stencil_destroy(sca_stencil_t *sten, int32_t *psw)
{
    sca_error_t err;
    if (sten == NULL) return NLCPY_ERROR_MEMORY;
    err = sca_stencil_destroy(*sten);
    if (err != SCA_ERROR_OK) {
        return NLCPY_ERROR_SCA;
    }
    free(sten);
    retrieve_fpe_flags(psw);
    return NLCPY_ERROR_OK;
}


uint64_t nlcpy_sca_code_destroy(sca_code_t *code, int32_t *psw)
{
    sca_error_t err;
    if (code == NULL) return NLCPY_ERROR_MEMORY;
    err = sca_code_destroy(*code);
    if (err != SCA_ERROR_OK) {
        return NLCPY_ERROR_SCA;
    }
    free(code);
    retrieve_fpe_flags(psw);
    return NLCPY_ERROR_OK;
}


define(<--@macro_sca_utility_optimize_leading@-->,<--@
uint64_t nlcpy_sca_utility_optimize_leading_$1(
    sca_int_t nx, sca_int_t ny, sca_int_t nz,
    sca_int_t *mx, sca_int_t *my, sca_int_t *mz,
    int32_t *psw)
{
    DBG_PRT_FUNC_NAME;
    DBG_PRT_I(nx);
    DBG_PRT_I(ny);
    DBG_PRT_I(nz);

    sca_error_t err;
    err = sca_utility_optimize_leading_$1(nx, ny, nz, 1, mx, my, mz);
    if (err != SCA_ERROR_OK) {
        return NLCPY_ERROR_SCA;
    }

    DBG_PRT_IP(mx);
    DBG_PRT_IP(my);
    DBG_PRT_IP(mz);

    retrieve_fpe_flags(psw);
    return NLCPY_ERROR_OK;

}
@-->)dnl
macro_sca_utility_optimize_leading(s)dnl
macro_sca_utility_optimize_leading(d)dnl


uint64_t nlcpy_sca_code_create(
    uint64_t *code_adr, sca_stencil_t *sten, sca_int_t nx, sca_int_t ny,
    sca_int_t nz, sca_int_t nw, int32_t *psw)
{
    DBG_PRT_FUNC_NAME;
    DBG_PRT_I(nx);
    DBG_PRT_I(ny);
    DBG_PRT_I(nz);
    DBG_PRT_I(nw);

    sca_error_t err;
    sca_code_t *code = (sca_code_t *)malloc(sizeof(sca_code_t));
    err = sca_code_create(code, *sten, nx, ny, nz, nw);
    if (err != SCA_ERROR_OK) {
        return NLCPY_ERROR_SCA;
    }
    *code_adr = (uint64_t)code;
    retrieve_fpe_flags(psw);
    return NLCPY_ERROR_OK;
}


/*
uint64_t nlcpy_sca_code_execute(sca_code_t *code, int32_t *psw)
{
    sca_error_t err;
    err = sca_code_execute(*code);
    if (err != SCA_ERROR_OK) {
        return NLCPY_ERROR_SCA;
    }
    retrieve_fpe_flags(psw);
    return NLCPY_ERROR_OK;
}
*/


uint64_t nlcpy_sca_code_execute(ve_arguments *args, int32_t *psw)
{
    sca_code_t *code = (sca_code_t *)(args->sca.code);
    sca_error_t err;
    err = sca_code_execute(*code);
    if (err != SCA_ERROR_OK) {
        return NLCPY_ERROR_SCA;
    }
    return NLCPY_ERROR_OK;
}


uint64_t nlcpy_sca_batch_run(uint64_t *codes, int64_t iteration, int64_t n_code)
{
    sca_error_t err;
    sca_code_t *code;
    for (int64_t i = 0; i < iteration; i++) {
        code = (sca_code_t *)codes[i % n_code];
        err = sca_code_execute(*code);
        if (err != SCA_ERROR_OK) {
            return NLCPY_ERROR_SCA;
        }
    }
    return NLCPY_ERROR_OK;
}


define(<--@macro_sca_set_elements@-->,<--@
uint64_t nlcpy_sca_set_elements_$1(
    sca_stencil_t *sten,
    ve_array *a_in,
    ve_array *a_loc,
    ve_array *a_fact,
    ve_array *a_coef,
    ve_array *a_coef_idx,
    ve_array *a_coef_leading,
    sca_int_t offset,
    sca_int_t nelm,
    sca_int_t elem_offset,
    sca_int_t sx_i,
    sca_int_t mx_i,
    sca_int_t my_i,
    sca_int_t mz_i,
    int32_t *psw
)
{
    assert(a_loc->ndim == 2);
    assert(a_fact->ndim == 1);
    assert(a_coef->size == a_coef_idx->size);
    assert(a_coef->dtype == ve_u64);
    assert(a_coef_idx->dtype == ve_i64);
    assert(a_coef_leading->dtype == ve_i64);

    DBG_PRT_FUNC_NAME;
    DBG_PRT_I(offset);
    DBG_PRT_I(nelm);
    DBG_PRT_I(elem_offset);
    DBG_PRT_I(sx_i);
    DBG_PRT_I(mx_i);
    DBG_PRT_I(my_i);
    DBG_PRT_I(mz_i);

    sca_error_t err;
    const $2 *d_in = (const $2 *)a_in->ve_adr;
    const sca_int_t *d_loc = (const sca_int_t *)a_loc->ve_adr;
    const $2 *d_fact = (const $2 *)a_fact->ve_adr;
    const uint64_t *d_coef = (const uint64_t *)a_coef->ve_adr;
    const sca_int_t *d_coef_idx = (const sca_int_t *)a_coef_idx->ve_adr;
    const sca_int_t *d_coef_leading = (const sca_int_t *)a_coef_leading->ve_adr;

    if (d_in == NULL || d_loc == NULL || d_fact == NULL || d_coef == NULL || d_coef_idx == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    err = sca_stencil_append_elements(*sten, nelm);
    if (err != SCA_ERROR_OK) {
        return NLCPY_ERROR_SCA;
    }
    for (sca_int_t i = 0; i < nelm; i++) {

        DBG_PRT("location = ");
        DBG_FOR(j, a_loc->shape[1]) {
            DBG_PRT("%d, ", d_loc[j+i*a_loc->shape[1]]);
        }
        DBG_PRT("\n");
        err = sca_stencil_set_location(
            *sten,
            i+elem_offset,
            d_loc[i*a_loc->shape[1] + 3],
            d_loc[i*a_loc->shape[1] + 2],
            d_loc[i*a_loc->shape[1] + 1],
            d_loc[i*a_loc->shape[1] + 0]
        );

        DBG_PRT("factor = %lf\n", d_fact[i]);
        err = sca_stencil_set_factor_$1(
            *sten,
            i+elem_offset,
            d_fact[i]
        );
        err = sca_stencil_set_input_array_$1(
            *sten,
            i+elem_offset,
            sx_i,
            mx_i,
            my_i,
            mz_i,
            d_in+offset
        );
        if (err != SCA_ERROR_OK) {
            return NLCPY_ERROR_SCA;
        }
    }
    if (a_coef->ndim > 0) {
        for (sca_int_t i = 0; i < a_coef->size; i++) {
            const $2 *coef = (const $2 *)d_coef[i];
            DBG_PRT_I(d_coef_idx[i] + elem_offset);
            DBG_PRT_FP(coef);
            if (d_coef_leading[i*5 + 0] == 1) {
                err = sca_stencil_set_coefficient_variable_$1(
                    *sten,
                    d_coef_idx[i] + elem_offset,
                    coef
                );
            } else {
                DBG_PRT("  sx_c = %ld\n", d_coef_leading[i*5 + 1]);
                DBG_PRT("  mx_c = %ld\n", d_coef_leading[i*5 + 2]);
                DBG_PRT("  my_c = %ld\n", d_coef_leading[i*5 + 3]);
                DBG_PRT("  mz_c = %ld\n", d_coef_leading[i*5 + 4]);
                err = sca_stencil_set_coefficient_array_$1(
                    *sten,
                    d_coef_idx[i] + elem_offset,
                    d_coef_leading[i*5 +  1],
                    d_coef_leading[i*5 +  2],
                    d_coef_leading[i*5 +  3],
                    d_coef_leading[i*5 +  4],
                    coef
                );
            }
            if (err != SCA_ERROR_OK) {
                return NLCPY_ERROR_SCA;
            }
        }
    }

    retrieve_fpe_flags(psw);
    return NLCPY_ERROR_OK;
}
@-->)dnl
macro_sca_set_elements(s, float)dnl
macro_sca_set_elements(d, double)dnl


define(<--@macro_sca_set_output_array@-->,<--@
uint64_t nlcpy_sca_set_output_array_$1(
    sca_stencil_t *sten,
    ve_array *a_out,
    sca_int_t sx_o,
    sca_int_t mx_o,
    sca_int_t my_o,
    sca_int_t mz_o,
    sca_int_t offset,
    int32_t *psw
)
{
    DBG_PRT_FUNC_NAME;
    DBG_PRT_I(offset);
    DBG_PRT_I(sx_o);
    DBG_PRT_I(mx_o);
    DBG_PRT_I(my_o);
    DBG_PRT_I(mz_o);

    sca_error_t err;

    $2 *d_out = ($2 *)a_out->ve_adr;
    if (d_out == NULL) {
        return NLCPY_ERROR_MEMORY;
    }

    err = sca_stencil_set_output_array_$1(
        *sten,
        sx_o,
        mx_o,
        my_o,
        mz_o,
        d_out+offset
    );
    if (err != SCA_ERROR_OK) {
        return NLCPY_ERROR_SCA;
    }

    retrieve_fpe_flags(psw);
    return NLCPY_ERROR_OK;
}
@-->)dnl
macro_sca_set_output_array(s, float)dnl
macro_sca_set_output_array(d, double)dnl

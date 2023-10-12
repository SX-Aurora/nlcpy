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
#include <math.h>
#include <alloca.h>
#include <complex.h>

#include "nlcpy.h"

define(<--@macro_simple_fnorm@-->,<--@
uint64_t nlcpy_simple_fnorm_$1(ve_array *x, ve_array *y, int32_t *psw)
{
    $2* const px = ($2*)x->ve_adr;
    $3* const py = ($3*)y->ve_adr;
    if (px == NULL || py == NULL) {
        return NLCPY_ERROR_MEMORY;
    }
#pragma omp single
{
    *py = 0;
}

#ifdef DEBUG_BARRIER
     nlcpy__sleep_thread();
#endif /* DEBUG_BARRIER */

#ifdef _OPENMP
    const int nt = omp_get_num_threads();
    const int it = omp_get_thread_num();
#else
    const int nt = 1;
    const int it = 0;
#endif

    $3 local_py = 0;
    const uint64_t i_s = x->size * it / nt;
    const uint64_t i_e = x->size * (it + 1) / nt;
    for (uint64_t i = i_s; i < i_e; i++) {
        $3 tmp = $5(px[i]);
        local_py += tmp * tmp;
    }
#pragma omp critical
{
    *py += local_py;
}
#pragma omp barrier

#ifdef DEBUG_BARRIER
     nlcpy__sleep_thread();
#endif /* DEBUG_BARRIER */

#pragma omp single
{
    *py = $4(*py);
}
    retrieve_fpe_flags(psw);
    return NLCPY_ERROR_OK;
}
@-->)dnl
macro_simple_fnorm(s, float, float, sqrtf)dnl
macro_simple_fnorm(d, double, double, sqrt)dnl
macro_simple_fnorm(c, float _Complex, float, sqrtf, cabsf)dnl
macro_simple_fnorm(z, double _Complex, double, sqrt, cabs)dnl


uint64_t nlcpy_simple_fnorm(ve_arguments *args, int32_t *psw) {
    uint64_t err = NLCPY_ERROR_OK;
    ve_array *x = &(args->simple_fnorm.x);
    ve_array *y = &(args->simple_fnorm.y);
    switch(x->dtype) {
    case ve_f32:
        err = nlcpy_simple_fnorm_s(x, y, psw);
        break;
    case ve_f64:
        err = nlcpy_simple_fnorm_d(x, y, psw);
        break;
    case ve_c64:
        err = nlcpy_simple_fnorm_c(x, y, psw);
        break;
    case ve_c128:
        err = nlcpy_simple_fnorm_z(x, y, psw);
        break;
    default:
        err = NLCPY_ERROR_DTYPE;
    }
    return (uint64_t)err;
}

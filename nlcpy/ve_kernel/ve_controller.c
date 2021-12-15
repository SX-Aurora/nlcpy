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

uint64_t call_binary_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_binary_func(funcnum)(args, psw);
    return res;
}

uint64_t call_unary_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_unary_func(funcnum)(args, psw);
    return res;
}

uint64_t call_indexing_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_indexing_func(funcnum)(args, psw);
    return res;
}


uint64_t call_creation_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_creation_func(funcnum)(args, psw);
    return res;
}

uint64_t call_manipulation_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_manipulation_func(funcnum)(args, psw);
    return res;
}

uint64_t call_cblas_wrapper_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_cblas_wrapper_func(funcnum)(args, psw);
    return res;
}

uint64_t call_linalg_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_linalg_func(funcnum)(args, psw);
    return res;
}

uint64_t call_reduce_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_reduce_func(funcnum)(args, psw);
    return res;
}

uint64_t call_reduceat_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_reduceat_func(funcnum)(args, psw);
    return res;
}

uint64_t call_accumulate_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_accumulate_func(funcnum)(args, psw);
    return res;
}

uint64_t call_outer_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_outer_func(funcnum)(args, psw);
    return res;
}

uint64_t call_at_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_at_func(funcnum)(args, psw);
    return res;
}

uint64_t call_searching_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_searching_func(funcnum)(args, psw);
    return res;
}

uint64_t call_sorting_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_sorting_func(funcnum)(args, psw);
    return res;
}

uint64_t call_math_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_math_func(funcnum)(args, psw);
    return res;
}

uint64_t call_random_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_random_func(funcnum)(args, psw);
    return res;
}

uint64_t call_sca_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_sca_func(funcnum)(args, psw);
    return res;
}

uint64_t call_mask_func(int64_t funcnum, ve_arguments *args, int32_t *psw) {
    uint64_t res;
    res = get_mask_func(funcnum)(args, psw);
    return res;
}


uint64_t run_request(request_package *pack, int32_t *psw) {
    uint64_t err;
    //printf("functype = %d\n", pack->functype);
    //printf("funcnum  = %d\n", pack->funcnum);
    switch (pack->functype) {
    case BINARY_OP:
        err = call_binary_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case UNARY_OP:
        err = call_unary_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case INDEXING_OP:
        err = call_indexing_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case CREATION_OP:
        err = call_creation_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case MANIPULATION_OP:
        err = call_manipulation_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case CBLAS_OP:
        err = call_cblas_wrapper_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case LINALG_OP:
        err = call_linalg_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case REDUCE_OP:
        err = call_reduce_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case REDUCEAT_OP:
        err = call_reduceat_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case ACCUMULATE_OP:
        err = call_accumulate_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case OUTER_OP:
        err = call_outer_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case AT_OP:
        err = call_at_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case SEARCHING_OP:
        err = call_searching_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case SORTING_OP:
        err = call_sorting_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case MATH_OP:
        err = call_math_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case RANDOM_OP:
        err = call_random_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case SCA_OP:
        err = call_sca_func(pack->funcnum, &(pack->arguments), psw);
        break;
    case MASK_OP:
        err = call_mask_func(pack->funcnum, &(pack->arguments), psw);
        break;
default:
        return NLCPY_ERROR_FUNCTYPE;
    }
    return err;
}


//uint64_t kernel_launcher(uint64_t req_adr, uint64_t nreq, int32_t *psw, double *ve_runtime) {
uint64_t kernel_launcher(uint64_t req_adr, uint64_t nreq, int32_t *psw) {
    uint64_t err = NLCPY_ERROR_OK;

    request_package *reqs = (request_package *)req_adr;

#ifdef _OPENMP
#pragma omp parallel
#endif /* _OPENMP */
{
    int i;
    uint64_t flag;
    int32_t lpsw = 0;
    *psw = 0;

    for(i = 0; i < nreq; i++) {
        flag = run_request(&reqs[i], &lpsw);
        //flag = run_request(&reqs[i], psw);
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
            *psw |= lpsw;
            if (flag != NLCPY_ERROR_OK) err |= flag;
        }
#ifdef _OPENMP
#pragma omp barrier
#endif /* _OPENMP */
        if (err != NLCPY_ERROR_OK) break;
    }
} /* omp parallel */

    return err;
}


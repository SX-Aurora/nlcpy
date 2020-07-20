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
/* NOTICE
 * this file will be deprecated! 
 */


#include "nlcpy.h"
#define NLCPY_MAX_NUM_THREADS 7
asl_sort_t sort[NLCPY_MAX_NUM_THREADS];



asl_error_t wrapper_sort_create_i32(const int it) {
    asl_error_t err;
    /* create sorter */
    err = asl_sort_create_i32(&sort[it], ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
    return err;
}

asl_error_t wrapper_sort_execute_i32(asl_int_t num, 
                                    asl_int32_t *kyi, 
                                    asl_int_t *vli,
                                    asl_int32_t *kyo,
                                    asl_int_t *vlo,
                                    const int it
) {
    asl_error_t err;
    if (!asl_sort_is_valid(sort[it])) {
        return ASL_ERROR_SORT_INVALID;
    }
    err = asl_sort_execute_i32(sort[it], num, kyi, vli, kyo, vlo);
    return err;
}



asl_error_t wrapper_sort_create_i64(const int it) {
    asl_error_t err;
    /* create sorter */
    err = asl_sort_create_i64(&sort[it], ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
    return err;
}

asl_error_t wrapper_sort_execute_i64(asl_int_t num, 
                                    asl_int64_t *kyi, 
                                    asl_int_t *vli,
                                    asl_int64_t *kyo,
                                    asl_int_t *vlo,
                                    const int it
) {
    asl_error_t err;
    if (!asl_sort_is_valid(sort[it])) {
        return ASL_ERROR_SORT_INVALID;
    }
    err = asl_sort_execute_i64(sort[it], num, kyi, vli, kyo, vlo);
    return err;
}



asl_error_t wrapper_sort_create_u32(const int it) {
    asl_error_t err;
    /* create sorter */
    err = asl_sort_create_u32(&sort[it], ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
    return err;
}

asl_error_t wrapper_sort_execute_u32(asl_int_t num, 
                                    asl_uint32_t *kyi, 
                                    asl_int_t *vli,
                                    asl_uint32_t *kyo,
                                    asl_int_t *vlo,
                                    const int it
) {
    asl_error_t err;
    if (!asl_sort_is_valid(sort[it])) {
        return ASL_ERROR_SORT_INVALID;
    }
    err = asl_sort_execute_u32(sort[it], num, kyi, vli, kyo, vlo);
    return err;
}



asl_error_t wrapper_sort_create_u64(const int it) {
    asl_error_t err;
    /* create sorter */
    err = asl_sort_create_u64(&sort[it], ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
    return err;
}

asl_error_t wrapper_sort_execute_u64(asl_int_t num, 
                                    asl_uint64_t *kyi, 
                                    asl_int_t *vli,
                                    asl_uint64_t *kyo,
                                    asl_int_t *vlo,
                                    const int it
) {
    asl_error_t err;
    if (!asl_sort_is_valid(sort[it])) {
        return ASL_ERROR_SORT_INVALID;
    }
    err = asl_sort_execute_u64(sort[it], num, kyi, vli, kyo, vlo);
    return err;
}



asl_error_t wrapper_sort_create_f32(const int it) {
    asl_error_t err;
    /* create sorter */
    err = asl_sort_create_s(&sort[it], ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
    return err;
}

asl_error_t wrapper_sort_execute_f32(asl_int_t num, 
                                    float *kyi, 
                                    asl_int_t *vli,
                                    float *kyo,
                                    asl_int_t *vlo,
                                    const int it
) {
    asl_error_t err;
    if (!asl_sort_is_valid(sort[it])) {
        return ASL_ERROR_SORT_INVALID;
    }
    err = asl_sort_execute_s(sort[it], num, kyi, vli, kyo, vlo);
    return err;
}



asl_error_t wrapper_sort_create_f64(const int it) {
    asl_error_t err;
    /* create sorter */
    err = asl_sort_create_d(&sort[it], ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
    return err;
}

asl_error_t wrapper_sort_execute_f64(asl_int_t num, 
                                    double *kyi, 
                                    asl_int_t *vli,
                                    double *kyo,
                                    asl_int_t *vlo,
                                    const int it
) {
    asl_error_t err;
    if (!asl_sort_is_valid(sort[it])) {
        return ASL_ERROR_SORT_INVALID;
    }
    err = asl_sort_execute_d(sort[it], num, kyi, vli, kyo, vlo);
    return err;
}


asl_error_t wrapper_sort_preallocate(asl_int_t n, const int it) {
    asl_error_t err;
    if (!asl_sort_is_valid(sort[it])) {
        return ASL_ERROR_SORT_INVALID;
    }
    err = asl_sort_preallocate(sort[it], n);
    return err;
}

asl_error_t wrapper_sort_set_key_long_stride(asl_int64_t std, const int it) {
    asl_error_t err;
    if (!asl_sort_is_valid(sort[it])) {
        return ASL_ERROR_SORT_INVALID;
    }
    err = asl_sort_set_input_key_long_stride(sort[it], std);
    if (err != ASL_ERROR_OK) return err;
    err = asl_sort_set_output_key_long_stride(sort[it], std);
    return err;
}

asl_error_t wrapper_sort_destroy(const int it) {
    asl_error_t err;
    err = asl_sort_destroy(sort[it]);
    return err;
}


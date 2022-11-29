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
#ifndef VE_SELECTOR_H_INCLUDED
#define VE_SELECTOR_H_INCLUDED

#include "ve_request.h"
#include "ve_funclist.h"

#ifdef __cplusplus
extern "C" {
#endif

binary_op get_binary_func(int64_t func_num);
unary_op  get_unary_func(int64_t func_num);
indexing_op get_indexing_func(int64_t func_num);
creation_op get_creation_func(int64_t func_num);
manipulation_op get_manipulation_func(int64_t func_num);
linalg_op get_linalg_func(int64_t func_num);
reduce_op get_reduce_func(int64_t func_num);
reduceat_op get_reduceat_func(int64_t func_num);
accumulate_op get_accumulate_func(int64_t func_num);
outer_op get_outer_func(int64_t func_num);
at_op get_at_func(int64_t func_num);
searching_op get_searching_func(int64_t func_num);
sorting_op get_sorting_func(int64_t func_num);
math_op get_math_func(int64_t func_num);
random_op get_random_func(int64_t func_num);
sca_op get_sca_func(int64_t func_num);
mask_op get_mask_func(int64_t func_num);


#ifdef __cplusplus
}
#endif


#endif /* VE_SELECTOR_H_INCLUDED */

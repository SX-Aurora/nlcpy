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
#ifndef VE_FUNCLIST_H_INCLUDED
#define VE_FUNCLIST_H_INCLUDED

#include "ve_array.h"
#include "ve_funcnum.h"

/****************************
 *
 *       BINARY OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*binary_op)(ve_arguments *, int32_t *);
/* function proto types */
#ifndef NO_OPERATOR
uint64_t nlcpy_add(ve_arguments *, int32_t *);
#ifndef ADD_ONLY
uint64_t nlcpy_subtract(ve_arguments *, int32_t *);
uint64_t nlcpy_multiply(ve_arguments *, int32_t *);
uint64_t nlcpy_divide(ve_arguments *, int32_t *);
uint64_t nlcpy_logaddexp(ve_arguments *, int32_t *);
uint64_t nlcpy_logaddexp2(ve_arguments *, int32_t *);
uint64_t nlcpy_true_divide(ve_arguments *, int32_t *);
uint64_t nlcpy_floor_divide(ve_arguments *, int32_t *);
uint64_t nlcpy_power(ve_arguments *, int32_t *);
uint64_t nlcpy_remainder(ve_arguments *, int32_t *);
uint64_t nlcpy_mod(ve_arguments *, int32_t *);
uint64_t nlcpy_fmod(ve_arguments *, int32_t *);
uint64_t nlcpy_divmod(ve_arguments *, int32_t *);
uint64_t nlcpy_heaviside(ve_arguments *, int32_t *);
uint64_t nlcpy_gcd(ve_arguments *, int32_t *);
uint64_t nlcpy_lcm(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_and(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_or(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_xor(ve_arguments *, int32_t *);
uint64_t nlcpy_left_shift(ve_arguments *, int32_t *);
uint64_t nlcpy_right_shift(ve_arguments *, int32_t *);
uint64_t nlcpy_greater(ve_arguments *, int32_t *);
uint64_t nlcpy_greater_equal(ve_arguments *, int32_t *);
uint64_t nlcpy_less(ve_arguments *, int32_t *);
uint64_t nlcpy_less_equal(ve_arguments *, int32_t *);
uint64_t nlcpy_not_equal(ve_arguments *, int32_t *);
uint64_t nlcpy_equal(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_and(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_or(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_xor(ve_arguments *, int32_t *);
uint64_t nlcpy_maximum(ve_arguments *, int32_t *);
uint64_t nlcpy_minimum(ve_arguments *, int32_t *);
uint64_t nlcpy_fmax(ve_arguments *, int32_t *);
uint64_t nlcpy_fmin(ve_arguments *, int32_t *);
uint64_t nlcpy_arctan2(ve_arguments *, int32_t *);
uint64_t nlcpy_hypot(ve_arguments *, int32_t *);
uint64_t nlcpy_copysign(ve_arguments *, int32_t *);
uint64_t nlcpy_nextafter(ve_arguments *, int32_t *);
uint64_t nlcpy_modf(ve_arguments *, int32_t *);
uint64_t nlcpy_ldexp(ve_arguments *, int32_t *);
uint64_t nlcpy_fexp(ve_arguments *, int32_t *);
#endif /* ADD_ONLY */
#endif /* NO_OPERATOR */




/****************************
 *
 *       UNARY OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*unary_op)(ve_arguments *, int32_t *);
/* function proto types */
#ifndef NO_OPERATOR
#ifndef ADD_ONLY
uint64_t nlcpy_negative(ve_arguments *, int32_t *);
uint64_t nlcpy_positive(ve_arguments *, int32_t *);
uint64_t nlcpy_absolute(ve_arguments *, int32_t *);
uint64_t nlcpy_fabs(ve_arguments *, int32_t *);
uint64_t nlcpy_rint(ve_arguments *, int32_t *);
uint64_t nlcpy_sign(ve_arguments *, int32_t *);
uint64_t nlcpy_conj(ve_arguments *, int32_t *);
uint64_t nlcpy_conjugate(ve_arguments *, int32_t *);
uint64_t nlcpy_exp(ve_arguments *, int32_t *);
uint64_t nlcpy_exp2(ve_arguments *, int32_t *);
uint64_t nlcpy_log(ve_arguments *, int32_t *);
uint64_t nlcpy_log2(ve_arguments *, int32_t *);
uint64_t nlcpy_log10(ve_arguments *, int32_t *);
uint64_t nlcpy_expm1(ve_arguments *, int32_t *);
uint64_t nlcpy_log1p(ve_arguments *, int32_t *);
uint64_t nlcpy_sqrt(ve_arguments *, int32_t *);
uint64_t nlcpy_square(ve_arguments *, int32_t *);
uint64_t nlcpy_cbrt(ve_arguments *, int32_t *);
uint64_t nlcpy_reciprocal(ve_arguments *, int32_t *);
uint64_t nlcpy_sin(ve_arguments *, int32_t *);
uint64_t nlcpy_cos(ve_arguments *, int32_t *);
uint64_t nlcpy_tan(ve_arguments *, int32_t *);
uint64_t nlcpy_arcsin(ve_arguments *, int32_t *);
uint64_t nlcpy_arccos(ve_arguments *, int32_t *);
uint64_t nlcpy_arctan(ve_arguments *, int32_t *);
uint64_t nlcpy_sinh(ve_arguments *, int32_t *);
uint64_t nlcpy_cosh(ve_arguments *, int32_t *);
uint64_t nlcpy_tanh(ve_arguments *, int32_t *);
uint64_t nlcpy_arcsinh(ve_arguments *, int32_t *);
uint64_t nlcpy_arccosh(ve_arguments *, int32_t *);
uint64_t nlcpy_arctanh(ve_arguments *, int32_t *);
uint64_t nlcpy_deg2rad(ve_arguments *, int32_t *);
uint64_t nlcpy_rad2deg(ve_arguments *, int32_t *);
uint64_t nlcpy_degrees(ve_arguments *, int32_t *);
uint64_t nlcpy_radians(ve_arguments *, int32_t *);
uint64_t nlcpy_invert(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_not(ve_arguments *, int32_t *);
uint64_t nlcpy_isfinite(ve_arguments *, int32_t *);
uint64_t nlcpy_isinf(ve_arguments *, int32_t *);
uint64_t nlcpy_isnan(ve_arguments *, int32_t *);
uint64_t nlcpy_signbit(ve_arguments *, int32_t *);
uint64_t nlcpy_spacing(ve_arguments *, int32_t *);
uint64_t nlcpy_floor(ve_arguments *, int32_t *);
uint64_t nlcpy_ceil(ve_arguments *, int32_t *);
uint64_t nlcpy_trunc(ve_arguments *, int32_t *);
uint64_t nlcpy_angle(ve_arguments *, int32_t *);
uint64_t nlcpy_erf(ve_arguments *, int32_t *);
uint64_t nlcpy_erfc(ve_arguments *, int32_t *);
#endif /* ADD_ONLY */
#endif /* NO_OPERATOR */



/****************************
 *
 *       INDEXING OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*indexing_op)(ve_arguments *, int32_t *);
/* function proto types */
uint64_t nlcpy_getitem_from_mask(ve_arguments *, int32_t *);
uint64_t nlcpy_setitem_from_mask(ve_arguments *, int32_t *);
uint64_t nlcpy_take(ve_arguments *, int32_t *);
uint64_t nlcpy_prepare_indexing(ve_arguments *, int32_t *);
uint64_t nlcpy_scatter_update(ve_arguments *, int32_t *);
uint64_t nlcpy_where(ve_arguments *, int32_t *);
uint64_t nlcpy_fill_diagonal(ve_arguments *, int32_t *);



/****************************
 *
 *       CREATION OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*creation_op)(ve_arguments *, int32_t *);
/* function proto types */
uint64_t nlcpy_arange(ve_arguments *, int32_t *);
uint64_t nlcpy_copy(ve_arguments *, int32_t *);
uint64_t nlcpy_copy_masked(ve_arguments *, int32_t *);
uint64_t nlcpy_eye(ve_arguments *, int32_t *);
uint64_t nlcpy_linspace(ve_arguments *, int32_t *);
uint64_t nlcpy_tri(ve_arguments *, int32_t *);



/****************************
 *
 *       MANIPULATION OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*manipulation_op)(ve_arguments *, int32_t *);
/* function proto types */
uint64_t nlcpy_block(ve_arguments *, int32_t *);
uint64_t nlcpy_delete(ve_arguments *, int32_t *);
uint64_t nlcpy_insert(ve_arguments *, int32_t *);
uint64_t nlcpy_repeat(ve_arguments *, int32_t *);
uint64_t nlcpy_roll(ve_arguments *, int32_t *);
uint64_t nlcpy_tile(ve_arguments *, int32_t *);



/****************************
 *
 *       LINALG OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*linalg_op)(ve_arguments *, int32_t *);
/* function proto types */
uint64_t nlcpy_dot(ve_arguments *, int32_t *);
uint64_t nlcpy_matmul(ve_arguments *, int32_t *);
uint64_t nlcpy_simple_fnorm(ve_arguments *, int32_t *);
uint64_t nlcpy_fnorm(ve_arguments *, int32_t *);



/****************************
 *
 *       REDUCE OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*reduce_op)(ve_arguments *, int32_t *);
/* function proto types */
#ifndef NO_OPERATOR
uint64_t nlcpy_add_reduce(ve_arguments *, int32_t *);
#ifndef ADD_ONLY
uint64_t nlcpy_subtract_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_multiply_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_floor_divide_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_true_divide_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_divide_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_mod_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_remainder_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_power_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_and_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_xor_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_or_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_and_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_xor_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_or_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_right_shift_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_left_shift_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_less_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_greater_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_less_equal_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_greater_equal_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_equal_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_not_equal_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_arctan2_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_hypot_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_logaddexp_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_logaddexp2_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_heaviside_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_maximum_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_minimum_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_copysign_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_fmax_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_fmin_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_fmod_reduce(ve_arguments *, int32_t *);
uint64_t nlcpy_nextafter_reduce(ve_arguments *, int32_t *);
#endif /* ADD_ONLY */
#endif /* NO_OPERATOR */


/****************************
 *
 *       REDUCEAT OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*reduceat_op)(ve_arguments *, int32_t *);


/****************************
 *
 *       ACCUMULATE OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*accumulate_op)(ve_arguments *, int32_t *);
/* function proto types */
#ifndef NO_OPERATOR
uint64_t nlcpy_add_accumulate(ve_arguments *, int32_t *);
#ifndef ADD_ONLY
uint64_t nlcpy_subtract_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_multiply_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_divide_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_logaddexp_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_logaddexp2_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_true_divide_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_floor_divide_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_power_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_remainder_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_mod_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_fmod_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_heaviside_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_and_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_or_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_xor_accumulate(ve_arguments *, int32_t *);
//uint64_t nlcpy_invert_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_left_shift_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_right_shift_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_greater_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_greater_equal_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_less_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_less_equal_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_not_equal_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_equal_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_and_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_or_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_xor_accumulate(ve_arguments *, int32_t *);
//uint64_t nlcpy_logical_not_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_maximum_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_minimum_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_fmax_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_fmin_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_arctan2_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_hypot_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_copysign_accumulate(ve_arguments *, int32_t *);
uint64_t nlcpy_nextafter_accumulate(ve_arguments *, int32_t *);
#endif /* ADD_ONLY */
#endif /* NO_OPERATOR */


/****************************
 *
 *       OUTER OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*outer_op)(ve_arguments *, int32_t *);
/* function proto types */
#ifndef NO_OPERATOR
uint64_t nlcpy_add_outer(ve_arguments *, int32_t *);
#ifndef ADD_ONLY
uint64_t nlcpy_add_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_subtract_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_multiply_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_floor_divide_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_true_divide_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_divide_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_power_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_and_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_xor_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_bitwise_or_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_and_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_xor_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_logical_or_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_right_shift_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_left_shift_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_mod_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_remainder_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_less_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_greater_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_less_equal_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_greater_equal_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_equal_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_not_equal_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_arctan2_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_hypot_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_logaddexp_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_logaddexp2_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_heaviside_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_maximum_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_minimum_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_copysign_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_fmax_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_fmin_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_fmod_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_nextafter_outer(ve_arguments *, int32_t *);
uint64_t nlcpy_ldexp_outer(ve_arguments *, int32_t *);
#endif /* ADD_ONLY */
#endif /* NO_OPERATOR */


/****************************
 *
 *       AT OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*at_op)(ve_arguments *, int32_t *);
/* function proto types */


/****************************
 *
 *       SEARCHING OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*searching_op)(ve_arguments *, int32_t *);
/* function proto types */
uint64_t nlcpy_nonzero(ve_arguments *, int32_t *);
uint64_t nlcpy_argmax(ve_arguments *, int32_t *);
uint64_t nlcpy_argmin(ve_arguments *, int32_t *);
uint64_t nlcpy_argwhere(ve_arguments *, int32_t *);


/****************************
 *
 *       SORTING OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*sorting_op)(ve_arguments *, int32_t *);
/* function proto types */
uint64_t nlcpy_sort(ve_arguments *, int32_t *);
uint64_t nlcpy_argsort(ve_arguments *, int32_t *);
uint64_t nlcpy_sort_multi(ve_arguments *, int32_t *);

/****************************
 *
 *       MATH OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*math_op)(ve_arguments *, int32_t *);
/* function proto types */
uint64_t nlcpy_diff(ve_arguments *, int32_t *);
uint64_t nlcpy_clip(ve_arguments *, int32_t *);


/****************************
 *
 *       RANDOM OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*random_op)(ve_arguments *, int32_t *);
/* function proto types */
uint64_t nlcpy_random_shuffle(ve_arguments *, int32_t *);


/****************************
 *
 *       SCA
 *
 * **************************/
/* function pointer */
typedef uint64_t (*sca_op)(ve_arguments *, int32_t *);
/* function proto types */
uint64_t nlcpy_sca_code_execute(ve_arguments *, int32_t *);


/****************************
 *
 *       MASK OPERATOR
 *
 * **************************/
/* function pointer */
typedef uint64_t (*mask_op)(ve_arguments *, int32_t *);
/* function proto types */
uint64_t nlcpy_domain_mask(ve_arguments *, int32_t *);


/****************************
 *
 *       ERROR FUNCS
 *
 * **************************/
uint64_t nlcpy__select_err(ve_arguments *, int32_t *);


#endif /* VE_FUNCLIST_H_INCLUDED */

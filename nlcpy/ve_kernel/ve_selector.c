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



binary_op get_binary_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
#ifndef NO_OPERATOR
    case VE_FUNC_ADD          :
        return nlcpy_add;
#ifndef ADD_ONLY
    case VE_FUNC_SUBTRACT     :
        return nlcpy_subtract;
    case VE_FUNC_MULTIPLY     :
        return nlcpy_multiply;
    case VE_FUNC_DIVIDE       :
        return nlcpy_divide;
    case VE_FUNC_LOGADDEXP    :
        return nlcpy_logaddexp;
    case VE_FUNC_LOGADDEXP2   :
        return nlcpy_logaddexp2;
    case VE_FUNC_TRUE_DIVIDE  :
        return nlcpy_true_divide;
    case VE_FUNC_FLOOR_DIVIDE :
        return nlcpy_floor_divide;
    case VE_FUNC_POWER        :
        return nlcpy_power;
    case VE_FUNC_REMAINDER    :
        return nlcpy_remainder;
    case VE_FUNC_MOD          :
        return nlcpy_mod;
    case VE_FUNC_FMOD         :
        return nlcpy_fmod;
    case VE_FUNC_DIVMOD       :
        //return nlcpy_divmod;
    case VE_FUNC_HEAVISIDE    :
        return nlcpy_heaviside;
    case VE_FUNC_GCD          :
        //return nlcpy_gcd;
    case VE_FUNC_LCM          :
        //return nlcpy_lcm;
    case VE_FUNC_BITWISE_AND  :
        return nlcpy_bitwise_and;
    case VE_FUNC_BITWISE_OR   :
        return nlcpy_bitwise_or;
    case VE_FUNC_BITWISE_XOR  :
        return nlcpy_bitwise_xor;
    case VE_FUNC_LEFT_SHIFT   :
        return nlcpy_left_shift;
    case VE_FUNC_RIGHT_SHIFT  :
        return nlcpy_right_shift;
    case VE_FUNC_GREATER      :
        return nlcpy_greater;
    case VE_FUNC_GREATER_EQUAL:
        return nlcpy_greater_equal;
    case VE_FUNC_LESS         :
        return nlcpy_less;
    case VE_FUNC_LESS_EQUAL   :
        return nlcpy_less_equal;
    case VE_FUNC_NOT_EQUAL    :
        return nlcpy_not_equal;
    case VE_FUNC_EQUAL        :
        return nlcpy_equal;
    case VE_FUNC_LOGICAL_AND  :
        return nlcpy_logical_and;
    case VE_FUNC_LOGICAL_OR   :
        return nlcpy_logical_or;
    case VE_FUNC_LOGICAL_XOR  :
        return nlcpy_logical_xor;
    case VE_FUNC_MAXIMUM      :
        return nlcpy_maximum;
    case VE_FUNC_MINIMUM      :
        return nlcpy_minimum;
    case VE_FUNC_FMAX         :
        return nlcpy_fmax;
    case VE_FUNC_FMIN         :
        return nlcpy_fmin;
    case VE_FUNC_ARCTAN2      :
        return nlcpy_arctan2;
    case VE_FUNC_HYPOT        :
        return nlcpy_hypot;
    case VE_FUNC_COPYSIGN     :
        return nlcpy_copysign;
    case VE_FUNC_NEXTAFTER    :
        return nlcpy_nextafter;
    case VE_FUNC_MODF         :
        //return nlcpy_modf;
    case VE_FUNC_LDEXP        :
        return nlcpy_ldexp;
    case VE_FUNC_FEXP         :
        //return nlcpy_fexp;
#endif /* ADD_ONLY */
#endif /* NO_OPERATOR */
    default                   :
        return nlcpy__select_err;
    }
}



unary_op get_unary_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
#ifndef NO_OPERATOR
#ifndef ADD_ONLY
    case VE_FUNC_NEGATIVE   :
        return nlcpy_negative;
    case VE_FUNC_POSITIVE   :
        return nlcpy_positive;
    case VE_FUNC_ABSOLUTE   :
        return nlcpy_absolute;
    case VE_FUNC_FABS       :
        return nlcpy_fabs;
    case VE_FUNC_RINT       :
        return nlcpy_rint;
    case VE_FUNC_SIGN       :
        return nlcpy_sign;
    case VE_FUNC_CONJ       :
        return nlcpy_conj;
    case VE_FUNC_CONJUGATE  :
        return nlcpy_conjugate;
    case VE_FUNC_EXP        :
        return nlcpy_exp;
    case VE_FUNC_EXP2       :
        return nlcpy_exp2;
    case VE_FUNC_LOG        :
        return nlcpy_log;
    case VE_FUNC_LOG2       :
        return nlcpy_log2;
    case VE_FUNC_LOG10      :
        return nlcpy_log10;
    case VE_FUNC_EXPM1      :
        return nlcpy_expm1;
    case VE_FUNC_LOG1P      :
        return nlcpy_log1p;
    case VE_FUNC_SQRT       :
        return nlcpy_sqrt;
    case VE_FUNC_SQUARE     :
        return nlcpy_square;
    case VE_FUNC_CBRT       :
        return nlcpy_cbrt;
    case VE_FUNC_RECIPROCAL :
        return nlcpy_reciprocal;
    case VE_FUNC_SIN        :
        return nlcpy_sin;
    case VE_FUNC_COS        :
        return nlcpy_cos;
    case VE_FUNC_TAN        :
        return nlcpy_tan;
    case VE_FUNC_ARCSIN     :
        return nlcpy_arcsin;
    case VE_FUNC_ARCCOS     :
        return nlcpy_arccos;
    case VE_FUNC_ARCTAN     :
        return nlcpy_arctan;
    case VE_FUNC_SINH       :
        return nlcpy_sinh;
    case VE_FUNC_COSH       :
        return nlcpy_cosh;
    case VE_FUNC_TANH       :
        return nlcpy_tanh;
    case VE_FUNC_ARCSINH    :
        return nlcpy_arcsinh;
    case VE_FUNC_ARCCOSH    :
        return nlcpy_arccosh;
    case VE_FUNC_ARCTANH    :
        return nlcpy_arctanh;
    case VE_FUNC_DEG2RAD    :
        return nlcpy_deg2rad;
    case VE_FUNC_RAD2DEG    :
        return nlcpy_rad2deg;
    case VE_FUNC_DEGREES    :
        return nlcpy_degrees;
    case VE_FUNC_RADIANS    :
        return nlcpy_radians;
    case VE_FUNC_INVERT     :
        return nlcpy_invert;
    case VE_FUNC_LOGICAL_NOT:
        return nlcpy_logical_not;
    case VE_FUNC_ISFINITE   :
        return nlcpy_isfinite;
    case VE_FUNC_ISINF      :
        return nlcpy_isinf;
    case VE_FUNC_ISNAN      :
        return nlcpy_isnan;
    case VE_FUNC_SIGNBIT    :
        return nlcpy_signbit;
    case VE_FUNC_SPACING    :
        return nlcpy_spacing;
    case VE_FUNC_FLOOR      :
        return nlcpy_floor;
    case VE_FUNC_CEIL       :
        return nlcpy_ceil;
    case VE_FUNC_TRUNC      :
        return nlcpy_trunc;
    case VE_FUNC_ANGLE      :
        return nlcpy_angle;
    case VE_FUNC_ERF        :
        return nlcpy_erf;
    case VE_FUNC_ERFC       :
        return nlcpy_erfc;
#endif /* ADD_ONLY */
#endif /* NO_OPERATOR */
    default:
        return nlcpy__select_err;
    }
}


indexing_op get_indexing_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
    case VE_FUNC_GETITEM_FROM_MASK:
        return nlcpy_getitem_from_mask;
    case VE_FUNC_SETITEM_FROM_MASK:
        return nlcpy_setitem_from_mask;
    case VE_FUNC_TAKE:
        return nlcpy_take;
    case VE_FUNC_PREPARE_INDEXING:
        return nlcpy_prepare_indexing;
    case VE_FUNC_SCATTER_UPDATE:
        return nlcpy_scatter_update;
    case VE_FUNC_WHERE:
        return nlcpy_where;
    case VE_FUNC_FILL_DIAGONAL:
        return nlcpy_fill_diagonal;
    default:
        return nlcpy__select_err;
    }
}


creation_op get_creation_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
    case VE_FUNC_ARANGE     :
        return nlcpy_arange;
    case VE_FUNC_COPY       :
        return nlcpy_copy;
    case VE_FUNC_EYE        :
        return nlcpy_eye;
    case VE_FUNC_LINSPACE   :
        return nlcpy_linspace;
    case VE_FUNC_COPY_MASKED :
        return nlcpy_copy_masked;
    case VE_FUNC_TRI        :
        return nlcpy_tri;
    default:
        return nlcpy__select_err;
    }
}


manipulation_op get_manipulation_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
    case VE_FUNC_BLOCK  :
        return nlcpy_block;
    case VE_FUNC_DELETE :
        return nlcpy_delete;
    case VE_FUNC_INSERT :
        return nlcpy_insert;
    case VE_FUNC_TILE   :
        return nlcpy_tile;
    case VE_FUNC_REPEAT   :
        return nlcpy_repeat;
    case VE_FUNC_ROLL   :
        return nlcpy_roll;
    default:
        return nlcpy__select_err;
    }
}


cblas_wrapper_op get_cblas_wrapper_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
    case VE_FUNC_CBLAS_SDOT       :
        return wrapper_cblas_sdot;
    case VE_FUNC_CBLAS_DDOT       :
        return wrapper_cblas_ddot;
    case VE_FUNC_CBLAS_CDOTU_SUB  :
        return wrapper_cblas_cdotu_sub;
    case VE_FUNC_CBLAS_ZDOTU_SUB  :
        return wrapper_cblas_zdotu_sub;
    case VE_FUNC_CBLAS_SGEMM      :
        return wrapper_cblas_sgemm;
    case VE_FUNC_CBLAS_DGEMM      :
        return wrapper_cblas_dgemm;
    case VE_FUNC_CBLAS_CGEMM      :
        return wrapper_cblas_cgemm;
    case VE_FUNC_CBLAS_ZGEMM      :
        return wrapper_cblas_zgemm;
    default:
        return nlcpy__select_err;
    }
}


linalg_op get_linalg_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
    case VE_FUNC_DOT       :
        return nlcpy_dot;
    case VE_FUNC_MATMUL    :
        return nlcpy_matmul;
    default:
        return nlcpy__select_err;
    }
}


reduce_op get_reduce_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
#ifndef NO_OPERATOR
    case VE_FUNC_ADD_REDUCE    :
        return nlcpy_add_reduce;
#ifndef ADD_ONLY
    case VE_FUNC_SUBTRACT_REDUCE:
        return nlcpy_subtract_reduce;
    case VE_FUNC_MULTIPLY_REDUCE:
        return nlcpy_multiply_reduce;
    case VE_FUNC_FLOOR_DIVIDE_REDUCE:
        return nlcpy_floor_divide_reduce;
    case VE_FUNC_TRUE_DIVIDE_REDUCE:
        return nlcpy_true_divide_reduce;
    case VE_FUNC_DIVIDE_REDUCE:
        return nlcpy_divide_reduce;
    case VE_FUNC_MOD_REDUCE:
        return nlcpy_mod_reduce;
    case VE_FUNC_REMAINDER_REDUCE:
        return nlcpy_remainder_reduce;
    case VE_FUNC_POWER_REDUCE:
        return nlcpy_power_reduce;
    case VE_FUNC_BITWISE_AND_REDUCE:
        return nlcpy_bitwise_and_reduce;
    case VE_FUNC_BITWISE_XOR_REDUCE:
        return nlcpy_bitwise_xor_reduce;
    case VE_FUNC_BITWSE_OR_REDUCE:
        return nlcpy_bitwise_or_reduce;
    case VE_FUNC_LOGICAL_AND_REDUCE    :
        return nlcpy_logical_and_reduce;
    case VE_FUNC_LOGICAL_XOR_REDUCE    :
        return nlcpy_logical_xor_reduce;
    case VE_FUNC_LOGICAL_OR_REDUCE    :
        return nlcpy_logical_or_reduce;
    case VE_FUNC_RIGHT_SHIFT_REDUCE:
        return nlcpy_right_shift_reduce;
    case VE_FUNC_LEFT_SHIFT_REDUCE:
        return nlcpy_left_shift_reduce;
    case VE_FUNC_LESS_REDUCE:
        return nlcpy_less_reduce;
    case VE_FUNC_GREATER_REDUCE:
        return nlcpy_greater_reduce;
    case VE_FUNC_LESS_EQUAL_REDUCE:
        return nlcpy_less_equal_reduce;
    case VE_FUNC_GREATER_EQUAL_REDUCE:
        return nlcpy_greater_equal_reduce;
    case VE_FUNC_EQUAL_REDUCE:
        return nlcpy_equal_reduce;
    case VE_FUNC_NOT_EQUAL_REDUCE:
        return nlcpy_not_equal_reduce;
    case VE_FUNC_ARCTAN2_REDUCE:
        return nlcpy_arctan2_reduce;
    case VE_FUNC_HYPOT_REDUCE:
        return nlcpy_hypot_reduce;
    case VE_FUNC_LOGADDEXP_REDUCE:
        return nlcpy_logaddexp_reduce;
    case VE_FUNC_LOGADDEXP2_REDUCE:
        return nlcpy_logaddexp2_reduce;
    case VE_FUNC_HEAVISIDE_REDUCE:
        return nlcpy_heaviside_reduce;
    case VE_FUNC_MAXIMUM_REDUCE    :
        return nlcpy_maximum_reduce;
    case VE_FUNC_MINIMUM_REDUCE    :
        return nlcpy_minimum_reduce;
    case VE_FUNC_COPYSIGN_REDUCE:
        return nlcpy_copysign_reduce;
    case VE_FUNC_FMAX_REDUCE:
        return nlcpy_fmax_reduce;
    case VE_FUNC_FMIN_REDUCE:
        return nlcpy_fmin_reduce;
    case VE_FUNC_FMOD_REDUCE:
        return nlcpy_fmod_reduce;
    case VE_FUNC_NEXTAFTER_REDUCE:
        return nlcpy_nextafter_reduce;
#endif /* ADD_ONLY */
#endif /* NO_OPERATOR */
    default:
        return nlcpy__select_err;
    }
}


reduceat_op get_reduceat_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
    default:
        return nlcpy__select_err;
    }
}


accumulate_op get_accumulate_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
#ifndef NO_OPERATOR
    case VE_FUNC_ADD_ACCUMULATE    :
        return nlcpy_add_accumulate;
#ifndef ADD_ONLY
    case VE_FUNC_SUBTRACT_ACCUMULATE    :
        return nlcpy_subtract_accumulate;
    case VE_FUNC_MULTIPLY_ACCUMULATE    :
        return nlcpy_multiply_accumulate;
    case VE_FUNC_DIVIDE_ACCUMULATE    :
        return nlcpy_divide_accumulate;
    case VE_FUNC_LOGADDEXP_ACCUMULATE    :
        return nlcpy_logaddexp_accumulate;
    case VE_FUNC_LOGADDEXP2_ACCUMULATE    :
        return nlcpy_logaddexp2_accumulate;
    case VE_FUNC_TRUE_DIVIDE_ACCUMULATE    :
        return nlcpy_true_divide_accumulate;
    case VE_FUNC_FLOOR_DIVIDE_ACCUMULATE    :
        return nlcpy_floor_divide_accumulate;
    case VE_FUNC_POWER_ACCUMULATE    :
        return nlcpy_power_accumulate;
    case VE_FUNC_REMAINDER_ACCUMULATE    :
        return nlcpy_remainder_accumulate;
    case VE_FUNC_MOD_ACCUMULATE    :
        return nlcpy_mod_accumulate;
    case VE_FUNC_FMOD_ACCUMULATE    :
        return nlcpy_fmod_accumulate;
    case VE_FUNC_HEAVISIDE_ACCUMULATE   :
        return nlcpy_heaviside_accumulate;
    case VE_FUNC_BITWISE_AND_ACCUMULATE    :
        return nlcpy_bitwise_and_accumulate;
    case VE_FUNC_BITWISE_OR_ACCUMULATE    :
        return nlcpy_bitwise_or_accumulate;
    case VE_FUNC_BITWISE_XOR_ACCUMULATE    :
        return nlcpy_bitwise_xor_accumulate;
//    case VE_FUNC_INVERT_ACCUMULATE    :
//        return nlcpy_invert_accumulate;
    case VE_FUNC_LEFT_SHIFT_ACCUMULATE    :
        return nlcpy_left_shift_accumulate;
    case VE_FUNC_RIGHT_SHIFT_ACCUMULATE    :
        return nlcpy_right_shift_accumulate;
    case VE_FUNC_GREATER_ACCUMULATE    :
        return nlcpy_greater_accumulate;
    case VE_FUNC_GREATER_EQUAL_ACCUMULATE    :
        return nlcpy_greater_equal_accumulate;
    case VE_FUNC_LESS_ACCUMULATE    :
        return nlcpy_less_accumulate;
    case VE_FUNC_LESS_EQUAL_ACCUMULATE    :
        return nlcpy_less_equal_accumulate;
    case VE_FUNC_NOT_EQUAL_ACCUMULATE    :
        return nlcpy_not_equal_accumulate;
    case VE_FUNC_EQUAL_ACCUMULATE    :
        return nlcpy_equal_accumulate;
    case VE_FUNC_LOGICAL_AND_ACCUMULATE    :
        return nlcpy_logical_and_accumulate;
    case VE_FUNC_LOGICAL_OR_ACCUMULATE    :
        return nlcpy_logical_or_accumulate;
    case VE_FUNC_LOGICAL_XOR_ACCUMULATE    :
        return nlcpy_logical_xor_accumulate;
//    case VE_FUNC_LOGICAL_NOT_ACCUMULATE    :
//        return nlcpy_logical_not_accumulate;
    case VE_FUNC_MAXIMUM_ACCUMULATE    :
        return nlcpy_maximum_accumulate;
    case VE_FUNC_MINIMUM_ACCUMULATE    :
        return nlcpy_minimum_accumulate;
    case VE_FUNC_FMAX_ACCUMULATE    :
        return nlcpy_fmax_accumulate;
    case VE_FUNC_FMIN_ACCUMULATE    :
        return nlcpy_fmin_accumulate;
    case VE_FUNC_ARCTAN2_ACCUMULATE    :
        return nlcpy_arctan2_accumulate;
    case VE_FUNC_HYPOT_ACCUMULATE    :
        return nlcpy_hypot_accumulate;
    case VE_FUNC_COPYSIGN_ACCUMULATE    :
        return nlcpy_copysign_accumulate;
    case VE_FUNC_NEXTAFTER_ACCUMULATE    :
        return nlcpy_nextafter_accumulate;
#endif /* ADD_ONLY */
#endif /* NO_OPERATOR */
    default:
        return nlcpy__select_err;
    }
}


outer_op get_outer_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
#ifndef NO_OPERATOR
    case VE_FUNC_ADD_OUTER    :
        return nlcpy_add_outer;
#ifndef ADD_ONLY
    case VE_FUNC_SUBTRACT_OUTER        : return nlcpy_subtract_outer;
    case VE_FUNC_MULTIPLY_OUTER        : return nlcpy_multiply_outer;
    case VE_FUNC_FLOOR_DIVIDE_OUTER    : return nlcpy_floor_divide_outer;
    case VE_FUNC_TRUE_DIVIDE_OUTER     : return nlcpy_true_divide_outer;
    case VE_FUNC_DIVIDE_OUTER          : return nlcpy_divide_outer;
    case VE_FUNC_POWER_OUTER           : return nlcpy_power_outer;
    case VE_FUNC_BITWISE_AND_OUTER     : return nlcpy_bitwise_and_outer;
    case VE_FUNC_BITWISE_XOR_OUTER     : return nlcpy_bitwise_xor_outer;
    case VE_FUNC_BITWISE_OR_OUTER      : return nlcpy_bitwise_or_outer;
    case VE_FUNC_LOGICAL_AND_OUTER     : return nlcpy_logical_and_outer;
    case VE_FUNC_LOGICAL_XOR_OUTER     : return nlcpy_logical_xor_outer;
    case VE_FUNC_LOGICAL_OR_OUTER      : return nlcpy_logical_or_outer;
    case VE_FUNC_RIGHT_SHIFT_OUTER     : return nlcpy_right_shift_outer;
    case VE_FUNC_LEFT_SHIFT_OUTER      : return nlcpy_left_shift_outer;
    case VE_FUNC_MOD_OUTER             : return nlcpy_mod_outer;
    case VE_FUNC_REMAINDER_OUTER       : return nlcpy_remainder_outer;
    case VE_FUNC_LESS_OUTER            : return nlcpy_less_outer;
    case VE_FUNC_GREATER_OUTER         : return nlcpy_greater_outer;
    case VE_FUNC_LESS_EQUAL_OUTER      : return nlcpy_less_equal_outer;
    case VE_FUNC_GREATER_EQUAL_OUTER   : return nlcpy_greater_equal_outer;
    case VE_FUNC_EQUAL_OUTER           : return nlcpy_equal_outer;
    case VE_FUNC_NOT_EQUAL_OUTER       : return nlcpy_not_equal_outer;
    case VE_FUNC_ARCTAN2_OUTER         : return nlcpy_arctan2_outer;
    case VE_FUNC_HYPOT_OUTER           : return nlcpy_hypot_outer;
    case VE_FUNC_LOGADDEXP_OUTER       : return nlcpy_logaddexp_outer;
    case VE_FUNC_LOGADDEXP2_OUTER      : return nlcpy_logaddexp2_outer;
    case VE_FUNC_HEAVISIDE_OUTER       : return nlcpy_heaviside_outer;
    case VE_FUNC_MAXIMUM_OUTER         : return nlcpy_maximum_outer;
    case VE_FUNC_MINIMUM_OUTER         : return nlcpy_minimum_outer;
    case VE_FUNC_COPYSIGN_OUTER        : return nlcpy_copysign_outer;
    case VE_FUNC_FMAX_OUTER            : return nlcpy_fmax_outer;
    case VE_FUNC_FMIN_OUTER            : return nlcpy_fmin_outer;
    case VE_FUNC_FMOD_OUTER            : return nlcpy_fmod_outer;
    case VE_FUNC_NEXTAFTER_OUTER       : return nlcpy_nextafter_outer;
    case VE_FUNC_LDEXP_OUTER           : return nlcpy_ldexp_outer;
#endif /* ADD_ONLY */
#endif /* NO_OPERATOR */
    default:
        return nlcpy__select_err;
    }
}


at_op get_at_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
    default:
        return nlcpy__select_err;
    }
}


searching_op get_searching_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
    case VE_FUNC_NONZERO      :
        return nlcpy_nonzero;
    case VE_FUNC_ARGMAX       :
        return nlcpy_argmax;
    case VE_FUNC_ARGMIN       :
        return nlcpy_argmin;
    case VE_FUNC_ARGWHERE     :
        return nlcpy_argwhere;
    default:
        return nlcpy__select_err;
    }
}


sorting_op get_sorting_func(int64_t func_num) {
    //printf("func_num = %d\n", func_num);
    switch (func_num) {
    case VE_FUNC_SORT      :
        return nlcpy_sort;
    case VE_FUNC_ARGSORT   :
        return nlcpy_argsort;
    case VE_FUNC_SORT_MULTI:
        return nlcpy_sort_multi;
    default:
        return nlcpy__select_err;
    }
}

math_op get_math_func(int64_t func_num) {
    switch (func_num) {
    case VE_FUNC_DIFF     :
        return nlcpy_diff;
    case VE_FUNC_CLIP     :
        return nlcpy_clip;
    default:
        return nlcpy__select_err;
    }
}

random_op get_random_func(int64_t func_num) {
    switch (func_num) {
    case VE_FUNC_SHUFFLE     :
        return nlcpy_random_shuffle;
    default:
        return nlcpy__select_err;
    }
}

sca_op get_sca_func(int64_t func_num) {
    switch (func_num) {
    case VE_FUNC_SCA_EXECUTE :
        return nlcpy_sca_code_execute;
    default:
        return nlcpy__select_err;
    }
}

mask_op get_mask_func(int64_t func_num) {
    switch (func_num) {
    case VE_FUNC_DOMAIN_MASK    :
        return nlcpy_domain_mask;
    default:
        return nlcpy__select_err;
    }
}

uint64_t nlcpy__select_err(ve_arguments *args, int32_t *psw) {
    return NLCPY_ERROR_FUNCNUM;
}


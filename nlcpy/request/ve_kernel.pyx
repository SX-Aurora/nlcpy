#
#  * The source code in this file is developed independently by NEC Corporation.
#
#  # NLCPy License #
#
#      Copyright (c) 2020-2021 NEC Corporation
#      All rights reserved.
#
#      Redistribution and use in source and binary forms, with or without
#      modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright notice,
#        this list of conditions and the following disclaimer in the documentation
#        and/or other materials provided with the distribution.
#      * Neither NEC Corporation nor the names of its contributors may be
#        used to endorse or promote products derived from this software
#        without specific prior written permission.
#
#      THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#      ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#      WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#      DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#      FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#      (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#      LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#      ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#      (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#      SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from nlcpy.request.ve_kernel cimport *


cdef dict funcNumList = {
    # binary functionss
    "nlcpy_add": ve_funcnum.VE_FUNC_ADD,
    "nlcpy_subtract": ve_funcnum.VE_FUNC_SUBTRACT,
    "nlcpy_multiply": ve_funcnum.VE_FUNC_MULTIPLY,
    "nlcpy_divide": ve_funcnum.VE_FUNC_DIVIDE,
    "nlcpy_logaddexp": ve_funcnum.VE_FUNC_LOGADDEXP,
    "nlcpy_logaddexp2": ve_funcnum.VE_FUNC_LOGADDEXP2,
    "nlcpy_true_divide": ve_funcnum.VE_FUNC_TRUE_DIVIDE,
    "nlcpy_floor_divide": ve_funcnum.VE_FUNC_FLOOR_DIVIDE,
    "nlcpy_power": ve_funcnum.VE_FUNC_POWER,
    "nlcpy_remainder": ve_funcnum.VE_FUNC_REMAINDER,
    "nlcpy_mod": ve_funcnum.VE_FUNC_MOD,
    "nlcpy_fmod": ve_funcnum.VE_FUNC_FMOD,
    # "nlcpy_divmod": ve_funcnum.VE_FUNC_DIVMOD,
    "nlcpy_heaviside": ve_funcnum.VE_FUNC_HEAVISIDE,
    # "nlcpy_gcd": ve_funcnum.VE_FUNC_GCD,
    # "nlcpy_lcm": ve_funcnum.VE_FUNC_LCM,
    "nlcpy_bitwise_and": ve_funcnum.VE_FUNC_BITWISE_AND,
    "nlcpy_bitwise_or": ve_funcnum.VE_FUNC_BITWISE_OR,
    "nlcpy_bitwise_xor": ve_funcnum.VE_FUNC_BITWISE_XOR,
    "nlcpy_left_shift": ve_funcnum.VE_FUNC_LEFT_SHIFT,
    "nlcpy_right_shift": ve_funcnum.VE_FUNC_RIGHT_SHIFT,
    "nlcpy_greater": ve_funcnum.VE_FUNC_GREATER,
    "nlcpy_greater_equal": ve_funcnum.VE_FUNC_GREATER_EQUAL,
    "nlcpy_less": ve_funcnum.VE_FUNC_LESS,
    "nlcpy_less_equal": ve_funcnum.VE_FUNC_LESS_EQUAL,
    "nlcpy_not_equal": ve_funcnum.VE_FUNC_NOT_EQUAL,
    "nlcpy_equal": ve_funcnum.VE_FUNC_EQUAL,
    "nlcpy_logical_and": ve_funcnum.VE_FUNC_LOGICAL_AND,
    "nlcpy_logical_or": ve_funcnum.VE_FUNC_LOGICAL_OR,
    "nlcpy_logical_xor": ve_funcnum.VE_FUNC_LOGICAL_XOR,
    "nlcpy_maximum": ve_funcnum.VE_FUNC_MAXIMUM,
    "nlcpy_minimum": ve_funcnum.VE_FUNC_MINIMUM,
    "nlcpy_fmax": ve_funcnum.VE_FUNC_FMAX,
    "nlcpy_fmin": ve_funcnum.VE_FUNC_FMIN,
    "nlcpy_arctan2": ve_funcnum.VE_FUNC_ARCTAN2,
    "nlcpy_hypot": ve_funcnum.VE_FUNC_HYPOT,
    "nlcpy_copysign": ve_funcnum.VE_FUNC_COPYSIGN,
    "nlcpy_nextafter": ve_funcnum.VE_FUNC_NEXTAFTER,
    # "nlcpy_modf": ve_funcnum.VE_FUNC_MODF,
    "nlcpy_ldexp": ve_funcnum.VE_FUNC_LDEXP,
    # "nlcpy_fexp": ve_funcnum.VE_FUNC_FEXP,

    # unary functions
    "nlcpy_negative": ve_funcnum.VE_FUNC_NEGATIVE,
    "nlcpy_positive": ve_funcnum.VE_FUNC_POSITIVE,
    "nlcpy_absolute": ve_funcnum.VE_FUNC_ABSOLUTE,
    "nlcpy_fabs": ve_funcnum.VE_FUNC_FABS,
    "nlcpy_rint": ve_funcnum.VE_FUNC_RINT,
    "nlcpy_sign": ve_funcnum.VE_FUNC_SIGN,
    "nlcpy_conj": ve_funcnum.VE_FUNC_CONJ,
    "nlcpy_conjugate": ve_funcnum.VE_FUNC_CONJUGATE,
    "nlcpy_exp": ve_funcnum.VE_FUNC_EXP,
    "nlcpy_exp2": ve_funcnum.VE_FUNC_EXP2,
    "nlcpy_log": ve_funcnum.VE_FUNC_LOG,
    "nlcpy_log2": ve_funcnum.VE_FUNC_LOG2,
    "nlcpy_log10": ve_funcnum.VE_FUNC_LOG10,
    "nlcpy_expm1": ve_funcnum.VE_FUNC_EXPM1,
    "nlcpy_log1p": ve_funcnum.VE_FUNC_LOG1P,
    "nlcpy_sqrt": ve_funcnum.VE_FUNC_SQRT,
    "nlcpy_square": ve_funcnum.VE_FUNC_SQUARE,
    "nlcpy_cbrt": ve_funcnum.VE_FUNC_CBRT,
    "nlcpy_reciprocal": ve_funcnum.VE_FUNC_RECIPROCAL,
    "nlcpy_sin": ve_funcnum.VE_FUNC_SIN,
    "nlcpy_cos": ve_funcnum.VE_FUNC_COS,
    "nlcpy_tan": ve_funcnum.VE_FUNC_TAN,
    "nlcpy_arcsin": ve_funcnum.VE_FUNC_ARCSIN,
    "nlcpy_arccos": ve_funcnum.VE_FUNC_ARCCOS,
    "nlcpy_arctan": ve_funcnum.VE_FUNC_ARCTAN,
    "nlcpy_sinh": ve_funcnum.VE_FUNC_SINH,
    "nlcpy_cosh": ve_funcnum.VE_FUNC_COSH,
    "nlcpy_tanh": ve_funcnum.VE_FUNC_TANH,
    "nlcpy_arcsinh": ve_funcnum.VE_FUNC_ARCSINH,
    "nlcpy_arccosh": ve_funcnum.VE_FUNC_ARCCOSH,
    "nlcpy_arctanh": ve_funcnum.VE_FUNC_ARCTANH,
    "nlcpy_deg2rad": ve_funcnum.VE_FUNC_DEG2RAD,
    "nlcpy_rad2deg": ve_funcnum.VE_FUNC_RAD2DEG,
    "nlcpy_degrees": ve_funcnum.VE_FUNC_DEGREES,
    "nlcpy_radians": ve_funcnum.VE_FUNC_RADIANS,
    "nlcpy_invert": ve_funcnum.VE_FUNC_INVERT,
    "nlcpy_logical_not": ve_funcnum.VE_FUNC_LOGICAL_NOT,
    "nlcpy_isfinite": ve_funcnum.VE_FUNC_ISFINITE,
    "nlcpy_isinf": ve_funcnum.VE_FUNC_ISINF,
    "nlcpy_isnan": ve_funcnum.VE_FUNC_ISNAN,
    "nlcpy_signbit": ve_funcnum.VE_FUNC_SIGNBIT,
    "nlcpy_spacing": ve_funcnum.VE_FUNC_SPACING,
    "nlcpy_floor": ve_funcnum.VE_FUNC_FLOOR,
    "nlcpy_ceil": ve_funcnum.VE_FUNC_CEIL,
    "nlcpy_trunc": ve_funcnum.VE_FUNC_TRUNC,
    "nlcpy_angle": ve_funcnum.VE_FUNC_ANGLE,
    "nlcpy_erf": ve_funcnum.VE_FUNC_ERF,
    "nlcpy_erfc": ve_funcnum.VE_FUNC_ERFC,

    #  indexing functions
    "nlcpy_getitem_from_mask": ve_funcnum.VE_FUNC_GETITEM_FROM_MASK,
    "nlcpy_setitem_from_mask": ve_funcnum.VE_FUNC_SETITEM_FROM_MASK,
    "nlcpy_take": ve_funcnum.VE_FUNC_TAKE,
    "nlcpy_prepare_indexing": ve_funcnum.VE_FUNC_PREPARE_INDEXING,
    "nlcpy_scatter_update": ve_funcnum.VE_FUNC_SCATTER_UPDATE,
    "nlcpy_where": ve_funcnum.VE_FUNC_WHERE,
    "nlcpy_fill_diagonal": ve_funcnum.VE_FUNC_FILL_DIAGONAL,

    #  creation functions
    "nlcpy_arange": ve_funcnum.VE_FUNC_ARANGE,
    "nlcpy_copy": ve_funcnum.VE_FUNC_COPY,
    "nlcpy_eye": ve_funcnum.VE_FUNC_EYE,
    "nlcpy_linspace": ve_funcnum.VE_FUNC_LINSPACE,
    "nlcpy_copy_masked": ve_funcnum.VE_FUNC_COPY_MASKED,
    "nlcpy_tri": ve_funcnum.VE_FUNC_TRI,

    #  manipulation functions
    "nlcpy_block": ve_funcnum.VE_FUNC_BLOCK,
    "nlcpy_delete": ve_funcnum.VE_FUNC_DELETE,
    "nlcpy_tile": ve_funcnum.VE_FUNC_TILE,
    "nlcpy_repeat": ve_funcnum.VE_FUNC_REPEAT,
    "nlcpy_insert": ve_funcnum.VE_FUNC_INSERT,
    "nlcpy_roll": ve_funcnum.VE_FUNC_ROLL,

    #  cblas wrapper functions
    "wrapper_cblas_sdot": ve_funcnum.VE_FUNC_CBLAS_SDOT,
    "wrapper_cblas_ddot": ve_funcnum.VE_FUNC_CBLAS_DDOT,
    "wrapper_cblas_cdotu_sub": ve_funcnum.VE_FUNC_CBLAS_CDOTU_SUB,
    "wrapper_cblas_zdotu_sub": ve_funcnum.VE_FUNC_CBLAS_ZDOTU_SUB,
    "wrapper_cblas_sgemm": ve_funcnum.VE_FUNC_CBLAS_SGEMM,
    "wrapper_cblas_dgemm": ve_funcnum.VE_FUNC_CBLAS_DGEMM,
    "wrapper_cblas_cgemm": ve_funcnum.VE_FUNC_CBLAS_CGEMM,
    "wrapper_cblas_zgemm": ve_funcnum.VE_FUNC_CBLAS_ZGEMM,

    #  linalg functions
    "nlcpy_dot": ve_funcnum.VE_FUNC_DOT,
    "nlcpy_matmul": ve_funcnum.VE_FUNC_MATMUL,

    #  reduce functions
    "nlcpy_add_reduce": ve_funcnum.VE_FUNC_ADD_REDUCE,
    "nlcpy_subtract_reduce": ve_funcnum.VE_FUNC_SUBTRACT_REDUCE,
    "nlcpy_multiply_reduce": ve_funcnum.VE_FUNC_MULTIPLY_REDUCE,
    "nlcpy_floor_divide_reduce": ve_funcnum.VE_FUNC_FLOOR_DIVIDE_REDUCE,
    "nlcpy_true_divide_reduce": ve_funcnum.VE_FUNC_TRUE_DIVIDE_REDUCE,
    "nlcpy_divide_reduce": ve_funcnum.VE_FUNC_DIVIDE_REDUCE,
    "nlcpy_mod_reduce": ve_funcnum.VE_FUNC_MOD_REDUCE,
    "nlcpy_remainder_reduce": ve_funcnum.VE_FUNC_REMAINDER_REDUCE,
    "nlcpy_power_reduce": ve_funcnum.VE_FUNC_POWER_REDUCE,
    "nlcpy_bitwise_and_reduce": ve_funcnum.VE_FUNC_BITWISE_AND_REDUCE,
    "nlcpy_bitwise_xor_reduce": ve_funcnum.VE_FUNC_BITWISE_XOR_REDUCE,
    "nlcpy_bitwise_or_reduce": ve_funcnum.VE_FUNC_BITWSE_OR_REDUCE,
    "nlcpy_logical_and_reduce": ve_funcnum.VE_FUNC_LOGICAL_AND_REDUCE,
    "nlcpy_logical_xor_reduce": ve_funcnum.VE_FUNC_LOGICAL_XOR_REDUCE,
    "nlcpy_logical_or_reduce": ve_funcnum.VE_FUNC_LOGICAL_OR_REDUCE,
    "nlcpy_right_shift_reduce": ve_funcnum.VE_FUNC_RIGHT_SHIFT_REDUCE,
    "nlcpy_left_shift_reduce": ve_funcnum.VE_FUNC_LEFT_SHIFT_REDUCE,
    "nlcpy_less_reduce": ve_funcnum.VE_FUNC_LESS_REDUCE,
    "nlcpy_greater_reduce": ve_funcnum.VE_FUNC_GREATER_REDUCE,
    "nlcpy_less_equal_reduce": ve_funcnum.VE_FUNC_LESS_EQUAL_REDUCE,
    "nlcpy_greater_equal_reduce": ve_funcnum.VE_FUNC_GREATER_EQUAL_REDUCE,
    "nlcpy_equal_reduce": ve_funcnum.VE_FUNC_EQUAL_REDUCE,
    "nlcpy_not_equal_reduce": ve_funcnum.VE_FUNC_NOT_EQUAL_REDUCE,
    "nlcpy_arctan2_reduce": ve_funcnum.VE_FUNC_ARCTAN2_REDUCE,
    "nlcpy_hypot_reduce": ve_funcnum.VE_FUNC_HYPOT_REDUCE,
    "nlcpy_logaddexp_reduce": ve_funcnum.VE_FUNC_LOGADDEXP_REDUCE,
    "nlcpy_logaddexp2_reduce": ve_funcnum.VE_FUNC_LOGADDEXP2_REDUCE,
    "nlcpy_heaviside_reduce": ve_funcnum.VE_FUNC_HEAVISIDE_REDUCE,
    "nlcpy_maximum_reduce": ve_funcnum.VE_FUNC_MAXIMUM_REDUCE,
    "nlcpy_minimum_reduce": ve_funcnum.VE_FUNC_MINIMUM_REDUCE,
    "nlcpy_copysign_reduce": ve_funcnum.VE_FUNC_COPYSIGN_REDUCE,
    "nlcpy_fmax_reduce": ve_funcnum.VE_FUNC_FMAX_REDUCE,
    "nlcpy_fmin_reduce": ve_funcnum.VE_FUNC_FMIN_REDUCE,
    "nlcpy_fmod_reduce": ve_funcnum.VE_FUNC_FMOD_REDUCE,
    "nlcpy_nextafter_reduce": ve_funcnum.VE_FUNC_NEXTAFTER_REDUCE,
    #  reduceat functions
    "nlcpy_add_reduceat": ve_funcnum.VE_FUNC_ADD_REDUCEAT,
    #  accumulate functions
    "nlcpy_add_accumulate": ve_funcnum.VE_FUNC_ADD_ACCUMULATE,
    "nlcpy_subtract_accumulate": ve_funcnum.VE_FUNC_SUBTRACT_ACCUMULATE,
    "nlcpy_multiply_accumulate": ve_funcnum.VE_FUNC_MULTIPLY_ACCUMULATE,
    "nlcpy_divide_accumulate": ve_funcnum.VE_FUNC_DIVIDE_ACCUMULATE,
    "nlcpy_logaddexp_accumulate": ve_funcnum.VE_FUNC_LOGADDEXP_ACCUMULATE,
    "nlcpy_logaddexp2_accumulate": ve_funcnum.VE_FUNC_LOGADDEXP2_ACCUMULATE,
    "nlcpy_true_divide_accumulate": ve_funcnum.VE_FUNC_TRUE_DIVIDE_ACCUMULATE,
    "nlcpy_floor_divide_accumulate": ve_funcnum.VE_FUNC_FLOOR_DIVIDE_ACCUMULATE,
    "nlcpy_power_accumulate": ve_funcnum.VE_FUNC_POWER_ACCUMULATE,
    "nlcpy_remainder_accumulate": ve_funcnum.VE_FUNC_REMAINDER_ACCUMULATE,
    "nlcpy_mod_accumulate": ve_funcnum.VE_FUNC_MOD_ACCUMULATE,
    "nlcpy_fmod_accumulate": ve_funcnum.VE_FUNC_FMOD_ACCUMULATE,
    "nlcpy_heaviside_accumulate": ve_funcnum.VE_FUNC_HEAVISIDE_ACCUMULATE,
    "nlcpy_bitwise_and_accumulate": ve_funcnum.VE_FUNC_BITWISE_AND_ACCUMULATE,
    "nlcpy_bitwise_or_accumulate": ve_funcnum.VE_FUNC_BITWISE_OR_ACCUMULATE,
    "nlcpy_bitwise_xor_accumulate": ve_funcnum.VE_FUNC_BITWISE_XOR_ACCUMULATE,
    # "nlcpy_invert_accumulate": ve_funcnum.VE_FUNC_INVERT_ACCUMULATE,
    "nlcpy_left_shift_accumulate": ve_funcnum.VE_FUNC_LEFT_SHIFT_ACCUMULATE,
    "nlcpy_right_shift_accumulate": ve_funcnum.VE_FUNC_RIGHT_SHIFT_ACCUMULATE,
    "nlcpy_greater_accumulate": ve_funcnum.VE_FUNC_GREATER_ACCUMULATE,
    "nlcpy_greater_equal_accumulate": ve_funcnum.VE_FUNC_GREATER_EQUAL_ACCUMULATE,
    "nlcpy_less_accumulate": ve_funcnum.VE_FUNC_LESS_ACCUMULATE,
    "nlcpy_less_equal_accumulate": ve_funcnum.VE_FUNC_LESS_EQUAL_ACCUMULATE,
    "nlcpy_not_equal_accumulate": ve_funcnum.VE_FUNC_NOT_EQUAL_ACCUMULATE,
    "nlcpy_equal_accumulate": ve_funcnum.VE_FUNC_EQUAL_ACCUMULATE,
    "nlcpy_logical_and_accumulate": ve_funcnum.VE_FUNC_LOGICAL_AND_ACCUMULATE,
    "nlcpy_logical_or_accumulate": ve_funcnum.VE_FUNC_LOGICAL_OR_ACCUMULATE,
    "nlcpy_logical_xor_accumulate": ve_funcnum.VE_FUNC_LOGICAL_XOR_ACCUMULATE,
    # "nlcpy_logical_not_accumulate": ve_funcnum.VE_FUNC_LOGICAL_NOT_ACCUMULATE,
    "nlcpy_maximum_accumulate": ve_funcnum.VE_FUNC_MAXIMUM_ACCUMULATE,
    "nlcpy_minimum_accumulate": ve_funcnum.VE_FUNC_MINIMUM_ACCUMULATE,
    "nlcpy_fmax_accumulate": ve_funcnum.VE_FUNC_FMAX_ACCUMULATE,
    "nlcpy_fmin_accumulate": ve_funcnum.VE_FUNC_FMIN_ACCUMULATE,
    "nlcpy_arctan2_accumulate": ve_funcnum.VE_FUNC_ARCTAN2_ACCUMULATE,
    "nlcpy_hypot_accumulate": ve_funcnum.VE_FUNC_HYPOT_ACCUMULATE,
    "nlcpy_copysign_accumulate": ve_funcnum.VE_FUNC_COPYSIGN_ACCUMULATE,
    "nlcpy_nextafter_accumulate": ve_funcnum.VE_FUNC_NEXTAFTER_ACCUMULATE,
    #  outer functions
    "nlcpy_add_outer": ve_funcnum.VE_FUNC_ADD_OUTER,
    "nlcpy_subtract_outer": ve_funcnum.VE_FUNC_SUBTRACT_OUTER,
    "nlcpy_multiply_outer": ve_funcnum.VE_FUNC_MULTIPLY_OUTER,
    "nlcpy_floor_divide_outer": ve_funcnum.VE_FUNC_FLOOR_DIVIDE_OUTER,
    "nlcpy_true_divide_outer": ve_funcnum.VE_FUNC_TRUE_DIVIDE_OUTER,
    "nlcpy_divide_outer": ve_funcnum.VE_FUNC_DIVIDE_OUTER,
    "nlcpy_power_outer": ve_funcnum.VE_FUNC_POWER_OUTER,
    "nlcpy_bitwise_and_outer": ve_funcnum.VE_FUNC_BITWISE_AND_OUTER,
    "nlcpy_bitwise_xor_outer": ve_funcnum.VE_FUNC_BITWISE_XOR_OUTER,
    "nlcpy_bitwise_or_outer": ve_funcnum.VE_FUNC_BITWISE_OR_OUTER,
    "nlcpy_logical_and_outer": ve_funcnum.VE_FUNC_LOGICAL_AND_OUTER,
    "nlcpy_logical_xor_outer": ve_funcnum.VE_FUNC_LOGICAL_XOR_OUTER,
    "nlcpy_logical_or_outer": ve_funcnum.VE_FUNC_LOGICAL_OR_OUTER,
    "nlcpy_right_shift_outer": ve_funcnum.VE_FUNC_RIGHT_SHIFT_OUTER,
    "nlcpy_left_shift_outer": ve_funcnum.VE_FUNC_LEFT_SHIFT_OUTER,
    "nlcpy_mod_outer": ve_funcnum.VE_FUNC_MOD_OUTER,
    "nlcpy_remainder_outer": ve_funcnum.VE_FUNC_REMAINDER_OUTER,
    "nlcpy_less_outer": ve_funcnum.VE_FUNC_LESS_OUTER,
    "nlcpy_greater_outer": ve_funcnum.VE_FUNC_GREATER_OUTER,
    "nlcpy_less_equal_outer": ve_funcnum.VE_FUNC_LESS_EQUAL_OUTER,
    "nlcpy_greater_equal_outer": ve_funcnum.VE_FUNC_GREATER_EQUAL_OUTER,
    "nlcpy_equal_outer": ve_funcnum.VE_FUNC_EQUAL_OUTER,
    "nlcpy_not_equal_outer": ve_funcnum.VE_FUNC_NOT_EQUAL_OUTER,
    "nlcpy_arctan2_outer": ve_funcnum.VE_FUNC_ARCTAN2_OUTER,
    "nlcpy_hypot_outer": ve_funcnum.VE_FUNC_HYPOT_OUTER,
    "nlcpy_logaddexp_outer": ve_funcnum.VE_FUNC_LOGADDEXP_OUTER,
    "nlcpy_logaddexp2_outer": ve_funcnum.VE_FUNC_LOGADDEXP2_OUTER,
    "nlcpy_heaviside_outer": ve_funcnum.VE_FUNC_HEAVISIDE_OUTER,
    "nlcpy_maximum_outer": ve_funcnum.VE_FUNC_MAXIMUM_OUTER,
    "nlcpy_minimum_outer": ve_funcnum.VE_FUNC_MINIMUM_OUTER,
    "nlcpy_copysign_outer": ve_funcnum.VE_FUNC_COPYSIGN_OUTER,
    "nlcpy_fmax_outer": ve_funcnum.VE_FUNC_FMAX_OUTER,
    "nlcpy_fmin_outer": ve_funcnum.VE_FUNC_FMIN_OUTER,
    "nlcpy_fmod_outer": ve_funcnum.VE_FUNC_FMOD_OUTER,
    "nlcpy_nextafter_outer": ve_funcnum.VE_FUNC_NEXTAFTER_OUTER,
    "nlcpy_ldexp_outer": ve_funcnum.VE_FUNC_LDEXP_OUTER,
    #  at functions
    "nlcpy_add_at": ve_funcnum.VE_FUNC_ADD_AT,
    #  searching functions
    "nlcpy_nonzero": ve_funcnum.VE_FUNC_NONZERO,
    "nlcpy_argmax": ve_funcnum.VE_FUNC_ARGMAX,
    "nlcpy_argmin": ve_funcnum.VE_FUNC_ARGMIN,
    "nlcpy_argwhere": ve_funcnum.VE_FUNC_ARGWHERE,
    #  sorting functions
    "nlcpy_sort": ve_funcnum.VE_FUNC_SORT,
    "nlcpy_argsort": ve_funcnum.VE_FUNC_ARGSORT,
    "nlcpy_sort_multi": ve_funcnum.VE_FUNC_SORT_MULTI,
    #  math functions
    "nlcpy_diff": ve_funcnum.VE_FUNC_DIFF,
    "nlcpy_clip": ve_funcnum.VE_FUNC_CLIP,
    #  random functions
    "nlcpy_random_shuffle": ve_funcnum.VE_FUNC_SHUFFLE,
    #  sca functions
    "nlcpy_sca_code_execute": ve_funcnum.VE_FUNC_SCA_EXECUTE,
    #  mask functions
    "nlcpy_domain_mask": ve_funcnum.VE_FUNC_DOMAIN_MASK,
}


cdef dict funcTypeList = {
    "binary_op": ve_functype.BINARY_OP,
    "unary_op": ve_functype.UNARY_OP,
    "indexing_op": ve_functype.INDEXING_OP,
    "creation_op": ve_functype.CREATION_OP,
    "manipulation_op": ve_functype.MANIPULATION_OP,
    "cblas_op": ve_functype.CBLAS_OP,
    "linalg_op": ve_functype.LINALG_OP,
    "reduce_op": ve_functype.REDUCE_OP,
    "reduceat_op": ve_functype.REDUCEAT_OP,
    "accumulate_op": ve_functype.ACCUMULATE_OP,
    "outer_op": ve_functype.OUTER_OP,
    "at_op": ve_functype.AT_OP,
    "searching_op": ve_functype.SEARCHING_OP,
    "sorting_op": ve_functype.SORTING_OP,
    "math_op": ve_functype.MATH_OP,
    "random_op": ve_functype.RANDOM_OP,
    "sca_op": ve_functype.SCA_OP,
    "mask_op": ve_functype.MASK_OP,
}


cpdef check_error(uint64_t err):
    if err == NLCPY_ERROR_OK:
        return
    elif err & NLCPY_ERROR_DTYPE:
        raise RuntimeError('invalid dtype was detected in VE kernel.')
    elif err & NLCPY_ERROR_NDIM:
        raise RuntimeError('invalid ndim was detected in VE kernel.')
    elif err & NLCPY_ERROR_MEMORY:
        raise RuntimeError('invalid ve_adr was detected in VE kernel.')
    elif err & NLCPY_ERROR_FUNCNUM:
        raise RuntimeError('invalid function number was detected in VE kernel.')
    elif err & NLCPY_ERROR_FUNCTYPE:
        raise RuntimeError('invalid function type was detected in VE kernel.')
    elif err & NLCPY_ERROR_INDEX:
        raise RuntimeError('invalid index was detected in VE kernel.')
    elif err & NLCPY_ERROR_ASL:
        raise RuntimeError('ASL error was detected in VE kernel.')
    elif err & NLCPY_ERROR_SCA:
        raise RuntimeError('SCA error was detected in VE kernel.')
    else:
        raise RuntimeError('unknown error was detected in VE kernel.')

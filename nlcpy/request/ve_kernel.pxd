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

from libc.stdint cimport *

cdef extern from "../ve_kernel/ve_funcnum.h":
    cdef enum ve_funcnum:
        # binary functions
        VE_FUNC_ADD
        VE_FUNC_SUBTRACT
        VE_FUNC_MULTIPLY
        VE_FUNC_DIVIDE
        VE_FUNC_LOGADDEXP
        VE_FUNC_LOGADDEXP2
        VE_FUNC_TRUE_DIVIDE
        VE_FUNC_FLOOR_DIVIDE
        VE_FUNC_POWER
        VE_FUNC_REMAINDER
        VE_FUNC_MOD
        VE_FUNC_FMOD
        VE_FUNC_DIVMOD
        VE_FUNC_HEAVISIDE
        VE_FUNC_GCD
        VE_FUNC_LCM
        VE_FUNC_BITWISE_AND
        VE_FUNC_BITWISE_OR
        VE_FUNC_BITWISE_XOR
        VE_FUNC_LEFT_SHIFT
        VE_FUNC_RIGHT_SHIFT
        VE_FUNC_GREATER
        VE_FUNC_GREATER_EQUAL
        VE_FUNC_LESS
        VE_FUNC_LESS_EQUAL
        VE_FUNC_NOT_EQUAL
        VE_FUNC_EQUAL
        VE_FUNC_LOGICAL_AND
        VE_FUNC_LOGICAL_OR
        VE_FUNC_LOGICAL_XOR
        VE_FUNC_MAXIMUM
        VE_FUNC_MINIMUM
        VE_FUNC_FMAX
        VE_FUNC_FMIN
        VE_FUNC_ARCTAN2
        VE_FUNC_HYPOT
        VE_FUNC_COPYSIGN
        VE_FUNC_NEXTAFTER
        VE_FUNC_MODF
        VE_FUNC_LDEXP
        VE_FUNC_FEXP

        # unary functions
        VE_FUNC_NEGATIVE
        VE_FUNC_POSITIVE
        VE_FUNC_ABSOLUTE
        VE_FUNC_FABS
        VE_FUNC_RINT
        VE_FUNC_SIGN
        VE_FUNC_CONJ
        VE_FUNC_CONJUGATE
        VE_FUNC_EXP
        VE_FUNC_EXP2
        VE_FUNC_LOG
        VE_FUNC_LOG2
        VE_FUNC_LOG10
        VE_FUNC_EXPM1
        VE_FUNC_LOG1P
        VE_FUNC_SQRT
        VE_FUNC_SQUARE
        VE_FUNC_CBRT
        VE_FUNC_RECIPROCAL
        VE_FUNC_SIN
        VE_FUNC_COS
        VE_FUNC_TAN
        VE_FUNC_ARCSIN
        VE_FUNC_ARCCOS
        VE_FUNC_ARCTAN
        VE_FUNC_SINH
        VE_FUNC_COSH
        VE_FUNC_TANH
        VE_FUNC_ARCSINH
        VE_FUNC_ARCCOSH
        VE_FUNC_ARCTANH
        VE_FUNC_DEG2RAD
        VE_FUNC_RAD2DEG
        VE_FUNC_DEGREES
        VE_FUNC_RADIANS
        VE_FUNC_INVERT
        VE_FUNC_LOGICAL_NOT
        VE_FUNC_ISFINITE
        VE_FUNC_ISINF
        VE_FUNC_ISNAN
        VE_FUNC_SIGNBIT
        VE_FUNC_SPACING
        VE_FUNC_FLOOR
        VE_FUNC_CEIL
        VE_FUNC_TRUNC
        VE_FUNC_ANGLE
        VE_FUNC_ERF
        VE_FUNC_ERFC

        # indexing functions
        VE_FUNC_GETITEM_FROM_MASK
        VE_FUNC_SETITEM_FROM_MASK
        VE_FUNC_TAKE
        VE_FUNC_PREPARE_INDEXING
        VE_FUNC_SCATTER_UPDATE
        VE_FUNC_WHERE
        VE_FUNC_FILL_DIAGONAL

        # creation functions
        VE_FUNC_ARANGE
        VE_FUNC_COPY
        VE_FUNC_EYE
        VE_FUNC_LINSPACE
        VE_FUNC_COPY_MASKED
        VE_FUNC_TRI

        # manipulation functions
        VE_FUNC_DELETE
        VE_FUNC_TILE
        VE_FUNC_REPEAT
        VE_FUNC_INSERT
        VE_FUNC_ROLL
        VE_FUNC_BLOCK

        # cblas wrapper functions
        VE_FUNC_CBLAS_SDOT
        VE_FUNC_CBLAS_DDOT
        VE_FUNC_CBLAS_CDOTU_SUB
        VE_FUNC_CBLAS_ZDOTU_SUB
        VE_FUNC_CBLAS_SGEMM
        VE_FUNC_CBLAS_DGEMM
        VE_FUNC_CBLAS_CGEMM
        VE_FUNC_CBLAS_ZGEMM

        # linalg functions
        VE_FUNC_DOT
        VE_FUNC_MATMUL

        # reduce functions
        VE_FUNC_ADD_REDUCE
        VE_FUNC_SUBTRACT_REDUCE
        VE_FUNC_MULTIPLY_REDUCE
        VE_FUNC_FLOOR_DIVIDE_REDUCE
        VE_FUNC_TRUE_DIVIDE_REDUCE
        VE_FUNC_DIVIDE_REDUCE
        VE_FUNC_MOD_REDUCE
        VE_FUNC_REMAINDER_REDUCE
        VE_FUNC_POWER_REDUCE
        VE_FUNC_BITWISE_AND_REDUCE
        VE_FUNC_BITWISE_XOR_REDUCE
        VE_FUNC_BITWSE_OR_REDUCE
        VE_FUNC_LOGICAL_AND_REDUCE
        VE_FUNC_LOGICAL_XOR_REDUCE
        VE_FUNC_LOGICAL_OR_REDUCE
        VE_FUNC_RIGHT_SHIFT_REDUCE
        VE_FUNC_LEFT_SHIFT_REDUCE
        VE_FUNC_LESS_REDUCE
        VE_FUNC_GREATER_REDUCE
        VE_FUNC_LESS_EQUAL_REDUCE
        VE_FUNC_GREATER_EQUAL_REDUCE
        VE_FUNC_EQUAL_REDUCE
        VE_FUNC_NOT_EQUAL_REDUCE
        VE_FUNC_ARCTAN2_REDUCE
        VE_FUNC_HYPOT_REDUCE
        VE_FUNC_LOGADDEXP_REDUCE
        VE_FUNC_LOGADDEXP2_REDUCE
        VE_FUNC_HEAVISIDE_REDUCE
        VE_FUNC_MAXIMUM_REDUCE
        VE_FUNC_MINIMUM_REDUCE
        VE_FUNC_COPYSIGN_REDUCE
        VE_FUNC_FMAX_REDUCE
        VE_FUNC_FMIN_REDUCE
        VE_FUNC_FMOD_REDUCE
        VE_FUNC_NEXTAFTER_REDUCE
        # reduceat functions
        VE_FUNC_ADD_REDUCEAT
        # accumulate functions
        VE_FUNC_ADD_ACCUMULATE
        VE_FUNC_SUBTRACT_ACCUMULATE
        VE_FUNC_MULTIPLY_ACCUMULATE
        VE_FUNC_DIVIDE_ACCUMULATE
        VE_FUNC_LOGADDEXP_ACCUMULATE
        VE_FUNC_LOGADDEXP2_ACCUMULATE
        VE_FUNC_TRUE_DIVIDE_ACCUMULATE
        VE_FUNC_FLOOR_DIVIDE_ACCUMULATE
        VE_FUNC_POWER_ACCUMULATE
        VE_FUNC_REMAINDER_ACCUMULATE
        VE_FUNC_MOD_ACCUMULATE
        VE_FUNC_FMOD_ACCUMULATE
        VE_FUNC_HEAVISIDE_ACCUMULATE
        VE_FUNC_BITWISE_AND_ACCUMULATE
        VE_FUNC_BITWISE_OR_ACCUMULATE
        VE_FUNC_BITWISE_XOR_ACCUMULATE
#         VE_FUNC_INVERT_ACCUMULATE
        VE_FUNC_LEFT_SHIFT_ACCUMULATE
        VE_FUNC_RIGHT_SHIFT_ACCUMULATE
        VE_FUNC_GREATER_ACCUMULATE
        VE_FUNC_GREATER_EQUAL_ACCUMULATE
        VE_FUNC_LESS_ACCUMULATE
        VE_FUNC_LESS_EQUAL_ACCUMULATE
        VE_FUNC_NOT_EQUAL_ACCUMULATE
        VE_FUNC_EQUAL_ACCUMULATE
        VE_FUNC_LOGICAL_AND_ACCUMULATE
        VE_FUNC_LOGICAL_OR_ACCUMULATE
        VE_FUNC_LOGICAL_XOR_ACCUMULATE
#         VE_FUNC_LOGICAL_NOT_ACCUMULATE
        VE_FUNC_MAXIMUM_ACCUMULATE
        VE_FUNC_MINIMUM_ACCUMULATE
        VE_FUNC_FMAX_ACCUMULATE
        VE_FUNC_FMIN_ACCUMULATE
        VE_FUNC_ARCTAN2_ACCUMULATE
        VE_FUNC_HYPOT_ACCUMULATE
        VE_FUNC_COPYSIGN_ACCUMULATE
        VE_FUNC_NEXTAFTER_ACCUMULATE
        # outer functions
        VE_FUNC_ADD_OUTER
        VE_FUNC_SUBTRACT_OUTER
        VE_FUNC_MULTIPLY_OUTER
        VE_FUNC_FLOOR_DIVIDE_OUTER
        VE_FUNC_TRUE_DIVIDE_OUTER
        VE_FUNC_DIVIDE_OUTER
        VE_FUNC_POWER_OUTER
        VE_FUNC_BITWISE_AND_OUTER
        VE_FUNC_BITWISE_XOR_OUTER
        VE_FUNC_BITWISE_OR_OUTER
        VE_FUNC_LOGICAL_AND_OUTER
        VE_FUNC_LOGICAL_XOR_OUTER
        VE_FUNC_LOGICAL_OR_OUTER
        VE_FUNC_RIGHT_SHIFT_OUTER
        VE_FUNC_LEFT_SHIFT_OUTER
        VE_FUNC_MOD_OUTER
        VE_FUNC_REMAINDER_OUTER
        VE_FUNC_LESS_OUTER
        VE_FUNC_GREATER_OUTER
        VE_FUNC_LESS_EQUAL_OUTER
        VE_FUNC_GREATER_EQUAL_OUTER
        VE_FUNC_EQUAL_OUTER
        VE_FUNC_NOT_EQUAL_OUTER
        VE_FUNC_ARCTAN2_OUTER
        VE_FUNC_HYPOT_OUTER
        VE_FUNC_LOGADDEXP_OUTER
        VE_FUNC_LOGADDEXP2_OUTER
        VE_FUNC_HEAVISIDE_OUTER
        VE_FUNC_MAXIMUM_OUTER
        VE_FUNC_MINIMUM_OUTER
        VE_FUNC_COPYSIGN_OUTER
        VE_FUNC_FMAX_OUTER
        VE_FUNC_FMIN_OUTER
        VE_FUNC_FMOD_OUTER
        VE_FUNC_NEXTAFTER_OUTER
        VE_FUNC_LDEXP_OUTER
        # at functions
        VE_FUNC_ADD_AT

        # searching functions
        VE_FUNC_NONZERO
        VE_FUNC_ARGMAX
        VE_FUNC_ARGMIN
        VE_FUNC_ARGWHERE

        # sorting functions
        VE_FUNC_SORT
        VE_FUNC_ARGSORT
        VE_FUNC_SORT_MULTI

        # math functions
        VE_FUNC_DIFF
        VE_FUNC_CLIP

        # random functions
        VE_FUNC_SHUFFLE

        # sca functions
        VE_FUNC_SCA_EXECUTE

        # mask functions
        VE_FUNC_DOMAIN_MASK

cdef dict funcNumList


cdef extern from "../ve_kernel/ve_functype.h":
    cdef enum ve_functype:
        BINARY_OP
        UNARY_OP
        INDEXING_OP
        CREATION_OP
        MANIPULATION_OP
        CBLAS_OP
        LINALG_OP
        REDUCE_OP
        REDUCEAT_OP
        ACCUMULATE_OP
        OUTER_OP
        AT_OP
        SEARCHING_OP
        SORTING_OP
        MATH_OP
        RANDOM_OP
        SCA_OP
        MASK_OP


cdef dict funcTypeList


cdef extern from "../ve_kernel/ve_array.h":
    cdef int NLCPY_MAXNDIM
    cdef int SIZEOF_VE_ARRAY
    cdef int N_VE_ARRAY_ELEMENTS
    cdef int VE_ADR_OFFSET
    cdef int NDIM_OFFSET
    cdef int SIZE_OFFSET
    cdef int SHAPE_OFFSET
    cdef int STRIDES_OFFSET
    cdef int DTYPE_OFFSET
    cdef int ITEMSIZE_OFFSET
    cdef int C_CONTIGUOUS_OFFSET
    cdef int F_CONTIGUOUS_OFFSET
    cdef int SCALAR_OFFSET = F_CONTIGUOUS_OFFSET

cdef extern from "../ve_kernel/ve_request.h":
    cdef uint64_t SIZEOF_REQUEST_PACKAGE
    cdef uint64_t N_REQUEST_PACKAGE

cdef extern from "../ve_kernel/ve_error.h":
    cdef uint64_t NLCPY_ERROR_OK
    cdef uint64_t NLCPY_ERROR_NDIM
    cdef uint64_t NLCPY_ERROR_DTYPE
    cdef uint64_t NLCPY_ERROR_MEMORY
    cdef uint64_t NLCPY_ERROR_FUNCNUM
    cdef uint64_t NLCPY_ERROR_FUNCTYPE
    cdef uint64_t NLCPY_ERROR_INDEX
    cdef uint64_t NLCPY_ERROR_ASL
    cdef uint64_t NLCPY_ERROR_SCA

cpdef check_error(uint64_t err)

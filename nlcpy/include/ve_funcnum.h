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
#ifndef VE_FUNCNUM_H_INCLUDED
#define VE_FUNCNUM_H_INCLUDED

enum ve_funcnum {
    /* binary functions */
    VE_FUNC_ADD              = 0x00000,
    VE_FUNC_SUBTRACT         = 0x00001,
    VE_FUNC_MULTIPLY         = 0x00002,
    VE_FUNC_DIVIDE           = 0x00003,
    VE_FUNC_LOGADDEXP        = 0x00004,
    VE_FUNC_LOGADDEXP2       = 0x00005,
    VE_FUNC_TRUE_DIVIDE      = 0x00006,
    VE_FUNC_FLOOR_DIVIDE     = 0x00007,
    VE_FUNC_POWER            = 0x00008,
    VE_FUNC_REMAINDER        = 0x00009,
    VE_FUNC_MOD              = 0x0000a,
    VE_FUNC_FMOD             = 0x0000b,
    VE_FUNC_DIVMOD           = 0x0000c,
    VE_FUNC_HEAVISIDE        = 0x0000d,
    VE_FUNC_GCD              = 0x0000e,
    VE_FUNC_LCM              = 0x0000f,
    VE_FUNC_BITWISE_AND      = 0x00010,
    VE_FUNC_BITWISE_OR       = 0x00011,
    VE_FUNC_BITWISE_XOR      = 0x00012,
    VE_FUNC_LEFT_SHIFT       = 0x00013,
    VE_FUNC_RIGHT_SHIFT      = 0x00014,
    VE_FUNC_GREATER          = 0x00015,
    VE_FUNC_GREATER_EQUAL    = 0x00016,
    VE_FUNC_LESS             = 0x00017,
    VE_FUNC_LESS_EQUAL       = 0x00018,
    VE_FUNC_NOT_EQUAL        = 0x00019,
    VE_FUNC_EQUAL            = 0x0001a,
    VE_FUNC_LOGICAL_AND      = 0x0001b,
    VE_FUNC_LOGICAL_OR       = 0x0001c,
    VE_FUNC_LOGICAL_XOR      = 0x0001d,
    VE_FUNC_MAXIMUM          = 0x0001e,
    VE_FUNC_MINIMUM          = 0x0001f,
    VE_FUNC_FMAX             = 0x00020,
    VE_FUNC_FMIN             = 0x00021,
    VE_FUNC_ARCTAN2          = 0x00022,
    VE_FUNC_HYPOT            = 0x00023,
    VE_FUNC_COPYSIGN         = 0x00024,
    VE_FUNC_NEXTAFTER        = 0x00025,
    VE_FUNC_MODF             = 0x00026,
    VE_FUNC_LDEXP            = 0x00027,
    VE_FUNC_FEXP             = 0x00028,
    /* unary functions */
    VE_FUNC_NEGATIVE         = 0x01000,
    VE_FUNC_POSITIVE         = 0x01001,
    VE_FUNC_ABSOLUTE         = 0x01002,
    VE_FUNC_FABS             = 0x01003,
    VE_FUNC_RINT             = 0x01004,
    VE_FUNC_SIGN             = 0x01005,
    VE_FUNC_CONJ             = 0x01006,
    VE_FUNC_CONJUGATE        = 0x01007,
    VE_FUNC_EXP              = 0x01008,
    VE_FUNC_EXP2             = 0x01009,
    VE_FUNC_LOG              = 0x0100a,
    VE_FUNC_LOG2             = 0x0100b,
    VE_FUNC_LOG10            = 0x0100c,
    VE_FUNC_EXPM1            = 0x0100d,
    VE_FUNC_LOG1P            = 0x0100e,
    VE_FUNC_SQRT             = 0x0100f,
    VE_FUNC_SQUARE           = 0x01010,
    VE_FUNC_CBRT             = 0x01011,
    VE_FUNC_RECIPROCAL       = 0x01012,
    VE_FUNC_SIN              = 0x01013,
    VE_FUNC_COS              = 0x01014,
    VE_FUNC_TAN              = 0x01015,
    VE_FUNC_ARCSIN           = 0x01016,
    VE_FUNC_ARCCOS           = 0x01017,
    VE_FUNC_ARCTAN           = 0x01018,
    VE_FUNC_SINH             = 0x0101b,
    VE_FUNC_COSH             = 0x0101c,
    VE_FUNC_TANH             = 0x0101d,
    VE_FUNC_ARCSINH          = 0x0101e,
    VE_FUNC_ARCCOSH          = 0x0101f,
    VE_FUNC_ARCTANH          = 0x01020,
    VE_FUNC_DEG2RAD          = 0x01021,
    VE_FUNC_RAD2DEG          = 0x01022,
    VE_FUNC_DEGREES          = 0x01023,
    VE_FUNC_RADIANS          = 0x01024,
    VE_FUNC_INVERT           = 0x01025,
    VE_FUNC_LOGICAL_NOT      = 0x01026,
    VE_FUNC_ISFINITE         = 0x01029,
    VE_FUNC_ISINF            = 0x0102a,
    VE_FUNC_ISNAN            = 0x0102b,
    VE_FUNC_SIGNBIT          = 0x0102c,
    VE_FUNC_SPACING          = 0x0102d,
    VE_FUNC_FLOOR            = 0x0102e,
    VE_FUNC_CEIL             = 0x0102f,
    VE_FUNC_TRUNC            = 0x01030,
    VE_FUNC_ANGLE            = 0x01033,
    VE_FUNC_ERF              = 0x01034,
    VE_FUNC_ERFC             = 0x01035,
    /* indexing functions */
    VE_FUNC_GETITEM_FROM_MASK = 0x02000,
    VE_FUNC_SETITEM_FROM_MASK = 0x02001,
    VE_FUNC_TAKE              = 0x02002,
    VE_FUNC_PREPARE_INDEXING  = 0x02003,
    VE_FUNC_SCATTER_UPDATE    = 0x02004,
    VE_FUNC_WHERE             = 0x02005,
    VE_FUNC_FILL_DIAGONAL     = 0x02006,
    /* creation functions */
    VE_FUNC_ARANGE           = 0x04000,
    VE_FUNC_COPY             = 0x04001,
    VE_FUNC_EYE              = 0x04002,
    VE_FUNC_LINSPACE         = 0x04003,
    VE_FUNC_COPY_MASKED      = 0x04004,
    VE_FUNC_TRI              = 0x04005,
    /* manipulation functions */
    VE_FUNC_TILE             = 0x05000,
    VE_FUNC_DELETE           = 0x05001,
    VE_FUNC_INSERT           = 0x05002,
    VE_FUNC_REPEAT           = 0x05003,
    VE_FUNC_ROLL             = 0x05004,
    VE_FUNC_BLOCK            = 0x05005,
    /* linalg functions */
    VE_FUNC_DOT              = 0x07000,
    VE_FUNC_MATMUL           = 0x07001,
    VE_FUNC_SIMPLE_FNORM     = 0x07002,
    VE_FUNC_FNORM            = 0x07003,
    /* reduce functions */
    VE_FUNC_ADD_REDUCE            = 0x08000,
    VE_FUNC_SUBTRACT_REDUCE       = 0x08001,
    VE_FUNC_MULTIPLY_REDUCE       = 0x08002,
    VE_FUNC_FLOOR_DIVIDE_REDUCE   = 0x08003,
    VE_FUNC_TRUE_DIVIDE_REDUCE    = 0x08004,
    VE_FUNC_DIVIDE_REDUCE         = 0x08005,
    VE_FUNC_MOD_REDUCE            = 0x08006,
    VE_FUNC_REMAINDER_REDUCE      = 0x08007,
    VE_FUNC_POWER_REDUCE          = 0x08008,
    VE_FUNC_BITWISE_AND_REDUCE    = 0x08009,
    VE_FUNC_BITWISE_XOR_REDUCE    = 0x0800a,
    VE_FUNC_BITWSE_OR_REDUCE      = 0x0800b,
    VE_FUNC_LOGICAL_AND_REDUCE    = 0x0800c,
    VE_FUNC_LOGICAL_XOR_REDUCE    = 0x0800d,
    VE_FUNC_LOGICAL_OR_REDUCE     = 0x0800e,
    VE_FUNC_RIGHT_SHIFT_REDUCE    = 0x0800f,
    VE_FUNC_LEFT_SHIFT_REDUCE     = 0x08010,
    VE_FUNC_LESS_REDUCE           = 0x08011,
    VE_FUNC_GREATER_REDUCE        = 0x08012,
    VE_FUNC_LESS_EQUAL_REDUCE     = 0x08013,
    VE_FUNC_GREATER_EQUAL_REDUCE  = 0x08014,
    VE_FUNC_EQUAL_REDUCE          = 0x08015,
    VE_FUNC_NOT_EQUAL_REDUCE      = 0x08016,
    VE_FUNC_ARCTAN2_REDUCE        = 0x08017,
    VE_FUNC_HYPOT_REDUCE          = 0x08018,
    VE_FUNC_LOGADDEXP_REDUCE      = 0x08019,
    VE_FUNC_LOGADDEXP2_REDUCE     = 0x0801a,
    VE_FUNC_HEAVISIDE_REDUCE      = 0x0801b,
    VE_FUNC_MAXIMUM_REDUCE        = 0x0801c,
    VE_FUNC_MINIMUM_REDUCE        = 0x0801d,
    VE_FUNC_COPYSIGN_REDUCE       = 0x0801e,
    VE_FUNC_FMAX_REDUCE           = 0x0801f,
    VE_FUNC_FMIN_REDUCE           = 0x08020,
    VE_FUNC_FMOD_REDUCE           = 0x08021,
    VE_FUNC_NEXTAFTER_REDUCE      = 0x08022,
    /* reduceat functions */
    VE_FUNC_ADD_REDUCEAT          = 0x09000,
    /* accumulate functions */
    VE_FUNC_ADD_ACCUMULATE           = 0x0a000,
    VE_FUNC_SUBTRACT_ACCUMULATE      = 0x0a001,
    VE_FUNC_MULTIPLY_ACCUMULATE      = 0x0a002,
    VE_FUNC_DIVIDE_ACCUMULATE        = 0x0a003,
    VE_FUNC_LOGADDEXP_ACCUMULATE     = 0x0a004,
    VE_FUNC_LOGADDEXP2_ACCUMULATE    = 0x0a005,
    VE_FUNC_TRUE_DIVIDE_ACCUMULATE   = 0x0a006,
    VE_FUNC_FLOOR_DIVIDE_ACCUMULATE  = 0x0a007,
    VE_FUNC_POWER_ACCUMULATE         = 0x0a008,
    VE_FUNC_REMAINDER_ACCUMULATE     = 0x0a009,
    VE_FUNC_MOD_ACCUMULATE           = 0x0a00a,
    VE_FUNC_FMOD_ACCUMULATE          = 0x0a00b,
    VE_FUNC_HEAVISIDE_ACCUMULATE     = 0x0a00c,
    VE_FUNC_BITWISE_AND_ACCUMULATE   = 0x0a00d,
    VE_FUNC_BITWISE_OR_ACCUMULATE    = 0x0a00e,
    VE_FUNC_BITWISE_XOR_ACCUMULATE   = 0x0a00f,
//    VE_FUNC_INVERT_ACCUMULATE        = 0x0a010,
    VE_FUNC_LEFT_SHIFT_ACCUMULATE    = 0x0a011,
    VE_FUNC_RIGHT_SHIFT_ACCUMULATE   = 0x0a012,
    VE_FUNC_GREATER_ACCUMULATE       = 0x0a013,
    VE_FUNC_GREATER_EQUAL_ACCUMULATE = 0x0a014,
    VE_FUNC_LESS_ACCUMULATE          = 0x0a015,
    VE_FUNC_LESS_EQUAL_ACCUMULATE    = 0x0a016,
    VE_FUNC_NOT_EQUAL_ACCUMULATE     = 0x0a017,
    VE_FUNC_EQUAL_ACCUMULATE         = 0x0a018,
    VE_FUNC_LOGICAL_AND_ACCUMULATE   = 0x0a019,
    VE_FUNC_LOGICAL_OR_ACCUMULATE    = 0x0a01a,
    VE_FUNC_LOGICAL_XOR_ACCUMULATE   = 0x0a01b,
//    VE_FUNC_LOGICAL_NOT_ACCUMULATE   = 0x0a01c,
    VE_FUNC_MAXIMUM_ACCUMULATE       = 0x0a01d,
    VE_FUNC_MINIMUM_ACCUMULATE       = 0x0a01e,
    VE_FUNC_FMAX_ACCUMULATE          = 0x0a01f,
    VE_FUNC_FMIN_ACCUMULATE          = 0x0a020,
    VE_FUNC_ARCTAN2_ACCUMULATE       = 0x0a021,
    VE_FUNC_HYPOT_ACCUMULATE         = 0x0a022,
    VE_FUNC_COPYSIGN_ACCUMULATE      = 0x0a023,
    VE_FUNC_NEXTAFTER_ACCUMULATE     = 0x0a024,
    /* outer functions */
    VE_FUNC_ADD_OUTER             = 0x0b000,
    VE_FUNC_SUBTRACT_OUTER        = 0x0b001,
    VE_FUNC_MULTIPLY_OUTER        = 0x0b002,
    VE_FUNC_FLOOR_DIVIDE_OUTER    = 0x0b003,
    VE_FUNC_TRUE_DIVIDE_OUTER     = 0x0b004,
    VE_FUNC_DIVIDE_OUTER          = 0x0b005,
    VE_FUNC_POWER_OUTER           = 0x0b006,
    VE_FUNC_BITWISE_AND_OUTER     = 0x0b007,
    VE_FUNC_BITWISE_XOR_OUTER     = 0x0b008,
    VE_FUNC_BITWISE_OR_OUTER      = 0x0b009,
    VE_FUNC_LOGICAL_AND_OUTER     = 0x0b00a,
    VE_FUNC_LOGICAL_XOR_OUTER     = 0x0b00b,
    VE_FUNC_LOGICAL_OR_OUTER      = 0x0b00c,
    VE_FUNC_RIGHT_SHIFT_OUTER     = 0x0b00d,
    VE_FUNC_LEFT_SHIFT_OUTER      = 0x0b00e,
    VE_FUNC_MOD_OUTER             = 0x0b00f,
    VE_FUNC_REMAINDER_OUTER       = 0x0b010,
    VE_FUNC_LESS_OUTER            = 0x0b011,
    VE_FUNC_GREATER_OUTER         = 0x0b012,
    VE_FUNC_LESS_EQUAL_OUTER      = 0x0b013,
    VE_FUNC_GREATER_EQUAL_OUTER   = 0x0b014,
    VE_FUNC_EQUAL_OUTER           = 0x0b015,
    VE_FUNC_NOT_EQUAL_OUTER       = 0x0b016,
    VE_FUNC_ARCTAN2_OUTER         = 0x0b017,
    VE_FUNC_HYPOT_OUTER           = 0x0b018,
    VE_FUNC_LOGADDEXP_OUTER       = 0x0b019,
    VE_FUNC_LOGADDEXP2_OUTER      = 0x0b01a,
    VE_FUNC_HEAVISIDE_OUTER       = 0x0b01b,
    VE_FUNC_MAXIMUM_OUTER         = 0x0b01c,
    VE_FUNC_MINIMUM_OUTER         = 0x0b01d,
    VE_FUNC_COPYSIGN_OUTER        = 0x0b01e,
    VE_FUNC_FMAX_OUTER            = 0x0b01f,
    VE_FUNC_FMIN_OUTER            = 0x0b020,
    VE_FUNC_FMOD_OUTER            = 0x0b021,
    VE_FUNC_NEXTAFTER_OUTER       = 0x0b022,
    VE_FUNC_LDEXP_OUTER           = 0x0b023,

    /* at functions */
    VE_FUNC_ADD_AT                = 0x0c000,

    /* searching functions */
    VE_FUNC_NONZERO          = 0x0d000,
    VE_FUNC_ARGMAX           = 0x0d001,
    VE_FUNC_ARGMIN           = 0x0d002,
    VE_FUNC_ARGWHERE         = 0x0d003,

    /* sorting functions */
    VE_FUNC_SORT             = 0x0e000,
    VE_FUNC_ARGSORT          = 0x0e001,
    VE_FUNC_SORT_MULTI       = 0x0e002,

    /* math functions */
    VE_FUNC_DIFF             = 0x0f000,
    VE_FUNC_CLIP             = 0x0f001,

    /* random functions */
    VE_FUNC_SHUFFLE          = 0x10000,

    /* sca functions */
    VE_FUNC_SCA_EXECUTE      = 0x20000,

    /* mask functions */
    VE_FUNC_DOMAIN_MASK      = 0x30000,

};


#endif /* VE_FUNCNUM_H_INCLUDED */

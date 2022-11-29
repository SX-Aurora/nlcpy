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


import numpy
from nlcpy.ufuncs import ufuncs
from nlcpy.ufuncs import casting
from nlcpy.ufuncs import err
from nlcpy.ufuncs import ufunc_docs


# ----------------------------------------------------------------------------
# ufunc operations
# see: https://docs.scipy.org/doc/numpy/reference/ufuncs.html
# ----------------------------------------------------------------------------

# math_operations
add = ufuncs.create_ufunc(
    'nlcpy_add',
    casting._add_types,
    err._add_error_check,
    doc=ufunc_docs._add_doc
)


subtract = ufuncs.create_ufunc(
    'nlcpy_subtract',
    casting._subtract_types,
    err._subtract_error_check,
    doc=ufunc_docs._subtract_doc
)


multiply = ufuncs.create_ufunc(
    'nlcpy_multiply',
    casting._multiply_types,
    err._multiply_error_check,
    doc=ufunc_docs._multiply_doc
)


true_divide = ufuncs.create_ufunc(
    'nlcpy_true_divide',
    casting._true_divide_types,
    err._true_divide_error_check,
    doc=ufunc_docs._true_divide_doc
)


# ufunc_operation(divide,orig_types,valid_error_check)dnl
divide = true_divide

logaddexp = ufuncs.create_ufunc(
    'nlcpy_logaddexp',
    casting._logaddexp_types,
    err._logaddexp_error_check,
    doc=ufunc_docs._logaddexp_doc
)


logaddexp2 = ufuncs.create_ufunc(
    'nlcpy_logaddexp2',
    casting._logaddexp2_types,
    err._logaddexp2_error_check,
    doc=ufunc_docs._logaddexp2_doc
)


floor_divide = ufuncs.create_ufunc(
    'nlcpy_floor_divide',
    casting._floor_divide_types,
    err._floor_divide_error_check,
    doc=ufunc_docs._floor_divide_doc
)


negative = ufuncs.create_ufunc(
    'nlcpy_negative',
    casting._negative_types,
    err._negative_error_check,
    doc=ufunc_docs._negative_doc
)


positive = ufuncs.create_ufunc(
    'nlcpy_positive',
    casting._positive_types,
    err._positive_error_check,
    doc=ufunc_docs._positive_doc
)


power = ufuncs.create_ufunc(
    'nlcpy_power',
    casting._power_types,
    err._power_error_check,
    doc=ufunc_docs._power_doc
)


remainder = ufuncs.create_ufunc(
    'nlcpy_remainder',
    casting._remainder_types,
    err._remainder_error_check,
    doc=ufunc_docs._remainder_doc
)


mod = remainder

fmod = ufuncs.create_ufunc(
    'nlcpy_fmod',
    casting._fmod_types,
    err._fmod_error_check,
    doc=ufunc_docs._fmod_doc
)


absolute = ufuncs.create_ufunc(
    'nlcpy_absolute',
    casting._absolute_types,
    err._absolute_error_check,
    doc=ufunc_docs._absolute_doc
)


abs = absolute

fabs = ufuncs.create_ufunc(
    'nlcpy_fabs',
    casting._fabs_types,
    err._fabs_error_check,
    doc=ufunc_docs._fabs_doc
)


rint = ufuncs.create_ufunc(
    'nlcpy_rint',
    casting._rint_types,
    err._rint_error_check,
    doc=ufunc_docs._rint_doc
)


sign = ufuncs.create_ufunc(
    'nlcpy_sign',
    casting._sign_types,
    err._sign_error_check,
    doc=ufunc_docs._sign_doc
)


heaviside = ufuncs.create_ufunc(
    'nlcpy_heaviside',
    casting._heaviside_types,
    err._heaviside_error_check,
    doc=ufunc_docs._heaviside_doc
)


conjugate = ufuncs.create_ufunc(
    'nlcpy_conjugate',
    casting._conjugate_types,
    err._conjugate_error_check,
    doc=ufunc_docs._conjugate_doc
)


conj = conjugate

exp = ufuncs.create_ufunc(
    'nlcpy_exp',
    casting._exp_types,
    err._exp_error_check,
    doc=ufunc_docs._exp_doc
)


exp2 = ufuncs.create_ufunc(
    'nlcpy_exp2',
    casting._exp2_types,
    err._exp2_error_check,
    doc=ufunc_docs._exp2_doc
)


log = ufuncs.create_ufunc(
    'nlcpy_log',
    casting._log_types,
    err._log_error_check,
    doc=ufunc_docs._log_doc
)


log2 = ufuncs.create_ufunc(
    'nlcpy_log2',
    casting._log2_types,
    err._log2_error_check,
    doc=ufunc_docs._log2_doc
)


log10 = ufuncs.create_ufunc(
    'nlcpy_log10',
    casting._log10_types,
    err._log10_error_check,
    doc=ufunc_docs._log10_doc
)


expm1 = ufuncs.create_ufunc(
    'nlcpy_expm1',
    casting._expm1_types,
    err._expm1_error_check,
    doc=ufunc_docs._expm1_doc
)


log1p = ufuncs.create_ufunc(
    'nlcpy_log1p',
    casting._log1p_types,
    err._log1p_error_check,
    doc=ufunc_docs._log1p_doc
)


sqrt = ufuncs.create_ufunc(
    'nlcpy_sqrt',
    casting._sqrt_types,
    err._sqrt_error_check,
    doc=ufunc_docs._sqrt_doc
)


square = ufuncs.create_ufunc(
    'nlcpy_square',
    casting._square_types,
    err._square_error_check,
    doc=ufunc_docs._square_doc
)


cbrt = ufuncs.create_ufunc(
    'nlcpy_cbrt',
    casting._cbrt_types,
    err._cbrt_error_check,
    doc=ufunc_docs._cbrt_doc
)


reciprocal = ufuncs.create_ufunc(
    'nlcpy_reciprocal',
    casting._reciprocal_types,
    err._reciprocal_error_check,
    doc=ufunc_docs._reciprocal_doc
)


# ufunc_operation(gcd)dnl
# ufunc_operation(lcm)dnl
# bit-twiddling functions
bitwise_and = ufuncs.create_ufunc(
    'nlcpy_bitwise_and',
    casting._bitwise_and_types,
    err._bitwise_and_error_check,
    doc=ufunc_docs._bitwise_and_doc
)


bitwise_or = ufuncs.create_ufunc(
    'nlcpy_bitwise_or',
    casting._bitwise_or_types,
    err._bitwise_or_error_check,
    doc=ufunc_docs._bitwise_or_doc
)


bitwise_xor = ufuncs.create_ufunc(
    'nlcpy_bitwise_xor',
    casting._bitwise_xor_types,
    err._bitwise_xor_error_check,
    doc=ufunc_docs._bitwise_xor_doc
)


invert = ufuncs.create_ufunc(
    'nlcpy_invert',
    casting._invert_types,
    err._invert_error_check,
    doc=ufunc_docs._invert_doc
)


bitwise_not = invert

left_shift = ufuncs.create_ufunc(
    'nlcpy_left_shift',
    casting._left_shift_types,
    err._left_shift_error_check,
    doc=ufunc_docs._left_shift_doc
)


right_shift = ufuncs.create_ufunc(
    'nlcpy_right_shift',
    casting._right_shift_types,
    err._right_shift_error_check,
    doc=ufunc_docs._right_shift_doc
)


# comparison functions
greater = ufuncs.create_ufunc(
    'nlcpy_greater',
    casting._greater_types,
    err._greater_error_check,
    doc=ufunc_docs._greater_doc
)


greater_equal = ufuncs.create_ufunc(
    'nlcpy_greater_equal',
    casting._greater_equal_types,
    err._greater_equal_error_check,
    doc=ufunc_docs._greater_equal_doc
)


less = ufuncs.create_ufunc(
    'nlcpy_less',
    casting._less_types,
    err._less_error_check,
    doc=ufunc_docs._less_doc
)


less_equal = ufuncs.create_ufunc(
    'nlcpy_less_equal',
    casting._less_equal_types,
    err._less_equal_error_check,
    doc=ufunc_docs._less_equal_doc
)


not_equal = ufuncs.create_ufunc(
    'nlcpy_not_equal',
    casting._not_equal_types,
    err._not_equal_error_check,
    doc=ufunc_docs._not_equal_doc
)


equal = ufuncs.create_ufunc(
    'nlcpy_equal',
    casting._equal_types,
    err._equal_error_check,
    doc=ufunc_docs._equal_doc
)


logical_and = ufuncs.create_ufunc(
    'nlcpy_logical_and',
    casting._logical_and_types,
    err._logical_and_error_check,
    doc=ufunc_docs._logical_and_doc
)


logical_or = ufuncs.create_ufunc(
    'nlcpy_logical_or',
    casting._logical_or_types,
    err._logical_or_error_check,
    doc=ufunc_docs._logical_or_doc
)


logical_xor = ufuncs.create_ufunc(
    'nlcpy_logical_xor',
    casting._logical_xor_types,
    err._logical_xor_error_check,
    doc=ufunc_docs._logical_xor_doc
)


logical_not = ufuncs.create_ufunc(
    'nlcpy_logical_not',
    casting._logical_not_types,
    err._logical_not_error_check,
    doc=ufunc_docs._logical_not_doc
)


minimum = ufuncs.create_ufunc(
    'nlcpy_minimum',
    casting._minimum_types,
    err._minimum_error_check,
    doc=ufunc_docs._minimum_doc
)


maximum = ufuncs.create_ufunc(
    'nlcpy_maximum',
    casting._maximum_types,
    err._maximum_error_check,
    doc=ufunc_docs._maximum_doc
)


fmax = ufuncs.create_ufunc(
    'nlcpy_fmax',
    casting._fmax_types,
    err._fmax_error_check,
    doc=ufunc_docs._fmax_doc
)


fmin = ufuncs.create_ufunc(
    'nlcpy_fmin',
    casting._fmin_types,
    err._fmin_error_check,
    doc=ufunc_docs._fmin_doc
)


# trigonometric functions
sin = ufuncs.create_ufunc(
    'nlcpy_sin',
    casting._sin_types,
    err._sin_error_check,
    doc=ufunc_docs._sin_doc
)


cos = ufuncs.create_ufunc(
    'nlcpy_cos',
    casting._cos_types,
    err._cos_error_check,
    doc=ufunc_docs._cos_doc
)


tan = ufuncs.create_ufunc(
    'nlcpy_tan',
    casting._tan_types,
    err._tan_error_check,
    doc=ufunc_docs._tan_doc
)


arcsin = ufuncs.create_ufunc(
    'nlcpy_arcsin',
    casting._arcsin_types,
    err._arcsin_error_check,
    doc=ufunc_docs._arcsin_doc
)


arccos = ufuncs.create_ufunc(
    'nlcpy_arccos',
    casting._arccos_types,
    err._arccos_error_check,
    doc=ufunc_docs._arccos_doc
)


arctan = ufuncs.create_ufunc(
    'nlcpy_arctan',
    casting._arctan_types,
    err._arctan_error_check,
    doc=ufunc_docs._arctan_doc
)


arctan2 = ufuncs.create_ufunc(
    'nlcpy_arctan2',
    casting._arctan2_types,
    err._arctan2_error_check,
    doc=ufunc_docs._arctan2_doc
)


hypot = ufuncs.create_ufunc(
    'nlcpy_hypot',
    casting._hypot_types,
    err._hypot_error_check,
    doc=ufunc_docs._hypot_doc
)


sinh = ufuncs.create_ufunc(
    'nlcpy_sinh',
    casting._sinh_types,
    err._sinh_error_check,
    doc=ufunc_docs._sinh_doc
)


cosh = ufuncs.create_ufunc(
    'nlcpy_cosh',
    casting._cosh_types,
    err._cosh_error_check,
    doc=ufunc_docs._cosh_doc
)


tanh = ufuncs.create_ufunc(
    'nlcpy_tanh',
    casting._tanh_types,
    err._tanh_error_check,
    doc=ufunc_docs._tanh_doc
)


arcsinh = ufuncs.create_ufunc(
    'nlcpy_arcsinh',
    casting._arcsinh_types,
    err._arcsinh_error_check,
    doc=ufunc_docs._arcsinh_doc
)


arccosh = ufuncs.create_ufunc(
    'nlcpy_arccosh',
    casting._arccosh_types,
    err._arccosh_error_check,
    doc=ufunc_docs._arccosh_doc
)


arctanh = ufuncs.create_ufunc(
    'nlcpy_arctanh',
    casting._arctanh_types,
    err._arctanh_error_check,
    doc=ufunc_docs._arctanh_doc
)


deg2rad = ufuncs.create_ufunc(
    'nlcpy_deg2rad',
    casting._deg2rad_types,
    err._deg2rad_error_check,
    doc=ufunc_docs._deg2rad_doc
)


rad2deg = ufuncs.create_ufunc(
    'nlcpy_rad2deg',
    casting._rad2deg_types,
    err._rad2deg_error_check,
    doc=ufunc_docs._rad2deg_doc
)


degrees = ufuncs.create_ufunc(
    'nlcpy_degrees',
    casting._degrees_types,
    err._degrees_error_check,
    doc=ufunc_docs._degrees_doc
)


radians = ufuncs.create_ufunc(
    'nlcpy_radians',
    casting._radians_types,
    err._radians_error_check,
    doc=ufunc_docs._radians_doc
)


# floating functions
isfinite = ufuncs.create_ufunc(
    'nlcpy_isfinite',
    casting._isfinite_types,
    err._isfinite_error_check,
    doc=ufunc_docs._isfinite_doc
)


isinf = ufuncs.create_ufunc(
    'nlcpy_isinf',
    casting._isinf_types,
    err._isinf_error_check,
    doc=ufunc_docs._isinf_doc
)


isnan = ufuncs.create_ufunc(
    'nlcpy_isnan',
    casting._isnan_types,
    err._isnan_error_check,
    doc=ufunc_docs._isnan_doc
)


signbit = ufuncs.create_ufunc(
    'nlcpy_signbit',
    casting._signbit_types,
    err._signbit_error_check,
    doc=ufunc_docs._signbit_doc
)


copysign = ufuncs.create_ufunc(
    'nlcpy_copysign',
    casting._copysign_types,
    err._copysign_error_check,
    doc=ufunc_docs._copysign_doc
)


nextafter = ufuncs.create_ufunc(
    'nlcpy_nextafter',
    casting._nextafter_types,
    err._nextafter_error_check,
    doc=ufunc_docs._nextafter_doc
)


spacing = ufuncs.create_ufunc(
    'nlcpy_spacing',
    casting._spacing_types,
    err._spacing_error_check,
    doc=ufunc_docs._spacing_doc
)


ldexp = ufuncs.create_ufunc(
    'nlcpy_ldexp',
    casting._ldexp_types,
    err._ldexp_error_check,
    doc=ufunc_docs._ldexp_doc
)


# ufunc_operation(frexp)dnl
floor = ufuncs.create_ufunc(
    'nlcpy_floor',
    casting._floor_types,
    err._floor_error_check,
    doc=ufunc_docs._floor_doc
)


ceil = ufuncs.create_ufunc(
    'nlcpy_ceil',
    casting._ceil_types,
    err._ceil_error_check,
    doc=ufunc_docs._ceil_doc
)


trunc = ufuncs.create_ufunc(
    'nlcpy_trunc',
    casting._trunc_types,
    err._trunc_error_check,
    doc=ufunc_docs._trunc_doc
)


# matmul
matmul = ufuncs.create_ufunc(
    'nlcpy_matmul',
    numpy.matmul.types,
    None,
    doc=ufunc_docs._matmul_doc
)


# end of operator functions

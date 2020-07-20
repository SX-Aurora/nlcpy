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
from nlcpy.ufunc import ufunc
from nlcpy.ufunc import casting
from nlcpy.ufunc import err


# ----------------------------------------------------------------------------
# ufunc operations
# see: https://docs.scipy.org/doc/numpy/reference/ufuncs.html
# ----------------------------------------------------------------------------

# math_operations
add = ufunc.create_ufunc(
    'nlcpy_add',
    numpy.add.types,
    err._add_error_check,
)


subtract = ufunc.create_ufunc(
    'nlcpy_subtract',
    casting._subtract_types,
    err._subtract_error_check,
)


multiply = ufunc.create_ufunc(
    'nlcpy_multiply',
    numpy.multiply.types,
    err._multiply_error_check,
)


true_divide = ufunc.create_ufunc(
    'nlcpy_true_divide',
    casting._true_divide_types,
    err._true_divide_error_check,
)


# ufunc_operation(divide,orig_types,valid_error_check)dnl
divide = true_divide

logaddexp = ufunc.create_ufunc(
    'nlcpy_logaddexp',
    numpy.logaddexp.types,
    err._logaddexp_error_check,
)


logaddexp2 = ufunc.create_ufunc(
    'nlcpy_logaddexp2',
    numpy.logaddexp2.types,
    err._logaddexp2_error_check,
)


floor_divide = ufunc.create_ufunc(
    'nlcpy_floor_divide',
    numpy.floor_divide.types,
    err._floor_divide_error_check,
)


negative = ufunc.create_ufunc(
    'nlcpy_negative',
    casting._negative_types,
    err._negative_error_check,
)


positive = ufunc.create_ufunc(
    'nlcpy_positive',
    casting._positive_types,
    err._positive_error_check,
)


power = ufunc.create_ufunc(
    'nlcpy_power',
    numpy.power.types,
    err._power_error_check,
)


remainder = ufunc.create_ufunc(
    'nlcpy_remainder',
    casting._remainder_types,
    err._remainder_error_check,
)


# ufunc_operation(mod,orig_types,valid_error_check)dnl
mod = remainder

fmod = ufunc.create_ufunc(
    'nlcpy_fmod',
    casting._fmod_types,
    err._fmod_error_check,
)


divmod = ufunc.create_ufunc(
    'nlcpy_divmod',
    numpy.divmod.types,
    err._divmod_error_check,
)


absolute = ufunc.create_ufunc(
    'nlcpy_absolute',
    numpy.absolute.types,
    err._absolute_error_check,
)


fabs = ufunc.create_ufunc(
    'nlcpy_fabs',
    casting._fabs_types,
    err._fabs_error_check,
)


rint = ufunc.create_ufunc(
    'nlcpy_rint',
    numpy.rint.types,
    err._rint_error_check,
)


sign = ufunc.create_ufunc(
    'nlcpy_sign',
    casting._sign_types,
    err._sign_error_check,
)


heaviside = ufunc.create_ufunc(
    'nlcpy_heaviside',
    numpy.heaviside.types,
    err._heaviside_error_check,
)


conjugate = ufunc.create_ufunc(
    'nlcpy_conjugate',
    numpy.conjugate.types,
    err._conjugate_error_check,
)


# ufunc_operation(conj,numpy_types,valid_error_check)dnl
conj = conjugate

exp = ufunc.create_ufunc(
    'nlcpy_exp',
    numpy.exp.types,
    err._exp_error_check,
)


exp2 = ufunc.create_ufunc(
    'nlcpy_exp2',
    numpy.exp2.types,
    err._exp2_error_check,
)


log = ufunc.create_ufunc(
    'nlcpy_log',
    numpy.log.types,
    err._log_error_check,
)


log2 = ufunc.create_ufunc(
    'nlcpy_log2',
    numpy.log2.types,
    err._log2_error_check,
)


log10 = ufunc.create_ufunc(
    'nlcpy_log10',
    numpy.log10.types,
    err._log10_error_check,
)


expm1 = ufunc.create_ufunc(
    'nlcpy_expm1',
    numpy.expm1.types,
    err._expm1_error_check,
)


log1p = ufunc.create_ufunc(
    'nlcpy_log1p',
    numpy.log1p.types,
    err._log1p_error_check,
)


sqrt = ufunc.create_ufunc(
    'nlcpy_sqrt',
    numpy.sqrt.types,
    err._sqrt_error_check,
)


square = ufunc.create_ufunc(
    'nlcpy_square',
    numpy.square.types,
    err._square_error_check,
)


cbrt = ufunc.create_ufunc(
    'nlcpy_cbrt',
    casting._cbrt_types,
    err._cbrt_error_check,
)


reciprocal = ufunc.create_ufunc(
    'nlcpy_reciprocal',
    numpy.reciprocal.types,
    err._reciprocal_error_check,
)


gcd = ufunc.create_ufunc(
    'nlcpy_gcd',
    numpy.gcd.types,
    None,
)


lcm = ufunc.create_ufunc(
    'nlcpy_lcm',
    numpy.lcm.types,
    None,
)


# bit-twiddling functions
bitwise_and = ufunc.create_ufunc(
    'nlcpy_bitwise_and',
    casting._bitwise_and_types,
    err._bitwise_and_error_check,
)


bitwise_or = ufunc.create_ufunc(
    'nlcpy_bitwise_or',
    casting._bitwise_or_types,
    err._bitwise_or_error_check,
)


bitwise_xor = ufunc.create_ufunc(
    'nlcpy_bitwise_xor',
    casting._bitwise_xor_types,
    err._bitwise_xor_error_check,
)


invert = ufunc.create_ufunc(
    'nlcpy_invert',
    casting._invert_types,
    err._invert_error_check,
)


left_shift = ufunc.create_ufunc(
    'nlcpy_left_shift',
    casting._left_shift_types,
    err._left_shift_error_check,
)


right_shift = ufunc.create_ufunc(
    'nlcpy_right_shift',
    casting._right_shift_types,
    err._right_shift_error_check,
)


# comparison functions
greater = ufunc.create_ufunc(
    'nlcpy_greater',
    numpy.greater.types,
    err._greater_error_check,
)


greater_equal = ufunc.create_ufunc(
    'nlcpy_greater_equal',
    numpy.greater_equal.types,
    err._greater_equal_error_check,
)


less = ufunc.create_ufunc(
    'nlcpy_less',
    numpy.less.types,
    err._less_error_check,
)


less_equal = ufunc.create_ufunc(
    'nlcpy_less_equal',
    numpy.less_equal.types,
    err._less_equal_error_check,
)


not_equal = ufunc.create_ufunc(
    'nlcpy_not_equal',
    numpy.not_equal.types,
    err._not_equal_error_check,
)


equal = ufunc.create_ufunc(
    'nlcpy_equal',
    numpy.equal.types,
    err._equal_error_check,
)


logical_and = ufunc.create_ufunc(
    'nlcpy_logical_and',
    numpy.logical_and.types,
    err._logical_and_error_check,
)


logical_or = ufunc.create_ufunc(
    'nlcpy_logical_or',
    numpy.logical_or.types,
    err._logical_or_error_check,
)


logical_xor = ufunc.create_ufunc(
    'nlcpy_logical_xor',
    numpy.logical_xor.types,
    err._logical_xor_error_check,
)


logical_not = ufunc.create_ufunc(
    'nlcpy_logical_not',
    numpy.logical_not.types,
    err._logical_not_error_check,
)


minimum = ufunc.create_ufunc(
    'nlcpy_minimum',
    numpy.minimum.types,
    err._minimum_error_check,
)


maximum = ufunc.create_ufunc(
    'nlcpy_maximum',
    numpy.maximum.types,
    err._maximum_error_check,
)


fmax = ufunc.create_ufunc(
    'nlcpy_fmax',
    numpy.fmax.types,
    err._fmax_error_check,
)


fmin = ufunc.create_ufunc(
    'nlcpy_fmin',
    numpy.fmin.types,
    err._fmin_error_check,
)


# trigonometric functions
sin = ufunc.create_ufunc(
    'nlcpy_sin',
    numpy.sin.types,
    err._sin_error_check,
)


cos = ufunc.create_ufunc(
    'nlcpy_cos',
    numpy.cos.types,
    err._cos_error_check,
)


tan = ufunc.create_ufunc(
    'nlcpy_tan',
    numpy.tan.types,
    err._tan_error_check,
)


arcsin = ufunc.create_ufunc(
    'nlcpy_arcsin',
    numpy.arcsin.types,
    err._arcsin_error_check,
)


arccos = ufunc.create_ufunc(
    'nlcpy_arccos',
    numpy.arccos.types,
    err._arccos_error_check,
)


arctan = ufunc.create_ufunc(
    'nlcpy_arctan',
    numpy.arctan.types,
    err._arctan_error_check,
)


arctan2 = ufunc.create_ufunc(
    'nlcpy_arctan2',
    casting._arctan2_types,
    err._arctan2_error_check,
)


hypot = ufunc.create_ufunc(
    'nlcpy_hypot',
    casting._hypot_types,
    err._hypot_error_check,
)


sinh = ufunc.create_ufunc(
    'nlcpy_sinh',
    numpy.sinh.types,
    err._sinh_error_check,
)


cosh = ufunc.create_ufunc(
    'nlcpy_cosh',
    numpy.cosh.types,
    err._cosh_error_check,
)


tanh = ufunc.create_ufunc(
    'nlcpy_tanh',
    numpy.tanh.types,
    err._tanh_error_check,
)


arcsinh = ufunc.create_ufunc(
    'nlcpy_arcsinh',
    numpy.arcsinh.types,
    err._arcsinh_error_check,
)


arccosh = ufunc.create_ufunc(
    'nlcpy_arccosh',
    numpy.arccosh.types,
    err._arccosh_error_check,
)


arctanh = ufunc.create_ufunc(
    'nlcpy_arctanh',
    numpy.arctanh.types,
    err._arctanh_error_check,
)


deg2rad = ufunc.create_ufunc(
    'nlcpy_deg2rad',
    casting._deg2rad_types,
    err._deg2rad_error_check,
)


rad2deg = ufunc.create_ufunc(
    'nlcpy_rad2deg',
    casting._rad2deg_types,
    err._rad2deg_error_check,
)


degrees = ufunc.create_ufunc(
    'nlcpy_degrees',
    casting._degrees_types,
    err._degrees_error_check,
)


radians = ufunc.create_ufunc(
    'nlcpy_radians',
    casting._radians_types,
    err._radians_error_check,
)


# floating functions
isfinite = ufunc.create_ufunc(
    'nlcpy_isfinite',
    numpy.isfinite.types,
    err._isfinite_error_check,
)


isinf = ufunc.create_ufunc(
    'nlcpy_isinf',
    numpy.isinf.types,
    err._isinf_error_check,
)


isnan = ufunc.create_ufunc(
    'nlcpy_isnan',
    numpy.isnan.types,
    err._isnan_error_check,
)


isnat = ufunc.create_ufunc(
    'nlcpy_isnat',
    numpy.isnat.types,
    err._isnat_error_check,
)


signbit = ufunc.create_ufunc(
    'nlcpy_signbit',
    numpy.signbit.types,
    err._signbit_error_check,
)


copysign = ufunc.create_ufunc(
    'nlcpy_copysign',
    numpy.copysign.types,
    err._copysign_error_check,
)


nextafter = ufunc.create_ufunc(
    'nlcpy_nextafter',
    numpy.nextafter.types,
    err._nextafter_error_check,
)


spacing = ufunc.create_ufunc(
    'nlcpy_spacing',
    numpy.spacing.types,
    err._spacing_error_check,
)


modf = ufunc.create_ufunc(
    'nlcpy_modf',
    numpy.modf.types,
    err._modf_error_check,
)


ldexp = ufunc.create_ufunc(
    'nlcpy_ldexp',
    numpy.ldexp.types,
    err._ldexp_error_check,
)


frexp = ufunc.create_ufunc(
    'nlcpy_frexp',
    numpy.frexp.types,
    None,
)


floor = ufunc.create_ufunc(
    'nlcpy_floor',
    casting._floor_types,
    err._floor_error_check,
)


ceil = ufunc.create_ufunc(
    'nlcpy_ceil',
    casting._ceil_types,
    err._ceil_error_check,
)


trunc = ufunc.create_ufunc(
    'nlcpy_trunc',
    numpy.trunc.types,
    err._trunc_error_check,
)


# matmul
matmul = ufunc.create_ufunc(
    'nlcpy_matmul',
    numpy.matmul.types,
    None,
)


# end of operator functions

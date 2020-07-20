divert(-1)
#
# * The documantaition in this file is based on NumPy Reference.
#   (https://numpy.org/doc/stable/reference/index.html)
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
# # NumPy License #
# 
#     Copyright (c) 2005-2020, NumPy Developers.
#     All rights reserved.
#     
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of the NumPy Developers nor the names of any contributors may be
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
divert(0)dnl
include(macro_ufunc.m4)dnl
changecom(%%)

##
# @name Math Operations
# @{
#
macro_add(Ufuncs)
macro_subtract(Ufuncs)
macro_multiply(Ufuncs)
macro_divide(Ufuncs)
macro_logaddexp(Ufuncs)
macro_logaddexp2(Ufuncs)
macro_true_divide(Ufuncs)
macro_floor_divide(Ufuncs)
macro_negative(Ufuncs)
macro_positive(Ufuncs)
macro_power(Ufuncs)
macro_remainder(Ufuncs)
macro_mod(Ufuncs)
macro_fmod(Ufuncs)
macro_absolute(Ufuncs)
macro_fabs(Ufuncs)
macro_rint(Ufuncs)
macro_heaviside(Ufuncs)
macro_conj(Ufuncs)
macro_conjugate(Ufuncs)
macro_exp(Ufuncs)
macro_exp2(Ufuncs)
macro_log(Ufuncs)
macro_log2(Ufuncs)
macro_log10(Ufuncs)
macro_log1p(Ufuncs)
macro_expm1(Ufuncs)
macro_sqrt(Ufuncs)
macro_square(Ufuncs)
macro_cbrt(Ufuncs)
macro_reciprocal(Ufuncs)

##@}

##
# @name Trigonometric Functions
# @{
#

macro_sin(Ufuncs)
macro_cos(Ufuncs)
macro_tan(Ufuncs)
macro_arcsin(Ufuncs)
macro_arccos(Ufuncs)
macro_arctan(Ufuncs)
macro_arctan2(Ufuncs)
macro_hypot(Ufuncs)
macro_sinh(Ufuncs)
macro_cosh(Ufuncs)
macro_tanh(Ufuncs)
macro_arcsinh(Ufuncs)
macro_arccosh(Ufuncs)
macro_arctanh(Ufuncs)
macro_deg2rad(Ufuncs)
macro_rad2deg(Ufuncs)

##@}

##
# @name Bit-twiddling Functions
# @{
#
macro_bitwise_and(Ufuncs)
macro_bitwise_or(Ufuncs)
macro_bitwise_xor(Ufuncs)
macro_invert(Ufuncs)
macro_left_shift(Ufuncs)
macro_right_shift(Ufuncs)
macro_greater(Ufuncs)
macro_greater_equal(Ufuncs)
macro_less(Ufuncs)
macro_less_equal(Ufuncs)
macro_not_equal(Ufuncs)
macro_equal(Ufuncs)
macro_logical_and(Ufuncs)
macro_logical_or(Ufuncs)
macro_logical_xor(Ufuncs)
macro_logical_not(Ufuncs)
macro_maximum(Ufuncs)
macro_minimum(Ufuncs)
macro_fmax(Ufuncs)
macro_fmin(Ufuncs)

##@}

##
# @name Floating Point Functions
# @{
#
macro_isfinite(Ufuncs)
macro_isinf(Ufuncs)
macro_isnan(Ufuncs)
macro_signbit(Ufuncs)
macro_copysign(Ufuncs)
macro_sign(Ufuncs)
macro_nextafter(Ufuncs)
macro_spacing(Ufuncs)
macro_ldexp(Ufuncs)
macro_floor(Ufuncs)
macro_ceil(Ufuncs)
macro_trunc(Ufuncs)

##@}
## @fn matmul (x1, x2, out=None, casting='same_kind', order='K', dtype=None, subok=False)
#
# @if(lang_ja)
# @else
# @brief Matrix product of two arrays.@n
#
# @param x1, x2 : <em>array_like</em>@n
# Input arrays, scalars not allowed.
#
# @param out : <em>@ref n-dimensional_array "ndarray" or None, @b optional </em>@n
# A location into which the result is stored. If provided, it must have a shape that the
# inputs broadcast to. If not provided or None, a freshly-allocated array is returned. 
# A tuple (possible only as a keyword argument) must have length equal to the number of outputs.
#
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The matrix product of the inputs. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension array.
#
# @sa
# @li @ref products::dot : Dot product of two arrays. 
# @note
# The behavior depends on the arguments in the following way.
# @li If both arguments are 2-D they are multiplied like conventional matrices.
# @li If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
# @li If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
#
# @attention
# @li x1.ndim > 2 or x2.ndim > 2 :  @em NotImplementedError occurs.@n
#
# @par Example
# For 2-D arrays it is the matrix product:
# @code
# >>> import nlcpy as vp
# >>> a = np.array([[1, 0],
#                   [0, 1]])
# >>> b = np.array([[4, 1],
#                   [2, 2]])
# >>> np.matmul(a, b)
# array([[4, 1],
#        [2, 2]])
# @endcode
# For 2-D mixed with 1-D, the result is the usual.
# @code
# >>> a = np.array([[1, 0],
#                   [0, 1]])
# >>> b = np.array([1, 2])
# >>> np.matmul(a, b)
# array([1, 2])
# >>> np.matmul(b, a)
# array([1, 2])
# @endcode
# @endif
#
def matmul (x1, x2, out=None, casting='same_kind', order='K', dtype=None, subok=False): pass

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

include(macros.m4)dnl
import numpy
from nlcpy.ufuncs import ufuncs
from nlcpy.ufuncs import casting
from nlcpy.ufuncs import err
from nlcpy.ufuncs import ufunc_docs


# ----------------------------------------------------------------------------
# ufunc operations
# see: https://docs.scipy.org/doc/numpy/reference/ufuncs.html
# ----------------------------------------------------------------------------

define(<--@ufunc_operation@-->,<--@dnl
$1 = ufuncs.create_ufunc(
    'nlcpy_$1',
ifelse(<--@$2@-->,<--@orig_types@-->,<--@dnl
    casting._$1_types,
@-->,<--@dnl
    numpy.$1.types,
@-->)dnl
ifelse(<--@$3@-->,<--@valid_error_check@-->,<--@dnl
    err._$1_error_check,
@-->,<--@dnl
    None,
@-->)dnl
    doc=ufunc_docs._$1_doc
)


@-->)dnl
# math_operations
ufunc_operation(add,numpy_types,valid_error_check)dnl
ufunc_operation(subtract,orig_types,valid_error_check)dnl
ufunc_operation(multiply,numpy_types,valid_error_check)dnl
ufunc_operation(true_divide,orig_types,valid_error_check)dnl
# ufunc_operation(divide,orig_types,valid_error_check)dnl
divide = true_divide

ufunc_operation(logaddexp,numpy_types,valid_error_check)dnl
ufunc_operation(logaddexp2,numpy_types,valid_error_check)dnl
ufunc_operation(floor_divide,numpy_types,valid_error_check)dnl
ufunc_operation(negative,orig_types,valid_error_check)dnl
ufunc_operation(positive,orig_types,valid_error_check)dnl
ufunc_operation(power,numpy_types,valid_error_check)dnl
ufunc_operation(remainder,orig_types,valid_error_check)dnl
# ufunc_operation(mod,orig_types,valid_error_check)dnl
mod = remainder

ufunc_operation(fmod,orig_types,valid_error_check)dnl
# ufunc_operation(divmod,numpy_types,valid_error_check)dnl
ufunc_operation(absolute,numpy_types,valid_error_check)dnl
ufunc_operation(fabs,orig_types,valid_error_check)dnl
ufunc_operation(rint,numpy_types,valid_error_check)dnl
ufunc_operation(sign,orig_types,valid_error_check)dnl
ufunc_operation(heaviside,numpy_types,valid_error_check)dnl
ufunc_operation(conjugate,numpy_types,valid_error_check)dnl
# ufunc_operation(conj,numpy_types,valid_error_check)dnl
conj = conjugate

ufunc_operation(exp,numpy_types,valid_error_check)dnl
ufunc_operation(exp2,numpy_types,valid_error_check)dnl
ufunc_operation(log,numpy_types,valid_error_check)dnl
ufunc_operation(log2,numpy_types,valid_error_check)dnl
ufunc_operation(log10,numpy_types,valid_error_check)dnl
ufunc_operation(expm1,numpy_types,valid_error_check)dnl
ufunc_operation(log1p,numpy_types,valid_error_check)dnl
ufunc_operation(sqrt,numpy_types,valid_error_check)dnl
ufunc_operation(square,numpy_types,valid_error_check)dnl
ufunc_operation(cbrt,orig_types,valid_error_check)dnl
ufunc_operation(reciprocal,numpy_types,valid_error_check)dnl
# ufunc_operation(gcd)dnl
# ufunc_operation(lcm)dnl
# bit-twiddling functions
ufunc_operation(bitwise_and,orig_types,valid_error_check)dnl
ufunc_operation(bitwise_or,orig_types,valid_error_check)dnl
ufunc_operation(bitwise_xor,orig_types,valid_error_check)dnl
ufunc_operation(invert,orig_types,valid_error_check)dnl
ufunc_operation(left_shift,orig_types,valid_error_check)dnl
ufunc_operation(right_shift,orig_types,valid_error_check)dnl
# comparison functions
ufunc_operation(greater,numpy_types,valid_error_check)dnl
ufunc_operation(greater_equal,numpy_types,valid_error_check)dnl
ufunc_operation(less,numpy_types,valid_error_check)dnl
ufunc_operation(less_equal,numpy_types,valid_error_check)dnl
ufunc_operation(not_equal,numpy_types,valid_error_check)dnl
ufunc_operation(equal,numpy_types,valid_error_check)dnl
ufunc_operation(logical_and,numpy_types,valid_error_check)dnl
ufunc_operation(logical_or,numpy_types,valid_error_check)dnl
ufunc_operation(logical_xor,numpy_types,valid_error_check)dnl
ufunc_operation(logical_not,numpy_types,valid_error_check)dnl
ufunc_operation(minimum,numpy_types,valid_error_check)dnl
ufunc_operation(maximum,numpy_types,valid_error_check)dnl
ufunc_operation(fmax,numpy_types,valid_error_check)dnl
ufunc_operation(fmin,numpy_types,valid_error_check)dnl
# trigonometric functions
ufunc_operation(sin,numpy_types,valid_error_check,
    Computes the element-wise sin.
)dnl
ufunc_operation(cos,numpy_types,valid_error_check)dnl
ufunc_operation(tan,numpy_types,valid_error_check)dnl
ufunc_operation(arcsin,numpy_types,valid_error_check)dnl
ufunc_operation(arccos,numpy_types,valid_error_check)dnl
ufunc_operation(arctan,numpy_types,valid_error_check)dnl
ufunc_operation(arctan2,orig_types,valid_error_check)dnl
ufunc_operation(hypot,orig_types,valid_error_check)dnl
ufunc_operation(sinh,numpy_types,valid_error_check)dnl
ufunc_operation(cosh,numpy_types,valid_error_check)dnl
ufunc_operation(tanh,numpy_types,valid_error_check)dnl
ufunc_operation(arcsinh,numpy_types,valid_error_check)dnl
ufunc_operation(arccosh,numpy_types,valid_error_check)dnl
ufunc_operation(arctanh,numpy_types,valid_error_check)dnl
ufunc_operation(deg2rad,orig_types,valid_error_check)dnl
ufunc_operation(rad2deg,orig_types,valid_error_check)dnl
ufunc_operation(degrees,orig_types,valid_error_check)dnl
ufunc_operation(radians,orig_types,valid_error_check)dnl
# floating functions
ufunc_operation(isfinite,numpy_types,valid_error_check)dnl
ufunc_operation(isinf,numpy_types,valid_error_check)dnl
ufunc_operation(isnan,numpy_types,valid_error_check)dnl
# ufunc_operation(isnat,numpy_types,valid_error_check)dnl
ufunc_operation(signbit,numpy_types,valid_error_check)dnl
ufunc_operation(copysign,numpy_types,valid_error_check)dnl
ufunc_operation(nextafter,numpy_types,valid_error_check)dnl
ufunc_operation(spacing,numpy_types,valid_error_check)dnl
# ufunc_operation(modf,numpy_types,valid_error_check)dnl
ufunc_operation(ldexp,numpy_types,valid_error_check)dnl
# ufunc_operation(frexp)dnl
ufunc_operation(floor,orig_types,valid_error_check)dnl
ufunc_operation(ceil,orig_types,valid_error_check)dnl
ufunc_operation(trunc,numpy_types,valid_error_check)dnl
# matmul
ufunc_operation(matmul)dnl
# end of operator functions

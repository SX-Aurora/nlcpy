divert(-1)dnl
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
include(macros.m4)dnl
dnl
define(<--@macro_ufunc@-->,<--@dnl
pushdef(<--@UFUNC@-->,       <--@$1@-->)dnl
pushdef(<--@NARY@-->,        <--@$2@-->)dnl
pushdef(<--@GROUP@-->,       <--@$3@-->)dnl
pushdef(<--@DESC_BRIEF@-->,  <--@$4@-->)dnl
pushdef(<--@DESC_INPUT@-->,  <--@$5@-->)dnl
pushdef(<--@DESC_RET@-->,    <--@$6@-->)dnl
pushdef(<--@DESC_SA@-->,     <--@$7@-->)dnl
pushdef(<--@DESC_NOTE@-->,   <--@$8@-->)dnl
pushdef(<--@DESC_EXAMPLE@-->,<--@$9@-->)dnl
ifelse(NARY,unary,<--@dnl
## @fn UFUNC (x, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=False)
@-->,<--@dnl
## @fn UFUNC (x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=False)
@-->)dnl
#
# @if(lang_ja)
# @else
# @brief DESC_BRIEF@n
#
ifelse(NARY,unary,<--@dnl
# @param x : <em>array_like</em>@n
@-->,<--@dnl
# @param x1, x2 : <em>array_like</em>@n
@-->)dnl
# DESC_INPUT
#
# @param out : <em>@ref n-dimensional_array "ndarray" or None, @b optional </em>@n
# A location into which the result is stored. If provided, it must have a shape that the
# inputs broadcast to. If not provided or None, a freshly-allocated array is returned. 
# A tuple (possible only as a keyword argument) must have length equal to the number of outputs.
#
# @param where : <em>array_like, @b optional </em>@n
# This condition is broadcast over the input. At locations where the condition is True, 
# the @em out array will be set to the ufunc result. Elsewhere, the @em out array will retain 
# its original value. Note that if an uninitialized @em out array is created via the default
# <span class="pre">out=None</span>, locations within it where the condition is False will remain uninitialized.
#
# @param **kwargs @n
# For other keyword-only arguments, see the section @ref Ufunc_optional.
#
DESC_RET
#
DESC_SA 
DESC_NOTE
#
# @par Example
DESC_EXAMPLE
# @endif
#
ifelse(NARY,unary,<--@dnl
def UFUNC (x, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=False): pass
@-->,<--@dnl
def UFUNC (x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=False): pass
@-->)dnl
popdef(<--@UFUNC@-->)dnl
popdef(<--@NARY@-->)dnl
popdef(<--@GROUP@-->)dnl
popdef(<--@DESC_BRIEF@-->)dnl
popdef(<--@DESC_INPUT@-->)dnl
popdef(<--@DESC_RET@-->)dnl
popdef(<--@DESC_SA@-->)dnl
popdef(<--@DESC_NOTE@-->)dnl
popdef(<--@DESC_EXAMPLE@-->)dnl
@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.add
###########################################################################
divert(0)dnl
define(<--@macro_add@-->,<--@dnl
macro_ufunc(<--@add@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise addition of the inputs.@-->,dnl
<--@The arrays to be added. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval UFUNC : <em>@ref n-dimensional_array "ndarray"</em>@n
# The sum of @em x1 and @em x2, element-wise. If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @note
# Equivalent to @em x1 + @em x2 in terms of array broadcasting.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.add(1.0, 4.0)
# array(5.)
# >>> x1 = vp.arange(9.0).reshape((3, 3))
# >>> x2 = vp.arange(3.0)
# >>> vp.add(x1, x2)
# array([[  0.,   2.,   4.],
#        [  3.,   5.,   7.],
#        [  6.,   8.,  10.]])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.subtract
###########################################################################
divert(0)dnl
define(<--@macro_subtract@-->,<--@dnl
macro_ufunc(<--@subtract@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise subtraction of the inputs.@-->,dnl
<--@The arrays to be substracted from each other. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval UFUNC : <em>@ref n-dimensional_array "ndarray"</em>@n
# The difference of @em x1 and @em x2, element-wise. If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @note
# Equivalent to @em x1 - @em x2 in terms of array broadcasting.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.subtract(1.0, 4.0)
# array(-3.)
# @endcode@n
# @code
# >>> x1 = vp.arange(9.0).reshape((3, 3))
# >>> x2 = vp.arange(3.0)
# >>> vp.subtract(x1, x2)
# array([[0., 0., 0.],
#        [3., 3., 3.],
#        [6., 6., 6.]])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.multiply
###########################################################################
divert(0)dnl
define(<--@macro_multiply@-->,<--@dnl
macro_ufunc(<--@multiply@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise multiplication of the inputs.@-->,dnl
<--@The arrays to be multiplied. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval UFUNC : <em>@ref n-dimensional_array "ndarray"</em>@n
# The product of @em x1 and @em x2, element-wise. If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @note
# Equivalent to @em x1 * @em x2 in terms of array broadcasting.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.multiply(2.0, 4.0)
# array(8.)
# @endcode@n
# @code
# >>> x1 = vp.arange(9.0).reshape((3, 3))
# >>> x2 = vp.arange(3.0)
# >>> vp.multiply(x1, x2)
# array([[ 0.,  1.,  4.],
#        [ 0.,  4., 10.],
#        [ 0.,  7., 16.]])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.divide
###########################################################################
divert(0)dnl
define(<--@macro_divide@-->,<--@dnl
macro_ufunc(<--@divide@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise division of the inputs.@-->,dnl
<--@@em x1 : Dividend array.@n @em x2 : Divisor array.@n If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval UFUNC : <em>@ref n-dimensional_array "ndarray"</em>@n
# The result that @em x1 is divided by @em x2 for each element. If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @note
# @li If the values of @em x2 are zero, its return values becomes @c nan (not @c inf) due to performance reasons.
# @li Equivalent to @em x1 / @em x2 in terms of array broadcasting.
# @li In Python 3.0 or later, <span class="pre">//</span> is the floor division operator and <span class="pre">/</span> is the true division operator. The <span class="pre">divide(x1, x2)</span> function is equivalent to the true division in Python.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> x = vp.arange(5)
# >>> vp.divide(x, 4)
# array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
# @endcode@n
# @code
# >>> x/4
# array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
# >>> x//4
# array([0, 0, 0, 0, 1])
# @endcode@n
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.logaddexp
###########################################################################
divert(0)dnl
define(<--@macro_logaddexp@-->,<--@dnl
macro_ufunc(<--@logaddexp@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise natural logarithm of @f$ exp(x1) + exp(x2) @f$.@-->,dnl
<--@Input arrays or scalars. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval UFUNC : <em>@ref n-dimensional_array "ndarray"</em>@n
# An ndarray, containing @f$ log(exp(x1) + exp(x2)) @f$ for each element. If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::logaddexp2 : Computes the element-wise base-2 logarithm of the inputs.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> prob1 = vp.log(1e-50)
# >>> prob2 = vp.log(2.5e-50)
# >>> prob12 = vp.logaddexp(prob1, prob2)
# >>> prob12
# array(-113.87649168)
# >>> vp.exp(prob12)
# array(3.5e-50)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.logaddexp2
###########################################################################
divert(0)dnl
define(<--@macro_logaddexp2@-->,<--@dnl
macro_ufunc(<--@logaddexp2@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise base-2 logarithm of the inputs.@-->,dnl
<--@Input arrays or scalars. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval UFUNC : <em>@ref n-dimensional_array "ndarray"</em>@n
# An ndarray, containing @f$ log_2(2^{x1} + 2^{x2}) @f$ for each element. If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::logaddexp : Computes the element-wise natural logarithm of @f$ exp(x1) + exp(x2) @f$.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> prob1 = vp.log2(1e-50)
# >>> prob2 = vp.log2(2.5e-50)
# >>> prob12 = vp.logaddexp2(prob1, prob2)
# >>> prob12
# array(-164.28904982)
# >>> 2**prob12
# array(3.5e-50)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.true_divide
###########################################################################
divert(0)dnl
define(<--@macro_true_divide@-->,<--@dnl
macro_ufunc(<--@true_divide@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise division of the inputs.@n Instead of the Python traditional 'floor division', this returns a true division. True division adjusts the output type to present the best answer, regardless of input types.@-->,dnl
<--@@em x1 : Dividend array.@n @em x2 : Divisor array.@n If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval UFUNC : <em>@ref n-dimensional_array "ndarray"</em>@n
# The result that @em x1 is divided by @em x2 for each element. If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @note
# @li If the values of @em x2 are zero, its return values becomes @c nan (not @c inf) due to performance reasons.
# @li Equivalent to @em x1 / @em x2 in terms of array broadcasting.
# @li In Python 3.0 or later, <span class="pre">//</span> is the floor division operator and <span class="pre">/</span> is the true division operator. The <span class="pre">true_divide(x1, x2)</span> function is equivalent to the true division in Python.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> x = vp.arange(5)
# >>> vp.true_divide(x, 4)
# array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
# @endcode@n
# @code
# >>> x/4
# array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
# >>> x//4
# array([0, 0, 0, 0, 1])
# @endcode@n
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.floor_divide
###########################################################################
divert(0)dnl
define(<--@macro_floor_divide@-->,<--@dnl
macro_ufunc(<--@floor_divide@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise floor divition of the inputs. It is equivalent to the Python <span class="pre">//</span> operator and pairs with the Python <span class="pre">%</span> (@ref remainder), function so that <span class="pre">a = a % b + b * (a // b)</span> up to roundoff.@-->,dnl
<--@@em x1 : Numerator.@n @em x2 : Denominator.@n If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# @em y = floor(@em x1 /@em x2). If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::remainder : Computes the element-wise %remainder of division.
# @li @ref ufuncs::divide : Computes the element-wise division of the inputs.
# @li @ref ufuncs::floor : Returns the %floor of the input, element-wise.
# @li @ref ufuncs::ceil : Returns the ceiling of the input, element-wise.
dnl@-->,<--@dnl
# @note
# @li In Python 3.0 or later, <span class="pre">//</span> is the floor division operator and <span class="pre">/</span> is the true division operator. The <span class="pre">floor_divide(x1, x2)</span> function is equivalent to the floor division in Python.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.floor_divide(7,3)
# array(2)
# >>> vp.floor_divide([1., 2., 3., 4.], 2.5)
# array([ 0.,  0.,  1.,  1.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.negative
###########################################################################
divert(0)dnl
define(<--@macro_negative@-->,<--@dnl
macro_ufunc(<--@negative@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes numerical negative, element-wise.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# @em y = -@em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.negative([1.,-1.])
# array([-1.,  1.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.positive
###########################################################################
divert(0)dnl
define(<--@macro_positive@-->,<--@dnl
macro_ufunc(<--@positive@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes numerical positive, element-wise.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# @em y = +@em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.negative([1.,-1.])
# array([1., -1.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.power
###########################################################################
divert(0)dnl
define(<--@macro_power@-->,<--@dnl
macro_ufunc(<--@power@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise exponentiation of the inputs.@-->,dnl
<--@@em x1 : The bases.@n @em x2 : The exponents.@n If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# @em x1 to the power of @em x2, element-wise. If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# Cube each element in a list.
# @code
# >>> import nlcpy as vp
# >>> x1 = vp.arange(6)
# >>> x1
# [0, 1, 2, 3, 4, 5]
# >>> vp.power(x1, 3)
# array([  0,   1,   8,  27,  64, 125])
# @endcode@n
# Raise the bases to different exponents.
# @code
# >>> x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
# >>> vp.power(x1, x2)
# array([ 0.,  1.,  8., 27., 16.,  5.])
# @endcode@n
# The effect of broadcasting.
# @code
# >>> x2 = vp.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
# >>> x2
# array([[1, 2, 3, 3, 2, 1],
#        [1, 2, 3, 3, 2, 1]])
# >>> vp.power(x1, x2)
# array([[ 0,  1,  8, 27, 16,  5],
#        [ 0,  1,  8, 27, 16,  5]])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.remainder
###########################################################################
divert(0)dnl
define(<--@macro_remainder@-->,<--@dnl
macro_ufunc(<--@remainder@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise remainder of division.@n Computes the remainder complementary to the floor_divide function. It is equivalent to the Python modulus operator <span class="pre">x1 % x2</span> and has the same sign as the divisor @em x2. @-->,dnl
<--@@em x1 : Dividend array.@n @em x2 : Divisor array.@n If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The element-wise remainder of the quotient ufuncs::floor_divide(x1, x2). If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
#@sa
# @li @ref ufuncs::floor_divide : Computes the element-wise %floor divition of the inputs.
# @li @ref ufuncs::fmod : Computes the element-wise %remainder of division.  
# @li @ref ufuncs::divide : Computes the element-wise division of the inputs. 
# @li @ref ufuncs::floor : Returns the %floor of the input, element-wise. 
dnl@-->,<--@dnl
# @note
# @li Returns 0 when @em x2 is 0 and both @em x1 and @em x2 are integers.@n
# @li @ref ufuncs::mod an alias of this function.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.remainder([4, 7], [2, 3])
# array([0, 1])
# >>> vp.remainder(vp.arange(7), 5)
# array([0, 1, 2, 3, 4, 0, 1])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.mod
###########################################################################
divert(0)dnl
define(<--@macro_mod@-->,<--@dnl
macro_ufunc(<--@mod@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise remainder of division.@n Computes the remainder complementary to the floor_divide function. It is equivalent to the Python modulus operator <span class="pre">x1 % x2</span> and has the same sign as the divisor @em x2. @-->,dnl
<--@@em x1 : Dividend array.@n @em x2 : Divisor array.@n If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The element-wise remainder of the quotient ufuncs::floor_divide(x1, x2). If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
#@sa
# @li @ref ufuncs::floor_divide : Computes the element-wise %floor divition of the inputs.
dnl@-->,<--@dnl
# @note
# @li Returns 0 when @em x2 is 0 and both @em x1 and @em x2 are integers.@n
# @li @ref ufuncs::remainder an alias of this function.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.mod([4, 7], [2, 3])
# array([0, 1])
# >>> vp.mod(vp.arange(7), 5)
# array([0, 1, 2, 3, 4, 0, 1])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.fmod
###########################################################################
divert(0)dnl
define(<--@macro_fmod@-->,<--@dnl
macro_ufunc(<--@fmod@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise remainder of division.@n This is the NLCPy implementation of the C library function fmod, the remainder has the same sign as the dividend @em x1. @-->,dnl
<--@@em x1 : Dividend array.@n @em x2 : Divisor array.@n If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The element-wise remainder of the quotient ufuncs::floor_divide(x1, x2). If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @li @ref ufuncs::remainder Equivalent of Python <span class="pre">%</span> operator.
dnl@-->,<--@dnl
# @note
# The result of the modulo operation for negative dividend and divisors is bound by conventions. For @ref fmod, the sign of result is the sign of the dividend, while for @ref remainder the sign of the result is the sign of the divisor. 
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.fmod([-3, -2, -1, 1, 2, 3], 2)
# array([-1,  0, -1,  1,  0,  1])
# >>> vp.remainder([-3, -2, -1, 1, 2, 3], 2)
# array([1, 0, 1, 1, 0, 1])
# @endcode@n
# @code
# >>> vp.fmod([5, 3], [2, 2.])
# array([ 1.,  1.])
# >>> a = vp.arange(-3, 3).reshape(3, 2)
# >>> a
# array([[-3, -2],
#        [-1,  0],
#        [ 1,  2]])
# >>> vp.fmod(a, [2,2])
# array([[-1,  0],
#        [-1,  0],
#        [ 1,  0]])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.absolute
###########################################################################
divert(0)dnl
define(<--@macro_absolute@-->,<--@dnl
macro_ufunc(<--@absolute@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise absolute value.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the absolute value for each element in @em x.
# For complex input, a + ib, the absolute value is @f$ \sqrt{ a^2 + b^2 }i@f$. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> x = vp.array([-1.2, 1.2])
# >>> vp.absolute(x)
# array([1.2, 1.2])
# >>> vp.absolute(1.2 + 1j)
# array(1.56204994)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.fabs
###########################################################################
divert(0)dnl
define(<--@macro_fabs@-->,<--@dnl
macro_ufunc(<--@fabs@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise absolute value.@n This function returns the absolute values (positive magnitude) of the data in x. Complex values are not handled, use absolute to find the absolute values of complex data.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the absolute value for each element in @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.fabs(-1)
# array(1.)
# >>> x = vp.array([-1.2, 1.2])
# >>> vp.fabs(x)
# array([ 1.2,  1.2])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.rint
###########################################################################
divert(0)dnl
define(<--@macro_rint@-->,<--@dnl
macro_ufunc(<--@rint@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise nearest integer.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the nearest integer for each element in @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::ceil : Returns the ceiling of the input, element-wise.
# @li @ref ufuncs::floor : Returns the %floor of the input, element-wise.
# @li @ref ufuncs::trunc : Returns the truncated value of the input, element-wise.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> a = vp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
# >>> vp.rint(a)
# array([-2., -2., -0.,  0.,  2.,  2.,  2.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.sign
###########################################################################
divert(0)dnl
define(<--@macro_sign@-->,<--@dnl
macro_ufunc(<--@sign@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Returns the element-wise indication of the sign of a number.@n The sign function returns <span class="pre">-1 if x < 0, 0 if x==0, 1 if x > 0</span>. @c nan is returned for @c nan inputs.@n
# For complex inputs, the sign function returns <span class="pre">sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j</span>. @c arary(nan+0j) is returned for complex @c nan inputs.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the sign for each element in @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::ceil : Returns the ceiling of the input, element-wise.
# @li @ref ufuncs::floor : Returns the %floor of the input, element-wise.
# @li @ref ufuncs::trunc : Returns the truncated value of the input, element-wise.
dnl@-->,<--@dnl
# @note There is more than one definition of sign in common use for complex numbers. The definition used here is equivalent to @f$x/\sqrt{x*x}@f$ which is different from a common alternative, @f$x/|x|@f$.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.sign([-5., 4.5])
# array([-1.,  1.])
# >>> vp.sign(0)
# array(0)
# >>> vp.sign(5-2j)
# array(1+0j)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.heaviside
###########################################################################
divert(0)dnl
define(<--@macro_heaviside@-->,<--@dnl
macro_ufunc(<--@heaviside@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes Heaviside step function.@n 
# The Heaviside step function is defined as follows:@n
# @f{eqnarray*}{ 
#   heaviside(x_1, x_2)  = \left\{  \begin{array}{ll}
#                            0    & ( x_1 < 0 ) \\
#                            x_2  & ( x_1 = 0 ) \\
#                            1    & ( x_1 > 0 ) \\
#                           \end{array}\right.
# @f}
# @-->,dnl
<--@@em x1 : Input an array or a scalar.@n @em x2 : The value of the function when @em x1 is 0.@n If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# Heaviside step function of @em x1 and @em x2, element-wise. If @em x1 and @em x2 are both scalars, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.heaviside([-1.5, 0, 2.0], 0.5)
# array([ 0. ,  0.5,  1. ])
# >>> vp.heaviside([-1.5, 0, 2.0], 1)
# array([ 0.,  1.,  1.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.conj
###########################################################################
divert(0)dnl
define(<--@macro_conj@-->,<--@dnl
macro_ufunc(<--@conj@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Returns the element-wise complex conjugate.@n The complex conjugate of a complex number is obtained by changing the sign of its imaginary part.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the complex conjugate for each element in @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @note
# @ref ufuncs::conj is an alias for @ref ufuncs::conjugate
# @code
# import nlcpy as vp
# >>> vp.conj is vp.conjugate
# True
# @endcode
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.conjugate(1+2j)
# array(1-2j)
# @endcode@n
# @code
# >>> vp.conj(1+2j)
# >>> x = vp.eye(2) + 1j * vp.eye(2)
# >>> vp.conj(x)
# array([[ 1.-1.j,  0.-0.j],
#        [ 0.-0.j,  1.-1.j]])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.conjugate
###########################################################################
divert(0)dnl
define(<--@macro_conjugate@-->,<--@dnl
macro_ufunc(<--@conjugate@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Returns the element-wise complex conjugate.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the complex conjugate for each element in @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @note
# @ref ufuncs::conjugate is an alias for @ref ufuncs::conj
# @code
# import nlcpy as vp
# >>> vp.conj is vp.conjugate
# True
# @endcode
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.conjugate(1+2j)
# array(1-2j)
# @endcode@n
# @code
# >>> vp.conjugate(1+2j)
# >>> x = vp.eye(2) + 1j * vp.eye(2)
# >>> vp.conjugate(x)
# array([[ 1.-1.j,  0.-0.j],
#        [ 0.-0.j,  1.-1.j]])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.exp
###########################################################################
divert(0)dnl
define(<--@macro_exp@-->,<--@dnl
macro_ufunc(<--@exp@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise exponential of the input array.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the exponential for each element in @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @ref ufuncs::expm1 : Computes <span class="pre">exp(x) - 1</span> for all elements in the array.
# @ref ufuncs::exp2 : Computes <span class="pre">2**x</span> for all elements in the array.
dnl@-->,<--@dnl
# @note
# The irrational number e is also known as Euler's number. It is approximately 2.718281, and is the base of the natural logarithm, <span class="pre">ln</span> (this means that, if @f$x = \ln y = \log_e y@f$, then @f$e^x = y@f$. For real input, @f$exp(x)@f$ is always positive.
# For complex arguments, <span class="pre">x = a + ib</span>, we can write @f$e^x = e^a e^{ib}@f$. The first term, @f$e^a@f$, is already known (it is the real argument, described above). The second term, @f$e^{ib}@f$, is @f$\cos b + i \sin b@f$, a function with magnitude 1 and a periodic phase.
#
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.exp([1+2j, 3+4j, 5+6j])
# array([ -1.13120438 +2.47172667j, -13.12878308-15.20078446j,
#        142.50190552-41.46893679j])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.exp2
###########################################################################
divert(0)dnl
define(<--@macro_exp2@-->,<--@dnl
macro_ufunc(<--@exp2@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise 2 to the power of @em x.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing 2 to the power of @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::power : Computes the element-wise exponentiation of the inputs.
dnl@-->,<--@dnl
# @attention
# @li dtype is a complex dtype("complex64", "complex128") :  @em TypeError occurs.@n
#
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.exp2([2, 3])
# array([ 4.,  8.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.log
###########################################################################
divert(0)dnl
define(<--@macro_log@-->,<--@dnl
macro_ufunc(<--@log@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise natural logarithm of <em>x</em>.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the natural logarithm of @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::log10 : Computes the element-wise base-10 logarithm of <em>x</em>.
# @li @ref ufuncs::log2 : Computes the element-wise base-2 logarithm of <em>x</em>.
# @li @ref ufuncs::log1p : Computes the element-wise natural logarithm of <em>1+x</em>.
dnl@-->,<--@dnl
# @note
# @li Logarithm is a multivalued function: for each @em x there is an infinite number of @em z such that <em>exp(z) = x</em>. The convention is to return the @em z whose imaginary part lies in <em>[-pi, pi]</em>.
# @li For real-valued input data types, @ref ufuncs::log always returns real output. For each value that cannot be expressed as a real number or infinity, it yields nan and sets the @em invalid floating point error flag.
# @li For complex-valued input, @ref ufuncs::log is a complex analytical function that has a branch cut <em>[-inf, 0]</em> and is continuous from above on it. @ref ufuncs::log handles the floating-point negative zero as an infinitesimal negative number, conforming to the C99 standard.
#
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.log([1, vp.e, vp.e**2, 0])
# array([  0.,   1.,   2., -Inf])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.log2
###########################################################################
divert(0)dnl
define(<--@macro_log2@-->,<--@dnl
macro_ufunc(<--@log2@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise base-2 logarithm of <em>x</em>.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the base-2 logarithm of @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::log : Computes the element-wise natural logarithm of <em>x</em>.
# @li @ref ufuncs::log10 : Computes the element-wise base-10 logarithm of <em>x</em>.
# @li @ref ufuncs::log1p : Computes the element-wise natural logarithm of <em>1+x</em>.
dnl@-->,<--@dnl
# @attention
# @li dtype is a complex dtype("complex64", "complex128") :  @em TypeError occurs.@n
#
# @note
# @li Logarithm is a multivalued function: for each @em x there is an infinite number of @em z such that <em>2**z = x</em>. The convention is to return the @em z whose imaginary part lies in i<em>[-pi, pi]</em>.
# @li For real-valued input data types, @ref ufuncs::log2 always returns real output. For each value that cannot be expressed as a real number or infinity, it yields nan and sets the @em invalid floating point error flag.
#
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> x = vp.array([0, 1, 2, 2**4])
# >>> vp.log2(x)
# array([-Inf,   0.,   1.,   4.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.log10
###########################################################################
divert(0)dnl
define(<--@macro_log10@-->,<--@dnl
macro_ufunc(<--@log10@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise base-10 logarithm of <em>x</em>.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the base-10 logarithm of @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::log : Computes the element-wise natural logarithm of <em>x</em>.
# @li @ref ufuncs::log2 : Computes the element-wise base-2 logarithm of <em>x</em>.
# @li @ref ufuncs::log1p : Computes the element-wise natural logarithm of <em>1+x</em>.
dnl@-->,<--@dnl
# @attention
# @li dtype is a complex dtype("complex64", "complex128") :  @em TypeError occurs.@n
#
# @note
# @li Logarithm is a multivalued function: for each @em x there is an infinite number of @em z such that <em>10**z = x</em>. The convention is to return the @em z whose imaginary part lies in <em>[-pi, pi]</em>.
# @li For real-valued input data types, @ref ufuncs::log10 always returns real output. For each value that cannot be expressed as a real number or infinity, it yields nan and sets the @em invalid floating point error flag.
#
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.log10([1e-15, -3.])
# array([-15.,  nan])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.log1p
###########################################################################
divert(0)dnl
define(<--@macro_log1p@-->,<--@dnl
macro_ufunc(<--@log1p@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise natural logarithm of <em>1+x</em>.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the natural logarithm of <em>x+1</em>.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::expm1 : Computes the element-wise exponential minus one.
dnl@-->,<--@dnl
# @attention
# @li dtype is a complex dtype("complex64", "complex128") :  @em TypeError occurs.@n
#
# @note
# @li For real-valued input, @ref ufuncs::log1p is accurate also for @em x so small that <em>1 + x == 1</em> in floating-point accuracy.
# @li Logarithm is a multivalued function: for each @em x there is an infinite number of @em z such that <em>exp(z) = 1 + x</em>. The convention is to return the @em z whose imaginary part lies in <em>[-pi, pi]</em>.
# @li For real-valued input data types, @ref ufuncs::log1p always returns real output. For each value that cannot be expressed as a real number or infinity, it yields nan and sets the @em invalid floating point error flag.
#
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.log1p(1e-99)
# array(1e-99)
# >>> vp.log(1 + 1e-99)
# array(0.)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.expm1
###########################################################################
divert(0)dnl
define(<--@macro_expm1@-->,<--@dnl
macro_ufunc(<--@expm1@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise exponential minus one.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the exponential minus one: <em>y = exp(x) - 1</em>.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::log1p : Computes the element-wise natural logarithm of <em>1+x</em>.
dnl@-->,<--@dnl
# @note
# @li For real-valued input, @ref ufuncs::log1p is accurate also for @em x so small that <em>1 + x == 1</em> in floating-point accuracy.
# @li Logarithm is a multivalued function: for each @em x there is an infinite number of @em z such that <em>exp(z) = 1 + x</em>. The convention is to return the @em z whose imaginary part lies in <em>[-pi, pi]</em>.
# @li For real-valued input data types, @ref ufuncs::log1p always returns real output. For each value that cannot be expressed as a real number or infinity, it yields nan and sets the @em invalid floating point error flag.
#
# @attention
# @li dtype is a complex dtype("complex64", "complex128") :  @em TypeError occurs.@n
#
dnl@-->,<--@dnl
# @code
# The true value of <span class="pre">exp(1e-10) - 1</span> is 1.00000000005e-10 to about 32 significant digits. This example shows the superiority of expm1 in this case.
# >>> import nlcpy as vp
# >>> vp.set_printoptions(16) # change displying digits
# >>> vp.expm1(1e-10)
# array(1.00000000005e-10)
# >>> vp.exp(1e-10) - 1
# array(1.000000082740371e-10)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.sqrt
###########################################################################
divert(0)dnl
define(<--@macro_sqrt@-->,<--@dnl
macro_ufunc(<--@sqrt@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise square-root of the input.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the square-root for each element of @em x.
# If elements of @em x are real with negative elements, this function returns <span class="pre">nan</span> in @em y.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.sqrt([1,4,9])
# array([ 1.,  2.,  3.])
# @endcode@n
# @code
# >>> vp.sqrt([4, -1, -3+4j])
# array([ 2.+0.j,  0.+1.j,  1.+2.j])
# @endcode@n
# @code
# >>> vp.sqrt([4, -1, vp.inf])
# array([ 2., nan, inf])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.square
###########################################################################
divert(0)dnl
define(<--@macro_square@-->,<--@dnl
macro_ufunc(<--@square@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise square of the input.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the square <em>(x*x)</em>.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::sqrt : Computes the element-wise square-root of the input.
# @li @ref ufuncs::power : Computes the element-wise exponentiation of the inputs.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.square([-1j, 1])
# array([-1.+0.j,  1.+0.j])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.cbrt
###########################################################################
divert(0)dnl
define(<--@macro_cbrt@-->,<--@dnl
macro_ufunc(<--@cbrt@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise cubic-root of the input.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the cubic-root for each element of @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.cbrt([1,8,27])
# array([ 1.,  2.,  3.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.reciprocal
###########################################################################
divert(0)dnl
define(<--@macro_reciprocal@-->,<--@dnl
macro_ufunc(<--@reciprocal@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise reciprocal of the input.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the reciprocal for each element of @em x.
# If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::sqrt : Computes the element-wise square-root of the input.
# @li @ref ufuncs::power : Computes the element-wise exponentiation of the inputs.
dnl@-->,<--@dnl
# @note
# @li This function is not designed to work with integers.
# @li For integer arguments with absolute value larger than 1 the result is always zero because of the way Python handles integer division. For integer zero the result is an overflow.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.reciprocal(2.)
# array(0.5)
# >>> vp.reciprocal([1, 2., 3.33])
# array([1.       , 0.5      , 0.3003003])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.sin
###########################################################################
divert(0)dnl
define(<--@macro_sin@-->,<--@dnl
macro_ufunc(<--@sin@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise sine.@-->,dnl
<--@Input an array or a scalar in radians.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The sine values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::arcsin : Computes the element-wise inverse sine.
# @li @ref ufuncs::sinh : Computes the element-wise hyperbolic sine.
# @li @ref ufuncs::cos : Computes the element-wise cosine.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# Print sine of one angle:
# @code
# >>> import nlcpy as vp
# >>> vp.sin(vp.pi/2.)
# array(1.)
# @endcode@n
# Print sines of an array of angles given in degrees:
# @code
# >>> vp.sin(vp.array((0., 30., 45., 60., 90.)) * vp.pi / 180. )
# array([0.        , 0.5       , 0.70710678, 0.8660254 , 1.        ])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.cos
###########################################################################
divert(0)dnl
define(<--@macro_cos@-->,<--@dnl
macro_ufunc(<--@cos@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise cosine.@-->,dnl
<--@Input an array or a scalar in radians.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The cosine values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::arccos : Computes the element-wise inverse cosine.
# @li @ref ufuncs::cosh : Computes the element-wise hyperbolic cosine.
# @li @ref ufuncs::sin : Computes the element-wise sine.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.cos(vp.array([0, vp.pi/2, vp.pi]))
# array([ 1.000000e+00,  6.123234e-17, -1.000000e+00])
# >>>
# >>> # Example of providing the optional output parameter
# >>> out1 = vp.array([0], dtype='d')
# >>> out2 = vp.cos([0.1], out=out1)
# >>> out2 is out1
# True
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.tan
###########################################################################
divert(0)dnl
define(<--@macro_tan@-->,<--@dnl
macro_ufunc(<--@tan@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise tangent.@-->,dnl
<--@Input an array or a scalar in radians.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The tangent values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::arctan : Computes the element-wise inverse tangent.
# @li @ref ufuncs::tanh : Computes the element-wise hyperbolic tangent.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> from math import pi
# >>> vp.tan(vp.array([-pi,pi/2,pi]))
# array([ 1.22464659e-16,  1.63312422e+16, -1.22464659e-16])
# >>>
# >>> # Example of providing the optional output parameter illustrating
# >>> # that what is returned is a reference to said parameter
# >>> out1 = vp.array([0], dtype='d')
# >>> out2 = vp.tan([0.1], out=out1)
# >>> out2 is out1
# True
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.arcsin
###########################################################################
divert(0)dnl
define(<--@macro_arcsin@-->,<--@dnl
macro_ufunc(<--@arcsin@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise inverse sine.@n The inverse of @ref ufuncs::sin so that, if y = sin(x), then x = arcsin(y).@-->,dnl
<--@@em y-coordinate on the unit circle. If x is real, the domain is limited to [-1,1].@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The inverse sine values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::sin : Computes the element-wise sine.
# @li @ref ufuncs::arccos : Computes the element-wise inverse cosine.
# @li @ref ufuncs::arctan : Computes the element-wise inverse tangent.
# @li @ref ufuncs::arctan2 : Computes the element-wise inverse tangent of <span class="pre">x1/x2</span> choosing the quadrant correctly.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.arcsin([-1, 0, 1])
# array([-1.57079633,  0.        ,  1.57079633])    # [-pi/2, 0, pi/2]
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.arccos
###########################################################################
divert(0)dnl
define(<--@macro_arccos@-->,<--@dnl
macro_ufunc(<--@arccos@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise inverse cosine.@n The inverse of @ref ufuncs::cos so that, if y = cos(x), then x = arccos(y).@-->,dnl
<--@@em x-coordinate on the unit circle. If x is real, the domain is limited to [-1,1].@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The inverse cosine values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::cos : Computes the element-wise cosine.
# @li @ref ufuncs::arcsin : Computes the element-wise inverse sine.
# @li @ref ufuncs::arctan : Computes the element-wise inverse tangent.
# @li @ref ufuncs::arctan2 : Computes the element-wise inverse tangent of <span class="pre">x1/x2</span> choosing the quadrant correctly.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.arccos([1, -1])
# rray([0.        , 3.14159265])   # [0, pi]
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.arctan
###########################################################################
divert(0)dnl
define(<--@macro_arctan@-->,<--@dnl
macro_ufunc(<--@arctan@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise inverse tangent.@n The inverse of @ref ufuncs::tan so that, if y = tan(x), then x = arctan(y).@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The inverse tangent values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::tan : Computes the element-wise tangent.
# @li @ref ufuncs::arcsin : Computes the element-wise inverse sine.
# @li @ref ufuncs::arccos : Computes the element-wise inverse cosine.
# @li @ref ufuncs::arctan2 : Computes the element-wise inverse tangent of <span class="pre">x1/x2</span> choosing the quadrant correctly.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.arctan([0, 1])
# array([0.        , 0.78539816])    # [0, pi/4]
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.arctan2
###########################################################################
divert(0)dnl
define(<--@macro_arctan2@-->,<--@dnl
macro_ufunc(<--@arctan2@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise inverse tangent of @em x1/@em x2 choosing the quadrant correctly.@n This function is not defined for complex-valued arguments; for the so-called argument of complex values, use @ref math::angle.@-->,dnl
<--@The values of @em x1 are @em y-coordinates.@n Also, The values of @em x2 are @em x-coordinates.@n  @em x1 and @em x2 must be real. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval angle : <em>@ref n-dimensional_array "ndarray"</em>@n
# Array of angles in radians, in the range <span class="pre">[-pi, pi]</span>. If @em x1 and @em x2 are both scalars, 
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::arctan : Computes the element-wise inverse tangent.
# @li @ref ufuncs::tan : Computes the element-wise tangent.
# @li @ref math::angle : Returns the angle of the complex argument.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# Consider four points in different quadrants:
# @code
# >>> import nlcpy as vp
# >>> x = vp.array([-1, +1, +1, -1])
# >>> y = vp.array([-1, -1, +1, +1])
# >>> vp.arctan2(y, x) * 180 / vp.pi
# array([-135.,  -45.,   45.,  135.])
# @endcode@n
# Note the order of the parameters. universal_funcitons::arctan2 is defined also when @em x2 = 0 and at several other special points, obtaining values in the range <span class="pre">[-pi, pi]</span>:
# @code
# >>> vp.arctan2([1., -1.], [0., 0.])
# array([ 1.57079633, -1.57079633])
# >>> vp.arctan2([0., 0., vp.inf], [+0., -0., vp.inf])
# array([0.        , 3.14159265, 0.78539816])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.hypot
###########################################################################
divert(0)dnl
define(<--@macro_hypot@-->,<--@dnl
macro_ufunc(<--@hypot@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the "legs" of a right triangle.@n Equivalent to sqrt(x1**2 + x2**2), element-wise.@-->,dnl
<--@Leg of the triangle(s). If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval z : <em>@ref n-dimensional_array "ndarray"</em>@n
# The leg of the triangle(s). If @em x1 and @em x2 are both scalars, 
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::arctan : Computes the element-wise inverse tangent.
# @li @ref ufuncs::tan : Computes the element-wise tangent.
# @li @ref math::angle : Returns the angle of the complex argument.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# Consider four points in different quadrants:
# @code
# >>> import nlcpy as vp
# >>> vp.hypot(3*vp.ones((3, 3)), 4*vp.ones((3, 3)))
# array([[ 5.,  5.,  5.],
#        [ 5.,  5.,  5.],
#        [ 5.,  5.,  5.]])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.sinh
###########################################################################
divert(0)dnl
define(<--@macro_sinh@-->,<--@dnl
macro_ufunc(<--@sinh@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise hyperbolic sine.@n Equivalent to <span class="pre">1/2 * (vp.exp(x) - vp.exp(-x))</span> or <span class="pre">-1j * vp.sin(1j*x)</span>.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The hyperbolic sine values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::arcsinh : Computes the element-wise inverse hyperbolic sine.
# @li @ref ufuncs::cosh : Computes the element-wise hyperbolic cosine. 
# @li @ref ufuncs::tanh : Computes the element-wise hyperbolic tangent.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.sinh(0)
# array(0.)
# >>> vp.sinh(vp.pi*1j/2)
# array(1j)
# >>> vp.sinh(vp.pi*1j) # (exact value is 0)
# array(1.2246063538223773e-016j)
# @endcode@n
# @code
# >>>>>> # Example of providing the optional output parameter
# >>> out1 = vp.array([0], dtype='d')
# >>> out2 = vp.sinh([0.1], out1)
# >>> out2 is out1
# True
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.cosh
###########################################################################
divert(0)dnl
define(<--@macro_cosh@-->,<--@dnl
macro_ufunc(<--@cosh@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise hyperbolic cosine.@n Equivalent to <span class="pre">1/2 * (vp.exp(x) + vp.exp(-x))</span> or <span class="pre">vp.cos(1j*x)</span>.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The hyperbolic cosine values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::arccosh : Computes the element-wise inverse hyperbolic cosine.
# @li @ref ufuncs::sinh : Computes the element-wise hyperbolic sine.
# @li @ref ufuncs::tanh : Computes the element-wise hyperbolic tangent.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.sinh(0)
# array(1.)
# >>> vp.sinh(vp.pi*1j/2)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.tanh
###########################################################################
divert(0)dnl
define(<--@macro_tanh@-->,<--@dnl
macro_ufunc(<--@tanh@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise hyperbolic tangent.@n Equivalent to <span class="pre">vp.sinh(x)/vp.cosh(x)i</span> or <span class="pre">-1j * vp.tan(1j*x)</span>.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The hyperbolic tangent values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::arctanh : Computes the element-wise inverse hyperbolic tangent.
# @li @ref ufuncs::sinh : Computes the element-wise hyperbolic sine.
# @li @ref ufuncs::cosh : Computes the element-wise hyperbolic cosine.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.tanh((0, vp.pi*1j, vp.pi*1j/2))
# array([0.+0.00000000e+00j, 0.-1.22464680e-16j, 0.+1.63312394e+16j])
# @endcode@n
# @code
# >>> # Example of providing the optional output parameter
# >>> out1 = vp.array([0], dtype='d')
# >>> out2 = vp.tanh([0.1], out1)
# >>> out2 is out1
# True
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.arcsinh
###########################################################################
divert(0)dnl
define(<--@macro_arcsinh@-->,<--@dnl
macro_ufunc(<--@arcsinh@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise inverse hyperbolic sine.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The inverse hyperbolic sine values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::sinh : Computes the element-wise hyperbolic sine.
# @li @ref ufuncs::arccosh : Computes the element-wise inverse hyperbolic cosine.
# @li @ref ufuncs::arctanh : Computes the element-wise inverse hyperbolic tangent.
dnl@-->,<--@dnl
# @note
# @li @ref ufuncs::arcsinh is a multivalued function: for each @em x there are infinitely many numbers @em z such that <em> sinh(z) = x</em>. The convention is to return the @em z whose imaginary part lies in <em>[-pi/2, pi/2]</em>.
# @li For real-valued input data types, @ref ufuncs::arcsinh always returns real output. For each value that cannot be expressed as a real number or infinity, it returns @c nan and sets the @em invalid floating point error flag.
# @li For complex-valued input, @ref ufuncs::arcsinh is a complex analytical function that has branch cuts <em>[1j, infj]</em> and <em>[-1j, -infj]</em> and is continuous from the right on the former and from the left on the latter.
#
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.arcsinh(vp.array([vp.e, 10.0]))
# array([ 1.72538256,  2.99822295])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.arccosh
###########################################################################
divert(0)dnl
define(<--@macro_arccosh@-->,<--@dnl
macro_ufunc(<--@arccosh@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise inverse hyperbolic cosine.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The inverse hyperbolic cosine values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::cosh : Computes the element-wise hyperbolic cosine.
# @li @ref ufuncs::arcsinh : Computes the element-wise inverse hyperbolic sine.
# @li @ref ufuncs::arctanh : Computes the element-wise inverse hyperbolic tangent.
dnl@-->,<--@dnl
# @note
# @li @ref ufuncs::arccosh is a multivalued function: for each @em x there are infinitely many numbers @em z such that <em> cosh(z) = x</em>. The convention is to return the @em z whose imaginary part lies in <em>[-pi, pi]</em> and the real part in <em>[-0, inf]</em>.
# @li For real-valued input data types, @ref ufuncs::arccosh always returns real output. For each value that cannot be expressed as a real number or infinity, it returns @c nan and sets the @em invalid floating point error flag.
# @li For complex-valued input, @ref ufuncs::arccosh is a complex analytical function that has a branch cut <em>[-inf, 1]</em> and is continuous from above on it.
#
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.arccosh([vp.e, 10.0])
# array([ 1.65745445,  2.99322285])
# >>> vp.arccosh(1)
# array(0.)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.arctanh
###########################################################################
divert(0)dnl
define(<--@macro_arctanh@-->,<--@dnl
macro_ufunc(<--@arctanh@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the element-wise inverse hyperbolic tangent.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The inverse hyperbolic tangent values for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::tanh : Computes the element-wise hyperbolic tangent.
# @li @ref ufuncs::arcsinh : Computes the element-wise inverse hyperbolic sine.
# @li @ref ufuncs::arccosh : Computes the element-wise inverse hyperbolic cosine.
dnl@-->,<--@dnl
# @note
# @li @ref ufuncs::arctanh is a multivalued function: for each @em x there are infinitely many numbers @em z such that <em> tanh(z) = x</em>. The convention is to return the @em z whose imaginary part lies in <em>[-pi/2, pi/2]</em>.
# @li For real-valued input data types, @ref ufuncs::arctanh always returns real output. For each value that cannot be expressed as a real number or infinity, it returns @c nan and sets the @em invalid floating point error flag.
# @li For complex-valued input, @ref ufuncs::arctanh is a complex analytical function that has branch cuts <em>[-1, -inf]</em> and <em>[1, inf]</em> and is continuous from above on the former and from below on the latter.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.arctanh([0, -0.5])
# array([ 0.        , -0.54930614])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.deg2rad
###########################################################################
divert(0)dnl
define(<--@macro_deg2rad@-->,<--@dnl
macro_ufunc(<--@deg2rad@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Converts angles from degrees to radians.@-->,dnl
<--@Input an array or a scalar containing angles in degrees.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The angles for each element of @em x. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
# The angles in radians. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::rad2deg : Converts angles from radians to degrees.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.deg2rad(180)
# array(3.14159265)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.rad2deg
###########################################################################
divert(0)dnl
define(<--@macro_rad2deg@-->,<--@dnl
macro_ufunc(<--@rad2deg@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Converts angles from radians to degrees.@-->,dnl
<--@Input an array or a scalar, containing angles in radians.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The angles in degrees. If @em x is a scalar, this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::deg2rad : Converts angles from degrees to radians.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.rad2deg(vp.pi/2)
# array(90.)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.bitwise_and
###########################################################################
divert(0)dnl
define(<--@macro_bitwise_and@-->,<--@dnl
macro_ufunc(<--@bitwise_and@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the bit-wise AND of two arrays element-wise.@n This ufunc implements the C/Python operator <span class="pre">&</span>.@-->,dnl
<--@Only integer and boolean types are handled. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# <em>y = x1 & x2</em>. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::logical_and : Computes the logical AND of two arrays element-wise.
# @li @ref ufuncs::bitwise_or : Computes the bit-wise OR of two arrays element-wise.
# @li @ref ufuncs::bitwise_xor : Computes the bit-wise XOR of two arrays element-wise.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# The number 13 is represented by 00001101. Likewise, 17 is represented by 00010001. The bit-wise AND of 13 and 17 is therefore 000000001, or 1:
# @code
# >>> import nlcpy as vp
# >>> vp.bitwise_and(13, 17)
# array(1)
# @endcode@n
# @code
# >>> vp.bitwise_and(14, 13)
# array(12)
# >>> vp.bitwise_and([14,3], 13)
# array([12,  1])
# @endcode@n
# @code
# >>> vp.bitwise_and([11,7], [4,25])
# array([0, 1])
# >>> vp.bitwise_and(vp.array([2,5,255]), vp.array([3,14,16]))
# array([ 2,  4, 16])
# >>> vp.bitwise_and([True, True], [False, True])
# array([False,  True])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.bitwise_or
###########################################################################
divert(0)dnl
define(<--@macro_bitwise_or@-->,<--@dnl
macro_ufunc(<--@bitwise_or@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the bit-wise OR of two arrays element-wise.@n This ufunc implements the C/Python operator <span class="pre">|</span>.@-->,dnl
<--@Only integer and boolean types are handled. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# <em>y = x1 | x2</em>. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::logical_or : Computes the logical OR of two arrays element-wise.
# @li @ref ufuncs::bitwise_and : Computes the bit-wise AND of two arrays element-wise.
# @li @ref ufuncs::bitwise_xor : Computes the bit-wise XOR of two arrays element-wise.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# The number 13 has the binaray representation 00001101. Likewise, 16 is represented by 00010000. The bit-wise OR of 13 and 16 is then 000111011, or 29:
# @code
# >>> import nlcpy as vp
# >>> vp.bitwise_or(13, 16)
# array(29)
# @endcode@n
# @code
# >>> vp.bitwise_or(32, 2)
# array(34)
# >>> vp.bitwise_or([33,3], 1)
# array([33,  5])
# >>> vp.bitwise_or([33, 4], [1, 2])
# array([33,  6])
# @endcode@n
# @code
# >>> vp.bitwise_or(vp.array([2, 5, 255]), vp.array([4, 4, 4]))
# array([  6,   5, 255])
# >>> vp.array([2, 5, 255]) | vp.array([4, 4, 4])
# array([  6,   5, 255])
# >>> vp.bitwise_or(vp.array([2, 5, 255, 2147483647], dtype=vp.int32),
# ...               vp.array([4, 4, 4, 2147483647], dtype=vp.int32))
# array([         6,          5,        255, 2147483647])
# >>> vp.bitwise_or([True, True], [False, True])
# array([ True,  True])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.bitwise_xor
###########################################################################
divert(0)dnl
define(<--@macro_bitwise_xor@-->,<--@dnl
macro_ufunc(<--@bitwise_xor@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the bit-wise XOR of two arrays element-wise.@n This ufunc implements the C/Python operator <span class="pre">^</span>.@-->,dnl
<--@Only integer and boolean types are handled. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# <em>y = x1 ^ x2</em>. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::logical_xor : Computes the logical XOR of two arrays element-wise.
# @li @ref ufuncs::bitwise_and : Computes the bit-wise AND of two arrays element-wise.
# @li @ref ufuncs::bitwise_or : Computes the bit-wise OR of two arrays element-wise.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# The number 13 is represented by 00001101. Likewise, 17 is represented by 00010001. The bit-wise XOR of 13 and 17 is therefore 00011100, or 28:
# @code
# >>> import nlcpy as vp
# >>> vp.bitwise_and(13, 17)
# array(28)
# @endcode@n
# @code
# >>> vp.bitwise_xor(31, 5)
# 26
# >>> vp.bitwise_xor([31,3], 5)
# array([26,  6])
# @endcode@n
# @code
# >>> vp.bitwise_xor([31,3], [5,6])
# array([26,  5])
# >>> vp.bitwise_xor([True, True], [False, True])
# array([ True, False])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.invert
###########################################################################
divert(0)dnl
define(<--@macro_invert@-->,<--@dnl
macro_ufunc(<--@invert@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the bit-wise NOT element-wise.@n This ufunc implements the C/Python operator <span class="pre">~</span>.@-->,dnl
<--@Only integer and boolean types are handled. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# <em>y = ~x</em>. If @em x is a scalar,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::bitwise_and : Computes the bit-wise AND of two arrays element-wise.
# @li @ref ufuncs::bitwise_or : Computes the bit-wise OR of two arrays element-wise.
# @li @ref ufuncs::bitwise_xor : Computes the bit-wise XOR of two arrays element-wise.
# @li @ref ufuncs::logical_not : Computes the logical NOT of the input array element-wise.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# We've seen that 13 is represented by 00001101. The invert or bit-wise NOT of 13 is then:
# @code
# >>> x = vp.invert(vp.array(13, dtype=vp.uint8))
# >>> x
# array(242)
# @endcode@n
# When using signed integer types the result is the two's complement of the result for the unsigned type:
# @code
# >>> vp.invert(vp.array([13], dtype=vp.int8))
# array([-14], dtype=int8)
# @endcode@n
# Booleans are accepted as well:
# @code
# >>> vp.invert(vp.array([True, False]))
# array([False,  True])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.left_shift
###########################################################################
divert(0)dnl
define(<--@macro_left_shift@-->,<--@dnl
macro_ufunc(<--@left_shift@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Shifts bits of an integer to the left, element-wise.@n Bits are shifted to the left by appending 0 at the right of @em x1. Because the internal representation of integer numbers is in binary format, this operation is equivalent to multiplying @f$ x1*2^{x2}@f$.@-->,dnl
<--@@em x1 : Input an array or a scalar.@n @em x2 : Number of zeros to append to @em x1. @em x2 has to be non-negative. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# @em x1 with bits shifted @em x2 times to the left. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @note
# @li If the values of @em x2 are greater equal than the bit-width of @em x1, this function returns zero.
# @li If the values of @em x2 are negative numbers, undefined values are returned.
#
# @sa
# @li @ref ufuncs::right_shift : Shifts bits of an integer to the right, element-wise.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.left_shift(5, 2)
# array(20)
# @endcode@n
# @code
# >>> vp.left_shift(5, [1,2,3])
# array([10, 20, 40])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.right_shift
###########################################################################
divert(0)dnl
define(<--@macro_right_shift@-->,<--@dnl
macro_ufunc(<--@right_shift@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Shifts bits of an integer to the right, element-wise.@n Because the internal representation of numbers is in binary format, this operation is equivalent to multiplying @f$ x1/2^{x2}@f$.@-->,dnl
<--@@em x1 : Input an array or a scalar.@n @em x2 : Number of bits to remove at the right of @em x1. @em x2 has to be non-negative. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# @em x1 with bits shifted @em x2 times to the right. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @note
# @li If the values of @em x2 are greater equal than the bit-width of @em x1, this function returns zero.
# @li If the values of @em x2 are negative numbers, undefined values are returned.
#
# @sa
# @li @ref ufuncs::left_shift : Shifts bits of an integer to the left, element-wise.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.right_shift(10, 1)
# array(5)
# @endcode@n
# @code
# >>> vp.right_shift(10, [1,2,3])
# array([5, 2, 1])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.greater
###########################################################################
divert(0)dnl
define(<--@macro_greater@-->,<--@dnl
macro_ufunc(<--@greater@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Returns (@em x1 > @em x2), element-wise.@-->,dnl
<--@Input arrays or scalars. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the result of the element-wise comparison of @em x1 and @em x2; the shape is determined by broadcasting. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::greater_equal : Returns (@em x1 >= @em x2), element-wise. 
# @li @ref ufuncs::less : Returns (@em x1 < @em x2), element-wise. 
# @li @ref ufuncs::less_equal : Returns (@em x1 <= @em x2), element-wise. 
# @li @ref ufuncs::not_equal : Returns (@em x1 != @em x2), element-wise. 
# @li @ref ufuncs::equal : Returns (@em x1 == @em x2), element-wise. 
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.greater([4,2],[2,2])
# array([ True, False])
# @endcode@n
# If the inputs are ndarrays, then vp.greater is equivalent to '>'.
# @code
# >>> a = vp.array([4,2])
# >>> b = vp.array([2,2])
# >>> a > b
# array([ True, False])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.greater_equal
###########################################################################
divert(0)dnl
define(<--@macro_greater_equal@-->,<--@dnl
macro_ufunc(<--@greater_equal@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Returns (@em x1 >= @em x2), element-wise.@-->,dnl
<--@Input arrays or scalars. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the result of the element-wise comparison of @em x1 and @em x2; the shape is determined by broadcasting. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::greater : Returns (@em x1 > @em x2), element-wise. 
# @li @ref ufuncs::less : Returns (@em x1 < @em x2), element-wise. 
# @li @ref ufuncs::less_equal : Returns (@em x1 <= @em x2), element-wise. 
# @li @ref ufuncs::not_equal : Returns (@em x1 != @em x2), element-wise. 
# @li @ref ufuncs::equal : Returns (@em x1 == @em x2), element-wise. 
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.greater_equal([4, 2, 1], [2, 2, 2])
# array([ True, True, False])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.less
###########################################################################
divert(0)dnl
define(<--@macro_less@-->,<--@dnl
macro_ufunc(<--@less@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Returns (@em x1 < @em x2), element-wise.@-->,dnl
<--@Input arrays or scalars. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the result of the element-wise comparison of @em x1 and @em x2; the shape is determined by broadcasting. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::greater : Returns (@em x1 > @em x2), element-wise. 
# @li @ref ufuncs::greater_equal : Returns (@em x1 >= @em x2), element-wise. 
# @li @ref ufuncs::less_equal : Returns (@em x1 <= @em x2), element-wise. 
# @li @ref ufuncs::not_equal : Returns (@em x1 != @em x2), element-wise. 
# @li @ref ufuncs::equal : Returns (@em x1 == @em x2), element-wise. 
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.less([1, 2], [2, 2])
# array([ True, False])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.less_equal
###########################################################################
divert(0)dnl
define(<--@macro_less_equal@-->,<--@dnl
macro_ufunc(<--@less_equal@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Returns (@em x1 <= @em x2), element-wise.@-->,dnl
<--@Input arrays or scalars. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the result of the element-wise comparison of @em x1 and @em x2; the shape is determined by broadcasting. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::greater : Returns (@em x1 > @em x2), element-wise. 
# @li @ref ufuncs::greater_equal : Returns (@em x1 >= @em x2), element-wise. 
# @li @ref ufuncs::less : Returns (@em x1 < @em x2), element-wise. 
# @li @ref ufuncs::not_equal : Returns (@em x1 != @em x2), element-wise. 
# @li @ref ufuncs::equal : Returns (@em x1 == @em x2), element-wise. 
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.less_equal([4, 2, 1], [2, 2, 2])
# array([False,  True,  True])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.not_equal
###########################################################################
divert(0)dnl
define(<--@macro_not_equal@-->,<--@dnl
macro_ufunc(<--@not_equal@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Returns (@em x1 != @em x2), element-wise.@-->,dnl
<--@Input arrays or scalars. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the result of the element-wise comparison of @em x1 and @em x2; the shape is determined by broadcasting. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::equal : Returns (@em x1 == @em x2), element-wise. 
# @li @ref ufuncs::greater_equal : Returns (@em x1 >= @em x2), element-wise. 
# @li @ref ufuncs::less_equal : Returns (@em x1 <= @em x2), element-wise. 
# @li @ref ufuncs::greater : Returns (@em x1 > @em x2), element-wise. 
# @li @ref ufuncs::less : Returns (@em x1 < @em x2), element-wise. 
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.not_equal([1.,2.], [1., 3.])
# array([False,  True])
# >>> vp.not_equal([1, 2], [[1, 3],[1, 4]])
# array([[False,  True],
#        [False,  True]])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.equal
###########################################################################
divert(0)dnl
define(<--@macro_equal@-->,<--@dnl
macro_ufunc(<--@equal@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Returns (@em x1 == @em x2), element-wise.@-->,dnl
<--@Input arrays or scalars. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# A ndarray, containing the result of the element-wise comparison of @em x1 and @em x2; the shape is determined by broadcasting. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::not_equal : Returns (@em x1 != @em x2), element-wise. 
# @li @ref ufuncs::greater_equal : Returns (@em x1 >= @em x2), element-wise. 
# @li @ref ufuncs::less_equal : Returns (@em x1 <= @em x2), element-wise. 
# @li @ref ufuncs::greater : Returns (@em x1 > @em x2), element-wise. 
# @li @ref ufuncs::less : Returns (@em x1 < @em x2), element-wise. 
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.equal([0, 1, 3], vp.arange(3))
# array([ True,  True, False])
# @endcode@n
# What is compared are values, not types. So an int (1) and an array of length one can evaluate as True:
# @code
# >>> vp.equal(1, vp.ones(1))
# array([ True])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.logical_and
###########################################################################
divert(0)dnl
define(<--@macro_logical_and@-->,<--@dnl
macro_ufunc(<--@logical_and@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the logical AND of two arrays element-wise.@-->,dnl
<--@Input arrays or scalars. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# Boolean result of the logical AND operation applied to the elements of @em x1 and @em x2; the shape is determined by broadcasting. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::logical_or : Computes the logical OR of two arrays element-wise.
# @li @ref ufuncs::logical_not : Computes the logical NOT of the input array element-wise.
# @li @ref ufuncs::logical_xor : Computes the logical XOR of two arrays element-wise.
# @li @ref ufuncs::bitwise_and : Computes the bit-wise AND of two arrays element-wise.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.logical_and(True, False)
# array(False)
# >>> vp.logical_and([True, False], [False, False])
# array([False, False])
# @endcode@n
# @code
# >>> x = vp.arange(5)
# >>> vp.logical_and(x>1, x<4)
# array([False, False,  True,  True, False])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.logical_or
###########################################################################
divert(0)dnl
define(<--@macro_logical_or@-->,<--@dnl
macro_ufunc(<--@logical_or@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the logical OR of two arrays element-wise.@-->,dnl
<--@Input arrays or scalars. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# Boolean result of the logical OR operation applied to the elements of @em x1 or @em x2; the shape is determined by broadcasting. If both @em x1 or @em x2 are scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::logical_and : Computes the logical AND of two arrays element-wise.
# @li @ref ufuncs::logical_not : Computes the logical NOT of the input array element-wise.
# @li @ref ufuncs::logical_xor : Computes the logical XOR of two arrays element-wise.
# @li @ref ufuncs::bitwise_and : Computes the bit-wise AND of two arrays element-wise.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.logical_or(True, False)
# array(True)
# >>> vp.logical_or([True, False], [False, False])
# array([True, False])
# @endcode@n
# @code
# >>> x = vp.arange(5)
# >>> x = vp.arange(5)
# >>> vp.logical_or(x < 1, x > 3)
# array([ True, False, False, False,  True])
# @endcode@n
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.logical_xor
###########################################################################
divert(0)dnl
define(<--@macro_logical_xor@-->,<--@dnl
macro_ufunc(<--@logical_xor@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the logical XOR of two arrays element-wise.@-->,dnl
<--@Input arrays or scalars. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# Boolean result of the logical XOR operation applied to the elements of @em x1 and @em x2; the shape is determined by broadcasting. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::logical_or : Computes the logical OR of two arrays element-wise.
# @li @ref ufuncs::logical_not : Computes the logical NOT of the input array element-wise.
# @li @ref ufuncs::logical_xor : Computes the logical XOR of two arrays element-wise.
# @li @ref ufuncs::bitwise_and : Computes the bit-wise AND of two arrays element-wise.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.logical_xor(True, False)
# array(True)
# >>> vp.logical_xor([True, True, False, False], [True, False, True, False])
# array([False,  True,  True, False])
# @endcode@n
# @code
# >>> x = vp.arange(5)
# >>> vp.logical_xor(x < 1, x > 3)
# array([ True, False, False, False,  True])
# @endcode@n
# Simple example showing support of broadcasting
# @code
# >>> vp.logical_xor(0, vp.eye(2))
# array([[ True, False],
#        [False,  True]])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.logical_not
###########################################################################
divert(0)dnl
define(<--@macro_logical_not@-->,<--@dnl
macro_ufunc(<--@logical_not@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Computes the logical NOT of the input array element-wise.@-->,dnl
<--@Logical NOT is applied to the elements of @em x.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# Boolean result with the same shape as @em x of the logical NOT operation on elements of @em x. If @em x is a scalar,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::logical_and : Computes the logical AND of two arrays element-wise.
# @li @ref ufuncs::logical_or : Computes the logical OR of two arrays element-wise.
# @li @ref ufuncs::logical_xor : Computes the logical XOR of two arrays element-wise.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.logical_not(3)
# array(False)
# >>> vp.logical_not([True, False, 0, 1])
# array([False,  True,  True, False])
# @endcode@n
# @code
# >>> x = vp.arange(5)
# >>> vp.logical_not(x<3)
# array([False, False, False,  True,  True])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.maximum
###########################################################################
divert(0)dnl
define(<--@macro_maximum@-->,<--@dnl
macro_ufunc(<--@maximum@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise maximum of the inputs.@n Compare two arrays and returns a new array containing the element-wise maxima. If one of the elements being compared is a NaN, then that element is returned. If both elements are NaNs then the first is returned. The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.@-->,dnl
<--@Input arrays or scalars, containing the elements to be compared. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The maximum of @em x1 and @em x2, element-wise. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::minimum : Computes the element-wise minimum of the inputs. Computes the element-wise minimum of the inputs.
# @li @ref ufuncs::fmax : Computes the element-wise maximum of the inputs
# @li @ref order::amax : Returns the maximum of an array or maximum along an axis.
# @li @ref order::nanmax : Returns the maximum of an array or maximum along an axis, ignoring any NaNs.
dnl@-->,<--@dnl
# @note
# The maximum is equivalent to <span class="pre">nlcpy.where(x1 >= x2, x1, x2)</span> when neither @em x1 nor @em x2 are nans, but it is faster and does proper broadcasting.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.maximum([2, 3, 4], [1, 5, 2])
# array([2, 5, 4])
# @endcode@n
# @code
# >>> vp.maximum(vp.eye(2), [0.5, 2]) # broadcasting
# array([[ 1. ,  2. ],
#        [ 0.5,  2. ]])
# @endcode@n
# @code
# >>> vp.maximum([vp.nan, 0, vp.nan], [0, vp.nan, vp.nan])
# array([nan, nan, nan])
# >>> vp.maximum(vp.Inf, 1)
# array(inf)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.minimum
###########################################################################
divert(0)dnl
define(<--@macro_minimum@-->,<--@dnl
macro_ufunc(<--@minimum@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise minimum of the inputs.@n Compare two arrays and returns a new array containing the element-wise minima. If one of the elements being compared is a NaN, then that element is returned. If both elements are NaNs then the first is returned. The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.@-->,dnl
<--@Input arrays or scalars, containing the elements to be compared. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The minimum of @em x1 and @em x2, element-wise. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::maximum : Computes the element-wise maximum of the inputs.
# @li @ref ufuncs::fmin : Computes the element-wise minimum of the inputs
# @li @ref order::amin : Returns the minimum of an array or minimum along an axis.
# @li @ref order::nanmin : Returns the minimum of an array or minimum along an axis, ignoring any NaNs.
dnl@-->,<--@dnl
# @note
# The mainmum is equivalent to <span class="pre">nlcpy.where(x1 <= x2, x1, x2)</span> when neither @em x1 nor @em x2 are nans, but it is faster and does proper broadcasting.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.minimum([2, 3, 4], [1, 5, 2])
# array([1, 3, 2])
# @endcode@n
# @code
# >>> vp.minimum(vp.eye(2), [0.5, 2]) # broadcasting
# array([[ 0.5,  0. ],
#        [ 0. ,  1. ]])
# @endcode@n
# @code
# >>> vp.minimum([vp.nan, 0, vp.nan],[0, vp.nan, vp.nan])
# array([nan, nan, nan])
# >>> vp.minimum(-vp.Inf, 1)
# array(-inf)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.fmax
###########################################################################
divert(0)dnl
define(<--@macro_fmax@-->,<--@dnl
macro_ufunc(<--@fmax@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise maximum of the inputs.@n Compare two arrays and returns a new array containing the element-wise maxima. Compare two arrays and returns a new array containing the element-wise maxima. If one of the elements being compared is a NaN, then the non-nan element is returned. If both elements are NaNs then the first is returned. The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are ignored when possible.@-->,dnl
<--@Input arrays or scalars, containing the elements to be compared. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The maximum of @em x1 and @em x2, element-wise. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::fmin : Computes the element-wise minimum of the inputs
# @li @ref ufuncs::maximum : Computes the element-wise maximum of the inputs.
# @li @ref order::amax : Returns the maximum of an array or maximum along an axis.
# @li @ref order::nanmax : Returns the maximum of an array or maximum along an axis, ignoring any NaNs.
dnl@-->,<--@dnl
# @note
# The fmax is equivalent to <span class="pre">nlcpy.where(x1 >= x2, x1, x2)</span> when neither @em x1 nor @em x2 are nans, but it is faster and does proper broadcasting.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.fmax([2, 3, 4], [1, 5, 2])
# array([2, 5, 4])
# @endcode@n
# @code
# >>> vp.fmax(vp.eye(2), [0.5, 2]) # broadcasting
# array([[ 1. ,  2. ],
#        [ 0.5,  2. ]])
# @endcode@n
# @code
# >>> vp.fmax([vp.nan, 0, vp.nan], [0, vp.nan, vp.nan])
# array([0., 0., nan])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.fmin
###########################################################################
divert(0)dnl
define(<--@macro_fmin@-->,<--@dnl
macro_ufunc(<--@fmin@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Computes the element-wise minimum of the inputs.@n Compare two arrays and returns a new array containing the element-wise minima. If one of the elements being compared is a NaN, then the non-nan element is returned. If both elements are NaNs then the first is returned. The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are ignored when possible.@-->,dnl
<--@Input arrays or scalars, containing the elements to be compared. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The fmin of @em x1 and @em x2, element-wise. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::fmax : Computes the element-wise maximum of the inputs
# @li @ref ufuncs::minimum : Computes the element-wise minimum of the inputs.
# @li @ref order::amin : Returns the minimum of an array or minimum along an axis.
# @li @ref order::nanmin : Returns the minimum of an array or minimum along an axis, ignoring any NaNs.
dnl@-->,<--@dnl
# @note
# The mainmum is equivalent to <span class="pre">nlcpy.where(x1 <= x2, x1, x2)</span> when neither @em x1 nor @em x2 are nans, but it is faster and does proper broadcasting.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.fmin([2, 3, 4], [1, 5, 2])
# array([1, 3, 2])
# @endcode@n
# @code
# >>> vp.fmin(vp.eye(2), [0.5, 2]) # broadcasting
# array([[ 0.5,  0. ],
#        [ 0. ,  1. ]])
# @endcode@n
# @code
# >>> vp.fmin([vp.nan, 0, vp.nan],[0, vp.nan, vp.nan])
# array([ 0.,  0., nan])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.isfinite
###########################################################################
divert(0)dnl
define(<--@macro_isfinite@-->,<--@dnl
macro_ufunc(<--@isfinite@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Tests whether input elements are neither @c inf nor @c nan, or not.@n The result is returned as a boolean array.@-->,dnl
<--@Input an array or a scalar, containing the elements to be tested.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# @c True where x is not positive infinity, negative infinity, or NaN; @c False otherwise. If @em x is a scalar,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::isinf : Tests whether input elements are @c inf, or not.
# @li @ref ufuncs::isnan : Tests whether input elements are @c nan, or not.
dnl@-->,<--@dnl
# @note
# Not a Number, positive infinity and negative infinity are considered to be non-finite.@n
# NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE754). This means that Not a Number is not equivalent to infinity. Also that positive infinity is not equivalent to negative infinity. But infinity is equivalent to positive infinity.
#
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.isfinite(1)
# array(True)
# >>> vp.isfinite(0)
# array(True)
# >>> vp.isfinite(vp.nan)
# array(False)
# >>> vp.isfinite(vp.inf)
# array(False)
# >>> vp.isfinite(vp.NINF)
# array(False)
# @endcode@n
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.isinf
###########################################################################
divert(0)dnl
define(<--@macro_isinf@-->,<--@dnl
macro_ufunc(<--@isinf@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Tests whether input elements are @c inf, or not.@n The result is returned as a boolean array.@-->,dnl
<--@Input an array or a scalar, containing the elements to be tested.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# @c True where x is not positive or negative infinity, @c False otherwise. If @em x is a scalar,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::isfinite : Tests whether input elements are neither @c inf nor @c nan, or not.
# @li @ref ufuncs::isnan : Tests whether input elements are @c nan, or not.
dnl@-->,<--@dnl
# @note
# NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE754).
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.isinf(vp.inf)
# array(True)
# >>> vp.isinf(vp.nan)
# array(False)
# >>> vp.isinf(vp.NINF)
# array(True)
# >>> vp.isinf([vp.inf, -vp.inf, 1.0, vp.nan])
# array([ True,  True, False, False])
# @endcode@n
# @code
# >>> x = vp.array([-vp.inf, 0., vp.inf])
# >>> y = vp.array([2, 2, 2])
# >>> vp.isinf(x)
# array([ True, False,  True])
# >>> vp.isinf(y)
# array([False, False, False])
# @endcode@n
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.isnan
###########################################################################
divert(0)dnl
define(<--@macro_isnan@-->,<--@dnl
macro_ufunc(<--@isnan@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Tests whether input elements are @c nan, or not.@n The result is returned as a boolean array.@-->,dnl
<--@Input an array or a scalar, containing the elements to be tested.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# @c True where x is @c nan, @c False otherwise. If @em x is a scalar,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::isfinite : Tests whether input elements are neither @c inf nor @c nan, or not.
# @li @ref ufuncs::isinf : Tests whether input elements are @c inf, or not.
dnl@-->,<--@dnl
# @note
# NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE754).
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.isnan(vp.nan)
# array(True)
# >>> vp.isnan(vp.inf)
# array(False)
# >>> vp.isnan(1)
# array(False)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.signbit
###########################################################################
divert(0)dnl
define(<--@macro_signbit@-->,<--@dnl
macro_ufunc(<--@signbit@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Returns @c True where signbit is set (less than zero), element-wise.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# @c An ndarray, containing the results. If @em x is a scalar,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.signbit(-1.2)
# array(True)
# >>> vp.signbit(vp.array([1, -2.3, 2.1]))
# array([False,  True, False])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.copysign
###########################################################################
divert(0)dnl
define(<--@macro_copysign@-->,<--@dnl
macro_ufunc(<--@copysign@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Changes the sign of x1 to that of x2, element-wise.@-->,dnl
<--@@em x1 : Values to change the sign of.@n @em x2 : The sign of @em x2 is copied to @em x1.@n If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The values of @em x1 with the sign of @em x2. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.copysign(1.3, -1)
# array(-1.3)
# >>> 1/vp.copysign(0, 1)
# array(inf)
# >>> 1/vp.copysign(0, -1)
# array(-inf)
# @endcode@n
# @code
# >>> vp.copysign([-1, 0, 1], -1.1)
# array([-1., -0., -1.])
# >>> vp.copysign([-1, 0, 1], vp.arange(3)-1)
# array([-1.,  0.,  1.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.nextafter
###########################################################################
divert(0)dnl
define(<--@macro_nextafter@-->,<--@dnl
macro_ufunc(<--@nextafter@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Returns the next floating-point value after @em x1 towards @em x2, element-wise.@-->,dnl
<--@@em x1 : Values to find the next representable value of.@n @em x2 : The direction where to look for the next representable value of @em x1.@n If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The next representable values of @em x1 in the direction of @em x2. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.set_printoptions(16)
# >>> vp.nextafter(1, 0)
# array(0.9999999999999999)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.spacing
###########################################################################
divert(0)dnl
define(<--@macro_spacing@-->,<--@dnl
macro_ufunc(<--@spacing@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Returns the distance between @em x and the nearest adjacent number, element-wise.@-->,dnl
<--@Input an array or a scalar.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The spacing of values of @em x. If @em x is a scalar,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @note
# It can be considered as a generalization of EPS: <span class="pre">spacing(vp.float64(1)) == nlcpy.finfo(vp.float64).eps</span>, and there should not be any representable number between <span class="pre">x + spacing(x)</span> and @em x for any finite @em x.
#
# Spacing of +- inf and NaN is NaN.
#
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.spacing(1) == vp.finfo(vp.float64).eps
# array(True)
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.ldexp
###########################################################################
divert(0)dnl
define(<--@macro_ldexp@-->,<--@dnl
macro_ufunc(<--@ldexp@-->,<--@binary@-->,<--@$1@-->,dnl
<--@Returns @f$ x1 * 2^{x2} @f$, element-wise.@-->,dnl
<--@@em x1 : Array of multipliers.@n @em x2 : Array of twos exponents. If <span class="pre">x1.shape != x2.shape</span>, they must be broadcastable to a common shape (which becomes the shape of the output).@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The result of @f$ x1 * 2^{x2} @f$. If @em x1 and @em x2 are both scalars,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @attention
# @li dtype is a complex dtype("complex64", "complex128") :  @em TypeError occurs.@n
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> vp.ldexp(5., vp.arange(4), dtype='float64')
# array([ 5., 10., 20., 40.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.floor
###########################################################################
divert(0)dnl
define(<--@macro_floor@-->,<--@dnl
macro_ufunc(<--@floor@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Returns the floor of the input, element-wise.@n The floor of the scalar @em x is the largest integer @em i, such that @em i <= @em x. It is often denoted as @f$\lfloor x \rfloor@f$.@-->,dnl
<--@Input arrays or scalars.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The floor of each element in @em x. If @em x is a scalar,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::ceil : Returns the ceiling of the input, element-wise.
# @li @ref ufuncs::trunc : Returns the truncated value of the input, element-wise.
# @li @ref ufuncs::rint : Computes the element-wise nearest integer.
dnl@-->,<--@dnl
# @note
# Some spreadsheet programs calculate the "floor-towards-zero", in other words <span class="pre">floor(-2.5) == -2</span>. NLCPy instead uses the definition of @ref floor where <span class="pre">floor(-2.5) == -3</span>.
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> a = vp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
# >>> vp.floor(a)
# array([-2., -2., -1.,  0.,  1.,  1.,  2.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.ceil
###########################################################################
divert(0)dnl
define(<--@macro_ceil@-->,<--@dnl
macro_ufunc(<--@ceil@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Returns the ceiling of the input, element-wise.@n The ceiling of the scalar @em x is the smallest integer @em i, such that @em i >= @em x. It is often denoted as @f$\lceil x \rceil@f$.@-->,dnl
<--@Input arrays or scalars.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The ceiling of each element in @em x. If @em x is a scalar,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::floor : Returns the %floor of the input, element-wise.
# @li @ref ufuncs::trunc : Returns the truncated value of the input, element-wise.
# @li @ref ufuncs::rint : Computes the element-wise nearest integer.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> a = vp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
# >>> vp.ceil(a)
# array([-1., -1., -0.,  1.,  2.,  2.,  2.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl
divert(-1)
###########################################################################
# nlcpy.trunc
###########################################################################
divert(0)dnl
define(<--@macro_trunc@-->,<--@dnl
macro_ufunc(<--@trunc@-->,<--@unary@-->,<--@$1@-->,dnl
<--@Returns the truncated value of the input, element-wise.@n The truncated value of the scalar @em x is the nearest integer @em i which is closer to zero than @em x is. In short, the fractional part of the signed number @em x is discarded.@-->,dnl
<--@Input arrays or scalars.@-->,dnl
<--@dnl
# @retval y : <em>@ref n-dimensional_array "ndarray"</em>@n
# The truncated value of each element in @em x. If @em x is a scalar,
# this function returns the result as a 0-dimension ndarray.
dnl@-->,<--@dnl
# @sa
# @li @ref ufuncs::ceil : Returns the ceiling of the input, element-wise.
# @li @ref ufuncs::floor : Returns the %floor of the input, element-wise.
# @li @ref ufuncs::rint : Computes the element-wise nearest integer.
dnl@-->,<--@dnl
dnl@-->,<--@dnl
# @code
# >>> import nlcpy as vp
# >>> a = vp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
# >>> vp.trunc(a)
# array([-1., -1., -0.,  0.,  1.,  1.,  2.])
# @endcode
dnl@-->)dnl
dnl@-->)dnl
dnl

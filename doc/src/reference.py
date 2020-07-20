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

## @defgroup Reference Reference
# Here is a list of functionalities provided by NLCPy.
# 

## 
# @defgroup n-dimensional_array The N-dimensional array(ndarray)
# @ingroup Reference
#
# @if(layng_ja)
# @else
# @details
# <h1> N-dimensional array class </h1>
# @ref core::\__init__ "nlcpy.ndarray" is the NLCPy counterpart of 
# <a href="https://numpy.org/doc/stable/reference/arrays.ndarray.html#">numpy.ndarray</a>.
# It provides an intuitive interface for a fixed-size multidimensional array which resides in a VE. 
#
# For the basic concept of @ref core::\__init__ "ndarray", please refer
# to the <a href="https://numpy.org/doc/stable/reference/arrays.ndarray.html#">numpy.ndarray</a>.
# <table>
# <tr><td>@ref core::\__init__ "nlcpy.ndarray"</td><td>N-dimensional array class for VE.</td></tr>
# </table>
#
# <h1> Array Indexing </h1>
# Arrays can be indexed using an extended Python slicing syntax,
# array[selection].
# For the basic concept of indexing arrays, please refer to the
# <a href="https://numpy.org/doc/stable/reference/arrays.indexing.html#arrays-indexing">NumPy Array Indexing</a>.
#
# <h3> Differences from NumPy </h3>
#
# @par Out-of-bounds indices
# NLCPy handles out-of-bounds indices differently by default from NumPy
# when using integer array indexing. 
# NumPy handles them by raising an error, but NLCPy wraps around them.
# @code
# >>> import numpy as np
# >>> import nlcpy as vp
# >>> nx = np.arange(3)
# >>> nx[[0, 1, 5]]
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# IndexError: index 5 is out of bounds for axis 0 with size 3
# >>> vx = vp.arange(3)
# >>> vx[[0, 1, 5]]
# array([0, 1, 2])
# @endcode
#
# @par Multiple boolean indices
# NLCPy does not support slices that consists of more than one boolean arrays.
#
# @endif
#
#

##
# @ingroup Reference
# @defgroup Constants Constants
#
# @if(layng_ja)
# @else
# @details The following table shows constants provided by NLCPy.
#      <table>
#      <tr><td>@ref nlcpy_inf "inf"</td><td rowspan="5">IEEE 754 floating point representation of positive infinity.</td></tr>
#      <tr><td>@ref nlcpy_Inf "Inf"</td></tr>
#      <tr><td>@ref nlcpy_Infinity "Infinity"</td></tr>
#      <tr><td>@ref nlcpy_infty "infty"</td></tr>
#      <tr><td>@ref nlcpy_PINF "PINF"</td></tr>
#      <tr><td>@ref nlcpy_NINF "NINF"</td><td></td>IEEE 754 floating point representation of negative infinity.</tr>
#      <tr><td>@ref nlcpy_nan "nan"</td><td rowspan="3">IEEE 754 floating point representation of Not a Number (NaN).</td></tr>
#      <tr><td>@ref nlcpy_NAN "NAN"</td></tr>
#      <tr><td>@ref nlcpy_NaN "NaN"</td></tr>
#      <tr><td>@ref nlcpy_NZERO "NZERO"</td><td></td>IEEE 754 floating point representation of negative zero. </tr>
#      <tr><td>@ref nlcpy_PZERO "PZERO"</td><td></td>IEEE 754 floating point representation of positive zero. </tr>
#      <tr><td>@ref nlcpy_e "e"</td><td>Euler's constant, base of natural logarithms, Napier's constant.</td></tr>
#      <tr><td>@ref nlcpy_euler_gamma "euler_gamma"</td><td>Euler's gamma</td></tr>
#      <tr><td>@ref nlcpy_pi "pi"</td><td>Circle ratio</td></tr>
#      </table>
# @endif
#

##
# @defgroup nlcpy_Inf nlcpy.Inf
# @par nlcpy.Inf
# IEEE 754 floating point representation of positive infinity.@n
# Use @ref nlcpy_inf "inf" because @ref nlcpy_Inf "Inf", @ref nlcpy_Infinity "Infinity", @ref nlcpy_PINF "PINF" and @ref nlcpy_infty "infty" are aliases for @ref nlcpy_inf "inf". For more details, see @ref nlcpy_inf "inf".
# @sa
# @li @ref nlcpy_inf "inf"
# @par
# 

##
# @defgroup nlcpy_Infinity nlcpy.Infinity
# @par nlcpy.Infinity
# IEEE 754 floating point representation of positive infinity.@n
# Use @ref nlcpy_inf "inf" because @ref nlcpy_Inf "Inf", @ref nlcpy_Infinity "Infinity", @ref nlcpy_PINF "PINF" and @ref nlcpy_infty "infty" are aliases for @ref nlcpy_inf "inf". For more details, see @ref nlcpy_inf "inf".
# @sa
# @li @ref nlcpy_inf "inf"
# @par
# 

##
# @defgroup nlcpy_NAN nlcpy.NAN
# @par nlcpy.NAN
# IEEE 754 floating point representation of Not a Number (NaN).@n
# @ref nlcpy_NaN "NaN" and @ref nlcpy_NAN "NAN" are equivalent definitions of @ref nlcpy_nan "nan". Please use @ref nlcpy_nan "nan" instead of @ref nlcpy_NAN "NAN".
# @sa 
# @li @ref nlcpy_nan "nan"
# @par
# 

##
# @defgroup nlcpy_NINF nlcpy.NINF
# @par nlcpy.NINF
# IEEE 754 floating point representation of negative infinity.@n
# @retval y : @em float @n
# A floating point representation of negative infinity.
# @sa
# @li @ref ufuncs::isinf : Tests whether input elements are @c inf, or not.
# @li @ref ufuncs::isnan : Tests whether input elements are @c nan, or not.
# @li @ref ufuncs::isfinite : Tests whether input elements are neither @c inf nor @c nan, or not.
# @note
# NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).@n
# This means that Not a Number is not equivalent to infinity.Also that positive infinity@n
# is not equivalent to negative infinity. But infinity is equivalent to positive infinity.
# @par Examples
# @code
# >>> import nlcpy as vp
# >>> vp.NINF
# -inf
# @endcode
# 

##
# @defgroup nlcpy_NZERO nlcpy.NZERO
# @par nlcpy.NZERO
# IEEE 754 floating point representation of negative zero.
# @retval y @b : @em float @n
# A floating point representation of negative zero.
# @sa
# @li @ref nlcpy_PZERO : Defines positive zero.
# @li @ref ufuncs::isinf : Tests whether input elements are @c inf, or not.
# @li @ref ufuncs::isnan : Tests whether input elements are @c nan, or not.
# @li @ref ufuncs::isfinite : Tests whether input elements are neither @c inf nor @c nan, or not.
# Not a Number, positive infinity and negative infinity.
# @note
# NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754). Negative zero is considered to be a finite number.
# @par Examples
# @code
# >>> import nlcpy as vp
# >>> vp.NZERO
# -0.0
# >>> vp.PZERO
# 0.0
# @endcode
# @n
# @code
# >>> import nlcpy as vp
# >>> vp.isfinite([vp.NZERO])
# array([ True])
# >>> vp.isnan([vp.NZERO])
# array([False])
# >>> vp.isinf([vp.NZERO])
# array([False])
# @endcode
# 

##
# @defgroup nlcpy_NaN nlcpy.NaN
# @par nlcpy.NaN
# IEEE 754 floating point representation of Not a Number (NaN).@n
# @ref nlcpy_NaN "NaN" and @ref nlcpy_NAN "NAN" are equivalent definitions of @ref nlcpy_nan "nan". Please use @ref nlcpy_nan "nan" instead of @ref nlcpy_NaN "NaN".
# @sa
# @li @ref nlcpy_nan "nan"
# @par
# 

##
# @defgroup nlcpy_PINF nlcpy.PINF
# @par nlcpy.PINF
# IEEE 754 floating point representation of positive infinity.@n
# Use @ref nlcpy_inf "inf" because @ref nlcpy_Inf "Inf", @ref nlcpy_Infinity "Infinity", @ref nlcpy_PINF "PINF" and @ref nlcpy_infty "infty" are aliases for @ref nlcpy_inf "inf". For more details, see @ref nlcpy_inf "inf".
# @sa
# @li @ref nlcpy_inf "inf"
# @par
# 

##
# @defgroup nlcpy_PZERO nlcpy.PZERO
# @par nlcpy.PZERO
# IEEE 754 floating point representation of positive zero.
# @retval y @b : @em float @n
# A floating point representation of positive zero.
# @sa
# @li @ref nlcpy_NZERO : Defines positive zero.
# @li @ref ufuncs::isinf : Tests whether input elements are @c inf, or not.
# @li @ref ufuncs::isnan : Tests whether input elements are @c nan, or not.
# @li @ref ufuncs::isfinite : Tests whether input elements are neither @c inf nor @c nan, or not.
# @note
# NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754). Positive zero is considered to be a finite number.
# @par Examples
# @code
# >>> import nlcpy as vp
# >>> vp.PZERO
# 0.0
# >>> vp.NZERO
# -0.0
# @endcode
# @n
# @code
# >>> import nlcpy as vp
# >>> vp.isfinite([vp.PZERO])
# array([ True])
# >>> vp.isnan([vp.PZERO])
# array([False])
# >>> vp.isinf([vp.NZERO])
# array([False])
# @endcode
# 

##
# @defgroup nlcpy_e nlcpy.e
# @par nlcpy.e
# Euler's constant, base of natural logarithms, Napier's constant.@n
# <span class="pre">e = 2.71828182845904523536028747135266249775724709369995...</span>
# @sa
# @li @ref ufuncs::exp : Computes the element-wise exponential of the input array.
# @li @ref ufuncs::log : Computes the element-wise natural logarithm of @em x.
# @par
# 

##
# @defgroup nlcpy_euler_gamma nlcpy.euler_gamma
# @par nlcpy.euler_gamma
# <span class="pre">&gamma; = 0.5772156649015328606065120900824024310421...</span>
# 

##
# @defgroup nlcpy_inf nlcpy.inf
# @par nlcpy.inf
# IEEE 754 floating point representation of positive infinity.
# @retval y @b : @em float @n
# A floating point representation of positive infinity.
# @sa
# @li @ref ufuncs::isinf : Tests whether input elements are @c inf, or not.
# @li @ref ufuncs::isnan : Tests whether input elements are @c nan, or not.
# @li @ref ufuncs::isfinite : Tests whether input elements are neither @c inf nor @c nan, or not.
# @note
# NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754). This means that Not a Number is not equivalent to infinity. Also that positive infinity is not equivalent to negative infinity. But infinity is equivalent to positive infinity.@n
# @ref nlcpy_Inf "Inf", @ref nlcpy_Infinity "Infinity", @ref nlcpy_PINF "PINF" and @ref nlcpy_infty "infty" are aliases for @ref nlcpy_inf "inf".
# @par Examples
# @code
# >>> import nlcpy as vp
# >>> vp.inf
# inf
# >>> vp.array([1]) / 0.
# array([inf])
# @endcode
# 

##
# @defgroup nlcpy_infty nlcpy.infty
# @par nlcpy.infty
# IEEE 754 floating point representation of positive infinity.@n
# Use @ref nlcpy_inf "inf" because @ref nlcpy_Inf "Inf", @ref nlcpy_Infinity "Infinity", @ref nlcpy_PINF "PINF" and @ref nlcpy_infty "infty" are aliases for @ref nlcpy_inf "inf". For more details, see @ref nlcpy_inf "inf".
# @sa
# @li @ref nlcpy_inf "inf"
# @par 
# 

##
# @defgroup nlcpy_nan nlcpy.nan
# @par nlcpy.nan
# IEEE 754 floating point representation of Not a Number (NaN).
# @retval y @n
# A floating point representation of Not a Number.
# @sa
# @li @ref ufuncs::isnan : Tests whether input elements are @c nan, or not.
# @li @ref ufuncs::isfinite : Tests whether input elements are neither @c inf nor @c nan, or not.
# @note
# NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754). This means that Not a Number is not equivalent to infinity.@n
# @ref nlcpy_NaN "NaN" and @ref nlcpy_NAN "NAN" are aliases of @ref nlcpy_nan "nan".
# @par Examples
# @code
# >>> import nlcpy as vp
# >>> vp.nan
# nan
# @endcode
# 

##
# @defgroup nlcpy_newaxis nlcpy.newaxis
# @par nlcpy.newaxis
# A convenient alias for None, useful for indexing arrays.
# @par Examples
# @code
# >>> import nlcpy as vp
# >>> vp.newaxis is None
# True
# >>> x = vp.arange(3)
# >>> x
# array([0, 1, 2])
# >>> x[:, vp.newaxis]
# array([[0],
#        [1],
#        [2]])
# >>> x[:, vp.newaxis, vp.newaxis]
# array([[[0]],
# 
#        [[1]],
# 
#        [[2]]])
# >>> x[:, vp.newaxis] * x
# array([[0, 0, 0],
#        [0, 1, 2],
#        [0, 2, 4]])
# @endcode
# @n
# Outer product, same as outer(x, y):
# @code
# >>> y = vp.arange(3, 6)
# >>> x[:, vp.newaxis] * y
# array([[ 0,  0,  0],
#        [ 3,  4,  5],
#        [ 6,  8, 10]])
# @endcode
# @n
# <span class="pre">x[newaxis, :]</span> is equivalent to <span class="pre">x[newaxis] and x[None]:</span>
# @code
# >>> x[vp.newaxis, :].shape
# (1, 3)
# >>> x[vp.newaxis].shape
# (1, 3)
# >>> x[None].shape
# (1, 3)
# >>> x[:, vp.newaxis].shape
# (3, 1)
# @endcode
# 

##
# @defgroup nlcpy_pi nlcpy.pi
# @par nlcpy.pi
# <span class="pre">pi = 3.1415926535897932384626433...</span>

## 
# @ingroup Reference
# @defgroup Array_creation_routines Array Creation Routines
#
# @details
# The following tables show array creation routines provided by NLCPy.
# ## Ones and Zeros ##
#      <table>
#      <tr><td>@ref basic::empty </td><td>Returns a new array of given shape and type, without initializing entries.</td></tr>
#      <tr><td>@ref basic::empty_like </td><td>Returns a new array with the same shape and type as a given array.</td></tr>
#      <tr><td>@ref basic::eye </td><td>Returns a 2-D array with ones on the diagonal and zeros elsewhere.</td></tr>
#      <tr><td>@ref basic::identity </td><td>Returns the identity array. </td></tr>
#      <tr><td>@ref basic::ones </td><td>Returns a new array of given shape and type, filled with ones.</td></tr>
#      <tr><td>@ref basic::ones_like </td><td>Returns an array of ones with the same shape and type as a given array.</td></tr>
#      <tr><td>@ref basic::zeros </td><td></td>Returns a new array of given shape and type, filled with zeros.</tr>
#      <tr><td>@ref basic::zeros_like </td><td>Returns an array of zeros with the same shape and type as a given array.</td></tr>
#      <tr><td>@ref basic::full </td><td></td>Returns a new array of given shape and type, filled with @em fill_value.</tr>
#      <tr><td>@ref basic::full_like </td><td>Returns a full array with the same shape and type as a given array.</td></tr>
#      </table>
# ## From Existing Data ##
#      <table>
#      <tr><td>@ref from_data::array </td><td>Creates an array.</td></tr>
#      <tr><td>@ref from_data::copy </td><td>Returns an array copy of the given object.</td></tr>
#      <tr><td>@ref from_data::asarray </td><td>Converts the input to an array.</td></tr>
#      <tr><td>@ref from_data::asanyarray </td><td>Converts the input to an array, but passes ndarray subclasses through.</td></tr>
#      </table>
# ## Numerical Ranges ##
#      <table>
#      <tr><td>@ref ranges::arange </td><td>Returns evenly spaced values within a given interval.</td></tr>
#      <tr><td>@ref ranges::linspace </td><td>Returns evenly spaced numbers over a specified interval. </td></tr>
#      </table>
# ## Building Matrices ##
#      <table>
#      <tr><td>@ref matrices::diag </td><td>Returns the indices of the elements that are non-zero.</td></tr>
#      </table>

## 
# @defgroup Ones_and_zeros Ones and Zeros

##
# @defgroup From_existing_data From Existing Data

## 
# @defgroup Numerical_ranges Numerical Ranges

## 
# @defgroup Building_matricies Building Matricies

## 
# @ingroup Reference
# @defgroup Array_manipulation_routines Array Manipulation Routines
#
# @details
# The following tables show array manipulation routines provided by NLCPy.
# ## Basic Operations ##
#      <table>
#      <tr><td>@ref basic::shape </td><td>Returns the shape of an array.</td></tr>
#      </table>
# ## Changing Array Shape ##
#      <table>
#      <tr><td>@ref shape::reshape </td><td>Gives a new shape to an array without changing its data.</td></tr>
#      <tr><td>@ref shape::ravel </td><td>Returns a contiguous flattened array. </td></tr>
#      </table>
# ## Transpose-like Operations ##
#      <table>
#      <tr><td>@ref trans::moveaxis </td><td>Moves axes of an array to new positions.</td></tr>
#      <tr><td>@ref trans::rollaxis </td><td>Rolls the specified axis backwards, until it lies in a given position.</td></tr>
#      <tr><td>@ref trans::transpose </td><td>Permutes the dimensions of an array.</td></tr>
#      </table>
# ## Changing Number of Dimensions ##
#      <table>
#      <tr><td>@ref dims::broadcast_to </td><td>Broadcasts an array to a new shape.</td></tr>
#      <tr><td>@ref dims::expand_dims </td><td>Expands the shape of an array.</td></tr>
#      <tr><td>@ref dims::squeeze </td><td>Removes single-dimensional entries from the shape of an array.</td></tr>
#      </table>
# ## Joining Arrays ##
#      <table>
#      <tr><td>@ref join::concatenate </td><td>Joins a sequence of arrays along an existing axis.</td></tr>
#      </table>
# ## Tiling Arrays ##
#      <table>
#      <tr><td>@ref tiling::tile </td><td>Constructs an array by repeating A the number of times given by reps.</td></tr>
#      </table>
# ## Adding and Removing Elements ##
#      <table>
#      <tr><td>@ref shape::reshape </td><td>Returns a new array with the specified shape.</td></tr>
#      </table>

##
# @defgroup Changing_array_shape Changing array shape

##
# @defgroup Transpose-like_operations Transpose-like operations

##
# @defgroup Changing_number_of_dimensions Changing number of dimensions

##
# @defgroup Joining_arrays Joining arrays

## 
# @defgroup Tiling_arrays Tiling arrays

##
# @defgroup Adding_and_removing_elements Adding and removing elements

##
# @ingroup Reference
# @defgroup Universal_functions Universal Functions(ufunc)
#
# @details
# A universal function (or @ref Ufunc_list "ufunc" for short) is a function that operates on ndarrays in an element-by-element fashion, supporting array broadcasting, type casting, and several other standard features. That is, a ufunc is a "vectorized" wrapper for a function that takes a fixed number of specific inputs and produces a fixed number of specific outputs.@n@n
# As with NumPy, Broadcasting rules are applied to input arrays of the universal functions of NLCPy.@n
#
# # Broadcasting #
# Each universal function takes array inputs and produces array outputs by performing the core function element-wise on the inputs (where an element is generally a scalar, but can be a vector or higher-order sub-array for generalized ufuncs). Standard broadcasting rules are applied so that inputs not sharing exactly the same shapes can still be usefully operated on. Broadcasting can be understood by four rules:@n
# <ol>
# <li> All input arrays with ndim smaller than the input array of largest ndim, have 1's prepended to their shapes.</li>
# <li> The size in each dimension of the output shape is the maximum of all the input sizes in that dimension.</li>
# <li> An input can be used in the calculation if its size in a particular dimension either matches the output size in that dimension, or has value exactly 1.</li>
# <li> If an input has a dimension size of 1 in its shape, the first data entry in that dimension will be used for all calculations along that dimension. In other words, the stepping machinery of the ufunc will simply not step along that dimension (the stride will be 0 for that dimension).</li>
# </ol>
#
# Broadcasting is used throughout NLCPy to decide how to handle disparately shaped arrays; for example, all arithmetic operations (+, -, *, etc.) between ndarrays broadcast the arrays before operation.
#
# A set of arrays is called "broadcastable" to the same shape if the above rules produce a valid result, i.e., one of the following is true:
# <ol>
# <li> The arrays all have exactly the same shape.</li>
# <li> The arrays all have the same number of dimensions and the length of each dimensions is either a common length or 1.</li>
# <li> The arrays that have too few dimensions can have their shapes prepended with a dimension of length 1 to satisfy property 2.</li>
# </ol>
#
# ## Examples ##
# If a.shape is (5,1), b.shape is (1,6), c.shape is (6,) and d.shape is () so that d is a scalar, then a, b, c, and d are all broadcastable to dimension (5,6); and
# <ul/>
# <li> a acts like a (5,6) array where a[:,0] is broadcast to the other columns,</li>
# <li> b acts like a (5,6) array where b[0,:] is broadcast to the other rows,</li>
# <li> c acts like a (1,6) array and therefore like a (5,6) array where c[:] is broadcast to every row, and finally,</li>
# <li> d acts like a (5,6) array where the single value is repeated.</li>
# </ul>
#
# # Methods #
# In NLCPy, @ref Ufunc_list "ufuncs" are instances of the @b nlcpy.ufunc class. 
# @ref Ufunc_list "nlcpy.ufunc" have four methods. However, these methods only make 
# sense on scalar ufuncs that take two input arguments and return one output argument.
# Attempting to call these methods on other ufuncs will cause a @em ValueError.
# <p>
# @li @ref ufunc::reduce reduces one of the dimension of the input array, by applying ufunc along one axis.
# @li @ref ufunc::accumulate accumulates the result of applying the operator to all elements.
# @li @ref ufunc::reduceat performs a (local) reduce with specified slices over a single axis.
# @li @ref ufunc::outer applies the ufunc @em op to all pairs (a, b) with a in @em A and b in @em B.
# </p>
# The current version of NLCPy does not provide <code>ufunc.at()</code>, which is supported by NumPy.
#
# @section Ufunc_optional Optional Keyword Arguments
# All @ref Ufunc_list "ufuncs" take optional keyword arguments. Most of these represent advanced usage and will not typically be used.
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
# @param casting : <em>{'no', 'equiv', 'safe', 'same_kind'}, @b optional </em>@n
# Controls what kind of data casting may occur.
#     <ul>
#     <li/> 'no' means the data types should not be cast at all.
#     <li/> 'equiv' means only byte-order changes are allowed.
#     <li/> 'safe' means only casts which can preserve values are allowed.
#     <li/> 'same_kind' means only safe casts or casts within a kind, like float64 to float32, are allowed.
#     </ul>
# NLCPy does NOT support 'unsafe', which is supported in NumPy.
#
# @param order : <em>character, @b optional </em>@n
# Specifies the calculation iteration order/memory layout of the output array. Defaults 
# to 'K'. 'C' means the output should be C-contiguous, 'F' means F-contiguous, 'A' 
# means F-contiguous if the inputs are F-contiguous and not also not C-contiguous, 
# C-contiguous otherwise, and 'K' means to match the element ordering of the inputs 
# as closely as possible.
#
# @param dtype : <em>dtype, @b optional </em>@n
# Overrides the dtype of the calculation and output arrays.
#
# @param subok : <em>bool, @b optional </em>@n
# Not implemented in NLCPy.

##
# @defgroup Ufunc_list ufunc
#
# @details
# The following tables show Universal functions(ufuncs) provided by NLCPy.
# For the overview of the ufuncs, please see @ref Universal_functions.
# ## Math Operations ##
#      <table>
#      <tr><td>@ref ufuncs::add          </td><td>Computes the element-wise addition of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::subtract     </td><td>Computes the element-wise subtraction of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::multiply     </td><td>Computes the element-wise multiplication of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::divide       </td><td>Computes the element-wise division of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::logaddexp    </td><td>Computes the element-wise natural logarithm of @f$ exp(x1) + exp(x2) @f$.</td></tr>
#      <tr><td>@ref ufuncs::logaddexp2   </td><td>Computes the element-wise base-2 logarithm of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::true_divide  </td><td>Computes the element-wise division of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::floor_divide </td><td>Computes the element-wise floor divition of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::negative     </td><td>Computes numerical negative, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::positive     </td><td>Computes numerical positive, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::power        </td><td>Computes the element-wise exponentiation of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::remainder    </td><td>Computes the element-wise remainder of division.</td></tr>
#      <tr><td>@ref ufuncs::mod          </td><td>Computes the element-wise remainder of division.</td></tr>
#      <tr><td>@ref ufuncs::fmod         </td><td>Computes the element-wise remainder of division.</td></tr>
#      <tr><td>@ref ufuncs::absolute     </td><td>Computes the element-wise absolute value.</td></tr>
#      <tr><td>@ref ufuncs::fabs         </td><td>Computes the element-wise absolute value.</td></tr>
#      <tr><td>@ref ufuncs::rint         </td><td>Computes the element-wise nearest integer.</td></tr>
#      <tr><td>@ref ufuncs::sign         </td><td>Computes the element-wise indication of the sign of a number.</td></tr>
#      <tr><td>@ref ufuncs::heaviside    </td><td>Computes the element-wise Heaviside step function.</td></tr>
#      <tr><td>@ref ufuncs::conj         </td><td>Returns the element-wise complex conjugate.</td></tr>
#      <tr><td>@ref ufuncs::conjugate    </td><td>Returns the element-wise complex conjugate.</td></tr>
#      <tr><td>@ref ufuncs::exp          </td><td>Computes the element-wise exponential of the input array.</td></tr>
#      <tr><td>@ref ufuncs::exp2         </td><td>Computes the element-wise 2 to the power of @em x.</td></tr>
#      <tr><td>@ref ufuncs::log          </td><td>Computes the element-wise natural logarithm of @em x.</td></tr>
#      <tr><td>@ref ufuncs::log2         </td><td>Computes the element-wise base-2 logarithm of @em x.</td></tr>
#      <tr><td>@ref ufuncs::log10        </td><td>Computes the element-wise base-10 logarithm of @em x.</td></tr>
#      <tr><td>@ref ufuncs::expm1        </td><td>Computes the element-wise exponential minus one.</td></tr>
#      <tr><td>@ref ufuncs::log1p        </td><td>Computes the element-wise natural logarithm of <em>1+x</em>.</td></tr>
#      <tr><td>@ref ufuncs::sqrt         </td><td>Computes the element-wise square-root of the input.</td></tr>
#      <tr><td>@ref ufuncs::square       </td><td>Computes the element-wise square of the input.</td></tr>
#      <tr><td>@ref ufuncs::cbrt         </td><td>Computes the element-wise cubic-root of the input.</td></tr>
#      <tr><td>@ref ufuncs::reciprocal   </td><td>Computes the element-wise reciprocal of the input.</td></tr>
#      </table>
#
# ## Trigonometric Functions ##
#      <table>
#      <tr><td>@ref ufuncs::sin     </td><td>Computes the element-wise sine.</td></tr>
#      <tr><td>@ref ufuncs::cos     </td><td>Computes the element-wise cosine.</td></tr>
#      <tr><td>@ref ufuncs::tan     </td><td>Computes the element-wise tangent.</td></tr>
#      <tr><td>@ref ufuncs::arcsin  </td><td>Computes the element-wise inverse sine.</td></tr>
#      <tr><td>@ref ufuncs::arccos  </td><td>Computes the element-wise inverse cosine.</td></tr>
#      <tr><td>@ref ufuncs::arctan  </td><td>Computes the element-wise inverse tangent.</td></tr>
#      <tr><td>@ref ufuncs::arctan2 </td><td>Computes the element-wise inverse tangent of @em x1/@em x2.</td></tr>
#      <tr><td>@ref ufuncs::hypot   </td><td>Computes the "legs" of a right triangle.</td></tr>
#      <tr><td>@ref ufuncs::sinh    </td><td>Computes the element-wise hyperbolic sine.</td></tr>
#      <tr><td>@ref ufuncs::cosh    </td><td>Computes the element-wise hyperbolic cosine.</td></tr>
#      <tr><td>@ref ufuncs::tanh    </td><td>Computes the element-wise hyperbolic tangent.</td></tr>
#      <tr><td>@ref ufuncs::arcsinh </td><td>Computes the element-wise inverse hyperbolic sine.</td></tr>
#      <tr><td>@ref ufuncs::arccosh </td><td>Computes the element-wise inverse hyperbolic cosine.</td></tr>
#      <tr><td>@ref ufuncs::arctanh </td><td>Computes the element-wise inverse hyperbolic tangent.</td></tr>
#      <tr><td>@ref ufuncs::deg2rad </td><td>Converts angles from degrees to radians.</td></tr>
#      <tr><td>@ref ufuncs::rad2deg </td><td>Converts angles from radians to degrees.</td></tr>
#      </table>
#
# ## Bit-Twiddling Functions ##
#      <table>
#      <tr><td>@ref ufuncs::bitwise_and   </td><td>Computes the bit-wise AND of two arrays element-wise.</td></tr>
#      <tr><td>@ref ufuncs::bitwise_or    </td><td>Computes the bit-wise OR of two arrays element-wise.</td></tr>
#      <tr><td>@ref ufuncs::bitwise_xor   </td><td>Computes the bit-wise XOR of two arrays element-wise.</td></tr>
#      <tr><td>@ref ufuncs::invert        </td><td>Computes the bit-wise NOT of the input array element-wise.</td></tr>
#      <tr><td>@ref ufuncs::logical_and   </td><td>Computes the logical AND of two arrays element-wise.</td></tr>
#      <tr><td>@ref ufuncs::logical_or    </td><td>Computes the logical OR of two arrays element-wise.</td></tr>
#      <tr><td>@ref ufuncs::logical_xor   </td><td>Computes the logical XOR of two arrays element-wise.</td></tr>
#      <tr><td>@ref ufuncs::logical_not   </td><td>Computes the logical NOT of the input array element-wise.</td></tr>
#      <tr><td>@ref ufuncs::left_shift    </td><td>Shifts bits of an integer to the left, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::right_shift   </td><td>Shifts bits of an integer to the right, element-wise.</td></tr>
#      </table>
#
# ## Comparison Functions ##
#      <table>
#      <tr><td>@ref ufuncs::greater       </td><td>Returns (@em x1 > @em x2), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::greater_equal </td><td>Returns (@em x1 >= @em x2), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::less          </td><td>Returns (@em x1 < @em x2), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::less_equal    </td><td>Returns (@em x1 <= @em x2), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::not_equal     </td><td>Returns (@em x1 != @em x2), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::equal         </td><td>Returns (@em x1 == @em x2), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::maximum       </td><td>Computes the element-wise maximum of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::minimum       </td><td>Computes the element-wise minimum of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::fmax          </td><td>Computes the element-wise maximum of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::fmin          </td><td>Computes the element-wise minimum of the inputs.</td></tr>
#      </table>
#
# ## Floating Point Functions ##
#      <table>
#      <tr><td>@ref ufuncs::isfinite  </td><td>Tests whether input elements are neither @c inf nor @c nan, or not.</td></tr>
#      <tr><td>@ref ufuncs::isinf     </td><td>Tests whether input elements are @c inf, or not.</td></tr>
#      <tr><td>@ref ufuncs::isnan     </td><td>Tests whether input elements are @c nan, or not.</td></tr>
#      <tr><td>@ref ufuncs::signbit   </td><td>Returns @c True where signbit is set (less than zero), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::copysign  </td><td>Changes the sign of @em x1 to that of @em x2, element-wise. </td></tr>
#      <tr><td>@ref ufuncs::nextafter </td><td>Returns the next floating-point value after @em x1 towards @em x2, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::spacing   </td><td>Returns the distance between x and the nearest adjacent number, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::ldexp     </td><td>Returns @f$ x1 * 2^{x2} @f$, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::floor   </td><td>Returns the %floor of the input, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::ceil    </td><td>Returns the ceilling of the input, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::trunc   </td><td>Returns the truncated value of the input, element-wise.</td></tr>
#      </table>
# 
# 

##
# @ingroup Reference
# @defgroup Mfuncs Mathematical Functions
#
# @details
# The following tables show mathematical functions provided by NLCPy.
# ## Trigonometric Functions ##
#      <table>
#      <tr><td>@ref ufuncs::sin     </td><td>Computes the element-wise sine.</td></tr>
#      <tr><td>@ref ufuncs::cos     </td><td>Computes the element-wise cosine.</td></tr>
#      <tr><td>@ref ufuncs::tan     </td><td>Computes the element-wise tangent.</td></tr>
#      <tr><td>@ref ufuncs::arcsin  </td><td>Computes the element-wise inverse sine.</td></tr>
#      <tr><td>@ref ufuncs::arccos  </td><td>Computes the element-wise inverse cosine.</td></tr>
#      <tr><td>@ref ufuncs::arctan  </td><td>Computes the element-wise inverse tangent.</td></tr>
#      <tr><td>@ref ufuncs::arctan2 </td><td>Computes the element-wise inverse tangent of @em x1/@em x2.</td></tr>
#      <tr><td>@ref ufuncs::hypot   </td><td>Computes the "legs" of a right triangle.</td></tr>
#      <tr><td>@ref ufuncs::deg2rad </td><td>Converts angles from degrees to radians.</td></tr>
#      <tr><td>@ref ufuncs::rad2deg </td><td>Converts angles from radians to degrees.</td></tr>
#      </table>
# 
# ## Hyperbolic Functions ##
#      <table>
#      <tr><td>@ref ufuncs::sinh    </td><td>Computes the element-wise hyperbolic sine.</td></tr>
#      <tr><td>@ref ufuncs::cosh    </td><td>Computes the element-wise hyperbolic cosine.</td></tr>
#      <tr><td>@ref ufuncs::tanh    </td><td>Computes the element-wise hyperbolic tangent.</td></tr>
#      <tr><td>@ref ufuncs::arcsinh </td><td>Computes the element-wise inverse hyperbolic sine.</td></tr>
#      <tr><td>@ref ufuncs::arccosh </td><td>Computes the element-wise inverse hyperbolic cosine.</td></tr>
#      <tr><td>@ref ufuncs::arctanh </td><td>Computes the element-wise inverse hyperbolic tangent.</td></tr>
#      </table>
# 
# ## Rounding ##
#      <table>
#      <tr><td>@ref ufuncs::floor   </td><td>Returns the %floor of the input, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::ceil    </td><td>Returns the ceilling of the input, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::trunc   </td><td>Returns the truncated value of the input, element-wise.</td></tr>
#      </table>
# 
# ## Sums, Products, Differences ##
#      <table>
#      <tr><td>@ref math::sum    </td><td>Sum of array elements over a given axis.</td></tr>
#      <tr><td>@ref math::cumsum </td><td>Returns the cumulative sum of the elements along a given axis.</td></tr>
#      <tr><td>@ref math::diff   </td><td>Calculates the n-th discrete difference along the given axis.</td></tr>
#      </table>
# 
# ## Exponents and Logarithms ##
#      <table>
#      <tr><td>@ref ufuncs::exp       </td><td>Computes the element-wise exponential of the input array.</td></tr>
#      <tr><td>@ref ufuncs::exp2      </td><td>Computes the element-wise 2 to the power of @em x.</td></tr>
#      <tr><td>@ref ufuncs::log       </td><td>Computes the element-wise natural logarithm of @em x.</td></tr>
#      <tr><td>@ref ufuncs::log2      </td><td>Computes the element-wise base-2 logarithm of @em x.</td></tr>
#      <tr><td>@ref ufuncs::log10     </td><td>Computes the element-wise base-10 logarithm of @em x.</td></tr>
#      <tr><td>@ref ufuncs::expm1     </td><td>Computes the element-wise exponential minus one.</td></tr>
#      <tr><td>@ref ufuncs::log1p     </td><td>Computes the element-wise natural logarithm of <em>1+x</em>.</td></tr>
#      <tr><td>@ref ufuncs::logaddexp </td><td>Computes the element-wise natural logarithm of @f$ exp(x1) + exp(x2) @f$.</td></tr>
#      <tr><td>@ref ufuncs::logaddexp2</td><td>Computes the element-wise base-2 logarithm of the inputs.</td></tr>
#      </table>
# 
# ## Floating Point Functions ##
#      <table>
#      <tr><td>@ref ufuncs::signbit   </td><td>Returns @c True where signbit is set (less than zero), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::copysign  </td><td>Changes the sign of @em x1 to that of @em x2, element-wise. </td></tr>
#      <tr><td>@ref ufuncs::nextafter </td><td>Returns the next floating-point value after @em x1 towards @em x2, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::spacing   </td><td>Returns the distance between x and the nearest adjacent number, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::ldexp     </td><td>Returns @f$ x1 * 2^{x2} @f$, element-wise.</td></tr>
#      </table>
# 
# ## Arithmetic Operations ##
#      <table>
#      <tr><td>@ref ufuncs::add          </td><td>Computes the element-wise addition of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::subtract     </td><td>Computes the element-wise subtraction of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::multiply     </td><td>Computes the element-wise multiplication of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::divide       </td><td>Computes the element-wise division of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::true_divide  </td><td>Computes the element-wise division of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::floor_divide </td><td>Computes the element-wise floor divition of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::negative     </td><td>Computes numerical negative, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::positive     </td><td>Computes numerical positive, element-wise.</td></tr>
#      <tr><td>@ref ufuncs::power        </td><td>Computes the element-wise exponentiation of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::remainder    </td><td>Computes the element-wise remainder of division.</td></tr>
#      <tr><td>@ref ufuncs::mod          </td><td>Computes the element-wise remainder of division.</td></tr>
#      <tr><td>@ref ufuncs::fmod         </td><td>Computes the element-wise remainder of division.</td></tr>
#      <tr><td>@ref ufuncs::absolute     </td><td>Computes the element-wise absolute value.</td></tr>
#      <tr><td>@ref ufuncs::reciprocal   </td><td>Computes the element-wise reciprocal of the input.</td></tr>
#      </table>
# 
# ## Handling Complex Numbers ##
#      <table>
#      <tr><td>@ref math::angle        </td><td>Returns the angle of the complex argument.</td></tr>
#      <tr><td>@ref math::real         </td><td>Returns the real part of the complex argument.</td></tr>
#      <tr><td>@ref math::imag         </td><td>Returns the imaginary part of the complex argument.</td></tr>
#      <tr><td>@ref ufuncs::conj        </td><td>Returns the element-wise complex conjugate.</td></tr>
#      <tr><td>@ref ufuncs::conjugate   </td><td>Returns the element-wise complex conjugate.</td></tr>
#      </table>
# 
# ## Miscellaneous ##
#      <table>
#      <tr><td>@ref ufuncs::sqrt      </td><td>Computes the element-wise square-root of the input.</td></tr>
#      <tr><td>@ref ufuncs::cbrt      </td><td>Computes the element-wise cubic-root of the input.</td></tr>
#      <tr><td>@ref ufuncs::square    </td><td>Computes the element-wise square of the input.</td></tr>
#      <tr><td>@ref ufuncs::absolute  </td><td>Computes the element-wise square-root of the input.</td></tr>
#      <tr><td>@ref ufuncs::fabs      </td><td>Computes the element-wise absolute value.</td></tr>
#      <tr><td>@ref ufuncs::sign      </td><td>Computes the element-wise indication of the sign of a number.</td></tr>
#      <tr><td>@ref ufuncs::heaviside </td><td>Computes the element-wise Heaviside step function.</td></tr>
#      <tr><td>@ref ufuncs::maximum   </td><td>Computes the element-wise maximum of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::minimum   </td><td>Computes the element-wise minimum of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::fmax      </td><td>Computes the element-wise maximum of the inputs.</td></tr>
#      <tr><td>@ref ufuncs::fmin      </td><td>Computes the element-wise minimum of the inputs.</td></tr>
#      </table>
# 

## @defgroup Indexing_routines Indexing Routines
# @ingroup Reference
# 
# @details
# The following tables show indexing routines provided by NLCPy.
# ## Generating Index Arrays ##
#      <table>
#      <tr><td>@ref searching::nonzero </td><td>Returns the indices of the elements that are non-zero.</td></tr>
#      <tr><td>@ref generate::where    </td><td>Returns elements chosen from @a x or @a y depending on @a condition.</td></tr>
#      </table>
# ## Indexing-like Operations ##
#      <table>
#      <tr><td>@ref matrices::diag </td><td>Returns the indices of the elements that are non-zero.</td></tr>
#      <tr><td>@ref indexing::diagonal </td><td>Returns specified diagonals.</td></tr>
#      <tr><td>@ref indexing::take </td><td>Takes elements from an array along an axis.</td></tr>
#      </table>

## @defgroup RANDOM Random Sampling
# @ingroup Reference
# @if(lang_ja)
# @else
# @details
# NLCPy random number routines produce pseudo random numbers and create sample from different statistical distributions.
# ## Generator ##
# The Generator provides access to a wide variety of probability distributions, and serves as a replacement for RandomState.@n@n
# An easy example of Generator is below:
# @code
# from nlcpy.random import Generator, MT19937
# rng = Generator(MT19937(12345))
# rng.standard_normal()
# @endcode
# And, an easy example of using default_rng is below:
# @code
# import nlcpy as vp
# rng = vp.random.default_rng()
# rng.standard_normal()
# @endcode
# @n
# The following tables show that nlcpy.random.Generator class has methods to generate random numbers:@n
# ### Functions ###
#      <table>
#      <tr><td>@ref _generator::default_rng </td><td>Constructs a new nlcpy.random.Generator with the default BitGenerator (MT19937).</td></tr>
#      </table>
# ### Simple Random Data ###
#      <table>
#      <tr><td>@ref _generator::Generator::bytes </td><td>Returns random bytes.</td></tr>
#      <tr><td>@ref _generator::Generator::integers </td><td>Returns random integers from low (inclusive) to high (exclusive), or if endpoint=True, low (inclusive) to high (inclusive).</td></tr>
#      <tr><td>@ref _generator::Generator::random </td><td>Returns random floats in the half-open interval <span class="pre">[0.0, 1.0)</span>.</td></tr>
#      </table>
# ### Permutations ###
#      <table>
#      <tr><td>@ref _generator::Generator::permutation </td><td>Randomly permutes a sequence, or returns a permuted range.</td></tr>
#      <tr><td>@ref _generator::Generator::shuffle </td><td>Modifies a sequence in-place by shuffling its contents.</td></tr>
#      </table>
# ### Distributions ###
#      <table>
#      <tr><td>@ref _generator::Generator::binomial </td><td>Draws samples from a binomial distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::exponential </td><td>Draws samples from an exponential distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::gamma </td><td>Draws samples from a Gamma distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::geometric </td><td>Draws samples from a geometric distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::gumbel </td><td>Draws samples from a Gumbel distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::logistic </td><td>Draws samples from a logistic distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::lognormal </td><td>Draws samples from a log-normal distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::normal </td><td>Draws random samples from a normal (Gaussian) distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::poisson </td><td>Draws samples from a Poisson distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::standard_cauchy </td><td>Draws samples from a standard Cauchy distribution with mode = 0.</td></tr>
#      <tr><td>@ref _generator::Generator::standard_exponential </td><td>Draws samples from a standard exponential distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::standard_gamma </td><td>Draws samples from a standard Gamma distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::standard_normal </td><td>Draws samples from a standard Normal distribution (mean=0, stdev=1).</td></tr>
#      <tr><td>@ref _generator::Generator::uniform </td><td>Draws samples from a uniform distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::weibull </td><td>Draws samples from a Weibull distribution.</td></tr>
#      </table>
# ## RandomState ##
# The RandomState provides access to legacy generators.@n@n
# An easy example of RandomState is below:
# @code
# # Uses the nlcpy.random.RandomState
# from nlcpy import random
# random.standard_normal()
# @endcode
# @n
# The following tables show that nlcpy.random.RandomState class has methods to generate random numbers:
# ### Seeding and State ###
#      <table>
#      <tr><td>@ref generator::RandomState::get_state </td><td>Returns a ndarray representing the internal state of the generator.</td></tr>
#      <tr><td>@ref generator::RandomState::seed </td><td>Reseeds a default bit generator(MT19937), which provide a stream of random bits.</td></tr>
#      <tr><td>@ref generator::RandomState::set_state </td><td>Sets the internal state of the generator from a ndarray.</td></tr>
#      </table>
# ### Simple Random Data ###
#      <table>
#      <tr><td>@ref generator::RandomState::bytes </td><td>Returns random bytes.</td></tr>
#      <tr><td>@ref generator::RandomState::rand </td><td>Random values in a given shape.</td></tr>
#      <tr><td>@ref generator::RandomState::randint </td><td>Returns random integers from low (inclusive) to high (exclusive).</td></tr>
#      <tr><td>@ref generator::RandomState::randn </td><td>Returns a sample (or samples) from a "standard normal" distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::random </td><td>Returns random floats in the half-open interval <span class="pre">[0.0, 1.0)</span>.</td></tr>
#      <tr><td>@ref generator::RandomState::random_integers </td><td>Random integers of type nlcpy.int64/nlcpy.int32 between low and high, inclusive.</td></tr>
#      <tr><td>@ref generator::RandomState::random_sample </td><td>Returns random floats in the half-open interval <span class="pre">[0.0, 1.0)</span>.</td></tr>
#      <tr><td>@ref generator::RandomState::ranf </td><td>This is an alias of random_sample.</td></tr>
#      <tr><td>@ref generator::RandomState::sample </td><td>This is an alias of random_sample.</td></tr>
#      <tr><td>@ref generator::RandomState::tomaxint </td><td>Random integers between 0 and numpy.iinfo(numpy.int).max, inclusive.</td></tr>
#      </table>
# ###Permutations###
#      <table>
#      <tr><td>@ref generator::RandomState::permutation </td><td>Randomly permutes a sequence, or returns a permuted range.</td></tr>
#      <tr><td>@ref generator::RandomState::shuffle </td><td>Modifies a sequence in-place by shuffling its contents.</td></tr>
#      </table>
# ###Distributions###
#      <table>
#      <tr><td>@ref generator::RandomState::binomial </td><td>Draws samples from a binomial distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::exponential </td><td>Draws samples from an exponential distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::gamma </td><td>Draws samples from a Gamma distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::geometric </td><td>Draws samples from a geometric distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::gumbel </td><td>Draws samples from a Gumbel distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::logistic </td><td>Draws samples from a logistic distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::lognormal </td><td>Draws samples from a log-normal distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::normal </td><td>Draws random samples from a normal (Gaussian) distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::poisson </td><td>Draws samples from a Poisson distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::standard_cauchy </td><td>Draws samples from a standard Cauchy distribution with mode = 0.</td></tr>
#      <tr><td>@ref generator::RandomState::standard_exponential </td><td>Draws samples from a standard exponential distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::standard_gamma </td><td>Draws samples from a standard Gamma distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::standard_normal </td><td>Draws samples from a standard Normal distribution (mean=0, stdev=1).</td></tr>
#      <tr><td>@ref generator::RandomState::uniform </td><td>Draws samples from a uniform distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::weibull </td><td>Draws samples from a Weibull distribution.</td></tr>
#      </table>
# @endif

## @defgroup ClassRandomStateTree RandomState
# @if(lang_ja)
# @else
# @details
# The Generator provides access to a wide variety of probability distributions, and serves as a replacement for RandomState.@n@n
# An easy example of RandomState is below:
# @code
# # Uses the nlcpy.random.RandomState
# from nlcpy import random
# random.standard_normal()
# @endcode
# @n
# The following tables show that nlcpy.random.RandomState class has methods to generate random numbers:@n
# ### Seeding and State ###
#      <table>
#      <tr><td>@ref generator::RandomState::get_state </td><td>Returns a ndarray representing the internal state of the generator.</td></tr>
#      <tr><td>@ref generator::RandomState::seed </td><td>Reseeds a default bit generator(MT19937), which provide a stream of random bits.</td></tr>
#      <tr><td>@ref generator::RandomState::set_state </td><td>Sets the internal state of the generator from a ndarray.</td></tr>
#      </table>
# ### Simple Random Data ###
#      <table>
#      <tr><td>@ref generator::RandomState::bytes </td><td>Returns random bytes.</td></tr>
#      <tr><td>@ref generator::RandomState::rand </td><td>Random values in a given shape.</td></tr>
#      <tr><td>@ref generator::RandomState::randint </td><td>Returns random integers from low (inclusive) to high (exclusive).</td></tr>
#      <tr><td>@ref generator::RandomState::randn </td><td>Returns a sample (or samples) from a "standard normal" distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::random </td><td>Returns random floats in the half-open interval <span class="pre">[0.0, 1.0)</span>.</td></tr>
#      <tr><td>@ref generator::RandomState::random_integers </td><td>Random integers of type nlcpy.int64/nlcpy.int32 between low and high, inclusive.</td></tr>
#      <tr><td>@ref generator::RandomState::random_sample </td><td>Returns random floats in the half-open interval <span class="pre">[0.0, 1.0)</span>.</td></tr>
#      <tr><td>@ref generator::RandomState::ranf </td><td>This is an alias of random_sample.</td></tr>
#      <tr><td>@ref generator::RandomState::sample </td><td>This is an alias of random_sample.</td></tr>
#      <tr><td>@ref generator::RandomState::tomaxint </td><td>Random integers between 0 and numpy.iinfo(numpy.int).max, inclusive.</td></tr>
#      </table>
# ###Permutations###
#      <table>
#      <tr><td>@ref generator::RandomState::permutation </td><td>Randomly permutes a sequence, or returns a permuted range.</td></tr>
#      <tr><td>@ref generator::RandomState::shuffle </td><td>Modifies a sequence in-place by shuffling its contents.</td></tr>
#      </table>
# ###Distributions###
#      <table>
#      <tr><td>@ref generator::RandomState::binomial </td><td>Draws samples from a binomial distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::exponential </td><td>Draws samples from an exponential distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::gamma </td><td>Draws samples from a Gamma distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::geometric </td><td>Draws samples from a geometric distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::gumbel </td><td>Draws samples from a Gumbel distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::logistic </td><td>Draws samples from a logistic distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::lognormal </td><td>Draws samples from a log-normal distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::normal </td><td>Draws random samples from a normal (Gaussian) distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::poisson </td><td>Draws samples from a Poisson distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::standard_cauchy </td><td>Draws samples from a standard Cauchy distribution with mode = 0.</td></tr>
#      <tr><td>@ref generator::RandomState::standard_exponential </td><td>Draws samples from a standard exponential distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::standard_gamma </td><td>Draws samples from a standard Gamma distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::standard_normal </td><td>Draws samples from a standard Normal distribution (mean=0, stdev=1).</td></tr>
#      <tr><td>@ref generator::RandomState::uniform </td><td>Draws samples from a uniform distribution.</td></tr>
#      <tr><td>@ref generator::RandomState::weibull </td><td>Draws samples from a Weibull distribution.</td></tr>
#      </table>
# @endif

## @defgroup ClassGeneratorTree Generator
# @if(lang_ja)
# @else
# @details
# The Generator provides access to a wide variety of probability distributions, and serves as a replacement for RandomState.@n@n
# An easy example of Generator is below:
# @code
# from nlcpy.random import Generator, MT19937
# rng = Generator(MT19937(12345))
# rng.standard_normal()
# @endcode
# And, an easy example of using default_rng is below:
# @code
# import nlcpy as vp
# rng = vp.random.default_rng()
# rng.standard_normal()
# @endcode
# @n
# The following tables show that nlcpy.random.Generator class has methods to generate random numbers:@n
# ### Functions ###
#      <table>
#      <tr><td>@ref _generator::default_rng </td><td>Constructs a new nlcpy.random.Generator with the default BitGenerator (MT19937).</td></tr>
#      </table>
# ### Simple Random Data ###
#      <table>
#      <tr><td>@ref _generator::Generator::bytes </td><td>Returns random bytes.</td></tr>
#      <tr><td>@ref _generator::Generator::integers </td><td>Returns random integers from low (inclusive) to high (exclusive), or if endpoint=True, low (inclusive) to high (inclusive).</td></tr>
#      <tr><td>@ref _generator::Generator::random </td><td>Returns random floats in the half-open interval <span class="pre">[0.0, 1.0)</span>.</td></tr>
#      </table>
# ### Permutations ###
#      <table>
#      <tr><td>@ref _generator::Generator::permutation </td><td>Randomly permutes a sequence, or returns a permuted range.</td></tr>
#      <tr><td>@ref _generator::Generator::shuffle </td><td>Modifies a sequence in-place by shuffling its contents.</td></tr>
#      </table>
# ### Distributions ###
#      <table>
#      <tr><td>@ref _generator::Generator::binomial </td><td>Draws samples from a binomial distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::exponential </td><td>Draws samples from an exponential distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::gamma </td><td>Draws samples from a Gamma distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::geometric </td><td>Draws samples from a geometric distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::gumbel </td><td>Draws samples from a Gumbel distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::logistic </td><td>Draws samples from a logistic distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::lognormal </td><td>Draws samples from a log-normal distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::normal </td><td>Draws random samples from a normal (Gaussian) distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::poisson </td><td>Draws samples from a Poisson distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::standard_cauchy </td><td>Draws samples from a standard Cauchy distribution with mode = 0.</td></tr>
#      <tr><td>@ref _generator::Generator::standard_exponential </td><td>Draws samples from a standard exponential distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::standard_gamma </td><td>Draws samples from a standard Gamma distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::standard_normal </td><td>Draws samples from a standard Normal distribution (mean=0, stdev=1).</td></tr>
#      <tr><td>@ref _generator::Generator::uniform </td><td>Draws samples from a uniform distribution.</td></tr>
#      <tr><td>@ref _generator::Generator::weibull </td><td>Draws samples from a Weibull distribution.</td></tr>
#      </table>
# @endif

##
# @ingroup Reference
# @defgroup Sorting_Searching_and_Counting Sorting and Searching
#
# @details
# The following tables show sorting and searching functions provided by NLCPy.
# ## Sorting ##
#      <table>
#      <tr><td>@ref sort::argsort    </td><td>Returns the indices that would sort an array.</td></tr>
#      <tr><td>@ref sort::sort     </td><td>Returns a sorted copy of an array.</td></tr>
#      </table>
# ## Searching ##
#      <table>
#      <tr><td>@ref searching::argmax   </td><td>Returns the indices of the maximum values along an axis.</td></tr>
#      <tr><td>@ref searching::argmin   </td><td>Returns the indices of the minimum values along an axis.</td></tr>
#      <tr><td>@ref searching::argwhere </td><td>Finds the indices of array elements that are non-zero, grouped by element.</td></tr>
#      <tr><td>@ref searching::nonzero  </td><td>Returns the indices of the elements that are non-zero.</td></tr>
#      </table>
# <!-- ## Counting ## -->

# @defgroup Sorting Sorting

# @defgroup Searching Searching

# @defgroup Counting Counting

##
# @ingroup Reference
# @defgroup Statistics Statistics
#
# @details
# The following tables show statistical functions provided by NLCPy.
# ## Order Statistics ##
#      <table>
#      <tr><td>@ref order::amin     </td><td>Returns the minimum of an array or minimum along an axis.</td></tr>
#      <tr><td>@ref order::amax     </td><td>Returns the maximum of an array or maximum along an axis.</td></tr>
#      <tr><td>@ref order::nanmin   </td><td>Returns minimum of an array or minimum along an axis, ignoring any NaNs.</td></tr>
#      <tr><td>@ref order::nanmax   </td><td>Returns maximum of an array or maximum along an axis, ignoring any NaNs.</td></tr>
#      </table>
# ## Averages and Variances ##
#      <table>
#      <tr><td>@ref function_base::average   </td><td>Computes the weighted average along the specified axis.</td></tr>
#      <tr><td>@ref function_base::mean      </td><td>Computes the arithmetic mean along the specified axis.</td></tr>
#      <tr><td>@ref function_base::median    </td><td>Computes the median along the specified axis.</td></tr>
#      <tr><td>@ref function_base::nanmean   </td><td>Computes the arithmetic mean along the specified axis, ignoring NaNs.</td></tr>
#      <tr><td>@ref function_base::nanstd    </td><td>Computes the standard deviation along the specified axis, while ignoring NaNs.</td></tr>
#      <tr><td>@ref function_base::nanvar    </td><td>Computes the variance along the specified axis, while ignoring NaNs.</td></tr>
#      <tr><td>@ref function_base::std       </td><td>Computes the standard deviation along the specified axis.</td></tr>
#      <tr><td>@ref function_base::var       </td><td>Computes the variance along the specified axis.</td></tr>
#      </table>
# ## Correlating ##
#      <table>
#      <tr><td>@ref function_base::corrcoef  </td><td>Returns Pearson product-moment correlation coefficients.</td></tr>
#      <tr><td>@ref function_base::cov       </td><td>Estimates a covariance matrix, given data and weights.</td></tr>
#      </table>
#

# @defgroup Order Order Statistics

# @defgroup Averages Averages and variances

# @defgroup Corr Correlating


##
# @ingroup Reference
# @defgroup Linear_algebra Linear Algebra
#
# @details
# The following table shows linear algebra routines provided by NLCPy.
# ## Matrix and Vector Products ##
#      <table>
#      <tr><td>@ref products::dot "dot"</td><td>Dot product of two arrays</td></tr>
#      <tr><td>@ref ufuncs::matmul "matmul"</td><td>Matrix product of two arrays.</td></tr>
#      </table>


##
# @ingroup Reference
# @defgroup Logic_functions Logic Functions
#
# @details
# The following tables show logical functions provided by NLCPy.
# ## Truth Value Testing ##
#      <table>
#      <tr><td>@ref testing::all "all"    </td><td>Tests whether all array elements along a given axis evaluate to True.</td></tr>
#      <tr><td>@ref testing::any "any"    </td><td>Tests whether any array elements along a given axis evaluate to True.</td></tr>
#      </table>
# ## Array Contents ##
#      <table>
#      <tr><td>@ref ufuncs::isfinite  </td><td>Tests whether input elements are neither @c inf nor @c nan, or not.</td></tr>
#      <tr><td>@ref ufuncs::isinf     </td><td>Tests whether input elements are @c inf, or not.</td></tr>
#      <tr><td>@ref ufuncs::isnan     </td><td>Tests whether input elements are @c nan, or not.</td></tr>
#      </table>
# ## Logical Operations ##
#      <table>
#      <tr><td>@ref ufuncs::logical_and   </td><td>Computes the logical AND of two arrays element-wise.</td></tr>
#      <tr><td>@ref ufuncs::logical_or    </td><td>Computes the logical OR of two arrays element-wise.</td></tr>
#      <tr><td>@ref ufuncs::logical_not   </td><td>Computes the logical NOT of the input array element-wise.</td></tr>
#      <tr><td>@ref ufuncs::logical_xor   </td><td>Computes the logical XOR of two arrays element-wise.</td></tr>
#      </table>
# ## Comparison ##
#      <table>
#      <tr><td>@ref ufuncs::greater       </td><td>Returns (@em x1 > @em x2), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::greater_equal </td><td>Returns (@em x1 >= @em x2), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::less          </td><td>Returns (@em x1 < @em x2), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::less_equal    </td><td>Returns (@em x1 <= @em x2), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::equal         </td><td>Returns (@em x1 == @em x2), element-wise.</td></tr>
#      <tr><td>@ref ufuncs::not_equal     </td><td>Returns (@em x1 != @em x2), element-wise.</td></tr>
#      </table>


## @defgroup RequestManaging Request Managing Routines
# @ingroup Reference
# @if(layng_ja)
# @else
# @details
# The following table show request managing routines provided by NLCPy.
#      <table>
#      <tr><td>@ref request::flush "flush" </td><td>Flushes stacked requests on VH to VE, and waits until VE exectuion is completed.</td></tr>
#      <tr><td>@ref request::get_offload_timing "get_offload_timing" </td><td>Gets kernel offload timing.</td></tr>
#      <tr><td>@ref request::set_offload_timing_lazy "set_offload_timing_lazy" </td><td>Sets kernel offload timing lazy.</td></tr>
#      <tr><td>@ref request::set_offload_timing_onthefly "set_offload_timing_onthefly" </td><td>Sets kernel offload timing on-the-fly.</td></tr>
#      </table>
#
# @endif



## @defgroup Profiling Profiling Routines
# @ingroup Reference
#
# @if(layng_ja)
# @else
# @details
# The following table show profiling routines provided by NLCPy.
#      <table>
#      <tr><td>@ref prof::get_run_stats "get_run_stats" </td><td>Gets dict of NLCPy run stats.</td></tr>
#      <tr><td>@ref prof::print_run_stats "print_run_stats" </td><td>Prints NLCPy run stats.</td></tr>
#      <tr><td>@ref prof::start_profiling "start_profiling" </td><td>Starts profiling.</td></tr>
#      <tr><td>@ref prof::stop_profiling "stop_profiling" </td><td>Stops profiling.</td></tr>
#      </table>
#
# @endif


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

# math_operations
_add_doc = '''
    Computes the element-wise addition of the inputs.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    add : ndarray
        The sum of *x1* and *x2*, element-wise. If *x1* and *x2* are both scalars, this
        function returns the result as a 0-dimension ndarray.

    Note
    ----
    Equivalent to `x1` + `x2` in terms of array broadcasting.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.add(1.0, 4.0)
    array(5.)
    >>> x1 = vp.arange(9.0).reshape((3, 3))
    >>> x2 = vp.arange(3.0)
    >>> vp.add(x1,x2)
    array([[ 0.,  2.,  4.],
           [ 3.,  5.,  7.],
           [ 6.,  8., 10.]])

'''
_subtract_doc = '''
    Computes the element-wise subtraction of the inputs.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be substracted from each other. If ``x1.shape != x2.shape``, they
        must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    subtract : ndarray
        The difference of *x1* and *x2*, element-wise. If *x1* and *x2* are both scalars,
        this function returns the result as a 0-dimension ndarray.

    Note
    ----
    Equivalent to `x1` - `x2` in terms of array broadcasting.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.subtract(1.0, 4.0)
    array(-3.)
    >>> x1 = vp.arange(9.0).reshape((3, 3))
    >>> x2 = vp.arange(3.0)
    >>> vp.subtract(x1,x2)
    array([[0., 0., 0.],
           [3., 3., 3.],
           [6., 6., 6.]])

'''
_multiply_doc = '''
    Computes the element-wise multiplication of the inputs.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be multiplied. If ``x1.shape != x2.shape``, they must be
        broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    multiply : ndarray
        The product of *x1* and *x2*, element-wise. If *x1* and *x2* are both scalars,
        this function returns the result as a 0-dimension ndarray.

    Note
    ----
    Equivalent to `x1` * `x2` in terms of array broadcasting.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.multiply(2.0, 4.0)
    array(8.)
    >>> x1 = vp.arange(9.0).reshape((3, 3))
    >>> x2 = vp.arange(3.0)
    >>> vp.multiply(x1,x2)
    array([[ 0.,  1.,  4.],
           [ 0.,  4., 10.],
           [ 0.,  7., 16.]])

'''
_divide_doc = '''
    Computes the element-wise division of the inputs.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is a dividend array and *x2 is a divisor array.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    divide : ndarray
        The result that *x1* is divided by *x2* for each element. If *x1* and *x2* are
        both scalars, this function returns the result as a 0-dimension ndarray.

    Note
    ----
    - If the values of `x2` are zero, the corresponding return values become nan
      (not inf) due to performance reasons.
    - Equivalent to `x1` / `x2` in terms of array broadcasting.
    - In Python 3.0 or later, ``//`` is the floor division operator and ``/`` is the true
      division operator. The ``divide(x1,x2)`` function is equivalent to the true
      division in Python.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(5)
    >>> vp.divide(x, 4)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> x/4
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> x//4
    array([0, 0, 0, 0, 1])

'''
_logaddexp_doc = '''
    Computes the element-wise natural logarithm of :math:`exp(x1) + exp(x2)`.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    logaddexp : ndarray
        An ndarray, containing :math:`log(exp(x1) + exp(x2))` for each element. If *x1*
        and *x2* are both scalars, this function returns the result as a 0-dimension
        ndarray.

    See Also
    --------
    logaddexp2 : Computes the element-wise base-2 logarithm of :math:`2^{x1} + 2^{x2}`.

    Examples
    --------
    >>> import nlcpy as vp
    >>> prob1 = vp.log(1e-50)
    >>> prob2 = vp.log(2.5e-50)
    >>> prob12 = vp.logaddexp(prob1, prob2)
    >>> prob12    # doctest: +SKIP
    array(-113.87649168)
    >>> vp.exp(prob12)  # doctest: +SKIP
    array(3.5e-50)

'''
_logaddexp2_doc = '''
    Computes the element-wise base-2 logarithm of :math:`2^{x1} + 2^{x2}`.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    logaddexp2 : ndarray
        An ndarray, containing :math:`log_2(2^{x1} + 2^{x2})` for each element. If *x1*
        and *x2* are both scalars, this function returns the result as a 0-dimension
        ndarray.

    See Also
    --------
    logaddexp : Computes the element-wise natural logarithm of
        :math:`exp(x1) + exp(x2)`.

    Examples
    --------
    >>> import nlcpy as vp
    >>> prob1 = vp.log2(1e-50)
    >>> prob2 = vp.log2(2.5e-50)
    >>> prob12 = vp.logaddexp2(prob1, prob2)
    >>> prob12 # doctest: +SKIP
    array(-164.28904982)
    >>> 2**prob12 # doctest: +SKIP
    array(3.5e-50)

'''
_true_divide_doc = '''
    Computes the element-wise division of the inputs.

    Instead of the Python traditional
    'floor division', this returns a true division. True division adjusts the output type
    to present the best answer, regardless of input types.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is a dividend array and *x2* is a divisor array.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    true_divide : ndarray
        The result that *x1* is divided by *x2* for each element. If *x1* and *x2* are
        both scalars, this function returns the result as a 0-dimension ndarray.

    Note
    ----
    - If the values of `x2` are zero, the corresponding return values become nan
      (not inf) due to performance reasons.
    - Equivalent to `x1` / `x2` in terms of array broadcasting.
    - In Python 3.0 or later, ``//`` is the floor division operator and ``/`` is the
      true division operator. The ``divide(x1,x2)`` function is equivalent to the true
      division in Python.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(5)
    >>> vp.true_divide(x, 4)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> x/4
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> x//4
    array([0, 0, 0, 0, 1])

'''
_floor_divide_doc = '''
    Computes the element-wise floor division of the inputs.

    It is equivalent to the Python ``//`` operator and pairs with the Python ``%``
    (:func:`remainder`), function so that ``a = a % b + b * (a // b)``
    up to roundoff.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is a numerator array and *x2* is a denominator array.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        *y* = floor(*x1*/*x2*). If *x1* and *x2* are both scalars, this function returns
        the result as a 0-dimension ndarray.

    Note
    ----
    - In Python 3.0 or later, ``//`` is the floor division operator and ``/`` is the
      true division operator. The ``floor_divide(x1,x2)`` function is equivalent to
      the floor division in Python.

    See Also
    --------
    remainder : Computes the element-wise remainder of division.
    divide : Computes the element-wise division of the inputs.
    floor : Returns the floor of the input, element-wise.
    ceil : Returns the ceiling of the input, element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.floor_divide(7,3)
    array(2)
    >>> vp.floor_divide([1., 2., 3., 4.], 2.5)
    array([0., 0., 1., 1.])

'''
_negative_doc = '''
    Computes numerical negative, element-wise.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        Returned array: *y* = -*x*. If *x* is a scalar, this function returns the
        result as a 0-dimension ndarray.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.negative([1.,-1.])
    array([-1.,  1.])

'''
_positive_doc = '''
    Computes numerical positive, element-wise.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        Returned array: *y* = +*x*. If *x* is a scalar, this function returns the
        result as a 0-dimension ndarray.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.positive([1.,-1.])
    array([ 1., -1.])

'''
_power_doc = '''
    Computes the element-wise exponentiation of the inputs.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is a base array and *x2* is an exponent array.
        If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The bases in *x1* raised to the exponents in *x2*.
        If *x1* and *x2* are both scalars, this function returns the result as a
        0-dimension ndarray.

    Examples
    --------

    Cube each element in a list.

    >>> import nlcpy as vp
    >>> x1 = vp.arange(6)
    >>> x1
    array([0, 1, 2, 3, 4, 5])
    >>> vp.power(x1, 3)
    array([  0,   1,   8,  27,  64, 125])

    Raise the bases to different exponents.

    >>> x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
    >>> vp.power(x1,x2)
    array([ 0.,  1.,  8., 27., 16.,  5.])

    The effect of broadcasting.

    >>> x2 = vp.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
    >>> x2
    array([[1, 2, 3, 3, 2, 1],
           [1, 2, 3, 3, 2, 1]])
    >>> vp.power(x1,x2)
    array([[ 0,  1,  8, 27, 16,  5],
           [ 0,  1,  8, 27, 16,  5]])

'''
_remainder_doc = '''
    Computes the element-wise remainder of division.

    Computes the remainder complementary to the :func:`floor_divide` function.
    It is equivalent to the Python modulus operator ``x1 % x2`` and has the same
    sign as the divisor *x2*.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is a dividend array and *x2* is a divisor array.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The element-wise remainder of the quotient ``floor_divide(x1,x2)``.
        If *x1* and *x2* are both scalars, this function returns the result as a
        0-dimension ndarray.

    Note
    ----
    - Returns 0 when `x2` is 0 and both `x1` and `x2` are integers.
    - mod : an alias of this function.

    See Also
    --------
    floor_divide : Computes the element-wise floor division of the inputs.
    fmod : Computes the element-wise remainder of division.
    divide : Computes the element-wise division of the inputs.
    floor : Returns the floor of the input, element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.remainder([4, 7], [2, 3])
    array([0, 1])
    >>> vp.remainder(vp.arange(7), 5)
    array([0, 1, 2, 3, 4, 0, 1])

'''
_mod_doc = '''
    Computes the element-wise remainder of division.

    Computes the remainder complementary to the :func:`floor_divide` function.
    It is equivalent to the Python modulus operator ``x1 % x2`` and has the same
    sign as the divisor *x2*.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is a dividend array and *x2* is a divisor array.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The element-wise remainder of the quotient ``floor_divide(x1,x2)``. If *x1*
        and *x2* are both scalars, this function returns the result as a 0-dimension
        ndarray.

    Note
    ----
    - Returns 0 when `x2` is 0 and both `x1` and `x2` are integers.
    - remainder : an alias of this function.

    See Also
    --------
    floor_divide : Computes the element-wise floor division of the inputs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.mod([4, 7], [2, 3])
    array([0, 1])
    >>> vp.mod(vp.arange(7), 5)
    array([0, 1, 2, 3, 4, 0, 1])

'''
_fmod_doc = '''
    Computes the element-wise remainder of division.

    This is the NLCPy implementation of the C library function fmod, the remainder
    has the same sign as the dividend *x1*.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is a dividend array and *x2* is a divisor array.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The element-wise remainder of the quotient ``floor_divide(x1,x2)``. If *x1*
        and *x2* are both scalars, this function returns the result as a 0-dimension
        ndarray.

    Note
    ----
    The result of the modulo operation for negative dividend and divisors is bound by
    conventions. For :func:`fmod`, the sign of result is the sign of the dividend, while
    for :func:`remainder` the sign of the result is the sign of the divisor.

    See Also
    --------
    remainder : Computes the element-wise remainder of division.
    divide : Computes the element-wise division of the inputs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.fmod([-3, -2, -1, 1, 2, 3], 2)
    array([-1,  0, -1,  1,  0,  1])
    >>> vp.remainder([-3, -2, -1, 1, 2, 3], 2)
    array([1, 0, 1, 1, 0, 1])
    >>> vp.fmod([5, 3], [2, 2.])
    array([1., 1.])
    >>> a = vp.arange(-3, 3).reshape(3, 2)
    >>> a
    array([[-3, -2],
           [-1,  0],
           [ 1,  2]])
    >>> vp.fmod(a, [2,2])
    array([[-1,  0],
           [-1,  0],
           [ 1,  0]])
'''
_divmod_doc = ''''''
_absolute_doc = '''
    Computes the element-wise absolute value.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the absolute value for each element in *x*. For complex
        input, ``a + ib``, the absolute value is :math:`\\sqrt{ a^2 + b^2 }`.
        If *x* is a scalar, this function returns the result as a 0-dimension ndarray.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.array([-1.2, 1.2])
    >>> vp.absolute(x)         # doctest: +SKIP
    array([1.2, 1.2])
    >>> vp.absolute(1.2 + 1j)  # doctest: +SKIP
    array(1.56204994)

'''
_fabs_doc = '''
    Computes the element-wise absolute value.

    This function returns the absolute values
    (positive magnitude) of the data in x. Complex values are not handled, use absolute
    to find the absolute values of complex data.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the absolute value for each element in *x*. If *x* is a
        scalar, this function returns the result as a 0-dimension ndarray.

    See Also
    --------
    fabs : Computes the element-wise absolute value.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.fabs(-1)
    array(1.)
    >>> x = vp.array([-1.2, 1.2])
    >>> vp.fabs(x)
    array([1.2, 1.2])

'''
_rint_doc = '''
    Computes the element-wise nearest integer.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the nearest integer for each element in *x*. If *x* is a
        scalar, this function returns the result as a 0-dimension ndarray.

    See Also
    --------
    ceil : Returns the ceiling of the input, element-wise.
    floor : Returns the floor of the input, element-wise.
    trunc : Returns the truncated value of the input, element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> vp.rint(a)
    array([-2., -2., -0.,  0.,  2.,  2.,  2.])

'''
_heaviside_doc = '''
    Computes Heaviside step function.

    The Heaviside step function is defined as follows::

                              0   if x1 < 0
        heaviside(x1, x2) =  x2   if x1 == 0
                              1   if x1 > 0

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        Heaviside step function of *x1* and *x2*, element-wise. If *x1* and *x2* are both
        scalars, this function returns the result as a 0-dimension ndarray.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.heaviside([-1.5, 0, 2.0], 0.5)
    array([0. , 0.5, 1. ])
    >>> vp.heaviside([-1.5, 0, 2.0], 1)
    array([0., 1., 1.])
'''
_conj_doc = '''
    Returns the element-wise complex conjugate.

    The complex conjugate of a complex number is obtained by changing the sign of
    its imaginary part.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the complex conjugate for each element in *x*. If *x* is a
        scalar, this function returns the result as a 0-dimension ndarray.

    Note
    ----
    :func:`nlcpy.conj` is an alias for :func:`nlcpy.conjugate`:

    >>> import nlcpy as vp
    >>> vp.conj is vp.conjugate
    True

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.conj(1+2j)
    array(1.-2.j)
    >>> x = vp.eye(2) + 1j * vp.eye(2)
    >>> vp.conj(x)
    array([[ 1.-1.j,  0.-0.j],
           [ 0.-0.j,  1.-1.j]])

'''
_conjugate_doc = '''
    Returns the element-wise complex conjugate.

    The complex conjugate of a complex number is obtained by changing the sign of
    its imaginary part.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the complex conjugate for each element in *x*. If *x* is a
        scalar, this function returns the result as a 0-dimension ndarray.

    Note
    ----
    :func:`nlcpy.conj` is an alias for :func:`nlcpy.conjugate`:

    >>> import nlcpy as vp
    >>> vp.conj is vp.conjugate
    True

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.conjugate(1+2j)
    array(1.-2.j)
    >>> x = vp.eye(2) + 1j * vp.eye(2)
    >>> vp.conjugate(x)
    array([[1.-1.j, 0.-0.j],
           [0.-0.j, 1.-1.j]])

'''
_exp_doc = '''
    Computes the exponential of all elements in the input array.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the exponential for each element in *x*. If *x* is a
        scalar, this function returns the result as a 0-dimension ndarray.

    See Also
    --------
    expm1 : Computes `exp(x) - 1` for all elements in the array.
    exp2 : Computes `2**x` for all elements in the array.

    Note
    ----
    The irrational number e is also known as Euler's number. It is approximately
    2.718281, and is the base of the natural logarithm, ``ln`` (this means that, if
    :math:`x = \\ln y = \\log_e y`, then :math:`e^x = y`. For real input, :math:`exp(x)`
    is always positive. For complex arguments, ``x = a + ib``, we can write
    :math:`e^x = e^a e^{ib}`. The first term, :math:`e^a`, is already known
    (it is the real argument, described above). The second term, :math:`e^{ib}`,
    is :math:`\\cos b + i \\sin b`, a function with magnitude 1 and a periodic phase.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.exp([1+2j, 3+4j, 5+6j])
    array([ -1.13120438 +2.47172667j, -13.12878308-15.20078446j,
           142.50190552-41.46893679j])

'''
_exp2_doc = '''
    Computes `2**x` for all elements in the array.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing `2**x`. If *x* is a scalar, this function
        returns the result as a 0-dimension ndarray.

    Restriction
    -----------
    - *dtype* is a complex dtype : *TypeError* occurs.

    See Also
    --------
    power : Computes the element-wise exponentiation of the inputs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.exp2([2, 3])
    array([4., 8.])

'''
_log_doc = '''
    Computes the element-wise natural logarithm of *x*.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the natural logarithm of *x*. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    Note
    ----
    - Logarithm is a multivalued function: for each `x` there is an infinite number of
      `z` such that `exp(z) = x`. The convention is to return the `z` whose imaginary
      part lies in [-pi, pi].
    - For real-valued input data types, :func:`log` always returns real output.
      For each value that cannot be expressed as a real number or infinity, it yields nan
      and sets the `invalid` floating point error flag.
    - For complex-valued input, :func:`log` is a complex analytical function that
      has a branch cut [-inf, 0] and is continuous from above on it. :func:`log`
      handles the floating-point negative zero as an infinitesimal negative number,
      conforming to the C99 standard.

    See Also
    --------
    log10 : Computes the element-wise base-10 logarithm of *x*.
    log2 : Computes the element-wise base-2 logarithm of *x*.
    log1p : Computes the element-wise natural logarithm of *1 + x*.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.log([1, vp.e, vp.e**2, vp.e**3]) #  doctest: +SKIP
    array([0., 1., 2., 3.])

'''
_log2_doc = '''
    Computes the element-wise base-2 logarithm of *x*.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the base-2 logarithm of *x*. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    Restriction
    -----------
    - *dtype* is a complex dtype : *TypeError* occurs.

    Note
    ----
    - Logarithm is a multivalued function: for each `x` there is an infinite number of
      `z` such that `2**z = x`. The convention is to return the `z` whose imaginary part
      lies in i[-pi, pi].
    - For real-valued input data types, :func:`log2` always returns real output.
      For each value that cannot be expressed as a real number or infinity, it yields nan
      and sets the `invalid` floating point error flag.

    See Also
    --------
    log : Computes the element-wise natural logarithm of *x*.
    log10 : Computes the element-wise base-10 logarithm of *x*.
    log1p : Computes the element-wise natural logarithm of *1 + x*.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.array([0, 1, 2, 2**4])
    >>> vp.log2(x)
    array([-inf,   0.,   1.,   4.])

'''
_log10_doc = '''
    Computes the element-wise base-10 logarithm of *x*.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the base-10 logarithm of *x*. NaNs are returned
        where *x* is negative. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    Restriction
    -----------
    - *dtype* is a complex dtype : *TypeError* occurs.

    Note
    ----
    - Logarithm is a multivalued function: for each `x` there is an infinite number of
      `z` such that `10**z = x`. The convention is to return the `z` whose imaginary part
      lies in [-pi, pi].
    - For real-valued input data types, :func:`log10` always returns real output.
      For each value that cannot be expressed as a real number or infinity, it yields nan
      and sets the `invalid` floating point error flag.

    See Also
    --------
    log : Computes the element-wise natural logarithm of *x*.
    log2 : Computes the element-wise base-2 logarithm of *x*.
    log1p : Computes the element-wise natural logarithm of *1 + x*.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.log10([1e-15, 3.])  # doctest: +SKIP
    array([-15.        ,   0.47712125])

'''
_log1p_doc = '''
    Computes the element-wise natural logarithm of *1 + x*.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the natural logarithm of *x* + 1. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    Restriction
    -----------
    - *dtype* is a complex dtype : *TypeError* occurs.

    Note
    ----
    - For real-valued input, :func:`log1p` is accurate also for `x` so small that
      `1 + x == 1` in floating-point accuracy.
    - Logarithm is a multivalued function: for each `x` there is an infinite number of
      `z` such that `exp(z) = 1 + x`. The convention is to return the `z` whose imaginary
      part lies in [-pi, pi].
    - For real-valued input data types, :func:`log1p` always returns real output.
      For each value that cannot be expressed as a real number or infinity, it yields nan
      and sets the `invalid` floating point error flag.

    See Also
    --------
    expm1 : Computes `exp(x) - 1` for all elements in the array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.log1p(1e-99)
    array(1.e-99)
    >>> vp.log(1 + 1e-99)
    array(0.)

'''
_expm1_doc = '''
    Computes `exp(x) - 1` for all elements in the array.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the exponential minus one: `y = exp(x) - 1`. If *x* is a
        scalar, this function returns the result as a 0-dimension ndarray.

    Restriction
    -----------
    - *dtype* is a complex dtype : *TypeError* occurs.

    Note
    ----
    This function provides greater precision than `exp(x) - 1` for small values of `x`.

    See Also
    --------
    log1p : Computes the element-wise natural logarithm of *1 + x*.

    Examples
    --------
    The true value of ``exp(1e-10) - 1`` is ``1.00000000005e-10`` to about 32 significant
    digits. This example shows the superiority of expm1 in this case.

    >>> import nlcpy as vp
    >>> vp.set_printoptions(16) # change displying digits
    >>> vp.expm1(1e-10)
    array(1.00000000005e-10)
    >>> vp.exp(1e-10) - 1
    array(1.000000082740371e-10)

'''
_sqrt_doc = '''
    Computes the element-wise square-root of the input.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the square-root for each element of *x*. If elements of *x*
        are real with negative elements, this function returns ``nan`` in *y*. If *x* is
        a scalar, this function returns the result as a 0-dimension ndarray.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.sqrt([1,4,9])
    array([1., 2., 3.])
    >>> vp.sqrt([4, -1, -3+4j])
    array([2.+0.j, 0.+1.j, 1.+2.j])
    >>> vp.sqrt([4, -1, vp.inf])
    array([ 2., nan, inf])

'''
_square_doc = '''
    Computes the element-wise square of the input.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the square `x*x`. If *x* is a scalar, this function
        returns the result as a 0-dimension ndarray.

    See Also
    --------
    sqrt : Computes the element-wise square-root of the input.
    power : Computes the element-wise exponentiation of the inputs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.square([-1j, 1])
    array([-1.+0.j,  1.+0.j])

'''
_cbrt_doc = '''
    Computes the element-wise cubic-root of the input.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the cubic-root for each element of *x*. If *x* is a scalar,
        this function returns the result as a 0-dimension ndarray.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.cbrt([1,8,27])
    array([1., 2., 3.])

'''
_reciprocal_doc = '''
    Computes the element-wise reciprocal of the input.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the reciprocal for each element of *x*. If *x* is a scalar,
        this function returns the result as a 0-dimension ndarray.

    Note
    ----
    - This function is not designed to work with integers.
    - For integer arguments with absolute value larger than 1 the result is always zero
      because of the way Python handles integer division. For integer zero the result is
      an overflow.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.reciprocal(2.)
    array(0.5)
    >>> vp.reciprocal([1, 2., 3.33])   # doctest: +SKIP
    array([1.       , 0.5      , 0.3003003])

'''
_gcd_doc = ''''''
_lcm_doc = ''''''

# trigonometric functions
_sin_doc = '''
    Computes the element-wise sine.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar in radians.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The sine values for each element of *x*. If *x* is a scalar, this function
        returns the result as a 0-dimension ndarray.

    See Also
    --------
    arcsin : Computes the element-wise inverse sine.
    sinh : Computes the element-wise hyperbolic sine.
    cos : Computes the element-wise cosine.

    Examples
    --------

    Print sine of one angle:

    >>> import nlcpy as vp
    >>> vp.sin(vp.pi/2.)     # doctest: +SKIP
    array(1.)

    Print sines of an array of angles given in degrees:

    >>> vp.sin(vp.array((0., 30., 45., 60., 90.)) * vp.pi / 180. )  # doctest: +SKIP
    array([0.        , 0.5       , 0.70710678, 0.8660254 , 1.        ])

'''
_cos_doc = '''
    Computes the element-wise cosine.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar in radians.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The cosine values for each element of *x*. If *x* is a scalar, this function
        returns the result as a 0-dimension ndarray.

    See Also
    --------
    arccos : Computes the element-wise inverse cosine.
    cosh : Computes the element-wise hyperbolic cosine.
    sin : Computes the element-wise sine.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.cos(vp.array([0, vp.pi/2, vp.pi]))   # doctest: +SKIP
    array([ 1.000000e+00,  6.123234e-17, -1.000000e+00])
    >>>
    >>> # Example of providing the optional output parameter
    >>> out1 = vp.array([0], dtype='d')
    >>> out2 = vp.cos([0.1], out=out1)
    >>> out2 is out1
    True

'''
_tan_doc = '''
    Computes the element-wise tangent.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar in radians.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The tangent values for each element of *x*. If *x* is a scalar, this function
        returns the result as a 0-dimension ndarray.

    See Also
    --------
    arctan : Computes the element-wise inverse tangent.
    tanh : Computes the element-wise hyperbolic tangent.

    Examples
    --------
    >>> import nlcpy as vp
    >>> from math import pi
    >>> vp.tan(vp.array([-pi,pi/2,pi]))  # doctest: +SKIP
    array([ 1.22464659e-16,  1.63312422e+16, -1.22464659e-16])
    >>>
    >>> # Example of providing the optional output parameter illustrating
    >>> # that what is returned is a reference to said parameter
    >>> out1 = vp.array([0], dtype='d')
    >>> out2 = vp.tan([0.1], out=out1)
    >>> out2 is out1
    True

'''
_arcsin_doc = '''
    Computes the element-wise inverse sine.

    The inverse of sin so that, if y = sin(x), then x = arcsin(y).

    Parameters
    ----------
    x : array_like
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The inverse sine values for each element of *x*. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    sin : Computes the element-wise sine.
    arccos : Computes the element-wise inverse cosine.
    arctan : Computes the element-wise inverse tangent.
    arctan2 : Computes the element-wise inverse tangent of *x1*/*x2* choosing the
        quadrant correctly.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.arcsin([-1, 0, 1])   # doctest: +SKIP
    array([-1.57079633,  0.        ,  1.57079633])    # [-pi/2, 0, pi/2]

'''
_arccos_doc = '''
    Computes the element-wise inverse cosine.

    The inverse of cos so that, if y = cos(x), then x = arccos(y).

    Parameters
    ----------
    x : array_like
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The inverse cosine values for each element of *x*. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    cos : Computes the element-wise cosine.
    arcsin : Computes the element-wise inverse sine.
    arctan : Computes the element-wise inverse tangent.
    arctan2 : Computes the element-wise inverse tangent of *x1*/*x2* choosing the
        quadrant correctly.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.arccos([1, -1])
    array([0.        , 3.14159265])

'''
_arctan_doc = '''
    Computes the element-wise inverse tangent.

    The inverse of tan so that, if y = tan(x), then x = arctan(y).

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The inverse tangent values for each element of *x*. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    tan : Computes the element-wise tangent.
    arcsin : Computes the element-wise inverse sine.
    arccos : Computes the element-wise inverse cosine.
    arctan2 : Computes the element-wise inverse tangent of *x1*/*x2* choosing the
        quadrant correctly.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.arctan([0, 1])    # doctest: +SKIP
    array([0.        , 0.78539816])  # [0, pi/4]

'''
_arctan2_doc = '''
    Computes the element-wise inverse tangent of *x1*/*x2* choosing the quadrant
    correctly.

    This function is not defined for complex-valued arguments; for the
    so-called argument of complex values, use angle.

    Parameters
    ----------
    x1, x2 : array_like
        The values of *x1* are *y-coordinates. Also, The values of *x2* are
        *x-coordinates. *x1* and *x2* must be real. If ``x1.shape != x2.shape``, they
        must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    angle : ndarray
        Array of angles in radians, in the range ``[-pi, pi]``. If *x1* and *x2* are both
        scalars, this function returns the result as a 0-dimension ndarray.

    See Also
    --------
    arctan : Computes the element-wise inverse tangent.
    tan : Computes the element-wise tangent.
    angle : Returns the angle of the complex argument.

    Examples
    --------

    Consider four points in different quadrants:

    >>> import nlcpy as vp
    >>> x = vp.array([-1, +1, +1, -1])
    >>> y = vp.array([-1, -1, +1, +1])
    >>> vp.arctan2(y, x) * 180 / vp.pi
    array([-135.,  -45.,   45.,  135.])

    Note the order of the parameters. :func:`arctan2` is defined also when
    *x2* = 0 and at several other special points, obtaining values in the range ``[-pi,
    pi]``:

    >>> vp.arctan2([1., -1.], [0., 0.])
    array([ 1.57079633, -1.57079633])
    >>> vp.arctan2([0., 0., vp.inf], [+0., -0., vp.inf])
    array([0.        , 3.14159265, 0.78539816])

'''
_hypot_doc = '''
    Computes the "legs" of a right triangle.

    Equivalent to sqrt(x1**2 + x2**2), element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Leg of the triangle(s). If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    z : ndarray
        The leg of the triangle(s). If *x1* and *x2* are both scalars, this function
        returns the result as a 0-dimension ndarray.

    See Also
    --------
    arctan : Computes the element-wise inverse tangent.
    tan : Computes the element-wise tangent.
    angle : Returns the angle of the complex argument.

    Examples
    --------

    Consider four points in different quadrants:

    >>> import nlcpy as vp
    >>> vp.hypot(3*vp.ones((3, 3)), 4*vp.ones((3, 3)))
    array([[5., 5., 5.],
           [5., 5., 5.],
           [5., 5., 5.]])

'''
_sinh_doc = '''
    Computes the element-wise hyperbolic sine.

    Equivalent to ``1/2 * (vp.exp(x) - vp.exp(-x))`` or ``-1j * vp.sin(1j*x)``.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The hyperbolic sine values for each element of *x*. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    arcsinh : Computes the element-wise inverse hyperbolic sine.
    cosh : Computes the element-wise hyperbolic cosine.
    tanh : Computes the element-wise hyperbolic tangent.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.sinh(0)
    array(0.)
    >>> vp.sinh(vp.pi*1j/2)
    array(0.+1.j)
    >>> vp.sinh(vp.pi*1j) # (exact value is 0)    # doctest: +SKIP
    array(-0.+1.2246468e-16j)
    >>> out1 = vp.array([0], dtype='d')
    >>> out2 = vp.sinh([0.1], out1)
    >>> out2 is out1
    True

'''
_cosh_doc = '''
    Computes the element-wise hyperbolic cosine.

    Equivalent to ``1/2 * (vp.exp(x) + vp.exp(-x))`` or ``vp.cos(1j*x)``.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The hyperbolic cosine values for each element of *x*. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    arccosh : Computes the element-wise inverse hyperbolic cosine.
    sinh : Computes the element-wise hyperbolic sine.
    tanh : Computes the element-wise hyperbolic tangent.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.sinh(0)
    array(0.)
    >>> vp.sinh(vp.pi*1j/2)
    array(0.+1.j)

'''
_tanh_doc = '''
    Computes the element-wise hyperbolic tangent.

    Equivalent to ``vp.sinh(x)/vp.cosh(x)`` or ``-1j * vp.tan(1j*x)``.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The hyperbolic tangent values for each element of *x*. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    arctanh : Computes the element-wise inverse hyperbolic tangent.
    sinh : Computes the element-wise hyperbolic sine.
    cosh : Computes the element-wise hyperbolic cosine.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.tanh((0, vp.pi*1j, vp.pi*1j/2))   # doctest: +SKIP
    array([0.+0.00000000e+00j, 0.-1.22464680e-16j, 0.+1.63312394e+16j])
    >>> # Example of providing the optional output parameter
    >>> out1 = vp.array([0], dtype='d')
    >>> out2 = vp.tanh([0.1], out1)
    >>> out2 is out1
    True

'''
_arcsinh_doc = '''
    Computes the element-wise inverse hyperbolic sine.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The inverse hyperbolic sine values for each element of *x*. If *x* is a scalar,
        this function returns the result as a 0-dimension ndarray.

    Note
    ----
    - :func:`arcsinh` is a multivalued function: for each `x` there are
      infinitely many numbers `z` such that  sinh(z) = x. The convention is to return
      the `z` whose imaginary part lies in [-pi/2, pi/2].
    - For real-valued input data types, :func:`arcsinh` always returns real
      output. For each value that cannot be expressed as a real number or infinity, it
      returns nan and sets the `invalid` floating point error flag.
    - For complex-valued input, :func:`arcsinh` is a complex analytical function
      that has branch cuts [1j, infj] and [-1j, -infj] and is continuous from the right
      on the former and from the left on the latter.

    See Also
    --------
    sinh : Computes the element-wise hyperbolic sine.
    arccosh : Computes the element-wise inverse hyperbolic cosine.
    arctanh : Computes the element-wise inverse hyperbolic tangent.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.arcsinh(vp.array([vp.e, 10.0]))  # doctest: +SKIP
    array([1.72538256, 2.99822295])

'''
_arccosh_doc = '''
    Computes the element-wise inverse hyperbolic cosine.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The inverse hyperbolic cosine values for each element of *x*. If *x* is a scalar,
        this function returns the result as a 0-dimension ndarray.

    Note
    ----
    - :func:`arccosh` is a multivalued function: for each `x` there are
      the infinitely many numbers `z` such that  cosh(z) = x. The convention is to return
      `z` whose imaginary part lies in [-pi, pi] and the real part in [-0, inf].
    - For real-valued input data types, :func:`arccosh` always returns real
      output. For each value that cannot be expressed as a real number or infinity, it
      returns nan and sets the `invalid` floating point error flag.
    - For complex-valued input, :func:`arccosh` is a complex analytical function
      that has a branch cut [-inf, 1] and is continuous from above on it.

    See Also
    --------
    cosh : Computes the element-wise hyperbolic cosine.
    arcsinh : Computes the element-wise inverse hyperbolic sine.
    arctanh : Computes the element-wise inverse hyperbolic tangent.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.arccosh([vp.e, 10.0])  # doctest: +SKIP
    array([1.65745445, 2.99322285])
    >>> vp.arccosh(1)
    array(0.)

'''
_arctanh_doc = '''
    Computes the element-wise inverse hyperbolic tangent.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The inverse hyperbolic tangent values for each element of *x*. If *x* is a
        scalar, this function returns the result as a 0-dimension ndarray.

    Note
    ----
    - :func:`arctanh` is a multivalued function: for each `x` there are
      infinitely many numbers `z` such that  tanh(z) = x. The convention is to return the
      `z` whose imaginary part lies in [-pi/2, pi/2].
    - For real-valued input data types, :func:`arctanh` always returns real
      output. For each value that cannot be expressed as a real number or infinity, it
      returns nan and sets the `invalid` floating point error flag.
    - For complex-valued input, :func:`arctanh` is a complex analytical function
      that has branch cuts [-1, -inf] and [1, inf] and is continuous from above on the
      former and from below on the latter.

    See Also
    --------
    tanh : Computes the element-wise hyperbolic tangent.
    arcsinh : Computes the element-wise inverse hyperbolic sine.
    arccosh : Computes the element-wise inverse hyperbolic cosine.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.arctanh([0, -0.5])
    array([ 0.        , -0.54930614])

'''
_degrees_doc = '''
    Converts angles from radians to degrees.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar, containing angles in radians.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The angles in degrees. If *x* is a scalar, this function returns the result as a
        0-dimension ndarray.

    See Also
    --------
    rad2deg : Converts angles from radians to degrees.
    radians : Converts angles from degrees to radians.
    deg2rad : Converts angles from degrees to radians.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.degrees(vp.pi/2)
    array(90.)

'''
_radians_doc = '''
    Converts angles from degrees to radians.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar containing angles in degrees.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The angles in radians. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    deg2rad : Converts angles from degrees to radians.
    degrees : Converts angles from radians to degrees.
    rag2deg : Converts angles from radians to degrees.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.radians(180)  # doctest: +SKIP
    array(3.14159265)

'''
_deg2rad_doc = '''
    Converts angles from degrees to radians.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar containing angles in degrees.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The angles in radians. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    radians : Converts angles from degrees to radians.
    rad2deg : Converts angles from radians to degrees.
    degrees : Converts angles from radians to degrees.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.deg2rad(180)
    array(3.14159265)

'''
_rad2deg_doc = '''
    Converts angles from radians to degrees.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar, containing angles in radians.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The angles in degrees. If *x* is a scalar, this function returns the result as a
        0-dimension ndarray.

    See Also
    --------
    degrees : Converts angles from radians to degrees.
    deg2rad : Converts angles from degrees to radians.
    radians : Converts angles from degrees to radians.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.rad2deg(vp.pi/2)
    array(90.)

'''

# bit-twiddling functions
_bitwise_and_doc = '''
    Computes the bit-wise AND of two arrays element-wise.

    This ufunc implements the C/Python operator ``&``.

    Parameters
    ----------
    x1, x2 : array_like
        Only integer and boolean types are handled. If ``x1.shape != x2.shape``, they
        must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        *y* = *x1* & *x2*. If *x1* and *x2* are both scalars, this function returns
        the result as a 0-dimension ndarray.

    See Also
    --------
    logical_and : Computes the logical AND of two arrays element-wise.
    bitwise_or : Computes the bit-wise OR of two arrays element-wise.
    bitwise_xor : Computes the bit-wise XOR of two arrays element-wise.

    Examples
    --------

    The number 13 is represented by 00001101. Likewise, 17 is represented by 00010001.
    The bit-wise AND of 13 and 17 is therefore 000000001, or 1:

    >>> import nlcpy as vp
    >>> vp.bitwise_and(13, 17)
    array(1)
    >>> vp.bitwise_and(14, 13)
    array(12)
    >>> vp.bitwise_and([14,3], 13)
    array([12,  1])
    >>> vp.bitwise_and([11,7], [4,25])
    array([0, 1])
    >>> vp.bitwise_and(vp.array([2,5,255]), vp.array([3,14,16]))
    array([ 2,  4, 16])
    >>> vp.bitwise_and([True, True], [False, True])
    array([False,  True])

'''
_bitwise_or_doc = '''
    Computes the bit-wise OR of two arrays element-wise.

    This ufunc implements the C/Python operator ``|``.

    Parameters
    ----------
    x1, x2 : array_like
        Only integer and boolean types are handled. If ``x1.shape != x2.shape``, they
        must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        *y* = *x1* | *x2*. If *x1* and *x2* are both scalars, this function returns
        the result as a 0-dimension ndarray.

    See Also
    --------
    logical_or : Computes the logical OR of two arrays element-wise.
    bitwise_and : Computes the bit-wise AND of two arrays element-wise.
    bitwise_xor : Computes the bit-wise XOR of two arrays element-wise.

    Examples
    --------

    The number 13 has the binaray representation 00001101. Likewise, 16 is represented by
    00010000. The bit-wise OR of 13 and 16 is then 000111011, or 29:

    >>> import nlcpy as vp
    >>> vp.bitwise_or(13, 16)
    array(29)
    >>> vp.bitwise_or(32, 2)
    array(34)
    >>> vp.bitwise_or([33,3], 1)
    array([33,  3])
    >>> vp.bitwise_or([33, 4], [1, 2])
    array([33,  6])
    >>> vp.bitwise_or(vp.array([2, 5, 255]), vp.array([4, 4, 4]))
    array([  6,   5, 255])
    >>> vp.array([2, 5, 255]) | vp.array([4, 4, 4])
    array([  6,   5, 255])
    >>> vp.bitwise_or(vp.array([2, 5, 255, 2147483647], dtype=vp.int32),
    ...               vp.array([4, 4, 4, 2147483647], dtype=vp.int32))
    array([         6,          5,        255, 2147483647], dtype=int32)
    >>> vp.bitwise_or([True, True], [False, True])
    array([ True,  True])

'''
_bitwise_xor_doc = '''
    Computes the bit-wise XOR of two arrays element-wise.

    This ufunc implements the C/Python operator ``^``.

    Parameters
    ----------
    x1, x2 : array_like
        Only integer and boolean types are handled. If ``x1.shape != x2.shape``, they
        must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        *y* = *x1* ^ *x2*. If *x1* and *x2* are both scalars, this function returns
        the result as a 0-dimension ndarray.

    See Also
    --------
    logical_xor : Computes the logical XOR of two arrays element-wise.
    bitwise_and : Computes the bit-wise AND of two arrays element-wise.
    bitwise_or : Computes the bit-wise OR of two arrays element-wise.

    Examples
    --------

    The number 13 is represented by 00001101. Likewise, 17 is represented by 00010001.
    The bit-wise XOR of 13 and 17 is therefore 00011100, or 28:

    >>> import nlcpy as vp
    >>> vp.bitwise_xor(13, 17)
    array(28)
    >>> vp.bitwise_xor(31, 5)
    array(26)
    >>> vp.bitwise_xor([31,3], 5)
    array([26,  6])
    >>> vp.bitwise_xor([31,3], [5,6])
    array([26,  5])
    >>> vp.bitwise_xor([True, True], [False, True])
    array([ True, False])

'''
_invert_doc = '''
    Computes the bit-wise NOT element-wise.

    This ufunc implements the C/Python operator ``~``.

    Parameters
    ----------
    x : array_like
        Only integer and boolean types are handled. If ``x1.shape != x2.shape``, they
        must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        *y* = ~ *x*. If *x* is a scalar, this function returns the result as a
        0-dimension ndarray.

    See Also
    --------
    bitwise_and : Computes the bit-wise AND of two arrays element-wise.
    bitwise_or : Computes the bit-wise OR of two arrays element-wise.
    bitwise_xor : Computes the bit-wise XOR of two arrays element-wise.
    logical_not : Computes the logical NOT of the input array element-wise.

    Examples
    --------

    >>> import nlcpy as vp
    >>> x = vp.invert(vp.array(13, dtype=vp.uint32))
    >>> x
    array(4294967282, dtype=uint32)

    When using signed integer types the result is the two's complement of the result for
    the unsigned type:

    >>> vp.invert(vp.array([13], dtype=vp.int32))
    array([-14], dtype=int32)

    Booleans are accepted as well:

    >>> vp.invert(vp.array([True, False]))
    array([False,  True])

'''
_left_shift_doc = '''
    Shifts bits of an integer to the left, element-wise.

    Bits are shifted to the left by appending 0 at the right of *x1*.
    Because the internal representation of integer numbers is in binary format,
    this operation is equivalent to multiplying :math:`x1*2^{x2}`.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is an input array or a scalar. *x2* is the number of zeros to append to
        *x1*. If ``x1.shape != x2.shape``, they must be
        broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        *x1* with bits shifted *x2* times to the left. If *x1* and *x2* are both scalars,
        this function returns the result as a 0-dimension ndarray.

    Note
    ----
    - If the values of `x2` are greater equal than the bit-width of `x1,` this function
      returns zero.
    - If the values of `x2` are negative numbers, undefined values are returned.

    See Also
    --------
    right_shift : Shifts bits of an integer to the right, element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.left_shift(5, 2)
    array(20)
    >>> vp.left_shift(5, [1,2,3])
    array([10, 20, 40])

'''
_right_shift_doc = '''
    Shifts bits of an integer to the right, element-wise.

    Because the internal representation of numbers is in binary format,
    this operation is equivalent to multiplying :math:`x1/2^{x2}`.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is an input array or a scalar. *x2* is the number of bits to remove at the
        right of *x1*. If ``x1.shape != x2.shape``, they must be
        broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        *x1* with bits shifted *x2* times to the right. If *x1* and *x2* are both
        scalars, this function returns the result as a 0-dimension ndarray.

    Note
    ----
    - If the values of `x2` are greater equal than the bit-width of `x1,` this function
      returns zero.
    - If the values of `x2` are negative numbers, undefined values are returned.

    See Also
    --------
    left_shift : Shifts bits of an integer to the left, element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.right_shift(10, 1)
    array(5)
    >>> vp.right_shift(10, [1,2,3])
    array([5, 2, 1])

'''

# comparison functions
_greater_doc = '''
    Returns (*x1* > *x2*), element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the result of the element-wise comparison of *x1* and *x2;*
        the shape is determined by broadcasting. If *x1* and *x2* are both scalars, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    greater_equal : Returns (*x1* >= *x2*), element-wise.
    less : Returns (*x1* < *x2*), element-wise.
    less_equal : Returns (*x1* <= *x2*), element-wise.
    not_equal : Returns (*x1* != *x2*), element-wise.
    equal : Returns (*x1* == *x2*), element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.greater([4,2],[2,2])
    array([ True, False])

    If the inputs are ndarrays, then vp.greater is equivalent to '>'.

    >>> a = vp.array([4,2])
    >>> b = vp.array([2,2])
    >>> a > b
    array([ True, False])

'''
_greater_equal_doc = '''
    Returns (*x1* >= *x2*), element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the result of the element-wise comparison of *x1* and *x2;*
        the shape is determined by broadcasting. If *x1* and *x2* are both scalars, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    greater : Returns (*x1* > *x2*), element-wise.
    less : Returns (*x1* < *x2*), element-wise.
    less_equal : Returns (*x1* <= *x2*), element-wise.
    not_equal : Returns (*x1* != *x2*), element-wise.
    equal : Returns (*x1* == *x2*), element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.greater_equal([4, 2, 1], [2, 2, 2])
    array([ True,  True, False])
'''
_less_doc = '''
    Returns (*x1* < *x2*), element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the result of the element-wise comparison of *x1* and *x2;*
        the shape is determined by broadcasting. If *x1* and *x2* are both scalars, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    greater : Returns (*x1* > *x2*), element-wise.
    greater_equal : Returns (*x1* >= *x2*), element-wise.
    less_equal : Returns (*x1* <= *x2*), element-wise.
    not_equal : Returns (*x1* != *x2*), element-wise.
    equal : Returns (*x1* == *x2*), element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.less([1, 2], [2, 2])
    array([ True, False])

'''
_less_equal_doc = '''
    Returns (*x1* <= *x2*), element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the result of the element-wise comparison of *x1* and *x2;*
        the shape is determined by broadcasting. If *x1* and *x2* are both scalars, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    greater : Returns (*x1* > *x2*), element-wise.
    greater_equal : Returns (*x1* >= *x2*), element-wise.
    less : Returns (*x1* < *x2*), element-wise.
    not_equal : Returns (*x1* != *x2*), element-wise.
    equal : Returns (*x1* == *x2*), element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.less_equal([4, 2, 1], [2, 2, 2])
    array([False,  True,  True])

'''
_not_equal_doc = '''
    Returns (*x1* != *x2*), element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the result of the element-wise comparison of *x1* and *x2;*
        the shape is determined by broadcasting. If *x1* and *x2* are both scalars, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    equal : Returns (*x1* == *x2*), element-wise.
    greater_equal : Returns (*x1* >= *x2*), element-wise.
    less_equal : Returns (*x1* <= *x2*), element-wise.
    greater : Returns (*x1* > *x2*), element-wise.
    less : Returns (*x1* < *x2*), element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.not_equal([1.,2.], [1., 3.])
    array([False,  True])
    >>> vp.not_equal([1, 2], [[1, 3],[1, 4]])
    array([[False,  True],
           [False,  True]])

'''
_equal_doc = '''
    Returns (*x1* == *x2*), element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the result of the element-wise comparison of *x1* and *x2;*
        the shape is determined by broadcasting. If *x1* and *x2* are both scalars, this
        function returns the result as a 0-dimension ndarray.

    See Also
    --------
    not_equal : Returns (*x1* != *x2*), element-wise.
    greater_equal : Returns (*x1* >= *x2*), element-wise.
    less_equal : Returns (*x1* <= *x2*), element-wise.
    greater : Returns (*x1* > *x2*), element-wise.
    less : Returns (*x1* < *x2*), element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.equal([0, 1, 3], vp.arange(3))
    array([ True,  True, False])

    What is compared are values, not types. So an int (1) and an array of length one can
    evaluate as True:

    >>> vp.equal(1, vp.ones(1))
    array([ True])

'''
_logical_and_doc = '''
    Computes the logical AND of two arrays element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        Boolean result of the logical AND operation applied to the elements of *x1* and
        *x2;* the shape is determined by broadcasting. If *x1* and *x2* are both scalars,
        this function returns the result as a 0-dimension ndarray.

    See Also
    --------
    logical_or : Computes the logical OR of two arrays element-wise.
    logical_not : Computes the logical NOT of the input array element-wise.
    logical_xor : Computes the logical XOR of two arrays element-wise.
    bitwise_and : Computes the bit-wise AND of two arrays element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.logical_and(True, False)
    array(False)
    >>> vp.logical_and([True, False], [False, False])
    array([False, False])
    >>> x = vp.arange(5)
    >>> vp.logical_and(x>1, x<4)
    array([False, False,  True,  True, False])

'''
_logical_or_doc = '''
    Computes the logical OR of two arrays element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        Boolean result of the logical OR operation applied to the elements of *x1* or
        *x2;* the shape is determined by broadcasting. If both *x1* or *x2* are scalars,
        this function returns the result as a 0-dimension ndarray.

    See Also
    --------
    logical_and : Computes the logical AND of two arrays element-wise.
    logical_not : Computes the logical NOT of the input array element-wise.
    logical_xor : Computes the logical XOR of two arrays element-wise.
    bitwise_and : Computes the bit-wise AND of two arrays element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.logical_or(True, False)
    array(True)
    >>> vp.logical_or([True, False], [False, False])
    array([ True, False])
    >>> x = vp.arange(5)
    >>> vp.logical_or(x < 1, x > 3)
    array([ True, False, False, False,  True])

'''
_logical_xor_doc = '''
    Computes the logical XOR of two arrays element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        Boolean result of the logical XOR operation applied to the elements of *x1* and
        *x2;* the shape is determined by broadcasting. If *x1* and *x2* are both scalars,
        this function returns the result as a 0-dimension ndarray.

    See Also
    --------
    logical_or : Computes the logical OR of two arrays element-wise.
    logical_not : Computes the logical NOT of the input array element-wise.
    logical_xor : Computes the logical XOR of two arrays element-wise.
    bitwise_and : Computes the bit-wise AND of two arrays element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.logical_xor(True, False)
    array(True)
    >>> vp.logical_xor([True, True, False, False], [True, False, True, False])
    array([False,  True,  True, False])
    >>> x = vp.arange(5)
    >>> vp.logical_xor(x < 1, x > 3)
    array([ True, False, False, False,  True])

    Simple example showing support of broadcasting

    >>> vp.logical_xor(0, vp.eye(2))
    array([[ True, False],
           [False,  True]])

'''
_logical_not_doc = '''
    Computes the logical NOT of the input array element-wise.

    Parameters
    ----------
    x : array_like
        Logical NOT is applied to the elements of *x*.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        Boolean result with the same shape as *x* of the logical NOT operation on
        elements of *x*. If *x* is a scalar, this function returns the result as a
        0-dimension ndarray.

    See Also
    --------
    logical_and : Computes the logical AND of two arrays element-wise.
    logical_or : Computes the logical OR of two arrays element-wise.
    logical_xor : Computes the logical XOR of two arrays element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.logical_not(3)
    array(False)
    >>> vp.logical_not([True, False, 0, 1])
    array([False,  True,  True, False])
    >>> x = vp.arange(5)
    >>> vp.logical_not(x<3)
    array([False, False, False,  True,  True])

'''
_maximum_doc = '''
    Computes the element-wise maximum of the inputs.

    Compare two arrays and returns a new array containing the element-wise maxima.
    If one of the elements being compared
    is a NaN, then that element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which are defined as
    at least one of the real or imaginary parts being a NaN. The net effect is that NaNs
    are propagated.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars, containing the elements to be compared.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The maximum of *x1* and *x2*, element-wise. If *x1* and *x2* are both scalars,
        this function returns the result as a 0-dimension ndarray.

    Note
    ----
    The maximum is equivalent to ``nlcpy.where(x1 >= x2, x1, x2)``  when neither
    `x1` nor `x2` are nans, but it is faster and does proper broadcasting.

    See Also
    --------
    minimum : Computes the element-wise minimum of the inputs. Computes the
        element-wise minimum of the inputs.
    fmax : Computes the element-wise maximum of the inputs
    amax : Returns the maximum of an array or maximum along an axis.
    nanmax : Returns the maximum of an array or maximum along an axis, ignoring
        any NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.maximum([2, 3, 4], [1, 5, 2])
    array([2, 5, 4])
    >>> vp.maximum(vp.eye(2), [0.5, 2]) # broadcasting
    array([[1. , 2. ],
           [0.5, 2. ]])
    >>> vp.maximum([vp.nan, 0, vp.nan], [0, vp.nan, vp.nan])
    array([nan, nan, nan])
    >>> vp.maximum(vp.Inf, 1)
    array(inf)

'''
_minimum_doc = '''
    Computes the element-wise minimum of the inputs.

    Compare two arrays and returns a new array containing the element-wise minima.
    If one of the elements being compared is a NaN, then that element is returned.
    If both elements are NaNs then the first is returned. The latter distinction is
    important for complex NaNs, which are defined as at least one of the real or
    imaginary parts being a NaN. The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars, containing the elements to be compared.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The minimum of *x1* and *x2*, element-wise. If *x1* and *x2* are both scalars,
        this function returns the result as a 0-dimension ndarray.

    Note
    ----
    The minmum is equivalent to ``nlcpy.where(x1 <= x2, x1, x2)``  when neither
    `x1` nor `x2` are nans, but it is faster and does proper broadcasting.

    See Also
    --------
    maximum : Computes the element-wise maximum of the inputs.
    fmin : Computes the element-wise minimum of the inputs
    amin : Returns the minimum of an array or minimum along an axis.
    nanmin : Returns the minimum of an array or minimum along an axis, ignoring
        any NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.minimum([2, 3, 4], [1, 5, 2])
    array([1, 3, 2])
    >>> vp.minimum(vp.eye(2), [0.5, 2]) # broadcasting
    array([[0.5, 0. ],
           [0. , 1. ]])
    >>> vp.minimum([vp.nan, 0, vp.nan],[0, vp.nan, vp.nan])
    array([nan, nan, nan])
    >>> vp.minimum(-vp.Inf, 1)
    array(-inf)

'''
_fmax_doc = '''
    Computes the element-wise maximum of the inputs.

    Compare two arrays and returns a new array containing the element-wise maxima.
    If one of the elements being compared is a  NaN, then the non-nan element is
    returned. If both elements are NaNs then the first is returned. The latter
    distinction is important for complex NaNs, which are defined as at least one of
    the real or imaginary parts being a NaN. The net effect is that NaNs are ignored
    when possible.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars, containing the elements to be compared.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The maximum of *x1* and *x2*, element-wise. If *x1* and *x2* are both scalars,
        this function returns the result as a 0-dimension ndarray.

    Note
    ----
    The fmax is equivalent to nlcpy.where(x1 >= x2, x1,x2) when neither `x1` nor `x2`
    are nans, but it is faster and does proper broadcasting.

    See Also
    --------
    fmin : Computes the element-wise minimum of the inputs
    maximum : Computes the element-wise maximum of the inputs.
    amax : Returns the maximum of an array or maximum along an axis.
    nanmax : Returns the maximum of an array or maximum along an axis, ignoring
        any NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.fmax([2, 3, 4], [1, 5, 2])
    array([2, 5, 4])
    >>> vp.fmax(vp.eye(2), [0.5, 2]) # broadcasting
    array([[1. , 2. ],
           [0.5, 2. ]])
    >>> vp.fmax([vp.nan, 0, vp.nan], [0, vp.nan, vp.nan])
    array([ 0.,  0., nan])

'''
_fmin_doc = '''
    Computes the element-wise minimum of the inputs.

    Compare two arrays and returns a new array containing the element-wise minima.
    If one of the elements being compared is a NaN, then the non-nan element is returned.
    If both elements are NaNs then the first is returned. The latter distinction is
    important for complex NaNs, which are defined as at least one of the real or
    imaginary parts being a NaN. The net effect is that NaNs are ignored when possible.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays or scalars, containing the elements to be compared.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The fmin of *x1* and *x2*, element-wise. If *x1* and *x2* are both scalars, this
        function returns the result as a 0-dimension ndarray.

    Note
    ----
    The minmum is equivalent to nlcpy.where(x1  when neither `x1` nor `x2` are nans, but
    it is faster and does proper broadcasting.

    See Also
    --------
    fmax : Computes the element-wise maximum of the inputs
    minimum : Computes the element-wise minimum of the inputs.
    amin : Returns the minimum of an array or minimum along an axis.
    nanmin : Returns the minimum of an array or minimum along an axis, ignoring
        any NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.fmin([2, 3, 4], [1, 5, 2])
    array([1, 3, 2])
    >>> vp.fmin(vp.eye(2), [0.5, 2]) # broadcasting
    array([[0.5, 0. ],
           [0. , 1. ]])
    >>> vp.fmin([vp.nan, 0, vp.nan],[0, vp.nan, vp.nan])
    array([ 0.,  0., nan])

'''

# floating functions
_isfinite_doc = '''
    Tests whether input elements are neither inf nor nan, or not.

    The result is returned as a boolean array.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar, containing the elements to be tested.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        ``True`` where x is not positive infinity, negative infinity, or NaN;
        ``False`` otherwise. If *x* is a scalar, this function returns the result as a
        0-dimension ndarray.

    Note
    ----
    Not a Number, positive infinity and negative infinity are considered to be
    non-finite.

    NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE754). This
    means that Not a Number is not equivalent to infinity. Also that positive infinity is
    not equivalent to negative infinity. But infinity is equivalent to positive infinity.

    See Also
    --------
    isinf : Tests whether input elements are inf, or not.
    isnan : Tests whether input elements are nan, or not.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.isfinite(1)
    array(True)
    >>> vp.isfinite(0)
    array(True)
    >>> vp.isfinite(vp.nan)
    array(False)
    >>> vp.isfinite(vp.inf)
    array(False)
    >>> vp.isfinite(vp.NINF)
    array(False)

'''
_isinf_doc = '''
    Tests whether input elements are inf, or not.

    The result is returned as a boolean array.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar, containing the elements to be tested.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        ``True`` where x is not positive or negative infinity, ``False`` otherwise.
        *x* is a scalar, this function returns the result as a 0-dimension ndarray.

    Note
    ----
    NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE754).

    See Also
    --------
    isfinite : Tests whether input elements are neither inf nor nan, or not.
    isnan : Tests whether input elements are nan, or not.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.isinf(vp.inf)
    array(True)
    >>> vp.isinf(vp.nan)
    array(False)
    >>> vp.isinf(vp.NINF)
    array(True)
    >>> vp.isinf([vp.inf, -vp.inf, 1.0, vp.nan])
    array([ True,  True, False, False])
    >>> x = vp.array([-vp.inf, 0., vp.inf])
    >>> y = vp.array([2, 2, 2])
    >>> vp.isinf(x)
    array([ True, False,  True])
    >>> vp.isinf(y)
    array([False, False, False])

'''
_isnan_doc = '''
    Tests whether input elements are nan, or not.

    The result is returned as a boolean array.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar, containing the elements to be tested.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        ``True`` where x is nan, ``False`` otherwise. If *x* is a scalar,
        this function returns the result as a 0-dimension ndarray.

    Note
    ----
    NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE754).

    See Also
    --------
    isfinite : Tests whether input elements are neither inf nor nan, or not.
    isinf : Tests whether input elements are inf, or not.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.isnan(vp.nan)
    array(True)
    >>> vp.isnan(vp.inf)
    array(False)
    >>> vp.isnan(1)
    array(False)

'''
_signbit_doc = '''
    Returns True where signbit is set (less than zero), element-wise.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        An ndarray, containing the results. If *x* is a scalar,
        this function returns the result as a 0-dimension ndarray.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.signbit(-1.2)
    array(True)
    >>> vp.signbit(vp.array([1, -2.3, 2.1]))
    array([False,  True, False])

'''
_copysign_doc = '''
    Changes the sign of x1 to that of x2, element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is values to change the sign. The sign of *x2* is copied to *x1*.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The values of *x1* with the sign of *x2*. If *x1* and *x2* are both scalars, this
        function returns the result as a 0-dimension ndarray.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.copysign(1.3, -1)
    array(-1.3)
    >>> 1/vp.copysign(0, 1)
    array(inf)
    >>> 1/vp.copysign(0, -1)
    array(-inf)
    >>> vp.copysign([-1, 0, 1], -1.1)
    array([-1., -0., -1.])
    >>> vp.copysign([-1, 0, 1], vp.arange(3)-1)
    array([-1.,  0.,  1.])

'''
_sign_doc = '''
    Returns the element-wise indication of the sign of a number.

    The sign function returns -1 if ``x < 0``, 0 if ``x==0``. `nan`
    is returned for `nan` inputs. For complex inputs, the sign function returns
    ``sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j``.
    `arary(nan+0j)` is returned for complex `nan` inputs.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        A ndarray, containing the sign for each element in *x*. If *x* is a scalar, this
        function returns the result as a 0-dimension ndarray.

    Note
    ----
    There is more than one definition of sign in common use for complex numbers. The
    definition used here is equivalent to :math:`x/\\sqrt{x*x}` which is different from a
    common alternative, :math:`x/|x|`.

    See Also
    --------
    ceil : Returns the ceiling of the input, element-wise.
    floor : Returns the floor of the input, element-wise.
    trunc : Returns the truncated value of the input, element-wise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.sign([-5., 4.5])
    array([-1.,  1.])
    >>> vp.sign(0)
    array(0)
    >>> vp.sign(5-2j)
    array(1.+0.j)

'''
_nextafter_doc = '''
    Returns the next floating-point value after *x1* towards *x2*, element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is values to find the next representable value. *x2* is the direction
        where to look for the next representable value of *x1*.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The next representable values of *x1* in the direction of *x2*. If *x1* and *x2*
        are both scalars, this function returns the result as a 0-dimension ndarray.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.set_printoptions(16)
    >>> vp.nextafter(1, 0)
    array(0.9999999999999999)

'''
_spacing_doc = '''
    Returns the distance between *x* and the nearest adjacent number, element-wise.

    Parameters
    ----------
    x : array_like
        Input an array or a scalar.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The spacing of values of *x*. If *x* is a scalar, this function returns the
        result as a 0-dimension ndarray.

    Note
    ----
    It can be considered as a generalization of EPS:
    ``nlcpy.spacing(vp.float64(1)) == nlcpy.finfo(vp.float64).eps``, and there should
    not be any representable number between ``x + spacing(x)`` and `x` for any finite
    `x`. Spacing of +- inf and NaN is NaN.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.spacing(1) == vp.finfo(vp.float64).eps
    array(True)

'''
_ldexp_doc = '''
    Returns :math:`x1 * 2^{x2}`, element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        *x1* is an array of multipliers. *x2* is an array of exponeniations.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The result of :math:`x1 * 2^{x2}`. If *x1* and *x2* are both scalars, this
        function returns the result as a 0-dimension ndarray.

    Restriction
    -----------
    - *dtype* is a complex dtype : *TypeError* occurs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.ldexp(5., vp.arange(4), dtype='float64')
    array([ 5., 10., 20., 40.])

'''
_floor_doc = '''
    Returns the floor of the input, element-wise.

    The floor of the scalar *x* is the largest integer *i*, such that *i* <= *x*.
    It is often denoted as :math:`\\lfloor x \\rfloor`.

    Parameters
    ----------
    x : array_like
        Input arrays or scalars.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The floor of each element in *x*. If *x* is a scalar, this function returns the
        result as a 0-dimension ndarray.

    Note
    ----
    Some spreadsheet programs calculate the "floor-towards-zero", in other words
    ``floor(-2.5) == -2``. NLCPy instead uses the definition of :func:`floor` where
    ``floor(-2.5) == -3``.

    See Also
    --------
    ceil : Returns the ceiling of the input, element-wise.
    trunc : Returns the truncated value of the input, element-wise.
    rint : Computes the element-wise nearest integer.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> vp.floor(a)
    array([-2., -2., -1.,  0.,  1.,  1.,  2.])

'''
_ceil_doc = '''
    Returns the ceiling of the input, element-wise.

    The ceiling of the scalar *x* is the smallest integer *i*, such that *i* >= *x*.
    It is often denoted as :math:`\\lceil x \\rceil`.

    Parameters
    ----------
    x : array_like
        Input arrays or scalars.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The ceiling of each element in *x*. If *x* is a scalar, this function returns the
        result as a 0-dimension ndarray.

    See Also
    --------
    floor : Returns the floor of the input, element-wise.
    trunc : Returns the truncated value of the input, element-wise.
    rint : Computes the element-wise nearest integer.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> vp.ceil(a)
    array([-1., -1., -0.,  1.,  2.,  2.,  2.])

'''
_trunc_doc = '''
    Returns the truncated value of the input, element-wise.

    The truncated value of the scalar *x* is the nearest integer *i* which is
    closer to zero than *x* is. In short, the fractional part of the signed number
    *x* is discarded.

    Parameters
    ----------
    x : array_like
        Input arrays or scalars.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned. A tuple (possible only as a keyword argument) must have length equal
        to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the condition is
        True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array
        will retain its original value. Note that if an uninitialized *out* array is
        created via the default ``out=None``, locations within it where the condition is
        False will remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the section
        :ref:`Optional Keyword Arguments <optional_keyword_arg>`.

    Returns
    -------
    y : ndarray
        The truncated value of each element in *x*. If *x* is a scalar, this function
        returns the result as a 0-dimension ndarray.

    See Also
    --------
    ceil : Returns the ceiling of the input, element-wise.
    floor : Returns the floor of the input, element-wise.
    rint : Computes the element-wise nearest integer.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> vp.trunc(a)
    array([-1., -1., -0.,  0.,  1.,  1.,  2.])

'''
_isnat_doc = ''''''
_modf_doc = ''''''
_frexp_doc = ''''''

# matmul
_matmul_doc = '''
    Matrix product of two arrays.

    Parameters
    ----------
    x1,x2 : array_like
        Input arrays, scalars not allowed.
    out : ndarray or None,  optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated
        array is returned. A tuple (possible only as a keyword argument) must have
        length equal to the number of outputs.

    Returns
    -------
    y : ndarray
        The matrix product of the inputs. If *x1* and *x2* are both scalars,
        this function returns the result as a 0-dimension array.

    Restriction
    -----------
    - ``x1.ndim > 2`` or ``x2.ndim > 2``: *NotImplementedError* occurs.

    Note
    ----
    The behavior depends on the arguments in the following way.

    - If both arguments are 2-D they are multiplied like conventional matrices.
    - If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to
      its dimensions. After matrix multiplication the prepended 1 is removed.
    - If the second argument is 1-D, it is promoted to a matrix by appending a 1 to
      its dimensions. After matrix multiplication the appended 1 is removed.

    See Also
    --------
    dot : Dot product of two arrays.

    Examples
    --------
    For 2-D arrays it is the matrix product:

    >>> import nlcpy as vp
    >>> a = vp.array([[1, 0],[0, 1]])
    >>> b = vp.array([[4, 1],[2, 2]])
    >>> vp.matmul(a, b)
    array([[4, 1],
           [2, 2]])

    For 2-D mixed with 1-D, the result is the usual.

    >>> a = vp.array([[1, 0],[0, 1]])
    >>> b = vp.array([1, 2])
    >>> vp.matmul(a, b)
    array([1, 2])
    >>> vp.matmul(b, a)
    array([1, 2])
'''

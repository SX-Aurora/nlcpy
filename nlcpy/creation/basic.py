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
import nlcpy
import numpy
import warnings

from nlcpy.request import request


# ----------------------------------------------------------------------------
# create ones and zeros arrays
# see: https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html
# ----------------------------------------------------------------------------
def empty(shape, dtype=float, order='C'):
    """Returns a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or sequence of int
        Shape of the empty array, e.g., (2, 3) or 2.
    dtype : dtype, optional
        Desired output dtype for the array, e.g, ``nlcpy.int64``.
        Default is ``nlcpy.float64``.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major (C-style) or column-major
        (Fortran-style) order in memory.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data of the given shape, dtype, and order.

    Note
    ----

    :func:`empty`, unlike :func:`zeros`, does not set the array values to zero, and may
    therefore be marginally faster. On the other hand, it requires the user to manually
    set all the values in the array, and should be used with caution.

    See Also
    --------
    empty_like : Returns a new array with the same shape and type
        as a given array.
    ones : Returns a new array of given shape and type, filled with ones.
    zeros : Returns a new array of given shape and type, filled with zeros.
    full : Returns a new array of given shape and type,
        filled with fill_value.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.empty([2, 2]) # doctest: +SKIP
    array([[0., 0.],
           [0., 0.]])          # They are not always zero. (uninitialized)
    >>> vp.empty([2, 2], dtype=int) # doctest: +SKIP
    array([[0, 0],
           [0, 0]])            # They are not always zero. (uninitialized)

    """
    return nlcpy.ndarray(shape=shape, dtype=dtype, order=order)


# ----------------------------------------------------------------------------
# not implemented routines
# default order value is temporarily set to 'C'.
# default subok value is temporarily set to False.
# ----------------------------------------------------------------------------

def empty_like(prototype, dtype=None, order='K', subok=False, shape=None):
    """Returns a new array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : array_like
        The shape and dtype of *prototype* define these same attributes of the returned
        array.
    dtype : dtype, optional
        Overrides the data type of the result.
    order : {'C', 'F'}, optional
        Overrides the memory layout of the result. 'C' means C-order, 'F' means F-order,
        'A' means 'F' if *prototype* is Fortran contiguous, 'C' otherwise. 'K' means
        match the layout of *prototype* as closely as possible.
    subok : bool, optional
        Not implemented.
    shape : int or sequence of ints, optional
        Overrides the shape of the result. If order='K' and the number of dimensions is
        unchanged, will try to keep order, otherwise, order='C' is implied.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same shape and type as
        *prototype*.

    Note
    ----

    This function does not initialize the returned array; to do that use
    :func:`zeros_like` or :func:`ones_like` instead. It may be marginally faster than the
    functions that do set the array values.

    See Also
    --------
    ones_like : Returns an array of ones with the same shape and type
        as a given array.
    zeros_like : Returns an array of zeros with the same shape
        and type as a given array.
    full_like : Returns a full array with the same shape
        and type as a given array.
    empty : Returns a new array of given shape and type,
        without initializing entries.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = ([1,2,3], [4,5,6])                         # a is array-like
    >>> vp.empty_like(a)    # doctest: +SKIP
    array([[0, 0, 0],
           [0, 0, 0]])                                 # uninitialized
    >>> a = vp.array([[1., 2., 3.],[4.,5.,6.]])
    >>> vp.empty_like(a)    # doctest: +SKIP
    array([[0., 0., 0.],
           [0., 0., 0.]])                              # uninitialized

    """
    if subok is not False:
        raise NotImplementedError('subok in empty_like is not implemented yet.')

    prototype = nlcpy.asanyarray(prototype)

    if shape is None:
        shape = prototype.shape
    if dtype is None:
        dtype = prototype.dtype
    if order is None or order in 'kKaA':
        if prototype._f_contiguous and not prototype._c_contiguous:
            order = 'F'
        else:
            order = 'C'

    if numpy.dtype(dtype).kind == 'V':
        raise NotImplementedError('void dtype in empty_like is not implemented yet.')

    out = nlcpy.ndarray(shape=shape, dtype=dtype, order=order)
    return out


def eye(N, M=None, k=0, dtype=float, order='C'):
    """Returns a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the output.
    M : int, optional
        Number of columns in the output. If None, defaults to *N*.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal, a positive
        value refers to an upper diagonal, and a negative value to a lower diagonal.
    dtype : dtype, optional
        Data-type of the returned array.
    order : {'C', 'F'}, optional
        Whether the output should be stored in row-major (C-style) or column-major
        (Fortran-style) order in memory.

    Returns
    -------
    I : ndarray
        An array where all elements are equal to zero, except for the k-th diagonal,
        whose values are equal to one.

    See Also
    --------
    identity : Returns the identity array.
    diag : Extracts a diagonal or construct a diagonal array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.eye(2, dtype=int)
    array([[1, 0],
           [0, 1]])
    >>> vp.eye(3, k=1)
    array([[0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 0.]])
    """
    if numpy.dtype(dtype).kind == 'V':
        raise NotImplementedError('void dtype in eye is not implemented yet.')
    if M is None:
        M = N

    out = nlcpy.ndarray(shape=(N, M), dtype=dtype, order=order)

    if order == 'F':
        N, M = M, N

    request._push_request(
        "nlcpy_eye",
        "creation_op",
        (out, int(N), int(M), int(k)),)

    return out


def identity(n, dtype=None):
    """Returns the identity array. The identity array is a square array with ones on the
    main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in *n* x *n* output.
    dtype : dtype, optional
        Data-type of the output. Defaults to ``nlcpy.float64``.

    Returns
    -------
    out : ndarray
        *n* x *n* array with its main diagonal set to 1, and all other elements 0.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.identity(3)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    """
    return eye(N=n, dtype=dtype)


def ones(shape, dtype=None, order='C'):
    """Returns a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : dtype, optional
        The desired dtype for the array, e.g, ``nlcpy.int64``.
        Default is ``nlcpy.float64``.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major (C-style) or column-major
        (Fortran-style) order in memory.

    Returns
    -------
    out : ndarray
        Array of ones with the given shape, dtype, and order.

    See Also
    --------
    ones_like : Returns an array of ones with the same shape and type
        as a given array.
    empty : Returns a new array of given shape and type,
        without initializing entries.
    zeros : Returns a new array of given shape and type, filled with zeros.
    full : Returns a new array of given shape and type,
        filled with fill_value.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.ones(5)
    array([1., 1., 1., 1., 1.])
    >>> vp.ones((5,), dtype=int)
    array([1, 1, 1, 1, 1])
    >>> vp.ones((2, 1))
    array([[1.],
           [1.]])
    >>> s = (2,2)
    >>> vp.ones(s)
    array([[1., 1.],
           [1., 1.]])

    """
    if numpy.dtype(dtype).kind == 'V':
        raise NotImplementedError('void dtype in ones is not implemented yet.')
    out = nlcpy.ndarray(shape=shape, dtype=dtype, order=order)
    out.fill(1)
    return out


def ones_like(a, dtype=None, order='K', subok=False, shape=None):
    """Returns an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and dtype of *a* define these same attributes of the returned array.
    dtype : dtype, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order, 'F' means F-order,
        'A' means 'F' if *a* is Fortran contiguous, 'C' otherwise. 'K' means match the
        layout of *a* as closely as possible.
    subok : bool, optional
        Not implemented.
    shape : int or sequence of ints, optional
        Overrides the shape of the result. If order='K' and the number of dimensions is
        unchanged, will try to keep order, otherwise, order='C' is implied.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as *a*.

    See Also
    --------
    empty_like : Returns a new array with the same shape and type
        as a given array.
    zeros_like : Returns an array of zeros with the same shape and type
        as a given array.
    full_like : Returns a full array with the same shape and type
        as a given array.
    ones : Returns a new array of given shape and type, filled with ones.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> vp.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]])
    >>> y = vp.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])
    >>> vp.ones_like(y)
    array([1., 1., 1.])

    """
    if subok is not False:
        raise NotImplementedError('subok in ones_like is not implemented yet.')

    a = nlcpy.asanyarray(a)

    if shape is None:
        shape = a.shape
    if dtype is None:
        dtype = a.dtype
    if numpy.dtype(dtype).kind == 'V':
        raise NotImplementedError('void dtype in ones_like is not implemented yet.')

    if order is None or order in 'kKaA':
        if a._f_contiguous and not a._c_contiguous:
            order = 'F'
        else:
            order = 'C'

    out = nlcpy.ndarray(shape=shape, dtype=dtype, order=order)
    out.fill(1)
    return out


def zeros(shape, dtype=float, order='C'):
    """Returns a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : dtype, optional
        The desired dtype for the array, e.g, ``nlcpy.int64``.
        Default is ``nlcpy.float64``.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major (C-style) or column-major
        (Fortran-style) order in memory.

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and order.

    See Also
    --------
    zeros_like : Returns an array of zeros with the same shape
        and type as a given array.
    empty : Returns a new array of given shape and type,
        without initializing entries.
    ones : Returns a new array of given shape and type,
        filled with ones.
    full : Returns a new array of given shape and type,
        filled with fill_value.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.zeros(5)
    array([0., 0., 0., 0., 0.])
    >>> vp.zeros((5,), dtype=int)
    array([0, 0, 0, 0, 0])
    >>> vp.zeros((2, 1))
    array([[0.],
           [0.]])
    >>> s = (2,2)
    >>> vp.zeros(s)
    array([[0., 0.],
           [0., 0.]])

    """
    if numpy.dtype(dtype).kind == 'V':
        raise NotImplementedError('void dtype in zeros is not implemented yet.')
    out = nlcpy.ndarray(shape=shape, dtype=dtype, order=order)
    out.fill(0)
    return out


def zeros_like(a, dtype=None, order='K', subok=False, shape=None):
    """Returns an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and dtype of *a* define these same attributes of the returned array.
    dtype : dtype, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order, 'F' means F-order,
        'A' means 'F' if *a* is Fortran contiguous, 'C' otherwise. 'K' means match the
        layout of *a* as closely as possible.
    subok : bool, optional
        Not implemented.
    shape : int or sequence of ints, optional
        Overrides the shape of the result. If order='K' and the number of dimensions is
        unchanged, will try to keep order, otherwise, order='C' is implied.

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as *a*.

    See Also
    --------
    empty_like : Returns a new array with the same shape and type
        as a given array.
    ones_like : Returns an array of ones with the same shape
        and type as a given array.
    full_like : Returns a full array with the same shape
        and type as a given array.
    zeros : Returns a new array of given shape and type,
        filled with zeros.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> vp.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])
    >>> y = vp.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])
    >>> vp.zeros_like(y)
    array([0., 0., 0.])

    """
    if subok is not False:
        raise NotImplementedError('subok in zeros_like is not implemented yet.')

    a = nlcpy.asanyarray(a)

    if shape is None:
        shape = a.shape

    if dtype is None:
        dtype = a.dtype
    if numpy.dtype(dtype).kind == 'V':
        raise NotImplementedError('void dtype in zeros_like is not implemented yet.')

    if order is None or order in 'kKaA':
        if a._f_contiguous and not a._c_contiguous:
            order = 'F'
        else:
            order = 'C'

    out = nlcpy.ndarray(shape=shape, dtype=dtype, order=order)
    out.fill(0)
    return out


def full(shape, fill_value, dtype=None, order='C'):
    """Returns a new array of given shape and type, filled with *fill_value*.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar
        Fill value.
    dtype : dtype, optional
        The desired dtype for the array, e.g, ``nlcpy.int64``.
        Default is ``nlcpy.float64``.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous (row- or
        column-wise) order in memory.

    Returns
    -------
    out : ndarray
        Array of *fill_value* with the given shape, dtype, and order.

    See Also
    --------
    full_like : Returns a full array with the same shape
        and type as a given array.
    empty : Returns a new array of given shape and type,
        without initializing entries.
    ones : Returns a new array of given shape and type,
        filled with ones.
    zeros : Returns a new array of given shape and type,
        filled with zeros.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.full((2, 2), vp.inf)
    array([[inf, inf],
           [inf, inf]])
    >>> vp.full((2, 2), 10)
    array([[10, 10],
           [10, 10]])

    """
    if numpy.dtype(dtype).kind == 'V':
        raise NotImplementedError('void dtype in full is not implemented yet.')

    if dtype is None:
        dtype = numpy.result_type(fill_value)
    else:
        dtype = nlcpy.dtype(dtype)

    if numpy.isscalar(fill_value):
        if numpy.iscomplex(fill_value):
            if dtype in ('complex64', 'complex128'):
                pass
            else:
                fill_value = numpy.real(fill_value)
                warnings.warn(
                    'Casting complex values to real discards the imaginary part',
                    numpy.ComplexWarning, stacklevel=2)
        out = nlcpy.ndarray(shape=shape, dtype=dtype, order=order)
        out.fill(fill_value)

    elif fill_value is None:
        raise NotImplementedError('fill_value in nlcpy.full is None')

    else:
        fill_value = nlcpy.asarray(fill_value)
        out = nlcpy.array(
            nlcpy.broadcast_to(fill_value, shape=shape), dtype=dtype, order=order)

    return out


def full_like(a, fill_value, dtype=None, order='K', subok=False, shape=None):
    """Returns a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and dtype of *a* define these same attributes of the returned array.
    fill_value : scalar
        Fill value.
    dtype : dtype, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order, 'F' means F-order,
        'A' means 'F' if *a* is Fortran contiguous, 'C' otherwise. 'K' means match the
        layout of *a* as closely as possible.
    subok : bool, optional
        Not implemented.
    shape : int or sequence of ints, optional
        Overrides the shape of the result. If order='K' and the number of dimensions is
        unchanged, will try to keep order, otherwise, order='C' is implied.

    Returns
    -------
    out : ndarray
        Array of *fill_value* with the same shape and type as *a*.

    See Also
    --------
    empty_like : Returns a new array with the same shape
        and type as a given array.
    ones_like : Returns an array of ones with the same shape
        and type as a given array.
    zeros_like : Returns an array of zeros with the same shape
        and type as a given array.
    full : Returns a new array of given shape and type,
        filled with fill_value.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(6, dtype=int)
    >>> vp.full_like(x, 1)
    array([1, 1, 1, 1, 1, 1])
    >>> vp.full_like(x, 0.1)
    array([0, 0, 0, 0, 0, 0])
    >>> vp.full_like(x, 0.1, dtype=vp.double)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    >>> vp.full_like(x, vp.nan, dtype=vp.double)
    array([nan, nan, nan, nan, nan, nan])
    >>> y = vp.arange(6, dtype=vp.double)
    >>> vp.full_like(y, 0.1)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    """
    if subok is not False:
        raise NotImplementedError('subok in full_like is not implemented yet.')

    a = nlcpy.asanyarray(a)
    if shape is None:
        shape = a.shape
    if dtype is None:
        dtype = a.dtype
    else:
        dtype = nlcpy.dtype(dtype)

    if numpy.dtype(dtype).kind == 'V':
        raise NotImplementedError('void dtype in full_like is not implemented yet.')

    if order is None or order in 'kKaA':
        if a._f_contiguous and not a._c_contiguous:
            order = 'F'
        else:
            order = 'C'

    if numpy.isscalar(fill_value):
        if numpy.iscomplex(fill_value):
            if dtype in ('complex64', 'complex128'):
                pass
            else:
                fill_value = numpy.real(fill_value)
                warnings.warn(
                    'Casting complex values to real discards the imaginary part',
                    numpy.ComplexWarning, stacklevel=2)
        out = nlcpy.ndarray(shape=shape, dtype=dtype, order=order)
        out.fill(fill_value)

    elif fill_value is None:
        raise NotImplementedError('fill_value in nlcpy.full_like is None')

    else:
        fill_value = nlcpy.asarray(fill_value)
        out = nlcpy.array(
            nlcpy.broadcast_to(fill_value, shape=shape), dtype=dtype, order=order)

    return out

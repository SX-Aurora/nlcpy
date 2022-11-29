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
from nlcpy.ufuncs import operations as ufunc_op
from nlcpy.linalg import cblas_wrapper
from nlcpy.request import request
from nlcpy.wrapper.numpy_wrap import numpy_wrap


@numpy_wrap
def dot(a, b, out=None):
    """Computes a dot product of two arrays.

    - If both *a* and *b* are 1-D arrays, it is inner product of vectors (without complex
      conjugation).
    - If both *a* and *b* are 2-D arrays, it is matrix multiplication, but using
      :func:`nlcpy.matmul` or ``a @ b`` is preferred.
    - If either *a* or *b* is 0-D (scalar), it is equivalent to multiply and using
      ``nlcpy.multiply(a,b)`` or ``a * b`` is preferred.
    - If *a* is an N-D array and *b* is a 1-D array, it is a sum product over the last
      axis of *a* and *b*.
    - If *a* is an N-D array and *b* is an M-D array (where ``M>=2``), it is a
      sum product over the last axis of *a* and the second-to-last axis of *b*:

      ``dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])``

    Parameters
    ----------
    a : array_like
        Input arrays or scalars.
    b : array_like
        Input arrays or scalars.
    out : ndarray, optional
        Output argument. This must have the exact kind that would be returned if it was
        not used. In particular, *out.dtype* must be the dtype that would be returned for
        *dot(a,b)*.

    Returns
    -------
    output : ndarray
        Returns the dot product of *a* and *b*. If *a* and *b* are both scalars or both
        1-D arrays then this function returns the result as a 0-dimention array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.dot(3, 4)
    array(12)

    Neither argument is complex-conjugated:

    >>> vp.dot([2j, 3j], [2j, 3j])
    array(-13.+0.j)

    For 2-D arrays it is the matrix product:

    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> vp.dot(a,b)
    array([[4, 1],
           [2, 2]])

    >>> a = vp.arange(3*4*5*6).reshape((3, 4, 5, 6))
    >>> b = vp.arange(3*4*5*6)[::-1].reshape((5, 4, 6, 3))
    >>> vp.dot(a, b)[2, 3, 2, 1, 2, 2]
    array(499128)
    >>> sum(a[2, 3, 2, :] * b[1, 2, :, 2])
    array(499128)

    """
    a = nlcpy.asanyarray(a)
    b = nlcpy.asanyarray(b)
    dtype_out = numpy.result_type(a.dtype, b.dtype)
    if out is not None:
        if not isinstance(out, nlcpy.ndarray):
            raise TypeError("'out' must be an array")
        if dtype_out != out.dtype:
            raise ValueError('output array is incorrect dtype')
    # if either a or b is 0-D array, it is equivalent to nlcpy.multiply
    if a.ndim == 0 or b.ndim == 0:
        return nlcpy.asanyarray(ufunc_op.multiply(a, b, out=out), order='C')
    # if both a and b are 1-D arrays, it is inner product of vectors
    if a.ndim == 1 and b.ndim == 1:
        return cblas_wrapper.cblas_dot(a, b, out=out)
    # if both a and b are 2-D arrays, it is matrix multiplication
    if a.ndim == 2 and b.ndim == 2:
        return cblas_wrapper.cblas_gemm(
            a, b, out=out, dtype=numpy.result_type(a.dtype, b.dtype))

    # if either a or b are N-D array, it is sum product over the
    # last(or second-last) axis.
    if b.ndim > 1:
        if a.shape[-1] != b.shape[-2]:
            raise ValueError('mismatch input shape')
        shape_out = a.shape[:-1] + b.shape[:-2] + (b.shape[-1],)
    else:
        if a.shape[-1] != b.shape[-1]:
            raise ValueError('mismatch input shape')
        shape_out = a.shape[:-1]

    if out is None:
        out = nlcpy.empty(shape_out, dtype=dtype_out)

    if out.dtype in (
        numpy.int8, numpy.int16,
        numpy.uint8, numpy.uint16, numpy.float16
    ):
        raise TypeError('output dtype \'%s\' is not supported' % dtype_out)
    elif out.shape != shape_out or not out.flags.c_contiguous:
        raise ValueError(
            'output array is not acceptable (must have the right datatype, '
            'number of dimensions, and be a C-Array)')

    out.fill(0)
    if a.size > 0 and b.size > 0:
        request._push_request(
            "nlcpy_dot",
            "linalg_op",
            (a, b, out),
        )
    return out


@numpy_wrap
def inner(a, b):
    """Computes an inner product of two arrays.

    Ordinary inner product of vectors for 1-D arrays (without complex conjugation).

    Parameters
    ----------
    a, b : array_like
        If *a* and *b* are nonscalar, their shape must match.

    Returns
    -------
    out : ndarray
        out.shape = a.shape[:-1] + b.shape[:-1]

    Restriction
    -----------
    If *a* or *b* is not 1-D array : *NotImplementedError* occurs.

    Note
    ----
    For vectors (1-D arrays) it computes the ordinary inner-product::

        import nlcpy as vp
        vp.inner(a, b) # equivalent to sum(a[:]*b[:])

    if *a* or *b* is scalar, in which case::

        vp.inner(a, b) # equivalent to a*b

    See Also
    --------
    dot : Computes a dot product of two arrays.

    Examples
    --------
    Ordinary inner product for vectors:

    >>> import nlcpy as vp
    >>> a = vp.array([1,2,3])
    >>> b = vp.array([0,1,0])
    >>> vp.inner(a, b)
    array(2)

    An example where b is a scalar:

    >>> vp.inner(vp.eye(2), 7)
    array([[7., 0.],
           [0., 7.]])

    """
    a = nlcpy.asanyarray(a)
    b = nlcpy.asanyarray(b)
    if a.ndim == 0 or b.ndim == 0:
        return ufunc_op.multiply(a, b)
    elif a.ndim == 1 and b.ndim == 1:
        return cblas_wrapper.cblas_dot(a, b)
    else:
        raise NotImplementedError("Only 1-D array is supported.")


def outer(a, b, out=None):
    """Computes an outer product of two vectors.

    Given two vectors, ``a = [a0, a1, ..., aM]`` and ``b = [b0, b1, ..., bN]``,
    the outer produce is::

        [[a0*b0 a0*b1 ... a0*bN]
        [a1*b0   .            ]
        [ ...         .       ]
        [aM*b0           aM*bN]]

    Parameters
    ----------
    a : (M,) array_like
        First input vector. Input is flattened if not already 1-dimensional.
    b : (N,) array_like
        Second input vector. Input is flattened if not already 1-dimensional.
    out : (M, N) ndarray, optional
        A location where the result is stored.

    Returns
    -------
    out : ndarray
        ``out[i, j] = a[i] * b[j]``

    See Also
    --------
    inner : Computes an inner product of two arrays.
    ufunc.outer : Applies the ufunc *op* to all pairs (a, b) with a in *A* and b in *B*.

    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:

    >>> import nlcpy as vp
    >>> rl = vp.outer(vp.ones((5,)), vp.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.]])
    >>> im = vp.outer(1j*vp.linspace(2, -2, 5), vp.ones((5,)))
    >>> im
    array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],
           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],
           [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])
    >>> grid = rl + im
    >>> grid
    array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],
           [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],
           [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],
           [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],
           [-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])

    """
    a = nlcpy.asanyarray(a)
    b = nlcpy.asanyarray(b)
    return nlcpy.multiply.outer(a.ravel(), b.ravel(), out=out, order='C')

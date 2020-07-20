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
from nlcpy.ufunc import operations as ufunc_op
from nlcpy.linalg import cblas_wrapper


def dot(a, b, out=None):
    """Computes a dot product of two arrays.

    - If both a and b are 1-D arrays, it is inner product of vectors (without complex
    conjugation).
    - If both a and b are 2-D arrays, it is matrix multiplication, but using
    ufuncs::matmul or a is preferred.
    - If either a or b is 0-D (scalar), it is equivalent to multiply and using
    nlcpy.multiply(a,b) or a * b is preferred.

    Args:
        a : array_like
            Input arrays or scalars.
        b : array_like
            Input arrays or scalars.
        out : `ndarray`, optional
            Output argument. This must have the exact kind that would be returned if it
            was not used. In particular, out.dtype must be the dtype that would be
            returned for dot(a,b).

    Returns:
        output : `ndarray`
            Returns the dot product of a and b. If a and b are both scalars or both 1-D
            arrays then this function returns the result as a 0-dimention array.

    Raises:
         a.ndim>2 or b.ndim>2 :
            NotImplementedError occurs.

    Examples:
        >>> import nlcpy as vp
        >>> vp.dot(3, 4)
        array(12)
        Neither argument is complex-conjugated:
        >>> vp.dot([2j, 3j], [2j, 3j])
        array([-13.+0.j])
        For 2-D arrays it is the matrix product:
        >>> a = [[1, 0], [0, 1]]
        >>> b = [[4, 1], [2, 2]]
        >>> vp.dot(a,b)
        array([[4, 1],
               [2, 2]])

    """
    a = nlcpy.asanyarray(a)
    b = nlcpy.asanyarray(b)
    if out is not None and \
            numpy.result_type(a.dtype, b.dtype) != out.dtype:
        raise ValueError('output array is incorrect dtype')
    # if either a or b is 0-D array, it is equivalent to nlcpy.multiply
    if a.ndim == 0 or b.ndim == 0:
        return ufunc_op.multiply(a, b, out=out)
    # if both a and b are 1-D arrays, it is inner product of vectors
    if a.ndim == 1 and b.ndim == 1:
        return cblas_wrapper.cblas_dot(a, b, out=out)
    # if both a and b are 2-D arrays, it is matrix multiplication
    if a.ndim == 2 and b.ndim == 2:
        return cblas_wrapper.cblas_gemm(
            a, b, out=out, dtype=numpy.result_type(a.dtype, b.dtype))

    # TODO:
    # if either a or b are N-D array, it is sum product over the
    # last(or second-last) axis.
    raise NotImplementedError(
        'array \'a\' or \'b\' are N-D array case is not implemented yet.')

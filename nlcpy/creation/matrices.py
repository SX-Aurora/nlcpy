#
# * The source code in this file is based on the soure code of CuPy.
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
# # CuPy License #
#
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#     THE SOFTWARE.
#
import nlcpy
import numpy
from nlcpy.request import request


# ----------------------------------------------------------------------------
# building matrices
# see: https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html
# ----------------------------------------------------------------------------

def diag(v, k=0):
    """Extracts a diagonal or constructs a diagonal array.

    Parameters
    ----------
    v : array_like
        If *v* is a 2-D array, return a copy of its *k-th* diagonal. If *v* is a 1-D
        array, return a 2-D array with *v* on the k-th diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use *k>0* for diagonals above the main
        diagonal, and *k<0* for diagonals below the main diagonal.

    Returns
    -------
    out : ndarray
        The extracted diagonal or constructed diagonal array.

    See Also
    --------
    diagonal : Returns specified diagonals.
    diagflat : Creates a two-dimensional array with the flattened input as a diagonal.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(9).reshape((3,3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> vp.diag(x)
    array([0, 4, 8])
    >>> vp.diag(x, k=1)
    array([1, 5])
    >>> vp.diag(x, k=-1)
    array([3, 7])
    >>> vp.diag(vp.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])

    """
    if isinstance(v, nlcpy.ndarray):
        ndim = v.ndim
    else:
        ndim = numpy.ndim(v)
        if ndim == 1:
            v = nlcpy.array(v)
        if ndim == 2:
            # to save bandwidth, don't copy non-diag elements to GPU
            v = numpy.array(v)

    if ndim == 1:
        size = v.size + abs(k)
        ret = nlcpy.zeros((size, size), dtype=v.dtype)
        ret.diagonal(k)[:] = v
        return ret
    elif ndim == 2:
        return v.diagonal(k).copy()
    else:
        raise ValueError('Input must be 1- or 2-d.')


def diagflat(v, k=0):
    """Creates a two-dimensional array with the flattened input as a diagonal.

    Parameters
    ----------
    v : array_like
        Input data, which is flattened and set as the *k*-th diagonal of the output.
    k : int, optional
        Diagonal to set; 0, the default, corresponds to the "main" diagonal, a positive
        (negative) *k* giving the number of the diagonal above (below) the main.

    Returns
    -------
    out : ndarray
        The 2-D output array.

    See Also
    --------
    diag : Extracts a diagonal or construct a diagonal array.
    diagonal : Returns specified diagonals.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.diagflat([[1,2], [3,4]])
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

    >>> vp.diagflat([1,2], 1)
    array([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])
    """
    v = nlcpy.asanyarray(v).ravel()
    return diag(v, k)


def tri(N, M=None, k=0, dtype=float):
    """An array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array. By default, *M* is taken equal to *N*.
    k : int, optional
        The sub-diagonal at and below which the array is filled. *k* = 0 is the main
        diagonal, while *k* < 0 is below it, and *k* > 0 is above. The default is 0.
    dtype : dtype, optional
        Data type of the returned array. The default is float.

    Returns
    -------
    tri : ndarray
        Array with its lower triangle filled with ones and zero elsewhere; in other
        words ``T[i,j] == 1`` for ``i <= j + k``, 0 otherwise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.tri(3, 5, 2, dtype=int)
    array([[1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1]])

    >>> vp.tri(3, 5, -1)
    array([[0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 1., 0., 0., 0.]])
    """
    if N < 0:
        N = 0
    else:
        N = int(N)
    if M is None:
        M = N
    elif M < 0:
        M = 0
    else:
        M = int(M)
    k = int(k)
    out = nlcpy.empty([N, M], dtype=dtype)
    if out.size:
        request._push_request(
            'nlcpy_tri',
            'creation_op',
            (out, k)
        )
    return out


def tril(m, k=0):
    """Lower triangle of an array.

    Returns a copy of an array with elements above the *k*-th diagonal zeroed.

    Parameters
    ----------
    m : array_like
        Input array.
    k : int, optional
        Diagonal above which to zero elements. *k* = 0 (the default) is the main
        diagonal, *k* < 0 is below it and *k* > 0 is above.

    Returns
    -------
    tri : ndarray
        Lower triangle of *m*, of same shape and data-type as *m*.

    See Also
    --------
    triu : Upper triangle of an array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    array([[ 0,  0,  0],
           [ 4,  0,  0],
           [ 7,  8,  0],
           [10, 11, 12]])
    """
    m = nlcpy.asanyarray(m)
    mask = nlcpy.tri(*m.shape[-2:], k=k, dtype=bool)
    return nlcpy.where(mask, m, numpy.dtype(m.dtype).type(0))


def triu(m, k=0):
    """Upper triangle of an array.

    Returns a copy of a matrix with the elements below the *k*-th diagonal zeroed.
    Please refer to the documentation for tril for further details.

    See Also
    --------
    tril : Lower triangle of an array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]])
    """
    m = nlcpy.asanyarray(m)
    mask = nlcpy.tri(*m.shape[-2:], k=k - 1, dtype=bool)
    return nlcpy.where(mask, numpy.dtype(m.dtype).type(0), m)

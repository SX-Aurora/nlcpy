#
# * The source code in this file is based on the soure code of CuPy.
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

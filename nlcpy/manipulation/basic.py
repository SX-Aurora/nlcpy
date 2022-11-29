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
from nlcpy.core import manipulation
from nlcpy.venode import transfer_array


def shape(a):
    """Returns the shape of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : tuple of ints
        The elements of the shape tuple give the lengths of the corresponding array
        dimensions.

    See Also
    --------
    ndarray.shape : Equivalent the array attribute.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.shape(vp.eye(3))
    (3, 3)
    >>> vp.shape([[1, 2]])
    (1, 2)
    >>> vp.shape([0])
    (1,)
    >>> vp.shape(0)
    ()

    """
    a = nlcpy.asanyarray(a)
    return a.shape


def copyto(dst, src, casting='same_kind', where=True):
    """Copies values from one array to another, broadcasting as necessary.

    Raises a TypeError if the `casting` rule is violated, and if `where` is provided, it
    selects which elements to copy.

    Parameters
    ----------
    dst : ndarray
        The array into which values are copied.
    src : array_like
        The array from which values are copied.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when copying.
        - 'no' means the data types should not be cast at all.
        - 'equiv' means only byte-order changes are allowed.
        - 'safe' means only casts which can preserve values are allowed.
        - 'same_kind' means only safe casts or casts within a kind, like float64 to
        float32, are allowed.
        - 'unsafe' means any data conversions may be done.
    where : array_like of bool, optional
        A boolean array which is broadcasted to match the dimensions of `dst`, and
        selects elements to copy from `src` to `dst` wherever it contains the value
        True.
    """

    # first argument must be nlcpy.ndarray
    if not isinstance(dst, nlcpy.ndarray):
        dst_type = "None" if dst is None else type(dst).__name__
        raise TypeError(
            "copyto() argument 1 must be nlcpy.ndarray, not {}".format(dst_type))

    if isinstance(src, nlcpy.ndarray):
        if src.venode == dst.venode:
            return manipulation._copyto(dst, src, casting, where)
        else:
            return transfer_array(src, dst.venode, dst)
    else:
        prev_ve = nlcpy.venode.VE()
        try:
            dst.venode.use()
            return manipulation._copyto(dst, src, casting, where)
        finally:
            prev_ve.use()

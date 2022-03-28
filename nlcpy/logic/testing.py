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


def all(a, axis=None, out=None, keepdims=nlcpy._NoValue):
    """Tests whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed. The default
        (*axis* = *None*) is to perform a logical AND over all the dimensions of the
        input array. *axis* may be negative, in which case it counts from the last to the
        first axis.
        If this is a tuple of ints, a reduction is performed on multiple axes.
    out : ndarray, optional
        Alternate output array in which to place the result. It must have the same shape
        as the expected output and its type is preserved (e.g., if ``dtype(out)`` is
        float, the result will consist of 0.0's and 1.0's).
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.

    Returns
    -------
    all : ndarray
        A new array is returned unless *out* is specified, in which case a reference to
        *out* is returned.

    Note
    ----

    Not a Number (NaN), positive infinity and negative infinity evaluate to `True`
    because these are not equal to zero.

    >>> import nlcpy as vp
    >>> vp.all([[True, False], [True, True]])
    array(False)
    >>> vp.all([[True,False],[True,True]], axis=0)
    array([ True, False])
    >>> vp.all([-1, 4, 5])
    array(True)
    >>> vp.all([1.0, vp.nan])
    array(True)
    >>> o=vp.array(False)
    >>> z=vp.all([-1, 4, 5], out=o)
    >>> id(z), id(o), z   # doctest: +SKIP
    (140052379774144, 140052379774144, array(True)) # may vary

    See Also
    --------
    any : Tests whether any array element along a given axis
        evaluates to True.

    """
    args = dict()
    if keepdims is not nlcpy._NoValue:
        args["keepdims"] = keepdims
    ret = nlcpy.logical_and.reduce(a, axis=axis, out=out, **args)
    return ret


def any(a, axis=None, out=None, keepdims=nlcpy._NoValue):
    """Tests whether any array elements along a given axis evaluate to True.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical OR reduction is performed. The default (axis =
        None) is to perform a logical OR over all the dimensions of the input array. axis
        may be negative, in which case it counts from the last to the first axis.
        If this is a tuple of ints, a reduction is performed on multiple axes.
    out : ndarray, optional
        Alternate output array in which to place the result. It must have the same shape
        as the expected output and its type is preserved (e.g., if ``dtype(out)`` is
        float, the result will consist of 0.0's and 1.0's).
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.

    Returns
    -------
    any : ndarray
        A new array is returned unless out is specified, in which case a reference to out
        is returned.

    Note
    ----

    Not a Number (NaN), positive infinity and negative infinity evaluate to True because
    these are not equal to zero.

    >>> import nlcpy as vp
    >>> vp.any([[True, False], [True, True]])
    array(True)
    >>> vp.any([[True,False],[False,False]], axis=0)
    array([ True, False])
    >>> vp.any([-1, 0, 5])
    array(True)
    >>> vp.any(vp.nan)
    array(True)
    >>> o=vp.array(False)
    >>> z=vp.any([-1, 4, 5], out=o)
    >>> z, o
    (array(True), array(True))
    >>> z is o
    True
    >>> id(z), id(o)  # doctest: +SKIP
    (140196865556152, 140196865556152)  # may vary

    See Also
    --------
    all : Tests whether all array element along a given axis
        evaluates to True.

    """
    args = dict()
    if keepdims is not nlcpy._NoValue:
        args["keepdims"] = keepdims
    ret = nlcpy.logical_or.reduce(a, axis=axis, out=out, **args)
    return ret

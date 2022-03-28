#
# * The source code in this file is based on the soure code of NumPy.
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

import nlcpy


def nanargmax(a, axis=None):
    """ Returns the indices of the maximum values in the specified axis ignoring NaNs.

    For all-NaN slices ``ValueError`` is raised. Warning: the results cannot be trusted
    if a slice contains only NaNs and -Infs.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate. By default flattened input is used.

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.

    See Also
    --------
    argmax : Returns the indices of the maximum values along an axis.
    nanargmin : Returns the indices of the minimum values in the specified axis
                ignoring NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[vp.nan, 4], [2, 3]])
    >>> vp.argmax(a)
    array(0)
    >>> vp.nanargmax(a)
    array(1)
    >>> vp.nanargmax(a, axis=0)
    array([1, 0])
    >>> vp.nanargmax(a, axis=1)
    array([1, 1])
    """
    a = nlcpy.array(a)
    if a.dtype.kind not in 'fc':
        return nlcpy.argmax(a, axis=axis)

    mask = nlcpy.isnan(a)
    if nlcpy.any(nlcpy.all(mask, axis=axis)):
        raise ValueError("All-NaN slice encountered")
    nlcpy.copyto(a, -nlcpy.inf, where=mask)
    return nlcpy.argmax(a, axis=axis)


def nanargmin(a, axis=None):
    """ Returns the indices of the minimum values in the specified axis ignoring NaNs.

    For all-NaN slices ``ValueError`` is raised. Warning: the results cannot be trusted
    if a slice contains only NaNs and -Infs.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate. By default flattened input is used.

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.

    See Also
    --------
    argmin : Returns the indices of the minimum values along an axis.
    nanargmax : Returns the indices of the maximum values in the specified axis
                ignoring NaNs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[vp.nan, 4], [2, 3]])
    >>> vp.argmin(a)
    array(0)
    >>> vp.nanargmin(a)
    array(2)
    >>> vp.nanargmin(a, axis=0)
    array([1, 1])
    >>> vp.nanargmin(a, axis=1)
    array([1, 0])
    """
    a = nlcpy.array(a)
    if a.dtype.kind not in 'fc':
        return nlcpy.argmin(a, axis=axis)

    mask = nlcpy.isnan(a)
    if nlcpy.any(nlcpy.all(mask, axis=axis)):
        raise ValueError("All-NaN slice encountered")
    nlcpy.copyto(a, nlcpy.inf, where=mask)
    return nlcpy.argmin(a, axis=axis)

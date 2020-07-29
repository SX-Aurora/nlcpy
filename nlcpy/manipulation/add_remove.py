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
from nlcpy import core


# ----------------------------------------------------------------------------
# adding and removing elements
# see: https://docs.scipy.org/doc/numpy/reference/
#             routines.array-manipulation.html#adding-and-removing-elements
# ----------------------------------------------------------------------------


def resize(a, new_shape):
    """Returns a new array with the specified shape. If the new array is larger than the
    original array, then the new array is filled filled with repeated copies of a. Note
    that this behavior is different from a.resize(new_shape) which fills with zeros
    instead of repeated copies of a.

    Args:
        a : array_like
            Array to be resized.
        new_shape : int or sequence of ints
            Shape of resized array.

    Returns:
        reshaped_array : `ndarray`
            The new array is formed from the data in the old array, repeated if necessary
            to fill out the required number of elements. The data are repeated in the
            order that they are stored in memory.

    Note:
        Warning: This functionality does not consider axes separately, i.e. it does not
        apply interpolation/extrapolation. It fills the return array array with the
        required number of elements, taken from a as they are laid out in memory,
        disregarding strides and axes. (This is in case the new shape is smaller. For
        larger, see above.) This functionality is therefore not suitable to resize
        images, or data where each axis represents a separate and distinct entity.

    Examples:
        >>> import nlcpy as vp
        >>> a=vp.array([[0,1],[2,3]])
        >>> vp.resize(a,(2,3))
        array([[0, 1, 2],
               [3, 0, 1]])
        >>> vp.resize(a,(1,4))
        array([[0, 1, 2, 3]])
        >>> vp.resize(a,(2,4))
        array([[0, 1, 2, 3],
               [0, 1, 2, 3]])

    """
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    a = nlcpy.ravel(a)
    Na = a.size
    total_size = core.internal.prod(new_shape)
    if Na == 0 or total_size == 0:
        return nlcpy.zeros(new_shape, a.dtype)

    n_copies = int(total_size / Na)
    extra = total_size % Na
    if extra != 0:
        n_copies = n_copies + 1
        extra = Na - extra

    a = nlcpy.concatenate((a,) * n_copies)
    if extra > 0:
        a = a[:-extra]

    return nlcpy.reshape(a, new_shape)


def trim_zeros(filt, trim='fb'):
    raise NotImplementedError('trim_zeros is not implemented yet.')


def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None):
    raise NotImplementedError('unique is not implemented yet.')

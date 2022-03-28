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
import numpy
import nlcpy


def split(ary, indices_or_sections, axis=0):
    """Splits an array into multiple sub-arrays.

    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
        If *indices_or_sections* is an integer, N, the array will be divided into N
        equal arrays along *axis*. If such a split is not possible, an error is raised.
        If *indices_or_sections* is a 1-D array of sorted integers, the entries indicate
        where along *axis* the array is split. For example, ``[2, 3]`` would,
        for ``axis=0``, result in

        - ary[:2]
        - ary[2:3]
        - ary[3:]

        If an index exceeds the dimension of the array along *axis*, an empty sub-array
        is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.

    Returns
    -------
    A list of sub-arrays.

    See Also
    --------
    hsplit : Splits an array into multiple sub-arrays horizontally (column-wise).
    vsplit : Splits an array into multiple sub-arrays vertically (row-wise).
    concatenate : Joins a sequence of arrays along an existing axis.
    stack : Joins a sequence of arrays along a new axis.
    hstack : Stacks arrays in sequence horizontally (column wise).
    vstack : Stacks arrays in sequence vertically (row wise).

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(9.0)
    >>> vp.split(x, 3)
    [array([0., 1., 2.]), array([3., 4., 5.]), array([6., 7., 8.])]

    >>> x = vp.arange(6)
    >>> vp.split(x, [3, 4, 7])
    [array([0, 1, 2]), array([3]), array([4, 5]), array([], dtype=int64)]
    """
    size = ary.shape[axis]
    if numpy.isscalar(indices_or_sections):
        if size % indices_or_sections:
            raise ValueError(
                'array split does not result in an equal division')
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError('number sections must be larger than 0.')
        Neach_section, extras = divmod(size, Nsections)
        section_sizes = ([0] +
                         extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        TH_cumsum = 2000
        if len(section_sizes) < TH_cumsum:
            div_points = numpy.array(section_sizes, dtype=nlcpy.intp).cumsum()
        else:
            div_points = nlcpy.array(section_sizes, dtype=nlcpy.intp).cumsum().tolist()
    else:
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [size]

    sub_arys = []
    sary = nlcpy.swapaxes(ary, axis, 0)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(nlcpy.swapaxes(sary[st:end], axis, 0))

    return sub_arys


def hsplit(ary, indices_or_sections):
    """Splits an array into multiple sub-arrays horizontally (column-wise).

    Please refer to the :func:`split` documentation. hsplit is equivalent to
    :func:`split` with ``axis=1``, the array is always split along the second axis
    regardless of the array dimension.

    See Also
    --------
    split : Splits an array into multiple sub-arrays.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(16.0).reshape(4, 4)
    >>> x
    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.],
           [12., 13., 14., 15.]])
    >>> vp.hsplit(x, 2)
    [array([[ 0.,  1.],
           [ 4.,  5.],
           [ 8.,  9.],
           [12., 13.]]), array([[ 2.,  3.],
           [ 6.,  7.],
           [10., 11.],
           [14., 15.]])]
    >>> vp.hsplit(x, vp.array([3, 6]))
    [array([[ 0.,  1.,  2.],
           [ 4.,  5.,  6.],
           [ 8.,  9., 10.],
           [12., 13., 14.]]), array([[ 3.],
           [ 7.],
           [11.],
           [15.]]), array([], shape=(4, 0), dtype=float64)]
    """
    ary = nlcpy.asanyarray(ary)
    if ary.ndim == 0:
        raise ValueError('hsplit only works on arrays of 1 or more dimensions')
    if ary.ndim > 1:
        return split(ary, indices_or_sections, 1)
    else:
        return split(ary, indices_or_sections, 0)


def vsplit(ary, indices_or_sections):
    """Splits an array into multiple sub-arrays vertically (row-wise).

    Please refer to the :func:`split` documentation. vsplit is equivalent to
    :func:`split` with ``axis=0`` (default), the array is always split along the
    first axis regardless of the array dimension.

    See Also
    --------
    split : Splits an array into multiple sub-arrays.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(16.0).reshape(4, 4)
    >>> x
    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.],
           [12., 13., 14., 15.]])
    >>> vp.vsplit(x, 2)
    [array([[0., 1., 2., 3.],
           [4., 5., 6., 7.]]), array([[ 8.,  9., 10., 11.],
           [12., 13., 14., 15.]])]
    >>> z1, z2, z3 = vp.vsplit(x, vp.array([3, 6]))
    >>> z1; z2; z3;
    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.]])
    array([[12., 13., 14., 15.]])
    array([], shape=(0, 4), dtype=float64)

    With a higher dimensional array the split is still along the first axis.

    >>> x = vp.arange(8.0).reshape(2, 2, 2)
    >>> x
    array([[[0., 1.],
            [2., 3.]],
    <BLANKLINE>
           [[4., 5.],
            [6., 7.]]])
    >>> vp.vsplit(x, 2)
    [array([[[0., 1.],
            [2., 3.]]]), array([[[4., 5.],
            [6., 7.]]])]
    """
    ary = nlcpy.asanyarray(ary)
    if ary.ndim < 2:
        raise ValueError('vsplit only works on arrays of 2 or more dimensions')
    return split(ary, indices_or_sections, 0)

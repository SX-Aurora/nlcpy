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
from nlcpy import veo
from nlcpy.request import request
from numpy import AxisError


def _lange(x, norm, axis):
    order = 'F' if x.flags.f_contiguous and not x.flags.c_contiguous else 'C'
    dtype = 'f' if x.dtype.char in 'fF' else 'd'
    if x.size == 0:
        shape = [x.shape[i] for i in set(range(x.ndim)) - set(axis)]
        return nlcpy.zeros(shape, dtype=dtype)
    if norm in (None, 'fro', 'f'):
        if x.ndim == 2 and (x._c_contiguous or x._f_contiguous):
            y = nlcpy.empty([], dtype=dtype)
            request._push_request(
                'nlcpy_simple_fnorm',
                'linalg_op',
                (x, y))
        else:
            shape = [x.shape[i] for i in set(range(x.ndim)) - set(axis)]
            y = nlcpy.empty(shape, dtype=dtype, order=order)
            work1 = nlcpy.empty(x.shape, dtype=dtype, order=order)
            shape = [x.shape[i] for i in range(x.ndim) if i != axis[0]]
            work2 = nlcpy.empty(shape, dtype=dtype)
            request._push_request(
                'nlcpy_fnorm',
                'linalg_op',
                (x, y, work1, work2, axis[0], axis[1]))
        return y
    if norm == nlcpy.inf:
        norm = 'I'
    else:
        norm = '1'
    x = nlcpy.asarray(nlcpy.moveaxis(x, (axis[0], axis[1]), (0, 1)), order='F')
    y = nlcpy.empty(x.shape[2:], dtype=dtype, order='F')
    lwork = x.shape[0] if norm == 'I' else 1
    work = nlcpy.empty(lwork, dtype=dtype)
    fpe = request._get_fpe_flag()
    args = (
        ord(norm),
        x,
        y,
        work,
        veo.OnStack(fpe, inout=veo.INTENT_OUT),
    )

    request._push_and_flush_request(
        'nlcpy_norm',
        args,
    )

    return nlcpy.asarray(y, order=order)


def norm(x, ord=None, axis=None, keepdims=False):
    """Returns matrix or vector norm.

    This function is able to return one of eight different matrix norms, or one of an
    infinite number of vector norms (described below), depending on the value of the
    ``ord`` parameter.

    Parameters
    ----------
    x : array_like
        Input array. If *axis* is None, *x* must be 1-D or 2-D.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Note``). inf means nlcpy's *inf* object.
    axis : {None, int, 2-tuple of ints}, optional
        If *axis* is an integer, it specifies the axis of *x* along which to compute the
        vector norms. If *axis* is a 2-tuple, it specifies the axes that hold 2-D
        matrices, and the matrix norms of these matrices are computed. If *axis* is None
        then either a vector norm (when *x* is 1-D) or a matrix norm (when *x* is 2-D) is
        returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the result as
        dimensions with size one. With this option the result will broadcast correctly
        against the original x.

    Returns
    -------
    n : ndarray
        Norm of the matrix or vector(s).

    Note
    ----
    For values of ``ord < 1``, the result is, strictly speaking, not a mathematical
    'norm', but it may still be useful for various numerical purposes. The following
    norms can be calculated:

    .. csv-table::
        :header: ord, norm for matrices, norm for vectors

        None, Frobenius norm, 2-norm
        'fro', Frobenius norm, \\-
        'nuc', nuclear norm, \\-
        inf, "max(sum(abs(x), axis=1))", max(abs(x))
        -inf, "min(sum(abs(x), axis=1))", min(abs(x))
        0, \\-, sum(x != 0)
        1, "max(sum(abs(x), axis=0))", as below
        -1, "min(sum(abs(x), axis=0))", as below
        2, 2-norm (largest sing. value), as below
        -2, smallest singular value, as below
        other, \\-, sum(abs(x)**ord)**(1./ord)

    The Frobenius norm is given by :math:`|A|_F = [\\sum_{i,j}abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.arange(9) - 4
    >>> a
    array([-4, -3, -2, -1,  0,  1,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])
    >>> vp.linalg.norm(a)    # doctest: +SKIP
    array(7.74596669)
    >>> vp.linalg.norm(b)    # doctest: +SKIP
    array(7.74596669)
    >>> vp.linalg.norm(b, 'fro')   # doctest: +SKIP
    array(7.74596669)
    >>> vp.linalg.norm(a, vp.inf)  # doctest: +SKIP
    array(4.)
    >>> vp.linalg.norm(b, vp.inf)  # doctest: +SKIP
    array(9.)
    >>> vp.linalg.norm(a, -vp.inf)  # doctest: +SKIP
    array(0.)
    >>> vp.linalg.norm(b, -vp.inf)  # doctest: +SKIP
    array(2.)
    >>> vp.linalg.norm(a, 1)   # doctest: +SKIP
    array(20.)
    >>> vp.linalg.norm(b, 1)   # doctest: +SKIP
    array(7.)
    >>> vp.linalg.norm(a, -1)  # doctest: +SKIP
    array(0.)
    >>> vp.linalg.norm(b, -1)  # doctest: +SKIP
    array(6.)
    >>> vp.linalg.norm(a, 2)   # doctest: +SKIP
    array(7.74596669)
    >>> vp.linalg.norm(b, 2)  # doctest: +SKIP
    array(7.34846923)
    >>> vp.linalg.norm(a, -2) # doctest: +SKIP
    array(0.)
    >>> vp.linalg.norm(b, -2) # doctest: +SKIP
    array(3.75757704e-16)
    >>> vp.linalg.norm(a, 3)  # doctest: +SKIP
    array(5.84803548)
    >>> vp.linalg.norm(a, -3) # doctest: +SKIP
    array(0.)

    Using the *axis* argument to compute vector norms:

    >>> c = vp.array([[ 1, 2, 3],
    ...               [-1, 1, 4]])
    >>> vp.linalg.norm(c, axis=0)   # doctest: +SKIP
    array([1.41421356, 2.23606798, 5.        ])
    >>> vp.linalg.norm(c, axis=1)   # doctest: +SKIP
    array([3.74165739, 4.24264069])
    >>> vp.linalg.norm(c, ord=1, axis=1)  # doctest: +SKIP
    array([6., 6.])

    Using the axis argument to compute matrix norms:

    >>> m = vp.arange(8).reshape(2,2,2)
    >>> vp.linalg.norm(m, axis=(1,2))   # doctest: +SKIP
    array([ 3.74165739, 11.22497216])
    >>> vp.linalg.norm(m[0, :, :]), vp.linalg.norm(m[1, :, :])  # doctest: +SKIP
    (array(3.74165739), array(11.22497216))

    """
    x = nlcpy.asarray(x)
    if x.dtype.char in '?ilIL':
        x = nlcpy.array(x, dtype='d')

    # Immediately handle some default, simple, fast, and common cases.
    if axis is None:
        ret = None
        ndim = x.ndim
        axis = tuple(range(x.ndim))
        if ord is None and x.ndim == 2:
            ret = _lange(x, ord, axis)
        elif ord is None or ord == 2 and x.ndim == 1:
            x = x.ravel()
            if x.dtype.char in 'FD':
                sqnorm = nlcpy.dot(x.real, x.real) + nlcpy.dot(x.imag, x.imag)
            else:
                sqnorm = nlcpy.dot(x, x)
            ret = nlcpy.sqrt(sqnorm)
        if ret is not None:
            if keepdims:
                ret = ret.reshape(ndim * [1])
            return ret
    elif not isinstance(axis, tuple):
        try:
            axis = (int(axis),)
        except Exception:
            raise TypeError("'axis' must be None, an integer or a tuple of integers")

    order = 'F' if x.flags.f_contiguous and not x.flags.c_contiguous else 'C'
    if len(axis) == 1:
        if ord == nlcpy.inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -nlcpy.inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            return nlcpy.sum((x != 0).astype(x.real.dtype), axis=axis, keepdims=keepdims)
        elif ord == 1:
            return nlcpy.add.reduce(abs(x), axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            s = (nlcpy.conj(x) * x).real
            ret = nlcpy.sqrt(nlcpy.add.reduce(s, axis=axis, keepdims=keepdims))
            return nlcpy.asarray(ret, order=order)
        else:
            try:
                ord + 1
            except TypeError:
                raise ValueError("Invalid norm order for vectors.")
            ret = abs(x) ** ord
            ret = nlcpy.add.reduce(ret, axis=axis, keepdims=keepdims)
            ret **= (1 / ord)
            if (keepdims or x.ndim > 1) and x.dtype.char in 'fF':
                ret = nlcpy.asarray(ret, dtype='f')
            else:
                ret = nlcpy.asarray(ret, dtype='d')
            return ret
    elif len(axis) == 2:
        for ax in axis:
            if ax < -x.ndim or ax >= x.ndim:
                msg = 'axis {} is out of bounds for array of dimension {}'
                raise AxisError(msg.format(ax, x.ndim))
        row_axis, col_axis = axis
        if row_axis < 0:
            row_axis += x.ndim
        if col_axis < 0:
            col_axis += x.ndim
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            y = nlcpy.moveaxis(x, (row_axis, col_axis), (-2, -1))
            ret = nlcpy.linalg.svd(y, compute_uv=0).max(axis=-1)
        elif ord == -2:
            y = nlcpy.moveaxis(x, (row_axis, col_axis), (-2, -1))
            ret = nlcpy.linalg.svd(y, compute_uv=0).min(axis=-1)
        elif ord == 1:
            if x.shape[col_axis] == 0:
                raise ValueError(
                    'zero-size array to '
                    'reduction operation maximum which has no identity'
                )
            ret = _lange(x, ord, (row_axis, col_axis))
        elif ord == nlcpy.inf:
            if x.shape[row_axis] == 0:
                raise ValueError(
                    'zero-size array to '
                    'reduction operation maximum which has no identity'
                )
            ret = _lange(x, ord, (row_axis, col_axis))
        elif ord in (None, 'fro', 'f'):
            ret = _lange(x, ord, (row_axis, col_axis))
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = nlcpy.add.reduce(abs(x), axis=row_axis).min(axis=col_axis)
        elif ord == -nlcpy.inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = nlcpy.add.reduce(abs(x), axis=col_axis).min(axis=row_axis)
        elif ord == 'nuc':
            y = nlcpy.moveaxis(x, (row_axis, col_axis), (-2, -1))
            ret = nlcpy.sum(nlcpy.linalg.svd(y, compute_uv=0), axis=-1)
        else:
            raise ValueError("Invalid norm order for matrices.")
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        ret = nlcpy.asarray(ret, order=order)
        return ret
    else:
        raise ValueError("Improper number of dimensions to norm.")

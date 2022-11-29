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
from nlcpy import veo
from nlcpy.request import request
from . import util


def solve(a, b):
    """Solves a linear matrix equation, or system of linear scalar equations.

    Computes the "exact" solution, *x*, of the well-determined, i.e., full rank, linear
    matrix equation :math:`ax = b`.

    Parameters
    ----------
    a : (..., M, M) array_like
        Coefficient matrix.
    b : {(..., M,), (..., M, K)} array_like
        Ordinate or "dependent variable" values.

    Returns
    -------
    x : {(..., M,), (..., M, K)} ndarray
        Solution to the system a x = b. Returned shape is identical to *b*.

    Note
    ----
    The solutions are computed using LAPACK routine ``_gesv``.

    `a` must be square and of full-rank, i.e., all rows (or, equivalently, columns) must
    be linearly independent; if either is not true, use :func:`lstsq` for the
    least-squares best "solution" of the system/equation.

    Examples
    --------
    Solve the system of equations ``3 * x0 + x1 = 9`` and ``x0 + 2 * x1 = 8``:

    >>> import nlcpy as vp
    >>> a = vp.array([[3,1], [1,2]])
    >>> b = vp.array([9,8])
    >>> x = vp.linalg.solve(a, b)
    >>> x
    array([2., 3.])

    """
    a = nlcpy.asarray(a)
    b = nlcpy.asarray(b)
    c_order = (a.flags.c_contiguous or a.ndim < 4 or
               a.ndim - b.ndim < 2 and b.flags.c_contiguous) and \
        not (a.ndim < b.ndim and not b.flags.c_contiguous)
    util._assertRankAtLeast2(a)
    util._assertNdSquareness(a)

    if a.ndim - 1 == b.ndim:
        if a.shape[-1] != b.shape[-1]:
            raise ValueError(
                'solve1: Input operand 1 has a mismatch in '
                'its core dimension 0, with gufunc signature (m,m),(m)->(m) '
                '(size {0} is different from {1})'.format(b.shape[-1], a.shape[-1]))
    elif b.ndim == 1:
        raise ValueError(
            'solve: Input operand 1 does not have enough dimensions '
            '(has 1, gufunc core with signature (m,m),(m,n)->(m,n) requires 2)')
    else:
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(
                'solve: Input operand 1 has a mismatch in '
                'its core dimension 0, with gufunc signature (m,m),(m,n)->(m,n) '
                '(size {0} is different from {1})'.format(b.shape[-2], a.shape[-1]))

    if b.ndim == 1 or a.ndim - 1 == b.ndim and a.shape[-1] == b.shape[-1]:
        tmp = 1
        _newaxis = (None, )
    else:
        tmp = 2
        _newaxis = (None, None)
    for i in range(1, min(a.ndim - 2, b.ndim - tmp) + 1):
        if a.shape[-2 - i] != b.shape[-tmp - i] and \
           1 not in (a.shape[-2 - i], b.shape[-tmp - i]):
            raise ValueError(
                'operands could not be broadcast together with '
                'remapped shapes [original->remapped]: {0}->({1}) '
                '{2}->({3}) and requested shape ({4})'.format(
                    str(a.shape).replace(' ', ''),
                    str(a.shape[:-2] + _newaxis).replace(' ', '').
                    replace('None', 'newaxis').strip('(,)'),
                    str(b.shape).replace(' ', ''),
                    str(b.shape[:-tmp] + _newaxis).replace(' ', '').
                    replace('None', 'newaxis').
                    replace('None', 'newaxis').strip('(,)'),
                    str(b.shape[-tmp:]).replace(' ', '').strip('(,)')
                ))

    if a.dtype.char in 'FD' or b.dtype.char in 'FD':
        dtype = 'complex128'
        if a.dtype.char in 'fF' and b.dtype.char in 'fF':
            x_dtype = 'complex64'
        else:
            x_dtype = 'complex128'
    else:
        dtype = 'float64'
        if a.dtype.char == 'f' and b.dtype.char == 'f':
            x_dtype = 'float32'
        else:
            x_dtype = 'float64'

    x_shape = b.shape
    if b.ndim == a.ndim - 1:
        b = b[..., nlcpy.newaxis]
    diff = abs(a.ndim - b.ndim)
    if a.ndim < b.ndim:
        bcast_shape = [b.shape[i] if b.shape[i] != 1 or i < diff
                       else a.shape[i - diff] for i in range(b.ndim - 2)]
    else:
        bcast_shape = [a.shape[i] if a.shape[i] != 1 or i < diff
                       else b.shape[i - diff] for i in range(a.ndim - 2)]
    bcast_shape_a = bcast_shape + list(a.shape[-2:])
    bcast_shape_b = bcast_shape + list(b.shape[-2:])
    a = nlcpy.broadcast_to(a, bcast_shape_a)
    if bcast_shape_b != list(b.shape):
        b = nlcpy.broadcast_to(b, bcast_shape_b)
        x_shape = b.shape
    if b.size == 0:
        return nlcpy.empty(x_shape, dtype=x_dtype)

    a = nlcpy.array(nlcpy.moveaxis(a, (-1, -2), (1, 0)), dtype=dtype, order='F')
    b = nlcpy.array(nlcpy.moveaxis(b, (-1, -2), (1, 0)), dtype=dtype, order='F')

    info = numpy.empty(1, dtype='l')
    fpe = request._get_fpe_flag()
    args = (
        a,
        b,
        veo.OnStack(info, inout=veo.INTENT_OUT),
        veo.OnStack(fpe, inout=veo.INTENT_OUT),
    )

    request._push_and_flush_request(
        'nlcpy_solve',
        args,
        callback=util._assertNotSingular(info)
    )

    if c_order:
        x = nlcpy.moveaxis(b, (1, 0), (-1, -2)).reshape(x_shape)
        return nlcpy.asarray(x, x_dtype, 'C')
    else:
        x = nlcpy.asarray(b, x_dtype)
        return nlcpy.moveaxis(x, (1, 0), (-1, -2)).reshape(x_shape)


def _lstsq_errchk(info):
    def _info_check(*args):
        if info == 1:
            raise util.LinAlgError('a singular value did not converge')
    return _info_check


def lstsq(a, b, rcond='warn'):
    """Returns the least-squares solution to a linear matrix equation.

    Solves the equation :math:`ax = b` by computing a vector *x* that minimizes the
    squared Euclidean 2-norm :math:`|b-ax|^2_2`. The equation may be under-, well-, or
    over-determined (i.e., the number of linearly independent rows of *a* can be less
    than, equal to, or greater than its number of linearly independent columns). If *a*
    is square and of full rank, then *x* (but for round-off error) is the "exact"
    solution of the equation.

    Parameters
    ----------
    a : (M,N) array_like
        "Coefficient" matrix.
    b : {(M,), (M, K)} array_like
        Ordinate or "dependent variable" values. If *b* is two-dimensional, the
        least-squares solution is calculated for each of the *K* columns of *b*.
    rcond : float, optional
        Cut-off ratio for small singular values of *a*. For the purposes of rank
        determination, singular values are treated as zero if they are smaller than
        *rcond* times the largest singular value of *a*.

    Returns
    -------
    x : {(N,), (N, K)} ndarray
        Least-squares solution. If *b* is two-dimensional, the solutions are in the *K*
        columns of *x*.
    residuals : {(1,), (K,), (0,)} ndarray
        Sums of residuals; squared Euclidean 2-norm for each column in ``b - a@x``. If
        the rank of *a* is < N or M <= N, this is an empty array. If *b* is
        1-dimensional, this is a (1,) shape array. Otherwise the shape is (K,).
    rank : *int*
        Rank of matrix *a*.
    s : (min(M, N),) ndarray
        Singular values of *a*.

    Note
    ----
    If `b` is a matrix, then all array results are returned as matrices.

    Examples
    --------
    .. plot::
        :align: center

        Fit a line, ``y = mx + c``, through some noisy data-points:

        >>> import nlcpy as vp
        >>> x = vp.array([0, 1, 2, 3])
        >>> y = vp.array([-1, 0.2, 0.9, 2.1])

        By examining the coefficients, we see that the line should have a gradient of
        roughly 1 and cut the y-axis at, more or less, -1.
        We can rewrite the line equation as ``y = Ap``, where ``A = [[x 1]]`` and
        ``p = [[m], [c]]``. Now use lstsq to solve for *p*:

        >>> A = vp.array((x, vp.ones(len(x)))).T
        >>> A
        array([[0., 1.],
               [1., 1.],
               [2., 1.],
               [3., 1.]])
        >>> m, c = vp.linalg.lstsq(A, y, rcond=None)[0]
        >>> m, c  # doctest: +SKIP
        (array(1.), array(-0.95)) # may vary

        Plot the data along with the fitted line:

        >>> import matplotlib.pyplot as plt
        >>> _ = plt.plot(x, y, 'o', label='Original data', markersize=15)
        >>> _ = plt.plot(x, m*x + c, 'r', label='Fitted line')
        >>> _ = plt.legend()
        >>> plt.show()

    """
    a = nlcpy.asarray(a)
    b = nlcpy.asarray(b)
    b_ndim = b.ndim
    if b_ndim == 1:
        b = b[:, nlcpy.newaxis]
    util._assertRank2(a, b)

    if a.shape[-2] != b.shape[-2]:
        raise util.LinAlgError('Incompatible dimensions')

    a_complex = a.dtype.char in 'FD'
    b_complex = b.dtype.char in 'FD'
    if a_complex or b_complex:
        if a.dtype.char in 'fF' and b.dtype.char in 'fF':
            x_dtype = 'F'
            f_dtype = 'f'
        else:
            x_dtype = 'D'
            f_dtype = 'd'
    else:
        if a.dtype.char == 'f' and b.dtype.char == 'f':
            x_dtype = 'f'
            f_dtype = 'f'
        else:
            x_dtype = 'd'
            f_dtype = 'd'

    m = a.shape[-2]
    n = a.shape[-1]
    k = b.shape[-1]
    minmn = min(m, n)
    maxmn = max(m, n)
    k_extend = (k == 0)
    if k_extend:
        b = nlcpy.zeros([m, 1], dtype=b.dtype)
        k = 1
    if minmn == 0:
        if n > 0:
            x = nlcpy.zeros([n, k], dtype=x_dtype)
            residuals = nlcpy.array([], dtype=f_dtype)
        else:
            x = nlcpy.array([], dtype=x_dtype)
            if b_complex:
                br = nlcpy.asarray(b.real, dtype='d')
                bi = nlcpy.asarray(b.imag, dtype='d')
                square_b = nlcpy.multiply(br, br) + nlcpy.multiply(bi, bi)
            else:
                square_b = nlcpy.multiply(b, b, dtype='d')
            residuals = nlcpy.add.reduce(square_b, dtype=f_dtype)
        return (x, residuals, 0, nlcpy.array([], dtype=f_dtype))

    if rcond == 'warn':
        rcond = -1.0
    if rcond is None:
        rcond = nlcpy.finfo(x_dtype).eps * max(m, n)
    rcond = nlcpy.asarray(rcond, dtype=f_dtype)

    a = nlcpy.array(a, dtype=x_dtype, order='F')
    if m < n:
        _b = nlcpy.empty([n, k], dtype=x_dtype, order='F')
        _b[:m, :] = b
        b = _b
    else:
        b = nlcpy.array(b, dtype=x_dtype, order='F')
    s = nlcpy.empty(min(m, n), dtype=f_dtype)
    rank = numpy.empty(1, dtype='l')

    nlvl = max(0, int(nlcpy.log(minmn / 26.0) / nlcpy.log(2)) + 1)
    mnthr = int(minmn * 1.6)
    mm = m
    lwork = 1
    if a_complex or b_complex:
        _tmp = 2
        wlalsd = minmn * k
        lrwork = 10 * maxmn + 2 * maxmn * 25 + 8 * maxmn * nlvl + 3 * 25 * k \
                            + max(26 ** 2, n * (1 + k) + 2 * k)
    else:
        _tmp = 3
        wlalsd = 9 * minmn + 2 * minmn * 25 + 8 * minmn * nlvl + minmn * k + 26 ** 2
        lrwork = 1

    if m >= n:
        if m >= mnthr:
            mm = n
            lwork = max(65 * n, n + 32 * k)
        lwork = max(lwork,
                    _tmp * n + (mm + n) * 64,
                    _tmp * n + 32 * k,
                    _tmp * n + 32 * n - 32,
                    _tmp * n + wlalsd)
    else:
        if n >= mnthr:
            lwork = max(lwork,
                        m * m + 4 * m + wlalsd,
                        m * m + 132 * m,
                        m * m + 4 * m + 32 * k,
                        m * m + (k + 1) * m,
                        m * m + n + m)
        else:
            lwork = max(lwork,
                        3 * m + wlalsd,
                        3 * m + 64 * (n + m),
                        3 * m + 32 * k)

    liwork = minmn * (11 + 3 * nlvl)
    work = nlcpy.empty(lwork, dtype=f_dtype)
    iwork = nlcpy.empty(liwork, dtype='l')
    rwork = nlcpy.empty(lrwork, dtype=f_dtype)
    info = numpy.empty(1, dtype='l')
    fpe = request._get_fpe_flag()
    args = (
        a,
        b,
        s,
        work,
        rwork,
        iwork,
        rcond,
        veo.OnStack(rank, inout=veo.INTENT_OUT),
        veo.OnStack(info, inout=veo.INTENT_OUT),
        veo.OnStack(fpe, inout=veo.INTENT_OUT),
    )

    request._push_and_flush_request(
        'nlcpy_lstsq',
        args,
        callback=_lstsq_errchk(info),
        sync=True
    )

    if rank < n or m <= n:
        residuals = nlcpy.array([], dtype=f_dtype)
    else:
        _b = b[n:]
        square_b = nlcpy.multiply(_b.real, _b.real) + nlcpy.multiply(_b.imag, _b.imag)
        residuals = nlcpy.add.reduce(square_b, dtype=f_dtype)
    if k_extend:
        x = b[..., :0]
        residuals = residuals[..., :0]
    elif b_ndim == 1:
        x = nlcpy.asarray(b[:n, 0], dtype=x_dtype, order='C')
    else:
        x = nlcpy.asarray(b[:n, :], dtype=x_dtype, order='C')

    return (x, residuals, rank[0], s)


def inv(a):
    """Computes the (multiplicative) inverse of a matrix.

    Given a square matrix *a*, return the matrix *ainv* satisfying

    ::

        dot(a, ainv) = dot(ainv, a) = eye(a.shape[0]).

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be inverted.

    Returns
    -------
    ainv : (..., M, M) ndarray
        (Multiplicative) inverse of the matrix *a*.

    Note
    ----
    Broadcasting rules apply, see the :ref:`nlcpy.linalg <nlcpy_linalg>`
    documentation for details.

    Examples
    --------
    >>> import nlcpy as vp
    >>> from nlcpy import testing
    >>> a = vp.array([[1., 2.], [3., 4.]])
    >>> ainv = vp.linalg.inv(a)
    >>> vp.testing.assert_allclose(vp.dot(a, ainv), vp.eye(2), atol=1e-8, rtol=1e-5)
    >>> vp.testing.assert_allclose(vp.dot(ainv, a), vp.eye(2), atol=1e-8, rtol=1e-5)

    Inverses of several matrices can be computed at once:

    >>> a = vp.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])
    >>> vp.linalg.inv(a)   # doctest: +SKIP
    array([[[-2.  ,  1.  ],
            [ 1.5 , -0.5 ]],
    <BLANKLINE>
           [[-1.25,  0.75],
            [ 0.75, -0.25]]])

    """
    a = nlcpy.asarray(a)
    # used to match the contiguous of result to numpy.
    c_order = a.flags.c_contiguous or sum([i > 1 for i in a.shape[:-2]]) < 2
    util._assertRankAtLeast2(a)
    util._assertNdSquareness(a)
    if a.dtype.char in 'FD':
        dtype = 'D'
        if a.dtype.char in 'fF':
            ainv_dtype = 'F'
        else:
            ainv_dtype = 'D'
    else:
        dtype = 'd'
        if a.dtype.char == 'f':
            ainv_dtype = 'f'
        else:
            ainv_dtype = 'd'
    if a.size == 0:
        return nlcpy.asarray(a, dtype=ainv_dtype)
    a = nlcpy.array(nlcpy.moveaxis(a, (-1, -2), (1, 0)), dtype=dtype, order='F')
    ipiv = nlcpy.empty(a.shape[-1], dtype='l')
    work = nlcpy.empty(a.shape[-1] * 256)
    info = numpy.empty(1, dtype='l')
    fpe = request._get_fpe_flag()
    args = (
        a,
        ipiv,
        work,
        veo.OnStack(info, inout=veo.INTENT_OUT),
        veo.OnStack(fpe, inout=veo.INTENT_OUT),
    )

    request._push_and_flush_request(
        'nlcpy_inv',
        args,
        callback=util._assertNotSingular(info)
    )

    if c_order:
        a = nlcpy.moveaxis(a, (1, 0), (-1, -2))
        return nlcpy.asarray(a, dtype=ainv_dtype, order='C')
    else:
        a = nlcpy.asarray(a, dtype=ainv_dtype)
        return nlcpy.moveaxis(a, (1, 0), (-1, -2))

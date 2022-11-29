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
import numpy
import warnings
from nlcpy import veo
from nlcpy.request import request
from . import util


def _take_along_axis(arr, indices, axis):
    shape = arr.shape
    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))
    fancy_index = []
    for dim, n in zip(dest_dims, shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1:]
            fancy_index.append(nlcpy.arange(n).reshape(ind_shape))
    return arr[tuple(fancy_index)]


def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """Singular Value Decomposition.

    When *a* is a 2D array, it is factorized as
    ``u @ nlcpy.diag(s) @ vh = (u * s) @ vh``,
    where *u* and *vh* are 2D unitary arrays and *s* is a 1D array of *a*'s singular
    values.
    When *a* is higher-dimensional, SVD is applied in stacked mode as explained below.

    Parameters
    ----------
    a : (..., M, N) array_like
        A real or complex array with a.ndim >= 2.
    full_matrices : bool, optional
        If True (default), *u* and *vh* have the shapes ``(..., M, M)`` and ``(..., N,
        N)``, respectively. Otherwise, the shapes are ``(..., M, K)`` and ``(..., K,
        N)``, respectively, where ``K = min(M, N)``.
    compute_uv : bool, optional
        Whether or not to compute *u* and *vh* in addition to *s*. True by default.
    hermitian : bool, optional
        If True, *a* is assumed to be Hermitian (symmetric if real-valued), enabling a
        more efficient method for finding singular values. Defaults to False.

    Returns
    -------
    u : {(..., M, M), (..., M, K)} ndarray
        Unitary array(s). The first ``a.ndim - 2`` dimensions have the same size as those
        of the input *a*. The size of the last two dimensions depends on the value of
        *full_matrices*. Only returned when *compute_uv* is True.
    s : (..., K) ndarray
        Vector(s) with the singular values, within each vector sorted in descending
        order. The first ``a.ndim - 2`` dimensions have the same size as those of the
        input *a*.
    vh : {(..., N, N), (..., K, N)} ndarray
        Unitary array(s). The first ``a.ndim - 2`` dimensions have the same size as those
        of the input *a*. The size of the last two dimensions depends on the value of
        *full_matrices*. Only returned when *compute_uv* is True.

    Note
    ----
    The decomposition is performed using LAPACK routine ``_gesdd``.

    SVD is usually described for the factorization of a 2D matrix :math:`A`.
    The higher-dimensional case will be discussed below. In the 2D case, SVD is written
    as
    :math:`A=USV^{H}`, where :math:`A = a`, :math:`U = u`, :math:`S = nlcpy.diag(s)`
    and :math:`V^{H} = vh`. The 1D array `s` contains the singular values of `a` and
    `u` and `vh` are unitary. The rows of `vh` are the eigenvectors of :math:`A^{H}A`
    and the columns of `u` are the eigenvectors of :math:`AA^{H}`. In both cases the
    corresponding (possibly non-zero) eigenvalues are given by ``s**2``.

    If `a` has more than two dimensions, then broadcasting rules apply, as explained in
    :ref:`Linear algebra on several matrices at once <linalg_several_matrices_at_once>`.
    This means that SVD is working in "stacked" mode: it iterates over all indices
    of the first ``a.ndim - 2`` dimensions and for each combination SVD is applied to the
    last two indices.

    Examples
    --------
    >>> import nlcpy as vp
    >>> from nlcpy import testing
    >>> a = vp.random.randn(9, 6) + 1j*vp.random.randn(9, 6)

    Reconstruction based on full SVD, 2D case:

    >>> u, s, vh = vp.linalg.svd(a, full_matrices=True)
    >>> u.shape, s.shape, vh.shape
    ((9, 9), (6,), (6, 6))
    >>> vp.testing.assert_allclose(a, vp.dot(u[:, :6] * s, vh))
    >>> smat = vp.zeros((9, 6), dtype=complex)
    >>> smat[:6, :6] = vp.diag(s)
    >>> vp.testing.assert_allclose(a, vp.dot(u, vp.dot(smat, vh)))

    Reconstruction based on reduced SVD, 2D case:

    >>> u, s, vh = vp.linalg.svd(a, full_matrices=False)
    >>> u.shape, s.shape, vh.shape
    ((9, 6), (6,), (6, 6))
    >>> vp.testing.assert_allclose(a, vp.dot(u * s, vh))
    >>> smat = vp.diag(s)
    >>> vp.testing.assert_allclose(a, vp.dot(u, vp.dot(smat, vh)))

    """
    a = nlcpy.asarray(a)
    util._assertRankAtLeast2(a)
    if hermitian:
        util._assertNdSquareness(a)

    # used to match the contiguous of result to numpy.
    c_order = a.flags.c_contiguous or sum([i > 1 for i in a.shape[:-2]]) < 2

    a_complex = a.dtype.char in 'FD'
    if a.dtype == 'F':
        dtype = 'F'
        f_dtype = 'f'
    elif a.dtype == 'D':
        dtype = 'D'
        f_dtype = 'd'
    elif a.dtype == 'f':
        dtype = 'f'
        f_dtype = 'f'
    else:
        dtype = 'd'
        f_dtype = 'd'

    if hermitian:
        if compute_uv:
            # lapack returns eigenvalues in reverse order, so to reconsist.
            s, u = nlcpy.linalg.eigh(a)
            signs = nlcpy.sign(s)
            s = abs(s)
            sidx = nlcpy.argsort(s)[..., ::-1]
            signs = _take_along_axis(signs, sidx, signs.ndim - 1)
            s = _take_along_axis(s, sidx, s.ndim - 1)
            u = _take_along_axis(u, sidx[..., None, :], u.ndim - 1)
            # singular values are unsigned, move the sign into v
            vt = nlcpy.conjugate(u * signs[..., None, :])
            vt = nlcpy.moveaxis(vt, -2, -1)
            return u, s, vt
        else:
            s = nlcpy.linalg.eigvalsh(a)
            s = nlcpy.sort(abs(s))[..., ::-1]
            return s

    m = a.shape[-2]
    n = a.shape[-1]
    min_mn = min(m, n)
    max_mn = max(m, n)
    if a.size == 0:
        s = nlcpy.empty(a.shape[:-2] + (min_mn,), f_dtype)
        if compute_uv:
            if full_matrices:
                u_shape = a.shape[:-1] + (m, )
                vt_shape = a.shape[:-2] + (n, n)
            else:
                u_shape = a.shape[:-1] + (min_mn, )
                vt_shape = a.shape[:-2] + (min_mn, n)
            u = nlcpy.empty(u_shape, dtype=dtype)
            vt = nlcpy.empty(vt_shape, dtype=dtype)
            return u, s, vt
        else:
            return s

    a = nlcpy.array(nlcpy.moveaxis(a, (-1, -2), (1, 0)), dtype=dtype, order='F')
    if compute_uv:
        if full_matrices:
            u = nlcpy.empty((m, m) + a.shape[2:], dtype=dtype, order='F')
            vt = nlcpy.empty((n, n) + a.shape[2:], dtype=dtype, order='F')
            job = 'A'
        else:
            u = nlcpy.empty((m, m) + a.shape[2:], dtype=dtype, order='F')
            vt = nlcpy.empty((min_mn, n) + a.shape[2:], dtype=dtype, order='F')
            job = 'S'
    else:
        u = nlcpy.empty(1, dtype=dtype)
        vt = nlcpy.empty(1, dtype=dtype)
        job = 'N'

    if a_complex:
        mnthr1 = int(min_mn * 17.0 / 9.0)
        if max_mn >= mnthr1:
            if job == 'N':
                lwork = 130 * min_mn
            elif job == 'S':
                lwork = (min_mn + 130) * min_mn
            else:
                lwork = max(
                    (min_mn + 130) * min_mn,
                    (min_mn + 1) * min_mn + 32 * max_mn,
                )
        else:
            lwork = 64 * (min_mn + max_mn) + 2 * min_mn
    else:
        mnthr = int(min_mn * 11.0 / 6.0)
        if m >= n:
            if m >= mnthr:
                if job == 'N':
                    lwork = 131 * n
                elif job == 'S':
                    lwork = max((131 + n) * n, (4 * n + 7) * n)
                else:
                    lwork = max(
                        (n + 131) * n,
                        (n + 1) * n + 32 * m,
                        (4 * n + 6) * n + m
                    )
            else:
                if job == 'N':
                    lwork = 64 * m + 67 * n
                elif job == 'S':
                    lwork = max(64 * m + 67 * n, (3 * n + 7) * n)
                else:
                    lwork = (3 * n + 7) * n
        else:
            if n >= mnthr:
                if job == 'N':
                    lwork = 131 * m
                elif job == 'S':
                    lwork = max((m + 131) * m, (4 * m + 7) * m)
                else:
                    lwork = max(
                        (m + 131) * m,
                        (m + 1) * m + 32 * n,
                        (4 * m + 7) * m
                    )
            else:
                if job == 'N':
                    lwork = 67 * m + 64 * n
                else:
                    lwork = max(67 * m + 64 * n, (3 * m + 7) * m)

    s = nlcpy.empty((min_mn,) + a.shape[2:], dtype=f_dtype, order='F')
    work = nlcpy.empty(lwork, dtype=dtype)
    if a_complex:
        if job == 'N':
            lrwork = 7 * n * m
        else:
            lrwork = min_mn * max(5 * min_mn + 7, 2 * max(m, n) + 2 * min_mn + 1)
    else:
        lrwork = 1
    rwork = nlcpy.empty(lrwork, dtype=f_dtype)
    iwork = nlcpy.empty(8 * min_mn, dtype='l')
    info = numpy.empty(1, dtype='l')
    fpe = request._get_fpe_flag()
    args = (
        ord(job),
        a,
        s,
        u,
        vt,
        work,
        rwork,
        iwork,
        veo.OnStack(info, inout=veo.INTENT_OUT),
        veo.OnStack(fpe, inout=veo.INTENT_OUT),
    )

    request._push_and_flush_request(
        'nlcpy_svd',
        args,
    )

    if c_order:
        s = nlcpy.asarray(nlcpy.moveaxis(s, 0, -1), order='C')
    else:
        s = nlcpy.moveaxis(s, 0, -1)
    if compute_uv:
        u = nlcpy.moveaxis(u, (1, 0), (-1, -2))
        if not full_matrices:
            u = u[..., :m, :min_mn]
        if c_order:
            u = nlcpy.asarray(u, dtype=dtype, order='C')
            vt = nlcpy.asarray(nlcpy.moveaxis(vt, (1, 0), (-1, -2)), dtype, order='C')
        else:
            vt = nlcpy.moveaxis(nlcpy.asarray(vt, dtype), (1, 0), (-1, -2))
        return u, s, vt
    else:
        return s


def cholesky(a):
    """Cholesky decomposition.

    Return the Cholesky decomposition, *L* * *L.H*, of the square matrix *a*, where *L*
    is lower-triangular and *.H* is the conjugate transpose operator (which is the
    ordinary transpose if *a* is real-valued). *a* must be Hermitian (symmetric if
    real-valued) and positive-definite. Only *L* is actually returned.

    Parameters
    ----------
    a : (..., M, M) array_like
        Hermitian (symmetric if all elements are real), positive-definite input matrix.

    Returns
    -------
    L : (..., M, M) ndarray
        Upper or lower-triangular Cholesky factor of *a*.

    Note
    ----
    The Cholesky decomposition is often used as a fast way of solving :math:`Ax = b`

    (when *A* is both Hermitian/symmetric and positive-definite).

    First, we solve for y in :math:`Ly = b`, and then for x in :math:`L.Hx = y`.

    Examples
    --------
    >>> import nlcpy as vp
    >>> A = vp.array([[1,-2j],[2j,5]])
    >>> A
    array([[ 1.+0.j, -0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> L = vp.linalg.cholesky(A)
    >>> L
    array([[1.+0.j, 0.+0.j],
           [0.+2.j, 1.+0.j]])
    >>> vp.dot(L, vp.conjugate(L.T)) # verify that L * L.H = A
    array([[1.+0.j, 0.-2.j],
           [0.+2.j, 5.+0.j]])

    """
    a = nlcpy.asarray(a)
    util._assertRankAtLeast2(a)
    util._assertNdSquareness(a)

    if a.dtype == 'F':
        dtype = 'D'
        L_dtype = 'F'
    elif a.dtype == 'D':
        dtype = 'D'
        L_dtype = 'D'
    elif a.dtype == 'f':
        dtype = 'd'
        L_dtype = 'f'
    else:
        dtype = 'd'
        L_dtype = 'd'

    if a.size == 0:
        return nlcpy.empty(a.shape, dtype=L_dtype)

    # used to match the contiguous of result to numpy.
    c_order = a.flags.c_contiguous or sum([i > 1 for i in a.shape[:-2]]) < 2

    a = nlcpy.array(nlcpy.moveaxis(a, (-1, -2), (1, 0)), dtype=dtype, order='F')
    info = numpy.empty(1, dtype='l')
    fpe = request._get_fpe_flag()
    args = (
        a,
        veo.OnStack(info, inout=veo.INTENT_OUT),
        veo.OnStack(fpe, inout=veo.INTENT_OUT),
    )

    request._push_and_flush_request(
        'nlcpy_cholesky',
        args,
        callback=util._assertPositiveDefinite(info)
    )

    if c_order:
        L = nlcpy.asarray(nlcpy.moveaxis(a, (1, 0), (-1, -2)), dtype=L_dtype, order='C')
    else:
        L = nlcpy.moveaxis(nlcpy.asarray(a, dtype=L_dtype), (1, 0), (-1, -2))
    return L


def qr(a, mode='reduced'):
    """Computes the qr factorization of a matrix.

    Factor the matrix *a* as *qr*, where *q* is orthonormal and *r* is upper-triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be factored.
    mode : {'reduced', 'complete', 'r', 'raw', 'full', 'economic'}, optional
        If K = min(M, N), then

        - 'reduced' : returns q, r with dimensions (M, K), (K, N) (default)
        - 'complete' : returns q, r with dimensions (M, M), (M, N)
        - 'r' : returns r only with dimensions (K, N)
        - 'raw' : returns h, tau with dimensions (N, M), (K,)
        - 'full' or 'f' : alias of 'reduced', deprecated
        - 'economic' or 'e' : returns h from 'raw', deprecated.

    Returns
    -------
    q : ndarray, optional
        A matrix with orthonormal columns. When mode = 'complete' the result is an
        orthogonal/unitary matrix depending on whether or not a is real/complex. The
        determinant may be either +/- 1 in that case.
    r : ndarray, optional
        The upper-triangular matrix.
    (h, tau) : ndarray, optional
        The array h contains the Householder reflectors that generate q along with r. The
        tau array contains scaling factors for the reflectors. In the deprecated
        'economic' mode only h is returned.

    Note
    ----
    This is an interface to the LAPACK routines ``dgeqrf``, ``zgeqrf``, ``dorgqr``,
    and ``zungqr``.

    For more information on the qr factorization, see for example:
    https://en.wikipedia.org/wiki/QR_factorization

    Note that when 'raw' option is specified the returned arrays are of type "float64" or
    "complex128" and the h array is transposed to be FORTRAN compatible.

    Examples
    --------
    >>> import numpy as np
    >>> import nlcpy as vp
    >>> from nlcpy import testing
    >>> a = vp.random.randn(9, 6)
    >>> q, r = vp.linalg.qr(a)
    >>> vp.testing.assert_allclose(a, vp.dot(q, r))  # a does equal qr
    >>> r2 = vp.linalg.qr(a, mode='r')
    >>> r3 = vp.linalg.qr(a, mode='economic')
    >>> # mode='r' returns the same r as mode='full'
    >>> vp.testing.assert_allclose(r, r2)
    >>> # But only triu parts are guaranteed equal when mode='economic'
    >>> vp.testing.assert_allclose(r, np.triu(r3[:6,:6], k=0))

    Example illustrating a common use of qr: solving of least squares problems

    What are the least-squares-best *m* and *y0* in ``y = y0 + mx`` for the following
    data: {(0,1), (1,0), (1,2), (2,1)}. (Graph the points and you’ll see that it should
    be y0 = 0, m = 1.) The answer is provided by solving the over-determined matrix
    equation ``Ax = b``, where::

        A = array([[0, 1], [1, 1], [1, 1], [2, 1]])
        x = array([[y0], [m]])
        b = array([[1], [0], [2], [1]])

    If A = qr such that q is orthonormal (which is always possible via Gram-Schmidt),
    then ``x = inv(r) * (q.T) * b``. (In practice, however, we simply use :func:`lstsq`.)

    >>> A = vp.array([[0, 1], [1, 1], [1, 1], [2, 1]])
    >>> A
    array([[0, 1],
           [1, 1],
           [1, 1],
           [2, 1]])
    >>> b = vp.array([1, 0, 2, 1])
    >>> q, r = vp.linalg.qr(A)
    >>> p = vp.dot(q.T, b)
    >>> vp.dot(vp.linalg.inv(r), p)
    array([1.1102230246251565e-16, 1.0000000000000002e+00])

    """
    if mode not in ('reduced', 'complete', 'r', 'raw'):
        if mode in ('f', 'full'):
            msg = "".join((
                  "The 'full' option is deprecated in favor of 'reduced'.\n",
                  "For backward compatibility let mode default."))
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
            mode = 'reduced'
        elif mode in ('e', 'economic'):
            msg = "The 'economic' option is deprecated."
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
            mode = 'economic'
        else:
            raise ValueError("Unrecognized mode '%s'" % mode)

    a = nlcpy.asarray(a)
    util._assertRank2(a)
    if a.dtype == 'F':
        dtype = 'D'
        a_dtype = 'F'
    elif a.dtype == 'D':
        dtype = 'D'
        a_dtype = 'D'
    elif a.dtype == 'f':
        dtype = 'd'
        a_dtype = 'f'
    else:
        dtype = 'd'
        a_dtype = 'd'

    m, n = a.shape
    if a.size == 0:
        if mode == 'reduced':
            return nlcpy.empty((m, 0), a_dtype), nlcpy.empty((0, n), a_dtype)
        elif mode == 'complete':
            return nlcpy.identity(m, a_dtype), nlcpy.empty((m, n), a_dtype)
        elif mode == 'r':
            return nlcpy.empty((0, n), a_dtype)
        elif mode == 'raw':
            return nlcpy.empty((n, m), dtype), nlcpy.empty((0,), dtype)
        else:
            return nlcpy.empty((m, n), a_dtype), nlcpy.empty((0,), a_dtype)

    a = nlcpy.asarray(a, dtype=dtype, order='F')
    k = min(m, n)
    if mode == 'complete':
        if m > n:
            x = nlcpy.empty((m, m), dtype=dtype, order='F')
            x[:m, :n] = a
            a = x
        r_shape = (m, n)
    elif mode in ('r', 'reduced', 'economic'):
        r_shape = (k, n)
    else:
        r_shape = 1
    jobq = 0 if mode in ('r', 'raw', 'economic') else 1
    tau = nlcpy.empty(k, dtype=dtype)
    r = nlcpy.zeros(r_shape, dtype=dtype)
    work = nlcpy.empty(n * 64, dtype=dtype)
    fpe = request._get_fpe_flag()
    args = (
        m, n, jobq,
        a,
        tau,
        r,
        work,
        veo.OnStack(fpe, inout=veo.INTENT_OUT),
    )

    request._push_and_flush_request(
        'nlcpy_qr',
        args,
    )

    if mode == 'raw':
        return a.T, tau

    if mode == 'r':
        return nlcpy.asarray(r, dtype=a_dtype)

    if mode == 'economic':
        return nlcpy.asarray(a, dtype=a_dtype)

    mc = m if mode == 'complete' else k
    q = nlcpy.asarray(a[:, :mc], dtype=a_dtype, order='C')
    r = nlcpy.asarray(r, dtype=a_dtype, order='C')
    return q, r

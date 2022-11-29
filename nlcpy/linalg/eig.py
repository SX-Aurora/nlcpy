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


def _geev(a, jobvr):
    a = nlcpy.asarray(a)
    util._assertRankAtLeast2(a)
    util._assertNdSquareness(a)

    # used to match the contiguous of result to numpy.
    c_order = a.flags.c_contiguous or sum([i > 1 for i in a.shape[:-2]]) < 2

    a_complex = a.dtype.char in 'FD'
    if a.dtype.char == 'F':
        dtype = 'D'
        f_dtype = 'f'
        c_dtype = 'F'
    elif a.dtype.char == 'D':
        dtype = 'D'
        f_dtype = 'd'
        c_dtype = 'D'
    else:
        dtype = 'd'
        if a.dtype.char == 'f':
            f_dtype = 'f'
            c_dtype = 'F'
        else:
            f_dtype = 'd'
            c_dtype = 'D'

    if a.size == 0:
        dtype = c_dtype if a_complex else f_dtype
        w = nlcpy.empty(shape=a.shape[:-1], dtype=dtype)
        if jobvr:
            vr = nlcpy.empty(shape=a.shape, dtype=dtype)
            return w, vr
        else:
            return w

    a = nlcpy.array(nlcpy.moveaxis(a, (-1, -2), (1, 0)), dtype=dtype, order='F')
    wr = nlcpy.empty(a.shape[1:], dtype=dtype, order='F')
    wi = nlcpy.empty(a.shape[1:], dtype=dtype, order='F')
    vr = nlcpy.empty(a.shape if jobvr else 1, dtype=dtype, order='F')
    vc = nlcpy.empty(a.shape if jobvr else 1, dtype='D', order='F')

    n = a.shape[0]
    work = nlcpy.empty(
        65 * n if a_complex else 66 * n, dtype=dtype, order='F')
    rwork = nlcpy.empty(2 * n if a_complex else 1, dtype='d', order='F')
    info = numpy.empty(1, dtype='l')
    fpe = request._get_fpe_flag()
    args = (
        a,
        wr,
        wi,
        vr,
        vc,
        work,
        rwork,
        ord('V') if jobvr else ord('N'),
        veo.OnStack(info, inout=veo.INTENT_OUT),
        veo.OnStack(fpe, inout=veo.INTENT_OUT),
    )

    request._push_and_flush_request(
        'nlcpy_eig',
        args,
    )

    if a_complex:
        w_complex = True
        w = wr
        vc = vr
    else:
        w_complex = nlcpy.any(wi)
        w = wr + wi * 1.0j
    if w_complex:
        if c_order:
            w = nlcpy.asarray(nlcpy.moveaxis(w, 0, -1), dtype=c_dtype, order='C')
        else:
            w = nlcpy.moveaxis(nlcpy.asarray(w, dtype=c_dtype), 0, -1)
    else:
        wr = w.real
        w = nlcpy.moveaxis(nlcpy.asarray(wr, dtype=f_dtype), 0, -1)

    if jobvr:
        if w_complex:
            if c_order:
                vr = nlcpy.asarray(
                    nlcpy.moveaxis(vc, (1, 0), (-1, -2)), dtype=c_dtype, order='C')
            else:
                vr = nlcpy.moveaxis(
                    nlcpy.asarray(vc, dtype=c_dtype), (1, 0), (-1, -2))
        else:
            if c_dtype == "F":
                vr = nlcpy.asarray(vc.real, dtype=f_dtype, order='C')
            else:
                vc = nlcpy.moveaxis(
                    nlcpy.asarray(vc, dtype=c_dtype), (1, 0), (-1, -2))
                vr = vc.real

    if jobvr:
        return w, vr
    else:
        return w


def _syevd(a, jobz, UPLO):
    a = nlcpy.asarray(a)
    util._assertRankAtLeast2(a)
    util._assertNdSquareness(a)
    UPLO = UPLO.upper()
    if UPLO not in 'UL':
        raise ValueError("UPLO argument must be 'L' or 'U'")

    # used to match the contiguous of result to numpy.
    c_order = a.flags.c_contiguous or sum([i > 1 for i in a.shape[:-2]]) < 2

    a_complex = a.dtype.char in 'FD'
    if a.dtype.char == 'F':
        dtype = 'F'
        f_dtype = 'f'
    elif a.dtype.char == 'D':
        dtype = 'D'
        f_dtype = 'd'
    else:
        if a.dtype.char == 'f':
            dtype = 'f'
            f_dtype = 'f'
        else:
            dtype = 'd'
            f_dtype = 'd'

    if a.size == 0:
        w = nlcpy.empty(shape=a.shape[:-1], dtype=f_dtype)
        if jobz:
            vr = nlcpy.empty(shape=a.shape, dtype=dtype)
            return w, vr
        else:
            return w

    a = nlcpy.array(nlcpy.moveaxis(a, (-1, -2), (1, 0)), dtype=dtype, order='F')
    w = nlcpy.empty(a.shape[1:], dtype=f_dtype, order='F')
    n = a.shape[0]
    if a.size > 1:
        if a_complex:
            lwork = max(2 * n + n * n, n + 48)
            lrwork = 1 + 5 * n + 2 * n * n if jobz else n
        else:
            lwork = max(2 * n + 32, 1 + 6 * n + 2 * n * n) if jobz else 2 * n + 32
            lrwork = 1
        liwork = 3 + 5 * n if jobz else 1
    else:
        lwork = 1
        lrwork = 1
        liwork = 1

    work = nlcpy.empty(lwork, dtype=dtype)
    rwork = nlcpy.empty(lrwork, dtype=f_dtype)
    iwork = nlcpy.empty(liwork, dtype='l')
    info = numpy.empty(1, dtype='l')
    fpe = request._get_fpe_flag()
    args = (
        a,
        w,
        work,
        rwork,
        iwork,
        ord('V') if jobz else ord('N'),
        ord(UPLO),
        veo.OnStack(info, inout=veo.INTENT_OUT),
        veo.OnStack(fpe, inout=veo.INTENT_OUT),
    )

    request._push_and_flush_request(
        'nlcpy_eigh',
        args,
    )

    if c_order:
        w = nlcpy.asarray(nlcpy.moveaxis(w, 0, -1), order='C')
    else:
        w = nlcpy.moveaxis(w, 0, -1)
    if jobz:
        if c_order:
            a = nlcpy.asarray(nlcpy.moveaxis(a, (1, 0), (-1, -2)), order='C')
        else:
            a = nlcpy.moveaxis(a, (1, 0), (-1, -2))
        return w, a
    else:
        return w


def eig(a):
    """Computes the eigenvalues and right eigenvectors of a square array.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrices for which the eigenvalues and right eigenvectors will be computed.

    Returns
    -------
    w : (..., M) ndarray
        The eigenvalues, each repeated according to its multiplicity. The eigenvalues are
        not necessarily ordered. The resulting array will be of complex type, unless the
        imaginary part is zero in which case it will be cast to a real type. Please note
        that there are cases where rounding errors affect the dtype of the array. When
        *a* is real the resulting eigenvalues will be real (0 imaginary part) or occur in
        conjugate pairs.
    v : (..., M, M) ndarray
        The normalized (unit "length") eigenvectors, such that the column ``v[:,i]`` is
        the eigenvector corresponding to the eigenvalue ``w[i]``.

    Note
    ----
    This is implemented using the ``_geev`` LAPACK routines which compute the eigenvalues
    and eigenvectors of general square arrays.

    The number `w` is an eigenvalue of `a` if there exists a vector `v` such that
    ``a @ v = w * v``.
    Thus, the arrays `a`, `w`, and `v` satisfy the equations
    ``a @ v[:,i] = w[i] * v[:,i]`` for `i` in {0, 1, ..., M-1}.

    The array `v` of eigenvectors may not be of maximum rank, that is, some of the
    columns may be linearly dependent, although round-off error may obscure that fact. If
    the eigenvalues are all different, then theoretically the eigenvectors are linearly
    independent. Likewise, the (complex-valued) matrix of eigenvectors `v` is unitary if
    the matrix `a` is normal, i.e., if ``dot(a, a.H) = dot(a.H, a)``, where `a.H` denotes
    the conjugate transpose of `a.`

    Finally, it is emphasized that `v` consists of the `right` (as in right-hand side)
    eigenvectors of `a.` A vector `y` satisfying ``y.T @ a = z * y.T`` for some number
    `z` is called a `left` eigenvector of `a,` and, in general, the left and right
    eigenvectors of a matrix are not necessarily the (perhaps conjugate) transposes of
    each other.

    See Also
    --------
    eigvals : Computes the eigenvalues of a general matrix.
    eigh : Computes the eigenvalues and eigenvectors of a complex Hermitian or a real
        symmetric matrix.
    eigvalsh : Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

    Examples
    --------
    (Almost) trivial example with real e-values and e-vectors.

    >>> import nlcpy as vp
    >>> w, v = vp.linalg.eig(vp.diag((1, 2, 3)))
    >>> w; v
    array([1., 2., 3.])
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Real matrix possessing complex e-values and e-vectors; note that the e-values are
    complex conjugates of each other.

    >>> w, v = vp.linalg.eig(vp.array([[1, -1], [1, 1]]))
    >>> w; v     # doctest: +SKIP
    array([1.+1.j, 1.-1.j])
    array([[0.70710678+0.j        , 0.70710678-0.j        ],
           [0.        -0.70710678j, 0.        +0.70710678j]])

    Complex-valued matrix with real e-values (but complex-valued e-vectors); note that
    ``a.conj().T == a``, i.e., *a* is Hermitian.

    >>> a = vp.array([[1, 1j], [-1j, 1]])
    >>> w, v = vp.linalg.eig(a)
    >>> w; v  # doctest: +SKIP
    array([2.+0.j, 0.+0.j])
    array([[ 0.        +0.70710678j,  0.70710678+0.j        ], # may vary
           [ 0.70710678+0.j        , -0.        +0.70710678j]])

    Be careful about round-off error!

    >>> a = vp.array([[1 + 1e-9, 0], [0, 1 - 1e-9]])
    >>> # Theor. e-values are 1 +/- 1e-9
    >>> w, v = vp.linalg.eig(a)
    >>> w; v   # doctest: +SKIP
    array([1., 1.])
    array([[1., 0.],
           [0., 1.]])

    """
    return _geev(a, True)


def eigvals(a):
    """Computes the eigenvalues of a general matrix.

    Main difference from :func:`eig` : the eigenvectors aren't returned.

    Parameters
    ----------
    a : (..., M, M) array_like
        A complex- or real-valued matrix whose eigenvalues will be computed.

    Returns
    -------
    w : (..., M) ndarray
        The eigenvalues, each repeated according to its multiplicity. They are not
        necessarily ordered, nor are they necessarily real for real matrices.

    Note
    ----
    This is implemented using the ``_geev`` LAPACK routines which compute the eigenvalues
    and eigenvectors of general square arrays.

    See Also
    --------
    eig : Computes the eigenvalues and right eigenvectors of a square array.
    eigh : Computes the eigenvalues and eigenvectors of a complex Hermitian or a real
        symmetric matrix.
    eigvalsh : Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

    Examples
    --------
    Illustration, using the fact that the eigenvalues of a diagonal matrix are its
    diagonal elements, that multiplying a matrix on the left by an orthogonal matrix,
    *Q*, and on the right by *Q.T* (the transpose of *Q*), preserves the eigenvalues of
    the "middle" matrix. In other words, if *Q* is orthogonal, then ``Q * A * Q.T`` has
    the same eigenvalues as ``A``:

    >>> import nlcpy as vp
    >>> x = vp.random.random()
    >>> Q = vp.array([[vp.cos(x), -vp.sin(x)], [vp.sin(x), vp.cos(x)]])
    >>> vp.linalg.norm(Q[0, :]), vp.linalg.norm(Q[1, :]), vp.dot(Q[0, :],Q[1, :])
    (array(1.), array(1.), array(0.))

    Now multiply a diagonal matrix by ``Q`` on one side and by ``Q.T`` on the other:

    >>> D = vp.diag((-1,1))
    >>> vp.linalg.eigvals(D)
    array([-1.,  1.])
    >>> A = vp.dot(Q, D)
    >>> A = vp.dot(A, Q.T)
    >>> vp.linalg.eigvals(A) # doctest: +SKIP
    array([ 1., -1.]) # random

    """
    return _geev(a, False)


def eigh(a, UPLO='L'):
    """Computes the eigenvalues and eigenvectors of a complex Hermitian or a real
    symmetric matrix.

    Returns two objects, a 1-D array containing the eigenvalues of *a*, and a 2-D square
    array or matrix (depending on the input type) of the corresponding eigenvectors (in
    columns).

    Parameters
    ----------
    a : (..., M, M) array_like
        Hermitian or real symmetric matrices whose eigenvalues and eigenvectors are to be
        computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular part of *a*
        ('L', default) or the upper triangular part ('U'). Irrespective of this value
        only the real parts of the diagonal will be considered in the computation to
        preserve the notion of a Hermitian matrix. It therefore follows that the
        imaginary part of the diagonal will always be treated as zero.

    Returns
    -------
    w : (..., M) ndarray
        The eigenvalues in ascending order, each repeated according to its multiplicity.
    v : (..., M, M) ndarray
        The column ``v[:, i]`` is the normalized eigenvector corresponding to the
        eigenvalue ``w[i]``.

    Note
    ----
    The eigenvalues/eigenvectors are computed using LAPACK routines ``_syevd``,
    ``_heevd``.

    The eigenvalues of real symmetric or complex Hermitian matrices are always real. The
    array `v` of (column) eigenvectors is unitary and `a,` `w,` and `v` satisfy the
    equations ``dot(a, v[:, i]) = w[i] * v[:, i]``.

    See Also
    --------
    eigvalsh : Computes the eigenvalues of a complex Hermitian or real symmetric matrix.
    eig : Computes the eigenvalues and right eigenvectors of a square array.
    eigvals : Computes the eigenvalues of a general matrix.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1, -2j], [2j, 5]])
    >>> a
    array([[ 1.+0.j, -0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> w, v = vp.linalg.eigh(a)
    >>> w; v    # doctest: +SKIP
    array([0.17157288, 5.82842712])
    array([[-0.92387953+0.j        , -0.38268343+0.j        ], # may vary
           [ 0.        +0.38268343j,  0.        -0.92387953j]])

    >>> # demonstrate the treatment of the imaginary part of the diagonal
    >>> a = vp.array([[5+2j, 9-2j], [0+2j, 2-1j]])
    >>> a
    array([[5.+2.j, 9.-2.j],
           [0.+2.j, 2.-1.j]])
    >>> # with UPLO='L' this is numerically equivalent to using LA.eig() with:
    >>> b = vp.array([[5.+0.j, 0.-2.j], [0.+2.j, 2.-0.j]])
    >>> b
    array([[5.+0.j, 0.-2.j],
           [0.+2.j, 2.+0.j]])
    >>> wa, va = vp.linalg.eigh(a)
    >>> wb, vb = vp.linalg.eig(b)
    >>> wa; wb                      # doctest: +SKIP
    array([1., 6.])
    array([6.+0.j, 1.+0.j])
    >>> va; vb                      # doctest: +SKIP
    array([[-0.4472136 -0.j        , -0.89442719+0.j        ],
           [ 0.        +0.89442719j,  0.        -0.4472136j ]])
    array([[ 0.89442719+0.j       ,  0.        +0.4472136j],
           [-0.        +0.4472136j,  0.89442719+0.j       ]])

    """
    return _syevd(a, True, UPLO)


def eigvalsh(a, UPLO='L'):
    """Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

    Main difference from :func:`eigh` : the eigenvectors are not computed.

    Parameters
    ----------
    a : (..., M, M) array_like
        A complex- or real-valued matrix whose eigenvalues are to be computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular part of a
        ('L', default) or the upper triangular part ('U'). Irrespective of this value
        only the real parts of the diagonal will be considered in the computation to
        preserve the notion of a Hermitian matrix. It therefore follows that the
        imaginary part of the diagonal will always be treated as zero.

    Returns
    -------
    w : (..., M) ndarray
        The eigenvalues in ascending order, each repeated according to its multiplicity.

    Note
    ----
    The eigenvalues are computed using LAPACK routines ``_syevd``, ``_heevd``.

    See Also
    --------
    eig : Computes the eigenvalues and right eigenvectors of a square array.
    eigvals : Computes the eigenvalues of a general matrix.
    eigh : Computes the eigenvalues and eigenvectors of a complex Hermitian or a real
        symmetric matrix.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1, -2j], [2j, 5]])
    >>> vp.linalg.eigvalsh(a)   # doctest: +SKIP
    array([0.17157288, 5.82842712])  # may vary

    >>> # demonstrate the treatment of the imaginary part of the diagonal
    >>> a = vp.array([[5+2j, 9-2j], [0+2j, 2-1j]])
    >>> a    # doctest: +SKIP
    array([[5.+2.j, 9.-2.j],
           [0.+2.j, 2.-1.j]])
    >>> # with UPLO='L' this is numerically equivalent to using vp.linalg.eigvals()
    >>> # with:
    >>> b = vp.array([[5.+0.j, 0.-2.j], [0.+2.j, 2.-0.j]])
    >>> b  # doctest: +SKIP
    array([[5.+0.j, 0.-2.j],
           [0.+2.j, 2.+0.j]])
    >>> wa = vp.linalg.eigvalsh(a)
    >>> wb = vp.linalg.eigvals(b)
    >>> wa; wb   # doctest: +SKIP
    array([1., 6.])
    array([6.+0.j, 1.+0.j])

    """
    return _syevd(a, False, UPLO)

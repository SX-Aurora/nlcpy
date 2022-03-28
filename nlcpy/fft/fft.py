#
# * The source code in this file is based on the soure code of NumPy and CuPy.
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

from nlcpy.fft import _fft


def fft(a, n=None, axis=-1, norm=None):
    """Computes the one-dimensional discrete fourier transform.

    This function computes the one-dimensional n-point discrete fourier transform (DFT)
    with the efficient fast fourier transform (FFT) algorithm.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    n : int,optional
        Length of the transformed axis of the output. If *n* is smaller than the length
        of the input, the input is cropped. If it is larger, the input is padded with
        zeros. If *n* is not given, the length of the input along the axis specified by
        *axis*  is used.
    axis : int,optional
        Axis over which to compute the FFT. If not given, the last axis is used. If
        *axis* is larger than the last axis of a, *IndexError* occurs.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are unscaled. It *norm* is
        set to "ortho", the return values will be scaled by :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis indicated by
        *axis* , or the last one if axis is not specified.

    Note
    ----
    FFT (fast fourier transform) refers to a way the discrete fourier transform (DFT) can
    be calculated efficiently, by using symmetries in the calculated terms. The symmetry
    is highest when *n* is a power of 2, and the transform is therefore most efficient
    for these sizes.

    See Also
    --------
    ifft : Computes the one-dimensional inverse discrete fourier transform.
    fft2 : Computes the 2-dimensional discrete fourier transform.
    fftn : Computes the n-dimensional discrete fourier transform.
    rfftn : Computes the n-dimensional discrete fourier transform for a real array.
    fftfreq : Returns the discrete fourier transform sample frequencies.

    Examples
    --------
    .. plot::
        :align: center

        >>> import nlcpy as vp
        >>> vp.fft.fft(vp.exp(2j * vp.pi * vp.arange(8) / 8))    # doctest: +SKIP
        array([-3.44509285e-16+1.14423775e-17j,  8.00000000e+00-8.52069395e-16j,
                2.33486982e-16+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j,
                9.95799250e-17+2.33486982e-16j,  0.00000000e+00+1.17281316e-16j,
                1.14423775e-17+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j])

        In this example, real input has an FFT which is Hermitian, i.e., symmetric
        in the real part and anti-symmetric in the imaginary part:

        >>> import matplotlib.pyplot as plt
        >>> t = vp.arange(256)
        >>> sp = vp.fft.fft(vp.sin(t))
        >>> freq = vp.fft.fftfreq(t.shape[-1])
        >>> _ = plt.plot(freq, sp.real, freq, sp.imag)
        >>> plt.show()

    """
    return _fft.fft(a, n, axis, norm)


def ifft(a, n=None, axis=-1, norm=None):
    """Computes the one-dimensional inverse discrete fourier transform.

    This function computes the inverse of the one-dimensional n-point discrete fourier
    transform computed by :func:`fft`.
    In other words, ``ifft( fft(a) ) == a`` to within numerical accuracy.
    The input should be ordered in the same way as is returned by :func:`fft`, i.e.,

    - ``a[0]`` should contain the zero frequency term,
    - ``a[1:n//2]`` should contain the positive-frequency terms,
    - ``a[n//2 + 1:]`` should contain the negative-frequency terms, in increasing order
      starting from the most negative frequency.

    For an even number of input points, ``A[n//2]`` represents the sum of the values at
    the positive and negative Nyquist frequencies, as the two are aliased together.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    n : int,optional
        Length of the transformed axis of the output. If *n* is smaller than the length
        of the input, the input is cropped. If it is larger, the input is padded with
        zeros. If *n* is not given, the length of the input along the axis specified by
        *axis* is used. See notes about padding issues.
    axis : int,optional
        Axis over which to compute the inverse DFT. If not given, the last axis is used.
        If *axis* is larger than the last axis of *a*, *IndexError* occurs.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are scaled by :math:`1/n`.
        It *norm* is set to "ortho", the return values will be scaled by
        :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis indicated by axis,
        or the last one if axis is not specified.

    Note
    ----
    If the input parameter *n* is larger than the size of the input, the input is padded
    by appending zeros at the end. Even though this is the common approach, it might lead
    to surprising results. If a different padding is desired, it must be performed before
    calling ifft.

    See Also
    --------
    fft : Computes the one-dimensional discrete fourier transform.
    ifft2 : Computes the 2-dimensional inverse discrete fourier transform.
    ifftn : Computes the n-dimensional inverse discrete fourier transform.

    Examples
    --------
    .. plot::
        :align: center

        >>> import nlcpy as vp
        >>> vp.fft.ifft([0, 4, 0, 0])
        array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j])

        Create and plot a band-limited signal with random phases:

        >>> import matplotlib.pyplot as plt
        >>> t = vp.arange(400)
        >>> n = vp.zeros((400,), dtype=complex)
        >>> n[40:60] = vp.exp(1j*vp.random.uniform(0, 2*vp.pi, (20,)))
        >>> s = vp.fft.ifft(n)
        >>> _ = plt.plot(t, s.real, 'b-', t, s.imag, 'r--')
        >>> _ = plt.legend(('real', 'imaginary'))
        >>> plt.show()

    """
    return _fft.ifft(a, n, axis, norm)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    """Computes the 2-dimensional discrete fourier transform.

    This function computes the n-dimensional discrete fourier transform over any axes in
    an m-dimensional array by means of the fast fourier transform (FFT). By default, the
    transform is computed over the last two axes of the input array, i.e., a
    2-dimensional FFT.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output (``s[0]`` refers to axis 0,
        ``s[1]`` to axis 1, etc.). This corresponds to ``n`` for ``fft(x, n)``. Along
        each axis, if the given shape is smaller than that of the input, the input is
        cropped. If it is larger, the input is padded with zeros. if s is not given, the
        shape of the input along the axes specified by axes is used. If s and axes have
        different length, or axes not given and ``len(s) != 2``, *ValueError* occurs.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last two axes are used. A
        repeated index in axes means the transform over that axis is performed multiple
        times. A one-element sequence means that a one-dimensional FFT is performed. If
        an element of axes is larger than than the number of axes of *a*, *IndexError*
        occurs.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are unscaled. It *norm* is
        set to "ortho", the return values will be scaled by :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes indicated by axes,
        or the last two axes if axes is not given.

    Note
    ----
    fft2 is just :func:`fftn` with a different default for axes.

    The output, analogously to :func:`fft`, contains the term for zero frequency in the
    low-order corner of the transformed axes, the positive frequency terms in the first
    half of these axes, the term for the Nyquist frequency in the middle of the axes and
    the negative frequency terms in the second half of the axes, in order of decreasingly
    negative frequency.

    See :func:`fftn` for details and a plotting example.

    See Also
    --------
    ifft2 : Computes the 2-dimensional inverse discrete fourier transform.
    fft : Computes the one-dimensional discrete fourier transform.
    fftn : Computes the n-dimensional discrete fourier transform.
    fftshift : Shifts the zero-frequency component to the center of the spectrum.

    Examples
    --------
    >>> import nlcpy as vp
    >>> import numpy as np
    >>> a = np.mgrid[:5, :5][0]
    >>> vp.fft.fft2(a)   # doctest: +SKIP
    array([[ 50. +0.0000000000000000e+00j,   0. +0.0000000000000000e+00j, # may vary
              0. +0.0000000000000000e+00j,   0. +0.0000000000000000e+00j,
              0. +0.0000000000000000e+00j],
           [-12.5+1.7204774005889668e+01j,   0. +8.8817841970012523e-16j,
              0. +8.8817841970012523e-16j,   0. +8.8817841970012523e-16j,
              0. +8.8817841970012523e-16j],
           [-12.5+4.0614962029113286e+00j,   0. -2.2204460492503131e-16j,
              0. -2.2204460492503131e-16j,   0. -2.2204460492503131e-16j,
              0. -2.2204460492503131e-16j],
           [-12.5-4.0614962029113286e+00j,   0. +2.2204460492503131e-16j,
              0. +2.2204460492503131e-16j,   0. +2.2204460492503131e-16j,
              0. +2.2204460492503131e-16j],
           [-12.5-1.7204774005889668e+01j,   0. -8.8817841970012523e-16j,
              0. -8.8817841970012523e-16j,   0. -8.8817841970012523e-16j,
              0. -8.8817841970012523e-16j]])

    """
    return _fft.fft2(a, s, axes, norm)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    """Computes the 2-dimensional inverse discrete fourier transform.

    This function computes the inverse of the 2-dimensional discrete fourier transform
    over any number of axes in an m-dimensional array by means of the fast fourier
    transform (FFT). In other words, ``ifft2(fft2(a)) == a`` to within numerical
    accuracy. By default, the inverse transform is computed over the last two axes of the
    input array.

    The input, analogously to :func:`ifft`, should be ordered in the same way as is
    returned by :func:`fft2`, i.e. It should have the term for zero frequency in the
    low-order corner of the two axes, the positive frequency terms in the first half
    of these axes, the term for the Nyquist frequency in the middle of the axes and
    the negative frequency terms in the second half of both axes, in order of
    decreasingly negative frequency.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each axis) of the output (``s[0]`` refers to axis 0, ``s[1]`` to
        axis 1, etc.). This corresponds to n for ``ifft(x, n)``. Along each axis, if the
        given shape is smaller than that of the input, the input is cropped. If it is
        larger, the input is padded with zeros. If s is not given, the shape of the input
        along the axes specified by axes is used. See notes for issue on ifft zero
        padding. If s and axes have different length, or axes not given and ``len(s) !=
        2``, *ValueError* occurs.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last two axes are used. A
        repeated index in axes means the transform over that axis is performed multiple
        times. A one-element sequence means that a one-dimensional FFT is performed. If
        an element of axes is larger than than the number of axes of a, *IndexError*
        occurs.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are scaled by :math:`1/n`.
        It *norm* is set to "ortho", the return values will be scaled by
        :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes indicated by axes,
        or the last two axes if axes is not given.

    Note
    ----
    :func:`ifft2` is just :func:`ifftn` with a different default for axes.

    Zero-padding, analogously with :func:`ifft`, is performed by appending zeros to the
    input along the specified dimension. Although this is the common approach, it might
    lead to surprising results. If another form of zero padding is desired, it must be
    performed before :func:`ifft2` is called.

    See Also
    --------
    fft2 : Computes the 2-dimensional discrete fourier transform.
    ifftn : Computes the n-dimensional inverse discrete fourier transform.
    fft : Computes the one-dimensional discrete fourier transform.
    ifft : Computes the one-dimensional inverse discrete fourier transform.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = 4 * vp.eye(4)
    >>> vp.fft.ifft2(a)    # doctest: +SKIP
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],    # may vary
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])

    """
    return _fft.ifft2(a, s, axes, norm)


def fftn(a, s=None, axes=None, norm=None):
    """Computes the n-dimensional discrete fourier transform.

    This function computes the n-dimensional discrete fourier transform over any number
    of axes in an m-dimensional array by means of the fast fourier transform (FFT).

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output (``s[0]`` refers to axis 0,
        ``s[1]`` to axis 1, etc.). This corresponds to n for ``fft(x, n)``. Along any
        axis, if the given shape is smaller than that of the input, the input is cropped.
        If it is larger, the input is padded with zeros. if s is not given, the shape of
        the input along the axes specified by axes is used. If s and axes have different
        length, *ValueError* occurs.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)`` axes are
        used, or all axes if s is also not specified. Repeated indices in axes means that
        the transform over that axis is performed multiple times. If an element of axes
        is larger than than the number of axes of a, *IndexError* occurs.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are unscaled. It *norm* is
        set to "ortho", the return values will be scaled by :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes indicated by axes,
        or by a combination of s and a, as explained in the parameters section above.

    Note
    ----
    The output, analogously to :func:`fft`, contains the term for zero frequency in the
    low-order corner of all axes, the positive frequency terms in the first half of all
    axes, the term for the Nyquist frequency in the middle of all axes and the negative
    frequency terms in the second half of all axes, in order of decreasingly negative
    frequency.

    See Also
    --------
    ifftn : Computes the n-dimensional inverse discrete fourier transform.
    fft : Computes the one-dimensional discrete fourier transform.
    rfftn : Computes the n-dimensional discrete fourier transform for a real array.
    fft2 : Computes the 2-dimensional discrete fourier transform.
    fftshift : Shifts the zero-frequency component to the center of the spectrum.

    Examples
    --------
    .. plot::
        :align: center

        >>> import nlcpy as vp
        >>> import numpy as np
        >>> a = np.mgrid[:3, :3, :3][0]
        >>> vp.fft.fftn(a, axes=(1, 2))
        array([[[ 0.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j]],
        <BLANKLINE>
               [[ 9.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j]],
        <BLANKLINE>
               [[18.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j]]])
        >>> vp.fft.fftn(a, (2, 2), axes=(0, 1))
        array([[[ 2.+0.j,  2.+0.j,  2.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j]],
        <BLANKLINE>
               [[-2.+0.j, -2.+0.j, -2.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j]]])

        >>> import matplotlib.pyplot as plt
        >>> [X, Y] = np.meshgrid(2 * vp.pi * vp.arange(200) / 12,
        ...                      2 * vp.pi * vp.arange(200) / 34)
        >>> S = vp.sin(X) + vp.cos(Y) + vp.random.uniform(0, 1, X.shape)
        >>> FS = vp.fft.fftn(S)
        >>> _ = plt.imshow(vp.log(vp.absolute(vp.fft.fftshift(FS))**2))
        >>> plt.show()

    """
    return _fft.fftn(a, s, axes, norm)


def ifftn(a, s=None, axes=None, norm=None):
    """Computes the n-dimensional inverse discrete fourier transform.

    This function computes the inverse of the n-dimensional discrete fourier transform
    over any number of axes in an m-dimensional array by means of the fast fourier
    transform (FFT). In other words, ``ifftn(fftn(a)) == a`` to within numerical
    accuracy.

    The input, analogously to :func:`ifft`, should be ordered in the same way as is
    returned by :func:`fftn`, i.e. it should have the term for zero frequency in all
    axes in the low-order corner, the positive frequency terms in the first half of
    all axes, the term for the Nyquist frequency in the middle of all axes and the
    negative frequency terms in the second half of all axes, in order of decreasingly
    negative frequency.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output (``s[0]`` refers to axis 0,
        ``s[1]`` to axis 1, etc.). This corresponds to ``n`` for ``ifft(x, n)``. Along
        any axis, if the given shape is smaller than that of the input, the input is
        cropped. If it is larger, the input is padded with zeros. if s is not given, the
        shape of the input along the axes specified by axes is used. See notes for issue
        on ifft zero padding. If s and axes have different length, *ValueError* occurs.
    axes : sequence of ints, optional
        Axes over which to compute the IFFT. If not given, the last ``len(s)`` axes are
        used, or all axes if s is also not specified. Repeated indices in axes means that
        the inverse transform over that axis is performed multiple times. If an element
        of *axes* is larger than than the number of axes of a, *IndexError* occurs.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are scaled by :math:`1/n`.
        It *norm* is set to "ortho", the return values will be scaled by
        :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes indicated by axes,
        or by a combination of s and a, as explained in the parameters section above.

    Note
    ----
    Zero-padding, analogously with :func:`ifft`, is performed by appending zeros to the
    input along the specified dimension. Although this is the common approach, it might
    lead to surprising results. If another form of zero padding is desired, it must be
    performed before ifftn is called.

    See Also
    --------
    fftn : Computes the n-dimensional discrete fourier transform.
    ifft : Computes the one-dimensional inverse discrete fourier transform.
    ifft2 : Computes the 2-dimensional inverse discrete fourier transform.
    ifftshift : The inverse of fftshift.

    Examples
    --------
    .. plot::
        :align: center

        >>> import nlcpy as vp
        >>> a = vp.eye(4)
        >>> vp.fft.ifftn(vp.fft.fftn(a, axes=(0,)), axes=(1,))  # doctest: +SKIP
        array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],     # may vary
               [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])

        Create and plot an image with band-limited frequency content:

        >>> import matplotlib.pyplot as plt
        >>> n = vp.zeros((200,200), dtype=complex)
        >>> n[60:80, 20:40] = vp.exp(1j*vp.random.uniform(0, 2*vp.pi, (20, 20)))
        >>> im = vp.fft.ifftn(n).real
        >>> _ = plt.imshow(im)
        >>> plt.show()

    """
    return _fft.ifftn(a, s, axes, norm)


def rfft(a, n=None, axis=-1, norm=None):
    """Computes the one-dimensional discrete fourier transform for a real array.

    This function computes the one-dimensional *n*-point discrete fourier transform
    (DFT) of a real-valued array by means of an efficient algorithm called the fast
    fourier transform (FFT).

    Parameters
    ----------
    a : array_like
        Input array.
    n : int,optional
        Number of points along transformation axis in the input to use. If *n* is smaller
        than the length of the input, the input is cropped. If it is larger, the input is
        padded with zeros. If *n* is not given, the length of the input along the axis
        specified by axis is used.
    axis : int,optional
        Axis over which to compute the FFT. If not given, the last axis is used. If
        *axes* is larger than the last axis of *a*, *IndexError* occurs.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are unscaled. It *norm* is
        set to "ortho", the return values will be scaled by :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis indicated by axis,
        or the last one if axis is not specified. If n is even, the length of the
        transformed axis is ``(n/2)+1``. If *n* is odd, the length is ``(n+1)/2``.

    Note
    ----
    When the DFT is computed for purely a real array, the output is Hermitian-symmetric,
    i.e. the negative frequency terms are just the complex conjugates of the
    corresponding positive-frequency terms, and the negative-frequency terms are
    therefore redundant. This function does not compute the negative frequency terms, and
    the length of the transformed axis of the output is therefore ``n//2 + 1``.

    When ``A = rfft(a)`` and fs is the sampling frequency, ``A[0]`` contains the
    zero-frequency term 0*fs, which is real due to Hermitian symmetry.

    If *n* is even, ``A[-1]`` contains the term representing both positive and negative
    Nyquist frequency (+fs/2 and -fs/2), and must also be purely real. If *n* is odd,
    there is no term at fs/2; ``A[-1]`` contains the largest positive frequency
    (fs/2*(n-1)/n), and is complex in the general case.

    If the input a contains an imaginary part, it is silently discarded.

    See Also
    --------
    irfft : Computes the inverse of the n-point DFT for a real array.
    fft : Computes the one-dimensional discrete fourier transform.
    fftn : Computes the n-dimensional discrete fourier transform.
    rfftn : Computes the n-dimensional discrete fourier transform for a real array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.fft.fft([0, 1, 0, 0])   # doctest: +SKIP
    array([ 1.+0.j,  0.-1.j, -1.+0.j,  0.+1.j])  # may vary
    >>> vp.fft.rfft([0, 1, 0, 0])  # doctest: +SKIP
    array([ 1.+0.j,  0.-1.j, -1.+0.j])   # may vary

    Notice how the final element of the :func:`fft` output is the complex conjugate
    of the second element, for a real array.
    For rfft, this symmetry is exploited to compute only the non-negative frequency
    terms.

    """
    return _fft.rfft(a, n, axis, norm)


def irfft(a, n=None, axis=-1, norm=None):
    """Computes the inverse of the n-point DFT for a real array.

    This function computes the inverse of the one-dimensional *n*-point discrete fourier
    transform of a real array computed by :func:`rfft`. In other words,
    ``irfft( rfft(a),len(a)) == a`` to within numerical accuracy.
    (See Notes below for why ``len(a)`` is necessary here.)

    The input is expected to be in the form returned by :func:`rfft`, i.e. the real
    zero-frequency term followed by the complex positive frequency terms in order of
    increasing frequency. Since the discrete fourier transform of a real array is
    Hermitian-symmetric, the negative frequency terms are taken to be the complex
    conjugates of the corresponding positive frequency terms.

    Parameters
    ----------
    a : array_like
        The input array.
    n : int,optional
        Length of the transformed axis of the output. For *n* output points, ``n//2+1``
        input points are necessary. If the input is longer than this, it is cropped. If
        it is shorter than this, it is padded with zeros. If *n* is not given, it is
        taken to be ``2*(m-1)`` where ``m`` is the length of the input along the axis
        specified by axis.
    axis : int,optional
        Axis over which to compute the FFT. If not given, the last axis is used. If
        *axes* is larger than the last axis of *a*, *IndexError* occurs.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are scaled by :math:`1/n`.
        It *norm* is set to "ortho", the return values will be scaled by
        :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis indicated by axis,
        or the last one if axis is not specified. The length of the transformed axis is
        *n*, or, if *n* is not given, ``2*(m-1)`` where ``m`` is the length of the
        transformed axis of the input. To get an odd number of output points, *n* must be
        specified.

    Note
    ----
    Returns the real valued *n*-point inverse discrete fourier transform of *a*, where
    *a* contains the non-negative frequency terms of a Hermitian-symmetric sequence.
    *n* is the length of the result, not the input.

    If you specify an *n* such that a must be zero-padded or truncated, the extra/removed
    values will be added/removed at high frequencies.
    One can thus resample a series to *m* points via fourier interpolation by:
    ``a_resamp = irfft( rfft(a), m )``.

    The correct interpretation of the hermitian input depends on the length of the
    original data, as given by *n*. This is because each input shape could correspond to
    either an odd or even length signal. By default, irfft assumes an even output length
    which puts the last entry at the Nyquist frequency; aliasing with its symmetric
    counterpart. By Hermitian symmetry, the value is thus treated as purely real. To
    avoid losing information, the correct length of the real array must be given.

    See Also
    --------
    rfft : Computes the one-dimensional discrete fourier transform for a real array.
    fft : Computes the one-dimensional discrete fourier transform.
    irfft2 : Computes the 2-dimensional inverse FFT of a real array.
    irfftn : Computes the inverse of the n-dimensional FFT of a real array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.fft.ifft([1, -1j, -1, 1j])  # doctest: +SKIP
    array([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j])  # may vary
    >>> vp.fft.irfft([1, -1j, -1])
    array([0., 1., 0., 0.])

    Notice how the last term in the input to the ordinary :func:`ifft` is the complex
    conjugate of the second term, and the output has zero imaginary part everywhere.
    When calling irfft, the negative frequencies are not specified, and the output array
    is purely real.

    """
    return _fft.irfft(a, n, axis, norm)


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    """Computes the 2-dimensional FFT of a real array.

    Parameters
    ----------
    a : array
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape of the FFT.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are unscaled. It *norm* is
        set to "ortho", the return values will be scaled by :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : ndarray
        The result of the real 2-D FFT.

    Note
    ----
    This is really just :func:`rfftn` with different default behavior. For more details
    see :func:`rfftn`.

    See Also
    --------
    rfftn : Computes the n-dimensional discrete fourier transform for a real array.

    """
    return _fft.rfft2(a, s, axes, norm)


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    """Computes the 2-dimensional inverse FFT of a real array.

    Parameters
    ----------
    a : arrayi_like
        The input array
    s : sequence of ints, optional
        Shape of the real output to the inverse FFT.
    axes : sequence of ints, optional
        The axes over which to compute the inverse fft. Default is the last two axes.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are scaled by :math:`1/n`.
        It *norm* is set to "ortho", the return values will be scaled by
        :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : ndarray
        The result of the inverse real 2-D FFT.

    Note
    ----

    This is really :func:`irfftn` with different defaults. For more details see
    :func:`irfftn`.

    See Also
    --------
    irfftn : Computes the inverse of the n-dimensional FFT of a real array.

    """
    return _fft.irfft2(a, s, axes, norm)


def rfftn(a, s=None, axes=None, norm=None):
    """Computes the n-dimensional discrete fourier transform for a real array.

    This function computes the n-dimensional discrete fourier transform over any number
    of axes in an m-dimensional real array by means of the fast fourier transform (FFT).
    By default, all axes are transformed, with the real transform performed over the last
    axis, while the remaining transforms are complex.

    Parameters
    ----------
    a : array_like
        Input array, taken to be real.
    s : sequence of int, optional
        Shape (length along each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        The final element of *s* corresponds to *n* for ``rfft(x, n)``, while for the
        remaining axes, it corresponds to *n* for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input, the input
        is cropped. If it is larger, the input is padded with zeros. If *s* is not given,
        the shape of the input along the axes specified by axes is used.
        If *s* and *axes* have different length, *ValueError* occurs.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)`` axes are
        used, or all axes if *s* is also not specified. If an element of *axes* is larger
        than than the number of axes of *a*, *IndexError* occurs.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are unscaled. It *norm* is
        set to "ortho", the return values will be scaled by :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes indicated by
        *axes*, or by a combination of *s* and *a*, as explained in the parameters
        section above. The length of the last axis transformed will be ``s[-1]//2+1``,
        while the remaining transformed axes will have lengths according to *s*, or
        unchanged from the input.

    Note
    ----
    The transform for a real array is performed over the last transformation axis, as by
    :func:`rfft`, then the transform over the remaining axes is performed as by
    :func:`fftn`. The order of the output is as for :func:`rfft` for the final
    transformation axis, and as for :func:`fftn` for the remaining transformation axes.
    See :func:`fft` for details, definitions and conventions used.

    See Also
    --------
    irfftn : Computes the inverse of the n-dimensional FFT of a real array.
    fft : Computes the one-dimensional discrete fourier transform.
    rfft : Computes the one-dimensional discrete fourier transform for a real array.
    fftn : Computes the n-dimensional discrete fourier transform.
    rfft2 : Computes the 2-dimensional FFT of a real array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.ones((2, 2, 2))
    >>> vp.fft.rfftn(a)       # doctest: +SKIP
    array([[[8.+0.j, 0.+0.j],     # may vary
            [0.+0.j, 0.+0.j]],
    <BLANKLINE>
           [[0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j]]])
    >>> vp.fft.rfftn(a, axes=(2, 0))   # doctest: +SKIP
    array([[[4.+0.j, 0.+0.j],     # may vary
            [4.+0.j, 0.+0.j]],
    <BLANKLINE>
           [[0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j]]])

    """
    return _fft.rfftn(a, s, axes, norm)


def irfftn(a, s=None, axes=None, norm=None):
    """Computes the inverse of the n-dimensional FFT of a real array.

    This function computes the inverse of the n-dimensional discrete fourier transform
    for a real array over any number of axes in an m-dimensional array by means of the
    fast fourier transform (FFT). In other words, ``irfftn( rfftn(a), a.shape ) == a`` to
    within numerical accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for
    :func:`irfft`, and for the same reason.)

    The input should be ordered in the same way as is returned by :func:`rfftn`, i.e.
    as for :func:`irfft` for the final transformation axis, and as for
    :func:`ifftn` along all the other axes.

    Parameters
    ----------
    a : array_like
        Input array.
    s : sequence of int, optional
        Shape (length of each transformed axis) of the output (``s[0]`` refers to axis 0,
        ``s[1]`` to axis 1, etc.). *s* is also the number of input points used along this
        axis, except for the last axis, where ``s[-1]//2+1`` points of the input are
        used. Along any axis, if the shape indicated by *s* is smaller than that of the
        input, the input is cropped. If it is larger, the input is padded with zeros. If
        *s* is not given, the shape of the input along the axes specified by axes is
        used. Except for the last axis which is taken to be ``2*(m-1)`` where ``m`` is
        the length of the input along that axis. If *s* and *axes* have different length,
        *ValueError* occurs.
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. If not given, the last len(s) axes
        are used, or all axes if *s* is also not specified. Repeated indices in *axes*
        means that the inverse transform over that axis is performed multiple times. If
        an element of *axes* is larger than than the number of axes of *a*, *IndexError*
        occurs.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are scaled by :math:`1/n`.
        It *norm* is set to "ortho", the return values will be scaled by
        :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axes indicated by
        *axes*, or by a combination of *s* or *a*, as explained in the parameters section
        above. The length of each transformed axis is as given by the corresponding
        element of *s*, or the length of the input in every axis except for the last one
        if *s* is not given. In the final transformed axis the length of the output when
        *s* is not given is ``2*(m-1)`` where ``m`` is the length of the final
        transformed axis of the input. To get an odd number of output points in the final
        axis, *s* must be specified.

    Note
    ----
    See :func:`fft` for definitions and conventions used.

    See :func:`rfft` for definitions and conventions used for a real array.

    The correct interpretation of the hermitian input depends on the shape of the
    original data, as given by `s`. This is because each input shape could correspond to
    either an odd or even length signal. By default, irfftn assumes an even output length
    which puts the last entry at the Nyquist frequency; aliasing with its symmetric
    counterpart. When performing the final complex to real transform, the last value is
    thus treated as purely real. To avoid losing information, the correct shape of the
    real array **must** be given.

    See Also
    --------
    rfftn : Computes the n-dimensional discrete fourier transform for a real array.
    fft : Computes the one-dimensional discrete fourier transform.
    irfft : Computes the inverse of the n-point DFT for a real array.
    irfft2 : Computes the 2-dimensional inverse FFT of a real array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.zeros((3, 2, 2))
    >>> a[0, 0, 0] = 3 * 2 * 2
    >>> vp.fft.irfftn(a)
    array([[[1., 1.],
            [1., 1.]],
    <BLANKLINE>
           [[1., 1.],
            [1., 1.]],
    <BLANKLINE>
           [[1., 1.],
            [1., 1.]]])


    """
    return _fft.irfftn(a, s, axes, norm)


def hfft(a, n=None, axis=-1, norm=None):
    """Computes the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum.

    Parameters
    ----------
    a : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output. For *n* output points, ``n//2 + 1``
        input points are necessary. If the input is longer than this, it is cropped. If
        it is shorter than this, it is padded with zeros. If *n* is not given, it is
        taken to be ``2*(m-1)`` where ``m`` is the length of the input along the axis
        specified by *axis*.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is used. If
        *axis* is larger than the last axis of *a*, *IndexError* occurs.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are unscaled. It *norm* is
        set to "ortho", the return values will be scaled by :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis indicated by
        *axis*, or the last one if *axis* is not specified. The length of the transformed
        axis is *n*, or, if *n* is not given, ``2*m - 2`` where m is the length of the
        transformed axis of the input. To get an odd number of output points, *n* must be
        specified, for instance as ``2*m - 1`` in the typical case,

    Note
    ----
    :func:`hfft`/:func:`ihfft` are a pair analogous to :func:`rfft`/:func:`irfft`, but
    for the opposite case: here the signal has Hermitian symmetry in the time domain
    and is real in the frequency domain. So here it's :func:`hfft` for which you must
    supply the length of the result if it is to be odd.

    - even: ``ihfft( hfft(a, 2*len(a) - 2) ) == a``, within roundoff error,
    - odd: ``ihfft( hfft(a, 2*len(a) - 1) ) == a``, within roundoff error.

    The correct interpretation of the hermitian input depends on the length of the
    original data, as given by `n`. This is because each input shape could correspond to
    either an odd or even length signal. By default, hfft assumes an even output length
    which puts the last entry at the Nyquist frequency; aliasing with its symmetric
    counterpart. By Hermitian symmetry, the value is thus treated as purely real. To
    avoid losing information, the shape of the full signal **must** be given.

    See Also
    --------
    rfft : Computes the one-dimensional discrete fourier transform for a real array.
    ihfft : Computes the inverse FFT of a signal that has Hermitian symmetry.

    Examples
    --------
    >>> import nlcpy as vp
    >>> signal = vp.array([1, 2, 3, 4, 3, 2])
    >>> vp.fft.fft(signal)     # doctest: +SKIP
    array([15.+0.j, -4.+0.j,  0.+0.j, -1.+0.j,  0.+0.j, -4.+0.j])    # may vary
    >>> vp.fft.hfft(signal[:4]) # Input first half of signal
    array([15., -4.,  0., -1.,  0., -4.])
    >>> vp.fft.hfft(signal, 6)  # Input entire signal and truncate
    array([15., -4.,  0., -1.,  0., -4.])

    >>> signal = vp.array([[1, 1.j], [-1.j, 2]])
    >>> vp.conj(signal.T) - signal   # check Hermitian symmetry
    array([[ 0.-0.j, -0.+0.j],
           [ 0.+0.j,  0.-0.j]])
    >>> freq_spectrum = vp.fft.hfft(signal)
    >>> freq_spectrum
    array([[ 1.,  1.],
           [ 2., -2.]])

    """
    return _fft.hfft(a, n, axis, norm)


def ihfft(a, n=None, axis=-1, norm=None):
    """Computes the inverse FFT of a signal that has Hermitian symmetry.

    Parameters
    ----------
    a : array_like
        Input array.
    n : int, optional
        Length of the inverse FFT, the number of points along transformation axis in the
        input to use. If *n* is smaller than the length of the input, the input is
        cropped. If it is larger, the input is padded with zeros. If *n* is not given,
        the length of the input along the axis specified by *axis* is used.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last axis is used.
    norm : {None, "ortho"},optional
        Normalization mode. By default(None), the transforms are scaled by :math:`1/n`.
        It *norm* is set to "ortho", the return values will be scaled by
        :math:`1/\\sqrt{n}`.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis indicated by
        *axis*, or the last one if *axis* is not specified. The length of the transformed
        axis is ``n//2 + 1``.

    Note
    ----
    :func:`hfft`/:func:`ihfft` are a pair analogous to :func:`rfft`/:func:`irfft`, but
    for the opposite case: here the signal has Hermitian symmetry in the time domain and
    is real in the frequency domain. So here it's :func:`hfft` for which you must supply
    the length of the result if it is to be odd:

    - even: ``ihfft( hfft(a, 2*len(a) - 2) ) == a``, within roundoff error,
    - odd: ``ihfft( hfft(a, 2*len(a) - 1) ) == a``, within roundoff error.

    See Also
    --------
    hfft : Computes the FFT of a signal that has Hermitian symmetry, i.e., a real
        spectrum.
    irfft : Computes the inverse FFT of a signal that has Hermitian symmetry.

    Examples
    --------
    >>> import nlcpy as vp
    >>> spectrum = vp.array([ 15, -4, 0, -1, 0, -4])
    >>> vp.fft.ifft(spectrum)     # doctest: +SKIP
    array([1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j, 3.+0.j, 2.+0.j])   # may vary
    >>> vp.fft.ihfft(spectrum)    # doctest: +SKIP
    array([1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j])    # may vary

    """
    return _fft.ihfft(a, n, axis, norm)


def fftfreq(n, d=1.0):
    """Returns the Discrete fourier transform sample frequencies.

    The returned float array *f* contains the frequency bin centers in cycles per unit of
    the sample spacing (with zero at the start). For instance, if the sample spacing is
    in seconds, then the frequency unit is cycles/second. Given a window length *n* and a
    sample spacing *d*::

        f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
        f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : ndarray
        Array of length *n* containing the sample frequencies.

    Examples
    --------
    >>> import nlcpy as vp
    >>> signal = vp.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> fourier = vp.fft.fft(signal)
    >>> n = signal.size
    >>> timestep = 0.1
    >>> freq = vp.fft.fftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  2.5 ,  3.75, -5.  , -3.75, -2.5 , -1.25])

    """
    return _fft.fftfreq(n, d)


def rfftfreq(n, d=1.0):
    """Returns the Discrete fourier transform sample frequencies (for usage with
    :func:`rfft`, :func:`irfft`).

    The returned float array *f* contains the frequency bin centers in cycles per unit of
    the sample spacing (with zero at the start). For instance, if the sample spacing is
    in seconds, then the frequency unit is cycles/second. Given a window length *n* and a
    sample spacing *d*::

        f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
        f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Unlike :func:`fftfreq` the Nyquist frequency component is considered to be positive.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : ndarray
        Array of length ``n//2 + 1`` containing the sample frequencies.

    Examples
    --------
    >>> import nlcpy as vp
    >>> signal = vp.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    >>> fourier = vp.fft.rfft(signal)
    >>> n = signal.size
    >>> sample_rate = 100
    >>> freq = vp.fft.fftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40., -50., -40., -30., -20., -10.])
    >>> freq = vp.fft.rfftfreq(n, d=1./sample_rate)
    >>> freq
    array([ 0., 10., 20., 30., 40., 50.])

    """
    return _fft.rfftfreq(n, d)


def fftshift(x, axes=None):
    """Shifts the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all). Note that
    ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to shift. Default is None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    See Also
    --------
    ifftshift : The inverse of fftshift.

    Examples
    --------
    >>> import nlcpy as vp
    >>> freqs = vp.fft.fftfreq(10, 0.1)
    >>> freqs
    array([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
    >>> vp.fft.fftshift(freqs)
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

    Shift the zero-frequency component only along the second axis:

    >>> freqs = vp.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> vp.fft.fftshift(freqs, axes=(1,))
    array([[-3., -2., -1.],
           [ 0.,  1.,  2.],
           [ 3.,  4., -4.]])
    """
    return _fft.fftshift(x, axes)


def ifftshift(x, axes=None):
    """The inverse of fftshift. Although identical for even-length *x*, the functions
    differ by one sample for odd-length *x*.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to calculate. Defaults to None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    See Also
    --------
    fftshift : Shifts the zero-frequency component to the center of the spectrum.

    Examples
    --------
    >>> import nlcpy as vp
    >>> freqs = vp.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> vp.fft.ifftshift(vp.fft.fftshift(freqs))
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])

    """
    return _fft.ifftshift(x, axes)

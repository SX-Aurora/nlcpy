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
# # CuPy License #
#
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#     THE SOFTWARE.
#
import numpy

import operator
import nlcpy
from nlcpy.request import request


# ----------------------------------------------------------------------------
# create arrays from numerical ranges
# see: https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html
# ----------------------------------------------------------------------------

# @name Numerical ranges
# @{

# TODO: check complex case

def arange(start, stop=None, step=1, dtype=None):
    """Returns evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)`` (in other words,
    the interval including *start* but excluding *stop*). If stop is None, values are
    ganerated within ``[0, start)``. For integer arguments the function is equivalent to
    the Python built-in *range* function, but returns an ndarray rather than a list.
    When using a non-integer step, such as 0.1, the results will often not be consistent.
    It is better to use :func:`linspace` for these cases.

    Parameters
    ----------
    start : number
        Start of interval. The interval includes this value.
    stop : number, optional
        End of interval. The interval does not include this value, except in some cases
        where step is not an integer and floating point round-off affects the length of
        *out*.
    step : number, optional
        Spacing between values. For any output *out*, this is the distance between two
        adjacent values, ``out[i+1] - out[i]``. The default step size is 1. If *step* is
        specified as a position argument, *start* must also be given.
    dtype : dtype, optional
        The type of the output array. If *dtype* is not given, infer the data type from
        the other input arguments.

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.
        For floating point arguments, the length of the result is ``ceil((stop -
        start)/step)``. Because of floating point overflow, this rule may result in the
        last element of *out* being greater than *stop*.

    See Also
    --------
    linspace : Returns evenly spaced numbers over a specified interval.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.arange(3)
    array([0, 1, 2])
    >>> vp.arange(3.0)
    array([0., 1., 2.])
    >>> vp.arange(3,7)
    array([3, 4, 5, 6])
    >>> vp.arange(3,7,2)
    array([3, 5])

    """
    if dtype is None:
        if any(numpy.dtype(type(val)).kind == 'f'
                for val in (start, stop, step)):
            dtype = float
        else:
            dtype = int

    if stop is None:
        stop = start
        start = 0

    if step is None:
        step = 1

    size = int(numpy.ceil((stop - start) / step))
    # size = int(numpy.ceil(numpy.ceil(stop - start) / step))
    if size <= 0:
        return nlcpy.empty((0,), dtype=dtype)

    if numpy.dtype(dtype).type == numpy.bool_:
        if size > 2:
            raise ValueError('no fill-function for data-type.')
        if size == 2:
            return nlcpy.array([start, start - step], dtype=numpy.bool_)
        else:
            return nlcpy.array([start], dtype=numpy.bool_)

    ret = nlcpy.empty((size,), dtype=dtype)
    if numpy.dtype(dtype).kind == 'f':
        typ = numpy.dtype('f8').type
    elif numpy.dtype(dtype).kind == 'c':
        typ = numpy.dtype('c16').type
    elif numpy.dtype(dtype).kind == 'u':
        typ = numpy.dtype('u8').type
    elif numpy.dtype(dtype).kind == 'i':
        typ = numpy.dtype('i8').type
    elif numpy.dtype(dtype).kind == 'b':
        typ = numpy.dtype('bool').type
    else:
        raise TypeError('detected invalid dtype.')

    request._push_request(
        "nlcpy_arange",
        "creation_op",
        (typ(start), typ(step), ret),)

    return ret


# ----------------------------------------------------------------------------
# Return evenly spaced numbers over a specified interval.
# see: https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html
# ----------------------------------------------------------------------------
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """Returns evenly spaced numbers over a specified interval.

    Returns *num* evenly spaced samples, calculated over the interval ``[start, stop]``.
    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The end value of the sequence, unless *endpoint* is set to False. In that case,
        the sequence consists of all but the last of ``num + 1`` evenly spaced samples,
        so that *stop* is excluded. Note that the step size changes when *endpoint* is
        False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, *stop* is the last sample. Otherwise, it is not included. Default is
        True.
    retstep : bool, optional
        If True, return (*samples*, *step*) where *step* is the spacing between samples.
    dtype : dtype, optional
        The type of the output array. If *dtype* is not given, infer the data type from
        the other input arguments.
    axis : int, optional
        The axis in the result to store the samples. Relevant only if start or stop are
        array-like. By default (0), the samples will be along a new axis inserted at the
        beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : ndarray
        There are *num* equally spaced samples in the closed interval ``[start, stop]``
        or the half-open interval ``[start, stop)`` (depending on whether *endpoint* is
        True or False).
    step : float, optional
        Only returned if *retstep* is True
        Size of spacing between samples.

    See Also
    --------
    arange : Returns evenly spaced values within a given interval.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.linspace(2.0, 3.0, num=5)
    array([2.  , 2.25, 2.5 , 2.75, 3.  ])
    >>> vp.linspace(2.0, 3.0, num=5, endpoint=False)
    array([2. , 2.2, 2.4, 2.6, 2.8])
    >>> vp.linspace(2.0, 3.0, num=5, retstep=True)
    (array([2.  , 2.25, 2.5 , 2.75, 3.  ]), array([0.25]))

    """
    num = operator.index(num)
    if num < 0:
        raise ValueError("Number of samples, %s, must be non-negative." % num)

    dtype_kind = numpy.dtype(dtype).kind
    if dtype_kind == 'V':
        raise NotImplementedError('void dtype in linspace is not implemented yet.')

    start = nlcpy.asarray(start)
    stop = nlcpy.asarray(stop)
    dt = numpy.result_type(start, stop, float(num))
    if start.dtype.char in '?iIlL' or stop.dtype.char in '?iIlL':
        dt = 'D' if dt.char in 'FD' else 'd'

    if dtype is None:
        dtype = dt

    start = nlcpy.asarray(start, dtype=dt)
    stop = nlcpy.asarray(stop, dtype=dt)
    delta = stop - start
    div = (num - 1) if endpoint else num
    if num == 0:
        ret = nlcpy.empty((num,) + delta.shape, dtype=dtype)
        if retstep:
            ret = (ret, nlcpy.NaN)
        return ret
    elif div == 0 or num == 1:
        ret = nlcpy.resize(start, (1,) + delta.shape).astype(dtype)
        if retstep:
            ret = (ret, stop)
        return ret
    else:
        ret = nlcpy.empty((num,) + delta.shape, dtype=dtype)

    delta = delta[nlcpy.newaxis]
    start = nlcpy.array(nlcpy.broadcast_to(start, delta.shape))
    stop = nlcpy.array(nlcpy.broadcast_to(stop, delta.shape))
    step = delta / div if div > 1 else delta
    denormal = nlcpy.zeros(1, dtype='l')
    request._push_request(
        "nlcpy_linspace",
        "creation_op",
        (ret, start, stop, delta, step, int(endpoint), denormal))
    if axis != 0:
        ret = nlcpy.moveaxis(ret, 0, axis)
    if retstep:
        ret = (ret, step)

    return ret


def meshgrid(*xi, **kwargs):
    """Returns coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields
    over N-D grids, given one-dimensional coordinate arrays x1, x2,..., xn.

    Parameters
    ----------
    x1, x2, ..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output. See Notes for
        more details.
    sparse : bool, optional
        If True a sparse grid is returned in order to conserve memory. Default is
        False.
    copy : bool, optional
        If False, a view into the original arrays are returned in order to conserve
        memory. Default is True. Please note that ``sparse=False, copy=False`` will
        likely return non-contiguous arrays. Furthermore, more than one element of a
        broadcast array may refer to a single memory location. If you need to write
        to the arrays, make copies first.

    Returns
    -------
    X1, X2, ..., XN : ndarray
        For vectors x1, x2, ..., 'xn' with lengths ``Ni=len(xi)``, return
        ``(N1, N2, N3, ..., Nn)`` shaped arrays if indexing='ij' or (N2, N1, N3, ...,
        Nn) shaped arrays if indexing='xy' with the elements of xi repeated to fill the
        matrix along the first dimension for x1, the second for x2 and so on.

    Note
    ----
    This function supports both indexing conventions through the indexing keyword
    argument. Giving the string 'ij' returns a meshgrid with matrix indexing, while
    'xy' returns a meshgrid with Cartesian indexing. In the 2-D case with inputs of
    length M and N, the outputs are of shape (N, M) for 'xy' indexing and (M, N) for
    'ij' indexing. In the 3-D case with inputs of length M, N and P, outputs are of
    shape (N, M, P) for 'xy' indexing and (M, N, P) for 'ij' indexing.
    The difference is illustrated by the following code snippet::

        import nlcpy as vp
        xv, yv = vp.meshgrid(x, y, sparse=False, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]
        xv, yv = vp.meshgrid(x, y, sparse=False, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    In the 1-D and 0-D case, the indexing and sparse keywords have no effect.

    Examples
    --------
    >>> import nlcpy as vp
    >>> nx, ny = (3, 2)
    >>> x = vp.linspace(0, 1, nx)
    >>> y = vp.linspace(0, 1, ny)
    >>> xv, yv = vp.meshgrid(x, y)
    >>> xv
    array([[0. , 0.5, 1. ],
           [0. , 0.5, 1. ]])
    >>> yv
    array([[0., 0., 0.],
           [1., 1., 1.]])
    >>> xv, yv = vp.meshgrid(x, y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[0. , 0.5, 1. ]])
    >>> yv
    array([[0.],
           [1.]])

    meshgrid is very useful to evaluate functions on a grid.

    >>> import matplotlib.pyplot as plt
    >>> x = vp.arange(-5, 5, 0.1)
    >>> y = vp.arange(-5, 5, 0.1)
    >>> xx, yy = vp.meshgrid(x, y, sparse=True)
    >>> z = vp.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    >>> h = plt.contourf(x,y,z)
    >>> plt.show()

    """
    copy_ = kwargs.pop('copy', True)
    sparse = kwargs.pop('sparse', False)
    indexing = kwargs.pop('indexing', 'xy')

    if kwargs:
        raise TypeError("meshgrid() got an unexpected keyword argument '%s'"
                        % (list(kwargs)[0],))

    if indexing not in ['xy', 'ij']:
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'.")

    ndim = len(xi)
    s0 = (1,) * ndim
    output = [nlcpy.asanyarray(x).reshape(s0[:i] + (-1, ) + s0[i + 1:])
              for i, x in enumerate(xi)]

    if not sparse:
        shape = [output[i].shape[i] for i in range(len(output))]

    if indexing == 'xy' and ndim > 1:
        output[0].shape = (1, -1) + s0[2:]
        output[1].shape = (-1, 1) + s0[2:]
        if not sparse:
            shape[0], shape[1] = shape[1], shape[0]

    if not sparse:
        output = [nlcpy.broadcast_to(x, shape) for x in output]

    if copy_:
        output = [x.copy() for x in output]

    return output


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """Returns numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start`` (*base* to the power
    of *start*) and ends with ``base ** stop`` (see *endpoint* below).

    Parameters
    ----------
    start : array_like
        ``base ** start`` is the starting value of the sequence.
    stop : array_like
        ``base ** stop`` is the final value of the sequence, unless endpoint is *False*.
        In that case, ``num + 1`` values are spaced over the interval in log-space, of
        which all but the last (a sequence of length *num*) are returned.
    num : int, optional
        Number of samples to generate. Default is 50.
    endpoint : bool, optional
        If true, *stop* is the last sample. Otherwise, it is not included. Default is
        True.
    base : float, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.
    dtype : dtype, optional
        The type of the output array. If dtype is not given, infer the data type from
        the other input arguments.
    axis : int, optional
        The axis in the result to store the samples. Relevant only if start or stop
        are array-like. By default (0), the samples will be along a new axis inserted
        at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : ndarray
        *num* samples, equally spaced on a log scale.

    Note
    ----
    Logspace is equivalent to the code

    ::

        >>> import nlcpy as vp
        >>> y = vp.linspace(start, stop, num=num, endpoint=endpoint)
        ...  # doctest: +SKIP
        >>> vp.power(base, y).astype(dtype)
        ...  # doctest: +SKIP

    See Also
    --------
    arange : Returns evenly spaced values within a given interval.
    linspace : Returns evenly spaced numbers over a specified interval.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.logspace(2.0, 3.0, num=4)  # doctest: +SKIP
    array([ 100.        ,  215.443469  ,  464.15888336, 1000.        ])
    >>> vp.logspace(2.0, 3.0, num=4, endpoint=False) # doctest: +SKIP
    array([100.        , 177.827941  , 316.22776602, 562.34132519])
    >>> vp.logspace(2.0, 3.0, num=4, base=2.0) # doctest: +SKIP
    array([4.        , 5.0396842 , 6.34960421, 8.        ])

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 10
    >>> x1 = vp.logspace(0.1, 1, N, endpoint=True)
    >>> x2 = vp.logspace(0.1, 1, N, endpoint=False)
    >>> y = vp.zeros(N)
    >>> plt.plot(x1, y, 'o') # doctest: +SKIP
    >>> plt.plot(x2, y + 0.5, 'o') # doctest: +SKIP
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1.0)
    >>> plt.show()

    """
    y = linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    ret = nlcpy.power(base, y)
    if dtype is None:
        return ret
    else:
        return ret.astype(dtype, copy=False)

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

import nlcpy
from nlcpy.core.core import on_VE, on_VH, on_VE_VH
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

    Values are generated within the half-open interval [start, stop) (in other words, the
    interval including start but excluding stop). If stop is None, values are ganerated
    within [0, start). For integer arguments the function is equivalent to the Python
    built-in range function, but returns an ndarray rather than a list.
    When using a non-integer step, such as 0.1, the results will often not be consistent.
    It is better to use `nlcpy.linspace` for these cases.

    Args:
        start : number
            Start of interval. The interval includes this value.
        stop : number, optional
            End of interval. The interval does not include this value, except in some
            cases where step is not an integer and floating point round-off affects the
            length of out.
        step : number, optional
            Spacing between values. For any output out, this is the distance between two
            adjacent values, out[i+1] - out[i]. The default step size is 1. If step is
            specified as a position argument, start must also be given.
        dtype : dtype, optional
            The type of the output array. If dtype is not given, infer the data type from
            the other input arguments.

    Returns:
        arange : `ndarray`
            Array of evenly spaced values.
            For floating point arguments, the length of the result is ceil((stop -
            start)/step). Because of floating point overflow, this rule may result in the
            last element of out being greater than stop.

    See Also:
        linspace : Returns evenly spaced numbers over a specified interval.

    Examples:
        >>> import nlcpy as vp
        >>> vp.arange(3)
        array([0, 1, 2])
        >>> vp.arange(3.0)
        array([ 0.,  1.,  2.])
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

    if ret._memloc in {on_VE, on_VE_VH}:
        request._push_request(
            "nlcpy_arange",
            "creation_op",
            (typ(start), typ(step), ret),)

    if ret._memloc in {on_VH, on_VE_VH}:
        del ret.vh_data
        ret.vh_data = numpy.arange(typ(start), typ(stop), typ(step),
                                   dtype=ret.dtype)

    return ret


# ----------------------------------------------------------------------------
# Return evenly spaced numbers over a specified interval.
# see: https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html
# ----------------------------------------------------------------------------
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """Returns evenly spaced numbers over a specified interval.

    Returns num evenly spaced samples, calculated over the interval [start, stop].
    The endpoint of the interval can optionally be excluded.

    Args:
        start : array_like
            The starting value of the sequence.
        stop : array_like
            The end value of the sequence, unless endpoint is set to False. In that case,
            the sequence consists of all but the last of num + 1 evenly spaced samples,
            so that stop is excluded. Note that the step size changes when endpoint is
            False.
        num : int, optional
            Number of samples to generate. Default is 50. Must be non-negative.
        endpoint : bool, optional
            If True, stop is the last sample. Otherwise, it is not included. Default is
            True.
        retstep : bool, optional
            If True, return (samples, step), where step is the spacing between samples.
        dtype : dtype, optional
            The type of the output array. If dtype is not given, infer the data type from
            the other input arguments.
        axis : int, optional
            The axis in the result to store the samples. Relevant only if start or stop
            are array-like. By default (0), the samples will be along a new axis inserted
            at the beginning. Use -1 to get an axis at the end.

    Returns:
        samples : `ndarray`
            There are num equally spaced samples in the closed interval [start, stop] or
            the half-open interval [start, stop) (depending on whether endpoint is True
            or False).
        step : float, optional
            Only returned if retstep is True
            Size of spacing between samples.

    See Also:
        arange : Returns evenly spaced values within a given interval.

    Examples:
        >>> import nlcpy as vp
        >>> vp.linspace(2.0, 3.0, num=5)
        array([2.  , 2.25, 2.5 , 2.75, 3.  ])
        >>> vp.linspace(2.0, 3.0, num=5, endpoint=False)
        array([2. ,  2.2,  2.4,  2.6,  2.8])
        >>> vp.linspace(2.0, 3.0, num=5, retstep=True)
        (array([2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

    """
    if num < 0:
        raise ValueError("Number of samples, %s, must be non-negative." % num)

    if dtype is None:
        if isinstance(start, complex) or isinstance(stop, complex):
            start = nlcpy.asanyarray(start) * (1.0 + 0.0j)
            stop = nlcpy.asanyarray(stop) * (1.0 + 0.0j)
        else:
            start = nlcpy.asanyarray(start) * 1.0
            stop = nlcpy.asanyarray(stop) * 1.0
    else:
        if numpy.dtype(dtype).kind == 'V':
            raise NotImplementedError('void dtype in linspace is not implemented yet.')
        if numpy.dtype(dtype).kind == 'c':
            start = nlcpy.asanyarray(start, dtype='complex128')
            stop = nlcpy.asanyarray(stop, dtype='complex128')
        else:
            start = nlcpy.asanyarray(start, dtype='float64')
            stop = nlcpy.asanyarray(stop, dtype='float64')

    div = (num - 1) if endpoint else num
    step = (stop - start) / div if div > 1 else stop - start

    dt = numpy.result_type(start, stop, float(num))
    if dtype is None:
        dtype = dt
    if numpy.dtype(dtype).kind == 'V':
        raise NotImplementedError('void dtype in linspace is not implemented yet.')

    if num == 0:
        ret = nlcpy.empty((num,) + step.shape, dtype=dtype)
        # TODO: replace with nlcpy.numeric.NaN
        if retstep:
            ret = (ret, float('nan'))
        return ret
    elif num == 1:
        # TODO: replace with nlcpy.numeric.NaN
        ret = nlcpy.resize(start, (1,) + step.shape)
        ret = nlcpy.array(ret, dtype=dtype, copy=False)
        if retstep:
            ret = (ret, float('nan'))
        return ret
    else:
        ret = nlcpy.empty((num,) + step.shape, dtype=dtype)

    retdata = ret

    if retdata._memloc in {on_VE, on_VE_VH}:
        request._push_request(
            "nlcpy_linspace",
            "creation_op",
            (ret, start, stop,
             int(num), int(endpoint), int(axis)))
        if retstep:
            if ret.ndim > 1:
                ret = (ret, step)
            else:
                step = step[0].get().item() if step.size > 1 else step.get().item()
                ret = (ret, step)
        if axis != 0:
            raise NotImplementedError('moveaxis is not implemented yet.')

    if retdata._memloc in {on_VH, on_VE_VH}:
        del retdata.vh_data
        del step.vh_data
        typ = numpy.dtype(dtype).type
        if retstep:
            (retdata.vh_data, step.vh_data) = numpy.linspace(typ(start),
                                                             typ(stop), num, endpoint,
                                                             typ(retstep), dtype, axis)
        else:
            retdata.vh_data = numpy.linspace(typ(start),
                                             typ(stop), num, endpoint,
                                             typ(retstep), dtype, axis)
    return ret

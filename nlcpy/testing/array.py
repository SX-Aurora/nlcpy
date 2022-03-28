#
# * The source code in this file is based on the soure code of CuPy.
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

import numpy.testing

from nlcpy import testing

# NumPy-like assertion functions that accept both NumPy and nlcpy arrays


def assert_allclose(actual, desired, rtol=1e-7, atol=0, err_msg='',
                    verbose=True):
    """Raises an AssertionError if objects are not equal up to desired tolerance.

    Args:
         actual(numpy.ndarray or nlcpy.ndarray): The actual object to check.
         desired(numpy.ndarray or nlcpy.ndarray): The desired, expected object.
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting
             values are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_allclose`

    """  # NOQA
    numpy.testing.assert_allclose(
        testing.asnumpy(actual), testing.asnumpy(desired),
        rtol=rtol, atol=atol, err_msg=err_msg, verbose=verbose)


def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    """Raises an AssertionError if objects are not equal up to desired precision.

    Args:
         x(numpy.ndarray or nlcpy.ndarray): The actual object to check.
         y(numpy.ndarray or nlcpy.ndarray): The desired, expected object.
         decimal(int): Desired precision.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting
             values are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_array_almost_equal`
    """  # NOQA
    numpy.testing.assert_array_almost_equal(
        testing.asnumpy(x), testing.asnumpy(y), decimal=decimal,
        err_msg=err_msg, verbose=verbose)


def assert_array_almost_equal_nulp(x, y, nulp=1):
    """Compare two arrays relatively to their spacing.

    Args:
         x(numpy.ndarray or nlcpy.ndarray): The actual object to check.
         y(numpy.ndarray or nlcpy.ndarray): The desired, expected object.
         nulp(int): The maximum number of unit in the last place for tolerance.

    .. seealso:: :func:`numpy.testing.assert_array_almost_equal_nulp`
    """
    numpy.testing.assert_array_almost_equal_nulp(
        testing.asnumpy(x), testing.asnumpy(y), nulp=nulp)


def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    """Check that all items of arrays differ in at most N Units in the Last Place.

    Args:
         a(numpy.ndarray or nlcpy.ndarray): The actual object to check.
         b(numpy.ndarray or nlcpy.ndarray): The desired, expected object.
         maxulp(int): The maximum number of units in the last place
             that elements of ``a`` and ``b`` can differ.
         dtype(numpy.dtype): Data-type to convert ``a`` and ``b`` to if given.

    .. seealso:: :func:`numpy.testing.assert_array_max_ulp`
    """  # NOQA
    numpy.testing.assert_array_max_ulp(
        testing.asnumpy(a), testing.asnumpy(b), maxulp=maxulp, dtype=dtype)


def assert_array_equal(x, y, err_msg='', verbose=True, strides_check=False):
    """Raises an AssertionError if two array_like objects are not equal.

    Args:
         x(numpy.ndarray or nlcpy.ndarray): The actual object to check.
         y(numpy.ndarray or nlcpy.ndarray): The desired, expected object.
         strides_check(bool): If ``True``, consistency of strides is also
             checked.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_array_equal`
    """
    numpy.testing.assert_array_equal(
        testing.asnumpy(x), testing.asnumpy(y), err_msg=err_msg,
        verbose=verbose)

    if strides_check:
        x_strides = x.strides
        y_strides = y.strides
        if x.dtype == numpy.dtype('bool'):
            y_strides = numpy.empty_like(y, dtype='i4').strides
        if x_strides != y_strides:
            msg = ['Strides are not equal:']
            if err_msg:
                msg = [msg[0] + ' ' + err_msg]
            if verbose:
                msg.append(' x: {}'.format(x_strides))
                msg.append(' y: {}'.format(y_strides))
            raise AssertionError('\n'.join(msg))


def assert_array_list_equal(xlist, ylist, err_msg='', verbose=True):
    """Compares lists of arrays pairwise with ``assert_array_equal``.

    Args:
         x(array_like): Array of the actual objects.
         y(array_like): Array of the desired, expected objects.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.

    Each element of ``x`` and ``y`` must be either :class:`numpy.ndarray`
    or :class:`nlcpy.ndarray`. ``x`` and ``y`` must have same length.
    Otherwise, this function raises ``AssertionError``.
    It compares elements of ``x`` and ``y`` pairwise
    with :func:`assert_array_equal` and raises error if at least one
    pair is not equal.

    .. seealso:: :func:`numpy.testing.assert_array_equal`
    """
    x_type = type(xlist)
    y_type = type(ylist)
    if x_type is not y_type:
        raise AssertionError(
            'Matching types of list or tuple are expected, '
            'but were different types '
            '(xlist:{} ylist:{})'.format(x_type, y_type))
    if x_type not in (list, tuple):
        raise AssertionError(
            'List or tuple is expected, but was {}'.format(x_type))
    if len(xlist) != len(ylist):
        raise AssertionError('List size is different')
    for x, y in zip(xlist, ylist):
        numpy.testing.assert_array_equal(
            testing.asnumpy(x), testing.asnumpy(y), err_msg=err_msg,
            verbose=verbose)


def assert_array_less(x, y, err_msg='', verbose=True):
    """Raises an AssertionError if array_like objects are not ordered by less than.

    Args:
         x(numpy.ndarray or nlcpy.ndarray): The smaller object to check.
         y(numpy.ndarray or nlcpy.ndarray): The larger object to compare.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_array_less`
    """  # NOQA
    numpy.testing.assert_array_less(
        testing.asnumpy(x), testing.asnumpy(y), err_msg=err_msg,
        verbose=verbose)

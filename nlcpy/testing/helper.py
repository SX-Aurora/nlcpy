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

from __future__ import absolute_import
from __future__ import print_function

import contextlib
import functools
import inspect
import os
import pkg_resources
import random
import sys
import traceback
import unittest
import warnings

import numpy
import re
import importlib
from distutils.version import StrictVersion

import nlcpy
from nlcpy import ndarray
from nlcpy.core import internal
from nlcpy.testing import array
from nlcpy.testing import parameterized
from nlcpy.testing import ufunc


def _call_func(self, impl, args, kw):
    try:
        result = impl(self, *args, **kw)
        assert result is not None
        error = None
        msg = None
        tb_str = None
    except Exception as e:
        _, _, tb = sys.exc_info()  # e.__traceback__ is py3 only
        if tb.tb_next is None:
            # failed before impl is called, e.g. invalid kw
            raise e
        result = None
        error = e
        msg = str(e)
        tb_str = traceback.format_exc()

    return result, error, msg, tb_str


def _get_numpy_errors():
    numpy_version = numpy.lib.NumpyVersion(numpy.__version__)

    errors = [
        AttributeError, Exception, IndexError, TypeError, ValueError,
        NotImplementedError, DeprecationWarning,
    ]
    if numpy_version >= '1.13.0':
        errors.append(numpy.AxisError)
    if numpy_version >= '1.15.0':
        errors.append(numpy.linalg.LinAlgError)

    return errors


_numpy_errors = _get_numpy_errors()


def _check_numpy_nlcpy_error_compatible(nlcpy_error, numpy_error):
    """Checks if try/except blocks are equivalent up to public error classes
    """

    errors = _numpy_errors

    # Prior to NumPy version 1.13.0, NumPy raises either `ValueError` or
    # `IndexError` instead of `numpy.AxisError`.
    numpy_axis_error = getattr(numpy, 'AxisError', None)
    nlcpy_axis_error = nlcpy.core.error._AxisError
    if isinstance(nlcpy_error, nlcpy_axis_error) and numpy_axis_error is None:
        if not isinstance(numpy_error, (ValueError, IndexError)):
            return False
        errors = list(set(errors) - set([IndexError, ValueError]))

    return all([isinstance(nlcpy_error, err) == isinstance(numpy_error, err)
                for err in errors])


def _check_nlcpy_numpy_error(self, nlcpy_error, nlcpy_msg, nlcpy_tb,
                             numpy_error, numpy_msg, numpy_tb,
                             accept_error=None, check_msg=False):
    # skip error check if nlcpy raise NotImplementedError
    if isinstance(nlcpy_error, NotImplementedError):
        return

    # For backward compatibility
    if accept_error is None:
        accept_error = ()

    if nlcpy_error is None and numpy_error is None:
        self.fail('Both nlcpy and numpy are expected to raise errors, but not')
    elif nlcpy_error is None:
        self.fail('Only numpy raises error\n\n' + numpy_tb)
    elif numpy_error is None:
        nlcpy_accept_msg = "is not supported as an"
        if nlcpy_accept_msg in nlcpy_msg:
            print(
                'Only nlcpy raises error\n',
                nlcpy_msg,
                '\nBut, ignored it.\n')
        else:
            self.fail('Only nlcpy raises error\n\n' + nlcpy_tb)

    elif not _check_numpy_nlcpy_error_compatible(nlcpy_error, numpy_error):
        msg = '''Different types of errors occurred

nlcpy
%s
numpy
%s
''' % (nlcpy_tb, numpy_tb)
        self.fail(msg)

    elif not (isinstance(nlcpy_error, accept_error)
              and isinstance(numpy_error, accept_error)):
        msg = '''Both nlcpy and numpy raise exceptions

nlcpy
%s
numpy
%s
''' % (nlcpy_tb, numpy_tb)
        self.fail(msg)

    elif check_msg and not (numpy_msg == nlcpy_msg):
        msg = '''Different message of errors occured

nlcpy
%s
numpy
%s
''' % (nlcpy_tb, numpy_tb)
        self.fail(msg)

    nlcpy.venode.synchronize_all_ve()


def _make_positive_mask(self, impl, args, kw):
    ks = [k for k, v in kw.items() if v in _unsigned_dtypes]
    for k in ks:
        kw[k] = numpy.intp
    # return nlcpy.asnumpy(impl(self, *args, **kw)) >= 0
    return asnumpy(impl(self, *args, **kw)) >= 0


def _contains_signed_and_unsigned(kw):
    vs = set(kw.values())
    return any(d in vs for d in _unsigned_dtypes) and \
        any(d in vs for d in _float_dtypes + _signed_dtypes)


def _make_decorator(check_func, name, type_check, accept_error, sp_name=None,
                    scipy_name=None, skip=False):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            # if sp_name:
            #     kw[sp_name] = nlcpyx.scipy.sparse
            # if scipy_name:
            #     kw[scipy_name] = nlcpyx.scipy
            kw[name] = nlcpy
            nlcpy_result, nlcpy_error, nlcpy_msg, nlcpy_tb = \
                _call_func(self, impl, args, kw)

            kw[name] = numpy
            if sp_name:
                import scipy.sparse
                kw[sp_name] = scipy.sparse
            if scipy_name:
                import scipy
                kw[scipy_name] = scipy
            numpy_result, numpy_error, numpy_msg, numpy_tb = \
                _call_func(self, impl, args, kw)

            if nlcpy_msg is not None:
                nlcpy_msg = re.sub(r'nlcpy', "numpy", nlcpy_msg)

            if nlcpy_error or numpy_error:
                _check_nlcpy_numpy_error(self, nlcpy_error, nlcpy_msg,
                                         nlcpy_tb, numpy_error, numpy_msg,
                                         numpy_tb, accept_error=accept_error)
                return

            if not isinstance(nlcpy_result, (tuple, list)):
                nlcpy_result = nlcpy_result,
            if not isinstance(numpy_result, (tuple, list)):
                numpy_result = numpy_result,

            # shape check
            for numpy_r, nlcpy_r in zip(numpy_result, nlcpy_result):
                if isinstance(numpy_r, numpy.ndarray):
                    assert numpy_r.shape == nlcpy_r.shape

            # type check
            if type_check:
                for numpy_r, nlcpy_r in zip(numpy_result, nlcpy_result):
                    if not isinstance(numpy_r, numpy.ndarray):
                        continue
                    if numpy_r.dtype != nlcpy_r.dtype:
                        msg = ['\n']
                        msg.append(
                            ' numpy.dtype: {}'.format(
                                numpy_r.dtype))
                        msg.append(
                            ' nlcpy.dtype: {}'.format(
                                nlcpy_r.dtype))
                        raise AssertionError('\n'.join(msg))

            # result check
            if not skip:
                for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                    if isinstance(numpy_r, numpy.ma.MaskedArray):
                        if numpy_r is numpy.ma.masked and nlcpy_r is numpy.ma.masked:
                            continue
                        msg = ['\n']
                        if not isinstance(nlcpy_r, nlcpy.ma.MaskedArray):
                            msg.append(' numpy.type: {}'.format(type(numpy_r)))
                            msg.append(' nlcpy.type: {}'.format(type(nlcpy_r)))
                            raise AssertionError('\n'.join(msg))
                        if (numpy_r is numpy.ma.masked) != (nlcpy_r is nlcpy.ma.masked):
                            msg.append(' numpy: {}'.format(numpy_r))
                            msg.append(' nlcpy: {}'.format(nlcpy_r))
                            raise AssertionError('\n'.join(msg))
                        check_func(nlcpy_r.data, numpy_r.data)
                        check_func(nlcpy_r.mask, numpy_r.mask)
                        check_func(nlcpy_r.fill_value, numpy_r.fill_value)
                        if numpy_r.sharedmask != nlcpy_r.sharedmask:
                            raise AssertionError("sharedmask is differ")
                        if numpy_r.hardmask != nlcpy_r.hardmask:
                            raise AssertionError("hardmask is differ")
                    else:
                        check_func(nlcpy_r, numpy_r)
        return test_func
    return decorator


def numpy_nlcpy_allclose(rtol=1e-7, atol=0, err_msg='', verbose=True,
                         name='xp', type_check=True, accept_error=False,
                         sp_name=None, scipy_name=None, contiguous_check=True):
    """Decorator that checks NumPy results and nlcpy ones are close.

    Args:
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``nlcpy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and nlcpy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``nlcpyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``nlcpyx.scipy`` module. If ``None``, no argument is given for
             the modules.
         contiguous_check(bool): If ``True``, consistency of contiguity is
             also checked.

    Decorated test fixture is required to return the arrays whose values are
    close between ``numpy`` case and ``nlcpy`` case.
    For example, this test case checks ``numpy.zeros`` and ``nlcpy.zeros``
    should return same value.

    >>> import unittest
    >>> from nlcpy import testing
    >>> class TestFoo(unittest.TestCase):
    ...
    ...     @testing.numpy_nlcpy_allclose()
    ...     def test_foo(self, xp):
    ...         # ...
    ...         # Prepare data with xp
    ...         # ...
    ...
    ...         xp_result = xp.zeros(10)
    ...         return xp_result

    .. seealso:: :func:`nlcpy.testing.assert_allclose`
    """
    def check_func(v, n):
        v_array = v
        n_array = n
        # if sp_name is not None:
        #     import scipy.sparse
        #     if nlcpyx.scipy.sparse.issparse(v):
        #         v_array = v.A
        #     if scipy.sparse.issparse(n):
        #         n_array = n.A
        array.assert_allclose(v_array, n_array, rtol, atol, err_msg, verbose)
        if contiguous_check and isinstance(n, numpy.ndarray):
            if n.flags.c_contiguous != v.flags.c_contiguous:
                raise AssertionError(
                    'The state of c_contiguous flag is false. '
                    '(nlcpy_result:{} numpy_result:{})'.format(
                        v.flags.c_contiguous, n.flags.c_contiguous))
            if n.flags.f_contiguous != v.flags.f_contiguous:
                raise AssertionError(
                    'The state of f_contiguous flag is false. '
                    '(nlcpy_result:{} numpy_result:{})'.format(
                        v.flags.f_contiguous, n.flags.f_contiguous))
    return _make_decorator(check_func, name, type_check, accept_error, sp_name,
                           scipy_name)


def numpy_nlcpy_array_almost_equal(decimal=6, err_msg='', verbose=True,
                                   name='xp', type_check=True,
                                   accept_error=False, sp_name=None,
                                   scipy_name=None):
    """Decorator that checks NumPy results and nlcpy ones are almost equal.

    Args:
         decimal(int): Desired precision.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``nlcpy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and nlcpy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``nlcpyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``nlcpyx.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`nlcpy.testing.assert_array_almost_equal`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``nlcpy``.

    .. seealso:: :func:`nlcpy.testing.assert_array_almost_equal`
    """
    def check_func(x, y):
        array.assert_array_almost_equal(
            x, y, decimal, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, accept_error, sp_name,
                           scipy_name)


def numpy_nlcpy_array_almost_equal_nulp(nulp=1, name='xp', type_check=True,
                                        accept_error=False, sp_name=None,
                                        scipy_name=None):
    """Decorator that checks results of NumPy and nlcpy are equal w.r.t. spacing.

    Args:
         nulp(int): The maximum number of unit in the last place for tolerance.
         name(str): Argument name whose value is either
             ``numpy`` or ``nlcpy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and nlcpy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True``, all error types are acceptable.
             If it is ``False``, no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``nlcpyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``nlcpyx.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`nlcpy.testing.assert_array_almost_equal_nulp`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``nlcpy``.

    .. seealso:: :func:`nlcpy.testing.assert_array_almost_equal_nulp`
    """  # NOQA
    def check_func(x, y):
        array.assert_array_almost_equal_nulp(x, y, nulp)
    return _make_decorator(check_func, name, type_check, accept_error, sp_name,
                           scipy_name=None)


def numpy_nlcpy_array_max_ulp(maxulp=1, dtype=None, name='xp', type_check=True,
                              accept_error=False, sp_name=None,
                              scipy_name=None):
    """Decorator that checks results of NumPy and nlcpy ones are equal w.r.t. ulp.

    Args:
         maxulp(int): The maximum number of units in the last place
             that elements of resulting two arrays can differ.
         dtype(numpy.dtype): Data-type to convert the resulting
             two array to if given.
         name(str): Argument name whose value is either
             ``numpy`` or ``nlcpy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and nlcpy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``nlcpyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``nlcpyx.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`assert_array_max_ulp`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``nlcpy``.

    .. seealso:: :func:`nlcpy.testing.assert_array_max_ulp`

    """  # NOQA
    def check_func(x, y):
        array.assert_array_max_ulp(x, y, maxulp, dtype)
    return _make_decorator(check_func, name, type_check, accept_error, sp_name,
                           scipy_name)


def numpy_nlcpy_array_equal(err_msg='', verbose=True, name='xp',
                            type_check=True, accept_error=False, sp_name=None,
                            scipy_name=None, strides_check=False):
    """Decorator that checks NumPy results and nlcpy ones are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``nlcpy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and nlcpy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``nlcpyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``nlcpyx.scipy`` module. If ``None``, no argument is given for
             the modules.
         strides_check(bool): If ``True``, consistency of strides is also
             checked.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`numpy_nlcpy_array_equal`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``nlcpy``.

    .. seealso:: :func:`nlcpy.testing.assert_array_equal`
    """
    def check_func(x, y):
        # if sp_name is not None:
        #     import scipy.sparse
        #     if nlcpyx.scipy.sparse.issparse(x):
        #         x = x.A
        #     if scipy.sparse.issparse(y):
        #         y = y.A
        array.assert_array_equal(x, y, err_msg, verbose, strides_check)

    return _make_decorator(check_func, name, type_check, accept_error, sp_name,
                           scipy_name)


def numpy_nlcpy_array_list_equal(
        err_msg='', verbose=True, name='xp', sp_name=None, scipy_name=None):
    """Decorator that checks the resulting lists of NumPy and nlcpy's one are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are appended
             to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``nlcpy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``nlcpyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``nlcpyx.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same list of arrays
    (except the type of array module) even if ``xp`` is ``numpy`` or ``nlcpy``.

    .. seealso:: :func:`nlcpy.testing.assert_array_list_equal`
    """  # NOQA
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            # if sp_name:
            #     kw[sp_name] = nlcpyx.scipy.sparse
            # if scipy_name:
            #     kw[scipy_name] = nlcpyx.scipy
            kw[name] = nlcpy
            x = impl(self, *args, **kw)

            if sp_name:
                import scipy.sparse
                kw[sp_name] = scipy.sparse
            if scipy_name:
                import scipy
                kw[scipy_name] = scipy
            kw[name] = numpy
            y = impl(self, *args, **kw)
            assert x is not None
            assert y is not None
            array.assert_array_list_equal(x, y, err_msg, verbose)
        return test_func
    return decorator


def numpy_nlcpy_array_less(err_msg='', verbose=True, name='xp',
                           type_check=True, accept_error=False, sp_name=None,
                           scipy_name=None):
    """Decorator that checks the nlcpy result is less than NumPy result.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``nlcpy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and nlcpy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``nlcpyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``nlcpyx.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the smaller array
    when ``xp`` is ``nlcpy`` than the one when ``xp`` is ``numpy``.

    .. seealso:: :func:`nlcpy.testing.assert_array_less`
    """
    def check_func(x, y):
        array.assert_array_less(x, y, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, accept_error, sp_name,
                           scipy_name)


def numpy_nlcpy_equal(name='xp', sp_name=None, scipy_name=None):
    """Decorator that checks NumPy results are equal to nlcpy ones.

    Args:
         name(str): Argument name whose value is either
             ``numpy`` or ``nlcpy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``nlcpyx.sciyp.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``nlcpyx.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same results
    even if ``xp`` is ``numpy`` or ``nlcpy``.
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            # if sp_name:
            #     kw[sp_name] = nlcpyx.scipy.sparse
            # if scipy_name:
            #     kw[scipy_name] = nlcpyx.scipy
            kw[name] = nlcpy
            nlcpy_result = impl(self, *args, **kw)

            if sp_name:
                import scipy.sparse
                kw[sp_name] = scipy.sparse
            if scipy_name:
                import scipy
                kw[scipy_name] = scipy
            kw[name] = numpy
            numpy_result = impl(self, *args, **kw)

            if nlcpy_result != numpy_result:
                message = '''Results are not equal:
nlcpy: %s
numpy: %s''' % (str(nlcpy_result), str(numpy_result))
                raise AssertionError(message)
        return test_func
    return decorator


def numpy_nlcpy_raises(name='xp', sp_name=None, scipy_name=None,
                       accept_error=Exception, ignore_msg=False):
    """Decorator that checks the NumPy and nlcpy throw same errors.

    Args:
         name(str): Argument name whose value is either
             ``numpy`` or ``nlcpy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``nlcpyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``nlcpyx.scipy`` module. If ``None``, no argument is given for
             the modules.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and nlcpy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         ignore_msg(bool): if True, raised error message check is skipped.

    Decorated test fixture is required throw same errors
    even if ``xp`` is ``numpy`` or ``nlcpy``.
    """

    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            # if sp_name:
            #     kw[sp_name] = nlcpyx.scipy.sparse
            # if scipy_name:
            #     kw[scipy_name] = nlcpyx.scipy
            kw[name] = nlcpy
            nlcpy_msg = None
            numpy_msg = None
            try:
                impl(self, *args, **kw)
                nlcpy.request.flush()
                nlcpy_error = None
                nlcpy_tb = None
            except Exception as e:
                nlcpy_error = e
                nlcpy_msg = str(e)
                nlcpy_tb = traceback.format_exc()
            if sp_name:
                import scipy.sparse
                kw[sp_name] = scipy.sparse
            if scipy_name:
                import scipy
                kw[scipy_name] = scipy
            kw[name] = numpy
            try:
                impl(self, *args, **kw)
                numpy_error = None
                numpy_tb = None
            except Exception as e:
                numpy_error = e
                numpy_msg = str(e)
                numpy_tb = traceback.format_exc()

            if ignore_msg:
                nlcpy_msg = None
                numpy_msg = None
            _check_nlcpy_numpy_error(self, nlcpy_error, nlcpy_msg,
                                     nlcpy_tb, numpy_error, numpy_msg,
                                     numpy_tb, accept_error=accept_error)
            del nlcpy_error, numpy_error
        return test_func
    return decorator


def for_dtypes(dtypes, name='dtype'):
    """Decorator for parameterized dtype test.

    Args:
         dtypes(list of dtypes): dtypes to be tested.
         name(str): Argument name to which specified dtypes are passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixture. Then, it runs the fixtures in parallel
    by passing the each element of ``dtypes`` to the named
    argument.
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            for dtype in dtypes:
                try:
                    if dtype is None:
                        kw[name] = None
                    else:
                        kw[name] = numpy.dtype(dtype).type
                    impl(self, *args, **kw)
                except unittest.SkipTest as e:
                    print('skipped: {} = {} ({})'.format(name, dtype, e))
                except Exception:
                    print(name, 'is', dtype)
                    raise

        return test_func
    return decorator


_complex_dtypes = (numpy.complex64, numpy.complex128)
_regular_float_dtypes = (numpy.float64, numpy.float32)
# _float_dtypes = _regular_float_dtypes + (numpy.float16,) # not
# implemented yet
_float_dtypes = _regular_float_dtypes
# _signed_dtypes = tuple(numpy.dtype(i).type for i in 'bhilq') # not implemented yet
# _unsigned_dtypes = tuple(numpy.dtype(i).type for i in 'BHILQ') # not
# implemented yet
_signed_dtypes = tuple(numpy.dtype(i).type for i in 'il')
_unsigned_dtypes = tuple(numpy.dtype(i).type for i in 'IL')
_int_dtypes = _signed_dtypes + _unsigned_dtypes
_int_bool_dtypes = _int_dtypes + (numpy.bool_,)
_regular_dtypes = _regular_float_dtypes + _int_bool_dtypes
_dtypes = _float_dtypes + _int_bool_dtypes


def _make_all_dtypes(no_float16, no_bool, no_complex):
    if no_float16:
        dtypes = _regular_float_dtypes
    else:
        dtypes = _float_dtypes

    if no_bool:
        dtypes += _int_dtypes
    else:
        dtypes += _int_bool_dtypes

    if not no_complex:
        dtypes += _complex_dtypes

    return dtypes


def for_all_dtypes(name='dtype', no_float16=True, no_bool=False,
                   no_complex=False):
    """Decorator that checks the fixture with all dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.
         no_float16(bool): If ``True``, ``numpy.float16`` is
             omitted from candidate dtypes.
         no_bool(bool): If ``True``, ``numpy.bool_`` is
             omitted from candidate dtypes.
         no_complex(bool): If ``True``, ``numpy.complex64`` and
             ``numpy.complex128`` are omitted from candidate dtypes.

    dtypes to be tested: ``numpy.complex64`` (optional),
    ``numpy.complex128`` (optional),
    ``numpy.float16`` (optional), ``numpy.float32``,
    ``numpy.float64``, ``numpy.dtype('b')``, ``numpy.dtype('h')``,
    ``numpy.dtype('i')``, ``numpy.dtype('l')``, ``numpy.dtype('q')``,
    ``numpy.dtype('B')``, ``numpy.dtype('H')``, ``numpy.dtype('I')``,
    ``numpy.dtype('L')``, ``numpy.dtype('Q')``, and ``numpy.bool_`` (optional).

    The usage is as follows.
    This test fixture checks if ``cPickle`` successfully reconstructs
    :class:`nlcpy.ndarray` for various dtypes.
    ``dtype`` is an argument inserted by the decorator.

    >>> import unittest
    >>> from nlcpy import testing
    >>> class TestNpz(unittest.TestCase):
    ...
    ...     @testing.for_all_dtypes()
    ...     def test_pickle(self, dtype):
    ...         a = testing.shaped_arange((2, 3, 4), dtype=dtype)
    ...         s = six.moves.cPickle.dumps(a)
    ...         b = six.moves.cPickle.loads(s)
    ...         testing.assert_array_equal(a, b)

    Typically, we use this decorator in combination with
    decorators that check consistency between NumPy and nlcpy like
    :func:`nlcpy.testing.numpy_nlcpy_allclose`.
    The following is such an example.

    >>> import unittest
    >>> from nlcpy import testing
    >>> class TestMean(unittest.TestCase):
    ...
    ...     @testing.for_all_dtypes()
    ...     @testing.numpy_nlcpy_allclose()
    ...     def test_mean_all(self, xp, dtype):
    ...         a = testing.shaped_arange((2, 3), xp, dtype)
    ...         return a.mean()

    .. seealso:: :func:`nlcpy.testing.for_dtypes`
    """
    return for_dtypes(_make_all_dtypes(no_float16, no_bool, no_complex),
                      name=name)


def for_float_dtypes(name='dtype', no_float16=False):
    """Decorator that checks the fixture with float dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.
         no_float16(bool): If ``True``, ``numpy.float16`` is
             omitted from candidate dtypes.

    dtypes to be tested are ``numpy.float16`` (optional), ``numpy.float32``,
    and ``numpy.float64``.

    .. seealso:: :func:`nlcpy.testing.for_dtypes`,
        :func:`nlcpy.testing.for_all_dtypes`
    """
    if no_float16:
        return for_dtypes(_regular_float_dtypes, name=name)
    else:
        return for_dtypes(_float_dtypes, name=name)


def for_signed_dtypes(name='dtype'):
    """Decorator that checks the fixture with signed dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.

    dtypes to be tested are ``numpy.dtype('b')``, ``numpy.dtype('h')``,
    ``numpy.dtype('i')``, ``numpy.dtype('l')``, and ``numpy.dtype('q')``.

    .. seealso:: :func:`nlcpy.testing.for_dtypes`,
        :func:`nlcpy.testing.for_all_dtypes`
    """
    return for_dtypes(_signed_dtypes, name=name)


def for_unsigned_dtypes(name='dtype'):
    """Decorator that checks the fixture with unsinged dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.

    dtypes to be tested are ``numpy.dtype('B')``, ``numpy.dtype('H')``,

     ``numpy.dtype('I')``, ``numpy.dtype('L')``, and ``numpy.dtype('Q')``.

    .. seealso:: :func:`nlcpy.testing.for_dtypes`,
        :func:`nlcpy.testing.for_all_dtypes`
    """
    return for_dtypes(_unsigned_dtypes, name=name)


def for_int_dtypes(name='dtype', no_bool=False):
    """Decorator that checks the fixture with integer and optionally bool dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.
         no_bool(bool): If ``True``, ``numpy.bool_`` is
             omitted from candidate dtypes.

    dtypes to be tested are ``numpy.dtype('b')``, ``numpy.dtype('h')``,
    ``numpy.dtype('i')``, ``numpy.dtype('l')``, ``numpy.dtype('q')``,
    ``numpy.dtype('B')``, ``numpy.dtype('H')``, ``numpy.dtype('I')``,
    ``numpy.dtype('L')``, ``numpy.dtype('Q')``, and ``numpy.bool_`` (optional).

    .. seealso:: :func:`nlcpy.testing.for_dtypes`,
        :func:`nlcpy.testing.for_all_dtypes`
    """  # NOQA
    if no_bool:
        return for_dtypes(_int_dtypes, name=name)
    else:
        return for_dtypes(_int_bool_dtypes, name=name)


def for_complex_dtypes(name='dtype'):
    """Decorator that checks the fixture with complex dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.

    dtypes to be tested are ``numpy.complex64`` and ``numpy.complex128``.

    .. seealso:: :func:`nlcpy.testing.for_dtypes`,
        :func:`nlcpy.testing.for_all_dtypes`
    """
    return for_dtypes(_complex_dtypes, name=name)


def for_dtypes_combination(types, names=('dtype',), full=None):
    """Decorator that checks the fixture with a product set of dtypes.

    Args:
         types(list of dtypes): dtypes to be tested.
         names(list of str): Argument names to which dtypes are passed.
         full(bool): If ``True``, then all combinations
             of dtypes will be tested.
             Otherwise, the subset of combinations will be tested
             (see the description below).

    Decorator adds the keyword arguments specified by ``names``
    to the test fixture. Then, it runs the fixtures in parallel
    with passing (possibly a subset of) the product set of dtypes.
    The range of dtypes is specified by ``types``.

    The combination of dtypes to be tested changes depending
    on the option ``full``. If ``full`` is ``True``,
    all combinations of ``types`` are tested.
    Sometimes, such an exhaustive test can be costly.
    So, if ``full`` is ``False``, only the subset of possible
    combinations is tested. Specifically, at first,
    the shuffled lists of ``types`` are made for each argument
    name in ``names``.
    Let the lists be ``D1``, ``D2``, ..., ``Dn``
    where :math:`n` is the number of arguments.
    Then, the combinations to be tested will be ``zip(D1, ..., Dn)``.
    If ``full`` is ``None``, the behavior is switched
    by setting the environment variable ``nlcpy_TEST_FULL_COMBINATION=1``.

    For example, let ``types`` be ``[float16, float32, float64]``
    and ``names`` be ``['a_type', 'b_type']``. If ``full`` is ``True``,
    then the decorated test fixture is executed with all
    :math:`2^3` patterns. On the other hand, if ``full`` is ``False``,
    shuffled lists are made for ``a_type`` and ``b_type``.
    Suppose the lists are ``(16, 64, 32)`` for ``a_type`` and
    ``(32, 64, 16)`` for ``b_type`` (prefixes are removed for short).
    Then the combinations of ``(a_type, b_type)`` to be tested are
    ``(16, 32)``, ``(64, 64)`` and ``(32, 16)``.
    """

    if full is None:
        full = int(os.environ.get('nlcpy_TEST_FULL_COMBINATION', '0')) != 0

    if full:
        combination = parameterized.product({name: types for name in names})
    else:
        ts = []
        for _ in range(len(names)):
            # Make shuffled list of types for each name
            t = list(types)
            random.shuffle(t)
            ts.append(t)

        combination = [dict(zip(names, typs)) for typs in zip(*ts)]

    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            for dtypes in combination:
                kw_copy = kw.copy()
                kw_copy.update(dtypes)

                try:
                    impl(self, *args, **kw_copy)
                except Exception:
                    print(dtypes)
                    raise

        return test_func
    return decorator


def for_all_dtypes_combination(names=('dtyes',),
                               no_float16=False, no_bool=False, full=None,
                               no_complex=False):
    """Decorator that checks the fixture with a product set of all dtypes.

    Args:
         names(list of str): Argument names to which dtypes are passed.
         no_float16(bool): If ``True``, ``numpy.float16`` is
             omitted from candidate dtypes.
         no_bool(bool): If ``True``, ``numpy.bool_`` is
             omitted from candidate dtypes.
         full(bool): If ``True``, then all combinations of dtypes
             will be tested.
             Otherwise, the subset of combinations will be tested
             (see description in :func:`nlcpy.testing.for_dtypes_combination`).
         no_complex(bool): If, True, ``numpy.complex64`` and
             ``numpy.complex128`` are omitted from candidate dtypes.

    .. seealso:: :func:`nlcpy.testing.for_dtypes_combination`
    """
    types = _make_all_dtypes(no_float16, no_bool, no_complex)
    return for_dtypes_combination(types, names, full)


def for_signed_dtypes_combination(names=('dtype',), full=None):
    """Decorator for parameterized test w.r.t. the product set of signed dtypes.

    Args:
         names(list of str): Argument names to which dtypes are passed.
         full(bool): If ``True``, then all combinations of dtypes
             will be tested.
             Otherwise, the subset of combinations will be tested
             (see description in :func:`nlcpy.testing.for_dtypes_combination`).

    .. seealso:: :func:`nlcpy.testing.for_dtypes_combination`
    """  # NOQA
    return for_dtypes_combination(_signed_dtypes, names=names, full=full)


def for_unsigned_dtypes_combination(names=('dtype',), full=None):
    """Decorator for parameterized test w.r.t. the product set of unsigned dtypes.

    Args:
         names(list of str): Argument names to which dtypes are passed.
         full(bool): If ``True``, then all combinations of dtypes
             will be tested.
             Otherwise, the subset of combinations will be tested
             (see description in :func:`nlcpy.testing.for_dtypes_combination`).

    .. seealso:: :func:`nlcpy.testing.for_dtypes_combination`
    """  # NOQA
    return for_dtypes_combination(_unsigned_dtypes, names=names, full=full)


def for_int_dtypes_combination(names=('dtype',), no_bool=False, full=None):
    """Decorator for parameterized test w.r.t. the product set of int and boolean.

    Args:
         names(list of str): Argument names to which dtypes are passed.
         no_bool(bool): If ``True``, ``numpy.bool_`` is
             omitted from candidate dtypes.
         full(bool): If ``True``, then all combinations of dtypes
             will be tested.
             Otherwise, the subset of combinations will be tested
             (see description in :func:`nlcpy.testing.for_dtypes_combination`).

    .. seealso:: :func:`nlcpy.testing.for_dtypes_combination`
    """  # NOQA
    if no_bool:
        types = _int_dtypes
    else:
        types = _int_bool_dtypes
    return for_dtypes_combination(types, names, full)


def for_all_axis(begin, end, name='axis'):
    """Decorator to parameterize tests with axis.

    Args:
         begin(int): begin of axis range to be tested.
         end(int): end of axis range to be tested.
         name(str): Argument name to which the specified axis is passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixtures. Then, the fixtures run by passing each element of
    ``axis`` created from axis range(begin to end) to the named argument.

    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            for axis in range(begin, end):
                try:
                    kw[name] = axis
                    impl(self, *args, **kw)
                except Exception:
                    print(name, 'is', axis)
                    raise

        return test_func
    return decorator


def for_broadcast(shapes, name='shape'):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            bshapes = shaped_rearrange_for_broadcast(shapes)
            for _shape in bshapes:
                try:
                    kw[name] = _shape
                    impl(self, *args, **kw)
                except Exception:
                    print(name, 'is', _shape)
                    raise
        return test_func
    return decorator


def set_random_seed(minval, maxval, name='seed'):
    """Decorator to parameterize tests with seed for random.

    Args:
         minval(int): Minimum value of seed.
         maxval(int): Maximum value of seed.
         name(str): Argument name to which the specified seed is passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixtures. Then, the fixtures run by passing each element of
    ``seed``  to the named argument.

    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            seed = numpy.random.randint(minval, maxval)
            try:
                kw[name] = seed
                impl(self, *args, **kw)
            except Exception:
                print(name, 'is', seed)
                raise

        return test_func
    return decorator


def for_orders(orders, name='order'):
    """Decorator to parameterize tests with order.

    Args:
         orders(list of order): orders to be tested.
         name(str): Argument name to which the specified order is passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixtures. Then, the fixtures run by passing each element of
    ``orders`` to the named argument.

    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            for order in orders:
                try:
                    kw[name] = order
                    impl(self, *args, **kw)
                except Exception:
                    print(name, 'is', order)
                    raise

        return test_func
    return decorator


def for_CF_orders(name='order'):
    """Decorator that checks the fixture with orders 'C' and 'F'.

    Args:
         name(str): Argument name to which the specified order is passed.

    .. seealso:: :func:`nlcpy.testing.for_all_dtypes`

    """
    return for_orders([None, 'C', 'F', 'c', 'f'], name)


def with_requires(*requirements):
    """Run a test case only when given requirements are satisfied.

    .. admonition:: Example

       This test case runs only when `numpy>=1.10` is installed.

       >>> from nlcpy import testing
       ... class Test(unittest.TestCase):
       ...     @testing.with_requires('numpy>=1.10')
       ...     def test_for_numpy_1_10(self):
       ...         pass

    Args:
        requirements: A list of string representing requirement condition to
            run a given test case.

    """
    ws = pkg_resources.WorkingSet()
    try:
        ws.require(*requirements)
        skip = False
    except pkg_resources.DistributionNotFound:
        for req in requirements:
            m = re.findall(r'(\w+)\s*([=|>|<|!]+)\s*(.*)', req)

            libname = m[0][0]
            sign = m[0][1]
            targver = m[0][2]

            lib = importlib.import_module(libname)
            statement = ('StrictVersion(lib.__version__) {} '
                         'StrictVersion(targver)'.format(sign))

            flag = eval(statement, {'lib': lib, 'targver': targver,
                                    'StrictVersion': StrictVersion})
            if flag is False:
                skip = True
                break
            else:
                skip = False
    except pkg_resources.ResolutionError:
        skip = True

    msg = 'requires: {}'.format(','.join(requirements))
    return unittest.skipIf(skip, msg)


def numpy_satisfies(version_range):
    """Returns True if numpy version satisfies the specified criteria.

    Args:
        version_range: A version specifier (e.g., `>=1.13.0`).
    """
    spec = 'numpy{}'.format(version_range)
    try:
        pkg_resources.require(spec)
    except pkg_resources.DistributionNotFound:
        m = re.findall(r'(\w+)\s*([=|>|<|!]+)\s*(.*)', spec)
        libname = m[0][0]
        sign = m[0][1]
        targver = m[0][2]

        lib = importlib.import_module(libname)
        statement = ('StrictVersion(lib.__version__) {} '
                     'StrictVersion(targver)'.format(sign))

        flag = eval(statement, {'lib': lib, 'targver': targver,
                                'StrictVersion': StrictVersion})
        return flag
    except pkg_resources.VersionConflict:
        return False
    return True


def shaped_arange(shape, xp=nlcpy, dtype=numpy.float32, order='C'):
    """Returns an array with given shape, array module, and dtype.

    Args:
         shape(tuple of int): Shape of returned ndarray.
         xp(numpy or nlcpy): Array module to use.
         dtype(dtype): Dtype of returned ndarray.
         order({'C', 'F'}): Order of returned ndarray.

    Returns:
         numpy.ndarray or nlcpy.ndarray:
         The array filled with :math:`1, \\cdots, N` with specified dtype
         with given shape, array module. Here, :math:`N` is
         the size of the returned array.
         If ``dtype`` is ``numpy.bool_``, evens (resp. odds) are converted to
         ``True`` (resp. ``False``).

    """
    dtype = xp.dtype(dtype)
    if type(shape) is int:
        a = xp.array(shape)
    else:
        a = xp.arange(1, internal.prod(shape) + 1, 1)
    if dtype == '?':
        a = a % 2 == 0
    elif dtype.kind == 'c':
        a = a + a * 1j
    if a.size > 1:
        return xp.asarray(a.astype(dtype).reshape(shape), order=order)
    else:
        return xp.asarray(a.astype(dtype), order=order)


def shaped_rearrange_for_broadcast(shapes):
    n = len(shapes)
    ret = []
    for i in range(n):
        for j in range(2 ** (len(shapes[i]) * 2)):
            _shape = []
            _s1 = []
            _s2 = []
            for k in range(len(shapes[i])):
                _s1.append(1 if (j & 2**k) != 0 else shapes[i][k])
                _s2.append(
                    1 if (j & 2**(k + len(shapes[i]))) != 0 else shapes[i][k])
            _shape.append([_s1, _s2])
            ret.append(_shape[0])
    return ret


def create_shape_and_axis_set(shapes, with_none=False):
    ret = []
    for shape in shapes:
        if with_none:
            ret.append([shape, None])
        if isinstance(shape, (list, tuple)):
            begin = -len(shape)
            end = len(shape)
        else:
            begin = -1
            end = 1
        for i in range(begin, end):
            ret.append([shape, i])

    return ret


def shaped_reverse_arange(shape, xp=nlcpy, dtype=numpy.float32):
    """Returns an array filled with decreasing numbers.

    Args:
         shape(tuple of int): Shape of returned ndarray.
         xp(numpy or nlcpy): Array module to use.
         dtype(dtype): Dtype of returned ndarray.

    Returns:
         numpy.ndarray or nlcpy.ndarray:
         The array filled with :math:`N, \\cdots, 1` with specified dtype
         with given shape, array module.
         Here, :math:`N` is the size of the returned array.
         If ``dtype`` is ``numpy.bool_``, evens (resp. odds) are converted to
         ``True`` (resp. ``False``).
    """
    dtype = numpy.dtype(dtype)
    size = internal.prod(shape)
    a = numpy.arange(size, 0, -1)
    if dtype == '?':
        a = a % 2 == 0
    elif dtype.kind == 'c':
        a = a + a * 1j
    if a.size > 0:
        return xp.array(a.astype(dtype).reshape(shape))
    else:
        return xp.array(a.astype(dtype))


def shaped_random(shape, xp=nlcpy, dtype=numpy.float32, scale=10, seed=0):
    """Returns an array filled with random values.

    Args:
         shape(tuple): Shape of returned ndarray.
         xp(numpy or nlcpy): Array module to use.
         dtype(dtype): Dtype of returned ndarray.
         scale(float): Scaling factor of elements.
         seed(int): Random seed.

    Returns:
         numpy.ndarray or nlcpy.ndarray: The array with
             given shape, array module,

    If ``dtype`` is ``numpy.bool_``, the elements are
    independently drawn from ``True`` and ``False``
    with same probabilities.
    Otherwise, the array is filled with samples
    independently and identically drawn
    from uniform distribution over :math:`[0, scale)`
    with specified dtype.
    """
    numpy.random.seed(seed)
    dtype = numpy.dtype(dtype)
    if dtype == '?':
        # return xp.asarray(numpy.random.randint(2, size=shape).astype(dtype))
        return xp.asarray(numpy.random.randint(2, size=shape), dtype=dtype)
    elif dtype.kind == 'c':
        a = numpy.random.rand(*shape) + 1j * numpy.random.rand(*shape)
        # return xp.asarray((a * scale).astype(dtype))
        return xp.asarray((a * scale), dtype=dtype)
    else:
        # return xp.asarray((numpy.random.rand(*shape) * scale).astype(dtype))
        return xp.asarray((numpy.random.rand(*shape) * scale), dtype=dtype)


def numpy_nlcpy_check_for_unary_ufunc(
        ops,
        shapes,
        order_x=[  # order for x
            'C',
            'F'],
        order_out=[  # order for out
            'C',
            'F'],
        order_where=[  # order for where
            'C',
            'F'],
        dtype_x=_float_dtypes,  # dtype for x
        dtype_out=_float_dtypes,  # dtype for out
        dtype_arg=_float_dtypes,  # dtype for ufunc argument
        minval=0,
        maxval=100,
        mode='array',
        is_out=False,
        is_where=False,
        is_dtype=False,
        is_broadcast=False,  # if True, out/where shape will be expanded twice
        name_xp='xp',
        name_in1='in1',
        name_axis='axis',
        name_indices='indices',
        name_out='out',
        name_where='where',
        name_op='op',
        name_dtype='dtype',
        ufunc_name='',
        axes=(0,),
        indices=(None,),
        keepdims=False,
        seed=None):
    """Decorator to parameterize tests unary operator.
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            if seed is not None:
                numpy.random.seed(seed)
            for op in ops:
                for shape in shapes:
                    ufunc._check_for_unary_with_create_param(
                        self, args, kw, op, shape, order_x, order_out,
                        order_where, dtype_x, dtype_out, dtype_arg,
                        minval, maxval, mode, is_out, is_where,
                        is_dtype, is_broadcast, name_xp, name_in1, name_axis,
                        name_indices, name_out, name_where, name_op, name_dtype, impl,
                        ufunc_name, axes, indices, keepdims)
        return test_func
    return decorator


def numpy_nlcpy_check_for_binary_ufunc(
        ops,
        shapes,
        order_x=[  # order for x
            'C',
            'F'],
        order_y=[  # order for y
            'C',
            'F'],
        order_out=[  # order for out
            'C',
            'F'],
        order_where=[  # order for where
            'C',
            'F'],
        order_arg=['K'],  # order for argument
        dtype_x=_float_dtypes,  # dtype for x
        dtype_y=_float_dtypes,  # dtype for y
        dtype_out=_float_dtypes,  # dtype for out
        dtype_arg=_float_dtypes,  # dtype for ufunc argument
        minval=0,
        maxval=100,
        mode='array_array',
        is_out=False,
        is_where=False,
        is_dtype=False,
        is_broadcast=False,  # if True, out/where shape will be expanded twice
        name_xp='xp',
        name_in1='in1',
        name_in2='in2',
        name_order='order',
        name_casting='casting',
        name_out='out',
        name_where='where',
        name_op='op',
        name_dtype='dtype',
        ufunc_name='',
        casting=['same_kind', ],
        seed=None):
    """Decorator to parameterize tests binary operator.
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            if seed is not None:
                numpy.random.seed(seed)
            for op in ops:
                for shape in shapes:
                    ufunc._check_for_binary_with_create_param(
                        self, args, kw, op, shape, order_x, order_y,
                        order_out, order_where, order_arg, dtype_x, dtype_y,
                        dtype_out, dtype_arg, minval, maxval, mode, is_out, is_where,
                        is_dtype, is_broadcast, name_xp, name_in1,
                        name_in2, name_order, name_casting, name_out, name_where,
                        name_op, name_dtype, ufunc_name, casting, impl)
        return test_func
    return decorator


class numpy_nlcpy_errstate(object):

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.np_err = numpy.geterr()
        self.vp_err = nlcpy.geterr()
        numpy.seterr(**self.kw)
        nlcpy.seterr(**self.kw)

    def __exit__(self, *_):
        numpy.seterr(**self.np_err)
        nlcpy.seterr(**self.vp_err)


@contextlib.contextmanager
def assert_warns(expected):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        yield

    if any(isinstance(m.message, expected) for m in w):
        return

    try:
        exc_name = expected.__name__
    except AttributeError:
        exc_name = str(expected)

    raise AssertionError('%s not triggerred' % exc_name)


class NumpyAliasTestBase(unittest.TestCase):

    @property
    def func(self):
        raise NotImplementedError()

    @property
    def nlcpy_func(self):
        return getattr(nlcpy, self.func)

    @property
    def numpy_func(self):
        return getattr(numpy, self.func)


class NumpyAliasBasicTestBase(NumpyAliasTestBase):

    def test_argspec(self):
        f = inspect.signature
        assert f(self.nlcpy_func) == f(self.numpy_func)

    def test_docstring(self):
        nlcpy_func = self.nlcpy_func
        numpy_func = self.numpy_func
        assert hasattr(nlcpy_func, '__doc__')
        assert nlcpy_func.__doc__ is not None
        assert nlcpy_func.__doc__ != ''
        assert nlcpy_func.__doc__ is not numpy_func.__doc__


class NumpyAliasValuesTestBase(NumpyAliasTestBase):

    def test_values(self):
        assert self.nlcpy_func(*self.args) == self.numpy_func(*self.args)


def asnumpy(a, order='C'):
    if isinstance(a, ndarray):
        return a.get(order=order)
    else:
        return numpy.asarray(a, order=order)

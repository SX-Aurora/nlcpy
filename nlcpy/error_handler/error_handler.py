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

import threading
import contextlib


_ERR_IGNORE = 0
_ERR_WARN = 1
_ERR_RAISE = 2
_ERR_CALL = 3
_ERR_PRINT = 4
_ERR_LOG = 5
_ERR_DEFAULT = 521

_SHIFT_DIVIDEBYZERO = 0
_SHIFT_OVERFLOW = 3
_SHIFT_UNDERFLOW = 6
_SHIFT_INVALID = 9

_errdict = {"ignore": _ERR_IGNORE,
            "warn": _ERR_WARN,
            "raise": _ERR_RAISE,
            "print": _ERR_PRINT}
_errdict_rev = {value: key for key, value in _errdict.items()}
_thread_local = threading.local()


class _ErrState:

    def __init__(self):
        self._errstate = _ERR_DEFAULT

    @staticmethod
    def get():
        try:
            errstate = _thread_local._errstate
        except AttributeError:
            errstate = _ErrState()
            _thread_local._errstate = errstate
        return errstate

    def set(self, state):
        self._errstate = state


# ----------------------------------------------------------------------------
# Set how floating-point errors are handled.
# see: https://docs.scipy.org/doc/numpy/reference/generated/
#                                            numpy.seterr.html#numpy.seterr
# ----------------------------------------------------------------------------

def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    """Sets how floating-point errors are handled.

    Parameters
    ----------
    all : {'ignore', 'warn', 'raise', 'print'}, optional
        Sets treatment for all types of floating-point errors at once:

            - ignore: Take no action when the exception occurs.
            - warn: Print a *RuntimeWarning*.
            - raise: Raise a *FloatingPointError*.
            - print: Print a warning directly to stdout.

        The default is not to change the current behavior.
    divide : {'ignore', 'warn', 'raise', 'print'}, optional
        Treatment for division by zero.
    over : {'ignore', 'warn', 'raise', 'print'}, optional
        Treatment for floating-point overflow.
    under : {'ignore', 'warn', 'raise', 'print'}, optional
        Treatment for floating-point underflow.
    invalid : {'ignore', 'warn', 'raise', 'print'}, optional
        Treatment for invalid floating-point operation.

    Returns
    -------
    old_settings : dict
        Dictionary containing the old settings.

    Restriction
    -----------
    - If the 'call' mode or the 'log' mode is specified for each parameter,
      *KeyError* occurs.

    Note
    ----
    - The floating-point exceptions are defined in the IEEE 754 standard:

        - Division by zero: infinite result obtained from finite numbers.
        - Overflow: result too large to be expressed.
        - Underflow: result so close to zero that some precision was lost.
        - Invalid operation: result is not an expressible number, typically
          indicates that a NaN was produced.

    See Also
    --------
    geterr : Gets the current way of handling floating-point errors.
    errstate : Context manager for floating-point error handling.

    Examples
    --------
    >>> import nlcpy as vp
    >>> old_settings = vp.seterr(all='ignore')  #seterr to known value
    >>> vp.seterr(over='raise')
    {'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
    >>> vp.seterr(**old_settings)  # reset to default
    {'divide': 'ignore', 'over': 'raise', 'under': 'ignore', 'invalid': 'ignore'}

    """

    old = geterr()

    if divide is None:
        divide = all or old['divide']
    if over is None:
        over = all or old['over']
    if under is None:
        under = all or old['under']
    if invalid is None:
        invalid = all or old['invalid']

    maskvalue = ((_errdict[divide] << _SHIFT_DIVIDEBYZERO) +
                 (_errdict[over] << _SHIFT_OVERFLOW) +
                 (_errdict[under] << _SHIFT_UNDERFLOW) +
                 (_errdict[invalid] << _SHIFT_INVALID))

    _ErrState.get().set(maskvalue)
    return old


# ----------------------------------------------------------------------------
# Get the current way of handling floating-point errors.
# see: https://docs.scipy.org/doc/numpy/reference/generated/
#                                             numpy.geterr.html#numpy.geterr
# ----------------------------------------------------------------------------

def geterr():
    """Gets the current way of handling floating-point errors.

    Returns
    -------
    res : dict
        A dictionary with keys "divide", "over", "under", and "invalid",
        whose values are from the strings  "ignore", "print", "warn", and "raise".
        The keys represent possible floating-point exceptions, and the values
        define how these exceptions are handled.
        The elements of the shape tuple give the lengths of the corresponding array
        dimensions.

    Note
    ----
    - For complete documentation of the types of floating-point exceptions and treatment
      options, see :func:`nlcpy.seterr`.

    See Also
    --------
    seterr : Sets how floating-point errors are handled.
    errstate : Context manager for floating-point error handling.

    Examples
    --------
    >>> import nlcpy as vp
    >>> from collections import OrderedDict
    >>> sorted(vp.geterr().items())
    [('divide', 'warn'), ('invalid', 'warn'), ('over', 'warn'), ('under', 'ignore')]
    >>> vp.arange(3.) / vp.arange(3.)
    array([nan,  1.,  1.])

    """
    maskvalue = _ErrState.get()._errstate
    mask = 7
    res = {}
    val = (maskvalue >> _SHIFT_DIVIDEBYZERO) & mask
    res['divide'] = _errdict_rev[val]
    val = (maskvalue >> _SHIFT_OVERFLOW) & mask
    res['over'] = _errdict_rev[val]
    val = (maskvalue >> _SHIFT_UNDERFLOW) & mask
    res['under'] = _errdict_rev[val]
    val = (maskvalue >> _SHIFT_INVALID) & mask
    res['invalid'] = _errdict_rev[val]
    return res


class errstate(contextlib.ContextDecorator):
    """Context manager for floating-point error handling.

    Using an instance of `errstate` as a context manager allows statements in
    that context to execute with a known error handling behavior. Upon entering
    the context the error handling is set with `seterr`, and upon exiting it is
    reset to what it was before.

    Parameters
    ----------
    kwargs : {divide, over, under, invalid}
        Keyword arguments. The valid keywords are the possible floating-point
        exceptions. Each keyword should have a string value that defines the
        treatment for the particular error. Possible values are
        {'ignore', 'warn', 'raise', 'print'}.

    See Also
    --------
    seterr : Sets how floating-point errors are handled.
    geterr : Gets the current way of handling floating-point errors.

    Examples
    --------
    >>> import nlcpy as vp
    >>> olderr = vp.seterr(all='ignore')  # Set error handling to known state.

    >>> vp.arange(3) / 0.
    array([nan, inf, inf])
    >>> with vp.errstate(divide='warn'):
    ...     vp.arange(3) / 0.  # doctest: +SKIP
    <stdin>:2: RuntimeWarning: divide by zero encountered \
in any of (nlcpy_arange, nlcpy_true_divide)
    array([nan, inf, inf])
    >>> vp.sqrt(-1)
    array(nan)
    >>> with vp.errstate(invalid='raise'):  # doctest: +SKIP
    ...     vp.sqrt(-1)
    Traceback (most recent call last):
    ...
    FloatingPointError: invalid value encountered in (nlcpy_sqrt)

    Outside the context the error handling behavior has not changed:

    >>> vp.geterr()
    {'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
    >>> _ = vp.seterr(**olderr)
    """

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.oldstate = seterr(**self.kwargs)

    def __exit__(self, *exc_info):
        seterr(**self.oldstate)

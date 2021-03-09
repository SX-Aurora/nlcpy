#
# * The source code in this file is developed independently by NEC Corporation.
#
# # NLCPy License #
#
#     Copyright (c) 2020-2021 NEC Corporation
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

import numpy


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
      *NotImplementedError* occurs.

    Note
    ----
    - This function is the wrapper function to utilize :func:`numpy.seterr`.
    - The floating-point exceptions are defined in the IEEE 754 standard:

        - Division by zero: infinite result obtained from finite numbers.
        - Overflow: result too large to be expressed.
        - Underflow: result so close to zero that some precision was lost.
        - Invalid operation: result is not an expressible number, typically
          indicates that a NaN was produced.

    See Also
    --------
    geterr : Gets the current way of handling floating-point errors.

    Examples
    --------
    >>> import nlcpy as vp
    >>> old_settings = vp.seterr(all='ignore')  #seterr to known value
    >>> vp.seterr(over='raise')
    {'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
    >>> vp.seterr(**old_settings)  # reset to default
    {'divide': 'ignore', 'over': 'raise', 'under': 'ignore', 'invalid': 'ignore'}

    """
    if all in ('call', 'log'):
        raise NotImplementedError('all=%s in seterr is not implemented yet.' % all)
    if divide in ('call', 'log'):
        raise NotImplementedError('divide=%s in seterr is not implemented yet.' % divide)
    if over in ('call', 'log'):
        raise NotImplementedError('over=%s in seterr is not implemented yet.' % over)
    if under in ('call', 'log'):
        raise NotImplementedError('under=%s in seterr is not implemented yet.' % under)
    if invalid in ('call', 'log'):
        raise NotImplementedError(
            'invalid=%s in seterr is not implemented yet.' % invalid)

    return numpy.seterr(all=all, divide=divide, over=over, under=under, invalid=invalid)


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
    - This function is the wrapper function to utilize :func:`numpy.geterr`.

    See Also
    --------
    seterr : Sets how floating-point errors are handled.

    Examples
    --------
    >>> import nlcpy as vp
    >>> from collections import OrderedDict
    >>> sorted(vp.geterr().items())
    [('divide', 'warn'), ('invalid', 'warn'), ('over', 'warn'), ('under', 'ignore')]
    >>> vp.arange(3.) / vp.arange(3.)
    array([nan,  1.,  1.])

    """
    return numpy.geterr()

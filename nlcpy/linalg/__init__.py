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

import numpy
from numpy.linalg import LinAlgError  # NOQA
from nlcpy.wrapper.numpy_wrap import _make_wrap_func # NOQA

# override numpy documentation
LinAlgError.__doc__ = '''
    Generic Python-exception-derived object raised by linalg functions.

    General purpose exception class, derived from Python's exception.
    Exception class, programmatically raised in linalg functions when a
    Linear Algebra-related condition would prevent further correct
    execution of the function.

    Parameters
    ----------
    None

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.linalg.inv(vp.zeros((2,2))) # doctest: +SKIP
    ...
    numpy.linalg.LinAlgError: Singular matrix
'''

from nlcpy.linalg import cblas_wrapper  # NOQA
from nlcpy.linalg import products  # NOQA
from nlcpy.linalg.solve import solve  # NOQA
from nlcpy.linalg.solve import lstsq  # NOQA
from nlcpy.linalg.solve import inv  # NOQA
from nlcpy.linalg.eig import eig  # NOQA
from nlcpy.linalg.eig import eigvals  # NOQA
from nlcpy.linalg.eig import eigh  # NOQA
from nlcpy.linalg.eig import eigvalsh  # NOQA
from nlcpy.linalg.norm import norm  # NOQA
from nlcpy.linalg.decomposition import svd  # NOQA
from nlcpy.linalg.decomposition import cholesky  # NOQA
from nlcpy.linalg.decomposition import qr  # NOQA


def __getattr__(attr):
    try:
        f = getattr(numpy.linalg, attr)
    except AttributeError as _err:
        raise AttributeError(
            "module 'nlcpy.linalg' has no attribute '{}'."
            .format(attr)) from _err
    if not callable(f):
        raise AttributeError(
            "module 'nlcpy.linalg' has no attribute '{}'.".format(attr))
    return _make_wrap_func(f)

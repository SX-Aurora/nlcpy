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


# ----------------------------------------------------------------------------
# Set how floating-point errors are handled.
# see: https://docs.scipy.org/doc/numpy/reference/generated/
#                                            numpy.seterr.html#numpy.seterr
# ----------------------------------------------------------------------------

def seterr(all=None, divide=None, over=None, under=None, invalid=None):
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
    return numpy.geterr()


# ----------------------------------------------------------------------------
# Set the floating-point error callback function or log object.
# see: https://docs.scipy.org/doc/numpy/reference/generated/
#                                      numpy.seterrcall.html#numpy.seterrcall
# ----------------------------------------------------------------------------

def seterrcall(func):
    return numpy.seterrcall(func)


# ----------------------------------------------------------------------------
# Return the current callback function used on floating-point errors.
# see: https://docs.scipy.org/doc/numpy/reference/generated/
#                                     numpy.geterrcall.html#numpy.geterrcall
# ----------------------------------------------------------------------------

def geterrcall():
    return numpy.geterrcall()

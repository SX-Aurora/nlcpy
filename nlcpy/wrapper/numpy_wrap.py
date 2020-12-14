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
import nlcpy


def numpy_wrap(func):
    def wrap_func(*args, **kwargs):
        is_out = False
        try:
            return func(*args, **kwargs)
        except NotImplementedError:
            f = getattr(numpy, func.__name__)
            # retrieve input ndarrays of NLCPy from VE
            largs = list(args)
            for i, _l in enumerate(largs):
                if isinstance(_l, nlcpy.ndarray):
                    largs[i] = _l.get()
            for k, v in kwargs.items():
                if isinstance(v, nlcpy.ndarray):
                    kwargs[k] = v.get()
                if k == 'out':
                    is_out = True
                    in_out = v
                else:
                    is_out = False

            # call NumPy function
            ret = f(*largs, **kwargs)
            # transfer the return values to VE
            if isinstance(ret, numpy.ndarray) or numpy.isscalar(ret):
                vp_ret = nlcpy.asarray(ret)
                if is_out:
                    in_out[...] = vp_ret
                return vp_ret
            elif hasattr(ret, "__iter__"):
                lret = list(ret)
                for i, _l in enumerate(lret):
                    if isinstance(_l, numpy.ndarray) or numpy.isscalar(_l):
                        lret[i] = nlcpy.asarray(_l)
                    else:
                        lret[i] = _l
                vp_ret = tuple(lret)
                if is_out:
                    raise NotImplementedError
                return vp_ret
            else:
                raise NotImplementedError

    return wrap_func

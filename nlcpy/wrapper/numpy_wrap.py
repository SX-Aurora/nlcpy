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
import nlcpy
import functools


def numpy_wrap(func):
    @functools.wraps(func)
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

            # call NumPy function
            ret = f(*largs, **kwargs)
            # transfer the return values to VE
            if isinstance(ret, numpy.ndarray) or \
                    numpy.isscalar(ret) and numpy.dtype(type(ret)).char in '?iIlLfFdD':
                vp_ret = nlcpy.asarray(ret)
                if is_out:
                    in_out[...] = vp_ret
                return vp_ret
            elif isinstance(ret, numpy.lib.npyio.NpzFile):
                return nlcpy.NpzFile(ret)
            elif isinstance(ret, dict):
                for key, val in ret.items():
                    ret[key] = nlcpy.asarray(val)
                return ret
            elif isinstance(ret, (list, tuple)):
                vpp = []
                for x in ret:
                    if isinstance(x, numpy.ndarray) is False:
                        if numpy.any(isinstance(x, numpy.ndarray)) is True:
                            b_ret = [nlcpy.asarray(i) for i in x]
                        else:
                            b_ret = x
                    else:
                        b_ret = nlcpy.asarray(x)
                    vpp.append(b_ret)
                vp_ret = vpp
                if isinstance(ret, tuple):
                    vp_ret = tuple(vp_ret)
                return vp_ret
            else:
                return ret

    return wrap_func

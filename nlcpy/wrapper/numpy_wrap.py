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
# # Mersenne Twister License #
#
#   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions
#   are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. The names of its contributors may not be used to endorse or promote
#        products derived from this software without specific prior written
#        permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
import nlcpy
import functools
from nlcpy.logging import _vp_logging
from nlcpy import _environment

MT19937_A = 0x9908b0df
MT19937_N = 624
MT19937_M = 397


def twist_full(s, pos):
    t = (((s[:-1] & 0x80000000) | (s[1:] & 0x7fffffff)) >> 1) ^ ((s[1:] & 1) * MT19937_A)
    if pos < MT19937_N:
        m0 = min(pos, MT19937_N - MT19937_M)
        m1 = min(pos, 2 * (MT19937_N - MT19937_M))
        m2 = min(pos, MT19937_N - 1)
        s[:m0] = s[MT19937_M:m0 + MT19937_M] ^ t[:m0]
        s[m0:m1] = s[m0 + MT19937_M - MT19937_N:m1 + MT19937_M - MT19937_N] ^ t[m0:m1]
        s[m1:m2] = s[m1 + MT19937_M - MT19937_N:m2 + MT19937_M - MT19937_N] ^ t[m1:m2]
    else:
        m0 = MT19937_N - MT19937_M
        m1 = 2 * (MT19937_N - MT19937_M)
        m2 = MT19937_N - 1
        s[:m0] = s[MT19937_M:MT19937_N] ^ t[:m0]
        s[m0:m1] = s[:m0] ^ t[m0:2 * m0]
        s[m1:m2] = s[m0:MT19937_M - 1] ^ t[m1:m2]
        u = (((s[-1] & 0x80000000) | (s[0] & 0x7fffffff)) >> 1) ^ (s[0] & 1) * MT19937_A
        s[MT19937_N - 1] = s[MT19937_M - 1] ^ u
    return numpy.concatenate([s[pos:], s[:pos]])


def numpy_wrap(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedError as e:
            if not _environment._is_numpy_wrap_enabled():
                raise e
            f = getattr(numpy, func.__name__)
            return _make_wrap_func(f)(*args, **kwargs)
    return wrap_func


def _make_wrap_func(f):
    def wrap_func(*args, **kwargs):
        dict_idx_out = {
            'around': (2,),
            'choose': (2,),
            'compress': (3,),
            'cumprod': (3,),
            'divmod': (2, 3),
            'dot': (2,),
            'fix': (1,),
            'float_power': (2,),
            'frexp': (1, 2),
            'gcd': (2,),
            'isnat': (1,),
            'isneginf': (1,),
            'isposinf': (1,),
            'lcm': (2,),
            'matmul': (2,),
            'modf': (1, 2),
            'nancumprod': (3,),
            'nancumsum': (3,),
            'nanprod': (3,),
            'nansum': (3,),
            'round_': (2,),
            'take': (3,),
            'trace': (5,),
        }
        if not _environment._is_numpy_wrap_enabled():
            raise NotImplementedError("{} is not implemented yet.".format(f.__name__))
        if _vp_logging._is_enable(_vp_logging.NPWRAP):
            _vp_logging.info(_vp_logging.NPWRAP,
                             "%s is replaced by numpy's one", f.__name__)

        # retrieve input ndarrays of NLCPy from VE
        in_out = []
        np_args = list(args)
        np_kwargs = kwargs.copy()
        if f.__name__ in ('apply_along_axis', 'from_function'):
            pass
        elif f.__name__ == 'piecewise':
            if len(np_args) > 1:
                np_args[1] = list(np_args[1])
                condlist = np_args[1]
            else:
                np_kwargs['condlist'] = list(kwargs['condlist'])
                condlist = np_kwargs['condlist']
            for i in range(len(condlist)):
                if isinstance(condlist[i], nlcpy.ndarray):
                    condlist[i] = condlist[i].get()
        else:
            idx_out = dict_idx_out.get(f.__name__)
            if idx_out is None:
                idx_out = (-1,)
            for i, _l in enumerate(np_args):
                if isinstance(_l, nlcpy.ndarray):
                    if i in idx_out:
                        in_out.append(_l)
                    np_args[i] = _l.get()
                if type(_l) is tuple:
                    _l = list(_l)
                    np_args[i] = _l
                if type(_l) is list:
                    for ii in range(len(_l)):
                        if isinstance(_l[ii], nlcpy.ndarray):
                            np_args[i][ii] = _l[ii].get()
                    if type(args[i]) is tuple:
                        np_args[i] = tuple(np_args[i])
            for k, v in np_kwargs.items():
                if isinstance(v, nlcpy.ndarray):
                    np_kwargs[k] = v.get()
                    if k == 'out':
                        in_out.append(v)
                if type(v) is tuple:
                    v = list(v)
                if type(v) is list:
                    for ii in range(len(v)):
                        if isinstance(v[ii], nlcpy.ndarray):
                            if k == 'out':
                                in_out.append(v[ii])
                            v[ii] = v[ii].get()
                    if type(kwargs[k]) is tuple:
                        np_kwargs[k] = tuple(v)

        if f.__name__ == 'lookfor':
            if len(np_args) < 2 and np_kwargs.get('module') is None:
                np_kwargs['module'] = 'nlcpy'

        if f.__name__ == 'info':
            if len(np_args) < 4 and np_kwargs.get('toplevel') is None:
                np_kwargs['toplevel'] = 'nlcpy'

        is_random_method = hasattr(numpy.random, f.__name__)
        if is_random_method:
            nrs = numpy.random.get_state()
            vrs = list(nlcpy.random.get_state())
            numpy.random.set_state(('MT19937', vrs[0][:624], 0, *vrs[1:]))

        # call NumPy function
        ret = f(*np_args, **np_kwargs)

        if is_random_method:
            nrs2 = numpy.random.get_state()
            if nrs2[2] > 0:
                nrs2 = list(nrs2)
                nrs2[1] = twist_full(nrs2[1], nrs2[2])
            nlcpy.copyto(vrs[0][:nrs2[1].size], nrs2[1])
            numpy.random.set_state(nrs)
            vrs[0][-1] = 0
            vrs[1:] = nrs2[3:]
            nlcpy.random.set_state(vrs)

        # transfer the return values to VE
        docopy = False
        if f.__name__ in ('at', 'put', 'putmask'):
            a = kwargs.get('a') if len(args) == 0 else args[0]
            if type(a) is nlcpy.ndarray:
                docopy = True
                na = np_kwargs.get('a') if len(args) == 0 else np_args[0]
        elif f.__name__ in ('put_along_axis', 'place'):
            a = kwargs.get('arr') if len(args) == 0 else args[0]
            if type(a) is nlcpy.ndarray:
                docopy = True
                na = np_kwargs.get('arr') if len(args) == 0 else np_args[0]
        elif f.__name__ == 'nan_to_num':
            a = kwargs.get('x') if len(args) == 0 else args[0]
            if type(a) is nlcpy.ndarray:
                if len(args) > 1:
                    docopy = not args[1]
                elif 'copy' in kwargs.keys():
                    docopy = not kwargs['copy']
                na = np_kwargs.get('x') if len(args) == 0 else np_args[0]
        if docopy:
            a[...] = na
        if isinstance(ret, numpy.ndarray) or \
                numpy.isscalar(ret) and numpy.dtype(type(ret)).char in '?iIlLfFdD':
            vp_ret = nlcpy.asarray(ret)
            if len(in_out) > 0:
                in_out[0][...] = vp_ret
                vp_ret = in_out[0]
            return vp_ret
        elif isinstance(ret, numpy.lib.npyio.NpzFile):
            return nlcpy.NpzFile(ret)
        elif isinstance(ret, dict):
            for key, val in ret.items():
                ret[key] = nlcpy.asarray(val)
            return ret
        elif isinstance(ret, (list, tuple)):
            vpp = []
            for i, x in enumerate(ret):
                if isinstance(x, numpy.ndarray) is False:
                    if numpy.any(isinstance(x, numpy.ndarray)) is True:
                        b_ret = [nlcpy.asarray(i) for i in x]
                    else:
                        b_ret = x
                else:
                    b_ret = nlcpy.asarray(x)
                    if len(in_out) > i:
                        in_out[i][...] = b_ret
                        b_ret = in_out[i]
                vpp.append(b_ret)
            vp_ret = vpp
            if isinstance(ret, tuple):
                vp_ret = tuple(vp_ret)
            return vp_ret
        else:
            return ret
    return wrap_func


def _make_wrap_method(f, arr):
    def wrap_method(*args, **kwargs):
        dict_idx_out = {
            'compress': 2,
            'cumprod': 2,
            'take': 2,
            'trace': 4,
            'round': 1,
        }
        if not _environment._is_numpy_wrap_enabled():
            raise NotImplementedError(
                "{}.{} is not implemented yet.".format(type(arr).__name__, f.__name__))
        if _vp_logging._is_enable(_vp_logging.NPWRAP):
            _vp_logging.info(_vp_logging.NPWRAP,
                             "%s.%s is replaced by numpy's one",
                             type(arr).__name__, f.__name__)

        # retrieve input ndarrays of NLCPy from VE
        in_out = None
        idx_out = dict_idx_out.get(f.__name__)
        if idx_out is None:
            idx_out = -1
        np_args = list(args)
        np_kwargs = kwargs.copy()
        for i, _l in enumerate(np_args):
            if isinstance(_l, nlcpy.ndarray):
                np_args[i] = _l.get()
            if type(_l) is tuple:
                _l = list(_l)
                np_args[i] = _l
            if type(_l) is list:
                for ii in range(len(_l)):
                    if isinstance(_l[ii], nlcpy.ndarray):
                        np_args[i][ii] = _l[ii].get()
                if type(args[i]) is tuple:
                    np_args[i] = tuple(np_args[i])
            if i == idx_out and isinstance(_l, nlcpy.ndarray):
                in_out = _l
        for k, v in np_kwargs.items():
            if isinstance(v, nlcpy.ndarray):
                np_kwargs[k] = v.get()
            if type(v) is tuple:
                v = list(v)
                np_kwargs[k] = v
            if type(v) is list:
                for ii in range(len(v)):
                    if isinstance(v[ii], nlcpy.ndarray):
                        np_kwargs[k][ii] = v[ii].get()
                if type(kwargs[k]) is tuple:
                    np_kwargs[k] = tuple(np_kwargs[k])
            if k == 'out' and isinstance(v, nlcpy.ndarray):
                in_out = v
        np_arr = arr.get()

        # call NumPy function
        ret = f(np_arr, *np_args, **np_kwargs)

        # transfer the return values to VE
        if f.__name__ in ('itemset', 'partition', 'put', 'setfield', 'sort') or \
           ret is np_arr:
            arr[...] = np_arr
            ret = arr
        if isinstance(ret, numpy.ndarray) or \
                numpy.isscalar(ret) and numpy.dtype(type(ret)).char in '?iIlLfFdD':
            vp_ret = nlcpy.asarray(ret)
            if in_out is not None:
                in_out[...] = vp_ret
                vp_ret = in_out
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
    return wrap_method

# distutils: language = c++
#
# * The source code in this file is based on the soure code of NumPy.
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
import numpy
import numbers
import warnings
import copy
import nlcpy
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport internal
from nlcpy.core.core cimport MemoryLocation
from nlcpy.core cimport core
from nlcpy.core cimport manipulation
from nlcpy.core cimport broadcast
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.core cimport dtype as _dtype
from nlcpy import veo
from nlcpy.manipulation.shape import reshape
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.wrapper.numpy_wrap import numpy_wrap
cimport cython
cimport cpython

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


cpdef nlcpy_wrapit(obj, shape):
    a = obj.ravel()
    result = a.reshape(shape)

    return result

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


cpdef nlcpy_hatmask(var):
    b = nlcpy.where(var, False, True)
    return b

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


cpdef nlcpy_replace_nan(a, val):
    a = nlcpy.asanyarray(a)

    if a.dtype == numpy.object_:
        mask = nlcpy.not_equal(a, a, dtype=bool)
    elif issubclass(a.dtype.type, nlcpy.inexact):
        mask = nlcpy.isnan(a)
    else:
        mask = None

    if mask is not None:
        a = nlcpy.array(a)
        nlcpy_copyto(a, val, mask)

    return a, mask

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


cpdef nlcpy_copyto(a, val, mask):
    if isinstance(a, nlcpy.core.core.ndarray):
        a[mask] = val
    else:
        a = a.dtype.type(val)
    return a

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


cpdef nlcpy_nan_mask(a, out=None):
    if a.dtype.kind not in 'fc':
        return True

    y = nlcpy.isnan(a, out=out)
    y = nlcpy.invert(y, out=y)
    return y

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


cpdef nlcpy_divide_by_count(a, b, out=None):
    try:
        if isinstance(a, nlcpy.core.core.ndarray):
            if out is None:
                return nlcpy.divide(a, b, out=a, casting='unsafe')
            else:
                return nlcpy.divide(a, b, out=out, casting='unsafe')
        else:
            if out is None:
                return a.dtype.type(a / b)
            else:
                return nlcpy.divide(a, b, out=out, casting='unsafe')
    except Exception as e:
        pass


# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


cpdef nlcpy_sq(axis, a):
    if axis is not None:
        ns = []
        for x in range(len(a.shape)):
            if x != axis:
                ns.append(a.shape[x])
        ans = nlcpy.reshape(a, tuple(ns))
    else:
        ans = a

    return ans

# ----------------------------------------------------------------------------
# local function
# ---------------------------------------------------------------------------


cpdef nlcpy_chk_type(a):
    if not isinstance(a, nlcpy.core.core.ndarray):
        return True

    if a is not None and a.dtype.kind in ('b', 'c', 'v'):
        raise NotImplementedError("dtype={} not supported".format(a.dtype))

    return True

# ----------------------------------------------------------------------------
# local function
# ---------------------------------------------------------------------------


cpdef nlcpy_countnan(a):
    a = nlcpy.asanyarray(a)
    ans = a[nlcpy.isnan(a)]
    return ans.size


# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


cpdef inner_convmeandata(data, shape, axis=None):
    save_shape = shape

    if not isinstance(data, nlcpy.core.core.ndarray):
        data = nlcpy.array(data)

    if data.ndim == 0:
        b_d = data.reshape(1, -1)
        ans = nlcpy.tile(b_d, save_shape)
    else:
        l_in = list(save_shape)
        l_x = list(save_shape)

        for x in range(len(l_in)):
            if x != axis:
                l_in[x] = 1

        for x in range(len(l_x)):
            if x == axis:
                l_x[x] = -1

        t = tuple(l_in)
        t_l = tuple(l_x)
        b_d = data.reshape(t_l)
        ans = nlcpy.tile(b_d, t)

    return ans

# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


cpdef nlcpy_median_nancheck(data, result, axis):
    if data.size == 0:
        return result

    n = nlcpy.isnan(data[..., -1])
    b = n.ravel()
    if n is True:
        result = data.dtype.type(numpy.nan)
    elif numpy.count_nonzero(b.get()) > 0:
        result[n] = nlcpy.nan

    return result


# ----------------------------------------------------------------------------
# local function
# ----------------------------------------------------------------------------


cpdef nlcpy_chk_axis(a, axis=None):
    ret = False

    if isinstance(a, nlcpy.core.core.ndarray) is True:
        if axis is None:
            ret = True
        else:
            if type(axis) is tuple:
                raise ValueError('tuple axis is not supported')
            elif axis >= a.ndim:
                raise ValueError("Nlcpy AxisError: axis {} is out of bounds"
                                 " for array of dimension {}".format(axis, a.ndim))
            else:
                ret = True
    else:
        a = core.argument_conversion(a)
        if axis is None:
            ret = True
        elif type(axis) is tuple:
            raise ValueError('tuple axis is not supported')
        else:
            if axis >= a.ndim:
                raise ValueError("Nlcpy AxisError: axis {} is out of bounds"
                                 " for array of dimension {}".format(axis, a.ndim))
            else:
                ret = True

    return ret

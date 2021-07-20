#
# * The source code in this file is based on the soure code of NumPy and CuPy.
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

from nlcpy.fft.libfft cimport *

import nlcpy
import numpy
import numbers
import operator
import itertools
import time
from nlcpy import veo, conjugate, empty, arange
from nlcpy.request import request
from nlcpy.request.ve_kernel cimport check_error
from numpy.compat import integer_types
from functools import reduce
from math import sqrt
from collections.abc import Sequence
integer_types = integer_types + (numpy.integer,)

_handle_info = None


def _set_handle_info(fn, size):
    global _handle_info
    _handle_info = (fn, size)


def _get_complex_ndarray(a):
    if isinstance(a, nlcpy.core.core.ndarray):
        key = numpy.dtype(a.dtype).name
        if key in ['float32']:
            a = nlcpy.asarray(a, dtype='complex64')
        elif key not in ['complex128', 'complex64']:
            a = nlcpy.asarray(a, dtype=complex)
        else:
            if not a._c_contiguous and not a._f_contiguous:
                a = nlcpy.array(a, dtype=key)
        return a
    else:
        if isinstance(a, numpy.ndarray):
            key = numpy.dtype(a.dtype).name
            if key in ['complex128', 'complex64']:
                a = nlcpy.array(a, dtype=a.dtype.name)
            elif key in ['float32']:
                a = nlcpy.array(a, dtype='complex64')
            else:
                a = nlcpy.array(a, dtype=complex)
            return a
        else:
            a = _ndarray_cast_helper(a, dtype=complex)
            if a is None:
                raise ValueError("First argument must be a complex or "
                                 "real sequence of single or double precision")
            return a


def _get_complex_ndarray_sub(a):
    if isinstance(a, nlcpy.core.core.ndarray):
        key = numpy.dtype(a.dtype).name
        if key not in ['complex128', 'complex64']:
            a = nlcpy.asarray(a, dtype=complex)
        else:
            if not a._c_contiguous and not a._f_contiguous:
                a = nlcpy.array(a, dtype=key)
        return a
    else:
        if isinstance(a, numpy.ndarray):
            key = numpy.dtype(a.dtype).name
            if key in ['complex128', 'complex64']:
                a = nlcpy.array(a, dtype=a.dtype.name)
            else:
                a = nlcpy.array(a, dtype=complex)
            return a
        else:
            a = _ndarray_cast_helper(a, dtype=complex)
            if a is None:
                raise ValueError("First argument must be a complex or "
                                 "real sequence of single or double precision")
            return a


def _get_real_ndarray(a):
    if isinstance(a, nlcpy.core.core.ndarray):
        key = numpy.dtype(a.dtype).name
        if key in ['complex64', 'complex128']:
            raise TypeError("1st argument must be a real sequence 1")
        else:
            if key not in ['float32', 'float64']:
                a = nlcpy.asarray(a, dtype=float)
            else:
                if not a._c_contiguous and not a._f_contiguous:
                    a = nlcpy.array(a, dtype=key)
        return a
    else:
        if isinstance(a, numpy.ndarray):
            key = numpy.dtype(a.dtype).name
            if key in ['complex128', 'complex64']:
                raise TypeError("1st argument must be a real sequence 1")
            elif key not in ['float32', 'float64']:
                a = nlcpy.asarray(a, dtype=float)
            else:
                a = nlcpy.asarray(a, dtype=key)
        else:
            a = _ndarray_cast_helper(a, dtype=float)
            if a is None:
                raise ValueError("1st argument must be a real sequence 2")
            key = nlcpy.dtype(a.dtype).name
            if key in ['complex64', 'complex128']:
                raise TypeError("1st argument must be a real sequence 1")
        return a


def _ndarray_cast_helper(a, dtype):
    try:
        return nlcpy.asarray(a, dtype=dtype)
    except Exception:
        return None


def _int_cast_helper(value):
    try:
        _v = int(value)
        return (_v, True)
    except Exception:
        return (value, False)


def _is_int_cast_for_iterable(param):
    for p in param:
        if not _int_cast_helper(p)[1]:
            return False
    return True


def _is_duplicated(param):
    before = len(param)
    _set = set(param)
    return (before != len(_set))


def _get_params_1d(a, n, axis, array_func=_get_complex_ndarray):
    if isinstance(axis, tuple):
        _len = len(axis)
        if _len > 1:
            raise TypeError("not all arguments converted during string formatting")
        elif _len == 0:
            raise TypeError("not enough arguments for format string")
        else:
            (axis, _valid) = _int_cast_helper(axis[0])
            if _valid:
                raise ValueError("Invalid axis (%d) specified."% axis)
            else:
                raise TypeError('%d format: a number is required, not {}'.format(
                                type(axis).__name__))

    (axis, _is_axis_cast) = _int_cast_helper(axis)
    if not _is_axis_cast:
        raise TypeError('%d format: a number is required, not {}'.format(
                        type(axis).__name__))

    a = array_func(a)

    if a is None or axis < -a.ndim or axis >= a.ndim:
        raise ValueError("Invalid axis (%d) specified."% axis)

    if axis < 0:
        axis += a.ndim

    if n is not None:
        if isinstance(n, (list, tuple)):
            raise TypeError('an integer is required (got type {})'.format(
                            type(n).__name__))
        (n, _is_n_cast) = _int_cast_helper(n)
        if not _is_n_cast:
            raise TypeError("can't convert {} to int".format(type(n).__name__))

    if (n is not None and n < 1) or (n is None and a.shape[axis] < 1):
        raise ValueError("Dimension n should be a positive integer "
                         "not larger than the shape of the array along "
                         "the chosen axis")

    return (a, n, axis)


def _get_params_nd(a, s, axes, array_func=_get_complex_ndarray, invreal=False):
    a = array_func(a)
    if a is None or\
       axes is not None and any((axis < -a.ndim or axis >= a.ndim) for axis in axes):
        raise ValueError("axes exceeds dimensionality of input")

    if s is None:
        shapeless = 1
        if axes is None:
            s = list(a.shape)
        else:
            if isinstance(axes, (list, tuple)):
                if not _is_int_cast_for_iterable(axes):
                    raise TypeError("tuple indices must be integers or slices, not str")
            else:
                raise TypeError('{} object is not iterable'.format(
                    type(axes).__name__))
            s = numpy.take(a.shape, axes)
    else:
        shapeless = 0
        if isinstance(s, (list, tuple)):
            if (axes is not None and len(axes) != len(s)):
                raise ValueError("Shape and axes have different lengths.")
            if not _is_int_cast_for_iterable(s):
                raise ValueError("when given, shape values must be integers")
            if any((si <= 0) for si in s):
                raise ValueError("Shape and axes have different lengths.")
        else:
            raise TypeError('{} object is not iterable'.format(
                type(axes).__name__))

    s = list(s)
    if axes is None:
        axes = tuple(range(0, len(s)))
    else:
        if isinstance(axes, (list, tuple)):
            if not _is_int_cast_for_iterable(axes):
                raise TypeError("tuple indices must be integers or slices, not str")
            if len(axes) != len(s) or\
               any((axis < -a.ndim or axis >= a.ndim) for axis in axes):
                raise ValueError("Shape and axes have different lengths.")
            if len(axes) > 0:
                axes = tuple(map(lambda x: x + a.ndim if x < 0 else x, axes))
        else:
            raise TypeError('{} object is not iterable'.format(
                            type(axes).__name__))

    if len(axes) != len(s):
        raise ValueError("Shape and axes have different lengths.")

    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2

    return (a, s, axes)


def _unitary(norm):
    if norm is None:
        return False
    if norm=="ortho":
        return True
    raise ValueError("Invalid norm value %s, should be None or \"ortho\"."
                     % norm)


def _generate_fn_key(dt, is_real, inv, dim_key):
    if is_real:
        if inv:
            return ("nlcpy_irfft_" + dim_key + "_c128_f64", "rfft_d_" + dim_key)\
                if dt.name == 'complex128'\
                else ("nlcpy_irfft_" + dim_key + "_c64_f32", "rfft_s_" + dim_key)
        else:
            return ("nlcpy_rfft_" + dim_key + "_f64_c128", "rfft_d_" + dim_key)\
                if dt.name == 'float64'\
                else ("nlcpy_rfft_" + dim_key + "_f32_c64", "rfft_s_" + dim_key)
    else:
        if inv:
            return ("nlcpy_ifft_" + dim_key + "_c128_c128", "fft_d_" + dim_key)\
                if dt.name == 'complex128'\
                else ("nlcpy_ifft_" + dim_key + "_c64_c64", "fft_s_" + dim_key)
        else:
            return ("nlcpy_fft_" + dim_key + "_c128_c128", "fft_d_" + dim_key)\
                if dt.name == 'complex128'\
                else ("nlcpy_fft_" + dim_key + "_c64_c64", "fft_s_" + dim_key)


def _raw_fft(a, n, axis, inv_norm, is_real=False, inv=False):
    _order = "F" if not a._c_contiguous and a._f_contiguous else "C"

    if is_real:
        if inv:
            nin = int(float(n / 2) + 1)
            nout = n
        else:
            nin = n
            nout = int(float(n / 2) + 1)
    else:
        nin = n
        nout = n

    # create out param
    t = a
    if a.shape[axis] != nin:
        s = list(a.shape)
        if s[axis] < nin:
            index = [slice(None)]*len(s)
            index[axis] = slice(0, s[axis])
            s[axis] = nin
            # zero padding
            z = nlcpy.zeros(s, a.dtype, order=_order)
            z[tuple(index)] = a
            a = nlcpy.asarray(z, order=_order)
        s[axis] = nout
        t = numpy.empty(s, dtype=a.dtype, order=_order)

    if is_real:
        _s = list(t.shape)
        _s[axis] = nout
        if inv:
            (out_shape, out_dtype) =\
                (tuple(_s), 'float32' if t.dtype.name == 'complex64' else 'float64')
        else:
            (out_shape, out_dtype) =\
                (tuple(_s), 'complex64' if t.dtype.name == 'float32' else 'complex128')
    else:
        (out_shape, out_dtype) = (t.shape, t.dtype)

    out = nlcpy.ndarray(shape=out_shape, dtype=out_dtype, order=_order)

    dt = numpy.dtype(a.dtype)
    fn, reuse_key = _generate_fn_key(dt, is_real, inv, "1d")
    is_reuse = (_handle_info is not None and (reuse_key, n) == _handle_info)

    fpe = request._get_fpe_flag()
    args = (a._ve_array,
            out._ve_array,
            axis,
            <int>n,
            <int>(1 if is_reuse else 0),
            veo.OnStack(fpe, inout=veo.INTENT_OUT))

    request._push_and_flush_request(
        fn,
        args,
    )

    _set_handle_info(reuse_key, n)

    if inv_norm != 1:
        nlcpy.multiply(out, (1/inv_norm), out=out)
        return out

    return out


def _raw_fft_nd(a, s, axes, inv_norm, is_real=False, inv=False):
    _order = "F" if not a._c_contiguous and a._f_contiguous else "C"

    nin = list(s)
    nout = list(s)
    if is_real and not inv:
        nout[-1] = int(float(nout[-1] / 2) + 1)

    # in-out shaping
    t = a
    out_shape = list(a.shape)
    in_shape = list(a.shape)
    index = [slice(None)]*len(in_shape)
    is_expand = False
    for ii in range(0, len(axes)):
        _axis = axes[ii]
        if a._shape[_axis] != nin[ii]:
            in_shape[_axis] = nin[ii]
            if a._shape[_axis] < nin[ii]:
                is_expand = True
                index[_axis] = slice(0, a._shape[_axis])
            else:
                index[_axis] = slice(0, in_shape[_axis])
        out_shape[_axis] = nout[ii]

    # zero padding (and cropping if needed)
    if is_expand:
        z = nlcpy.zeros(in_shape, a.dtype, order=_order)
        z[index] = a[index]
        a = z
    t = numpy.empty(out_shape, dtype=a.dtype, order=_order)

    _axes = nlcpy.array(axes, dtype=int)
    _size = tuple(map(lambda axis: in_shape[axis], axes))
    s_in = nlcpy.array(_size, dtype=int)

    if is_real:
        _s = list(t.shape)
        if inv:
            (out_shape, out_dtype) =\
                (tuple(_s), 'float32' if t.dtype.name == 'complex64' else 'float64')
        else:
            (out_shape, out_dtype) =\
                (tuple(_s), 'complex64' if t.dtype.name == 'float32' else 'complex128')
    else:
        (out_shape, out_dtype) = (t.shape, t.dtype)
    out = nlcpy.ndarray(shape=out_shape, dtype=out_dtype, order=_order)

    fft_dim = len(axes)
    if fft_dim == 2:
        dim_key = "2d"
    elif fft_dim == 3:
        dim_key = "3d"
    else:
        dim_key = "nd"

    dt = numpy.dtype(a.dtype)
    fn, reuse_key = _generate_fn_key(dt, is_real, inv, dim_key)
    is_reuse = (_handle_info is not None and (reuse_key, _size) == _handle_info)

    fpe = request._get_fpe_flag()
    args = (a._ve_array,
            out._ve_array,
            _axes._ve_array,
            s_in._ve_array,
            <int>(1 if is_reuse else 0),
            veo.OnStack(fpe, inout=veo.INTENT_OUT))

    request._push_and_flush_request(
        fn,
        args,
    )

    _set_handle_info(reuse_key, _size)

    if inv_norm != 1:
        nlcpy.multiply(out, (1/inv_norm), out=out)
        return out
    return out


def fft(a, n=None, axis=-1, norm=None):
    (a, n, axis) = _get_params_1d(a, n, axis)

    if n is None:
        n = a.shape[axis]
    inv_norm = 1
    if norm is not None and _unitary(norm):
        inv_norm = sqrt(n)
    return _raw_fft(a, n=n, axis=axis, inv_norm=inv_norm)


def ifft(a, n=None, axis=-1, norm=None):
    (a, n, axis) = _get_params_1d(a, n, axis)

    if n is None:
        n = a.shape[axis]
    if norm is not None and _unitary(norm):
        inv_norm = sqrt(max(n, 1))
    else:
        inv_norm = n
    return _raw_fft(a, n=n, axis=axis, inv_norm=inv_norm, inv=True)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    return fftn(a, s, axes, norm)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    return ifftn(a, s, axes, norm)


def fftn(a, s=None, axes=None, norm=None):
    if axes is not None and isinstance(axes, (list, tuple)):
        if len(tuple(axes)) == 0:
            return a

    (a, s, axes) = _get_params_nd(a, s, axes)

    if len(axes) == 1:
        return fft(a, n=s[0], axis=axes[0], norm=norm)
    else:
        inv_norm=1
        if norm is not None and _unitary(norm):
            inv_norm = reduce(lambda x, y: x*y, list(map(lambda _s: sqrt(_s), s)))
        return _raw_fft_nd(a, s, axes, inv_norm)


def ifftn(a, s=None, axes=None, norm=None):
    if axes is not None and isinstance(axes, (list, tuple)):
        if len(tuple(axes)) == 0:
            return a

    (a, s, axes) = _get_params_nd(a, s, axes)

    if len(axes) == 1:
        return ifft(a, n=s[0], axis=axes[0], norm=norm)
    else:
        if norm is not None and _unitary(norm):
            inv_norm = reduce(lambda x, y: x*y,
                              list(map(lambda _s: sqrt(max(_s, 1)), s)))
        else:
            inv_norm = reduce(lambda x, y: x*y, s)
        return _raw_fft_nd(a, s, axes, inv_norm, inv=True)


def rfft(a, n=None, axis=-1, norm=None):
    (a, n, axis) = _get_params_1d(a, n, axis, _get_real_ndarray)

    if n is None:
        n = a.shape[axis]
    inv_norm = 1
    if norm is not None and _unitary(norm):
        inv_norm = sqrt(n)
    return _raw_fft(a, n=n, axis=axis, inv_norm=inv_norm, is_real=True)


def irfft(a, n=None, axis=-1, norm=None):
    (a, n, axis) = _get_params_1d(a, n, axis, _get_complex_ndarray_sub)

    if n is None:
        n = (a.shape[axis] - 1) * 2
    inv_norm = n

    if norm is not None and _unitary(norm):
        inv_norm = sqrt(n)
    return _raw_fft(a, n=n, axis=axis,
                    inv_norm=inv_norm, is_real=True, inv=True)


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    return rfftn(a, s, axes, norm)


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    return irfftn(a, s, axes, norm)


def rfftn(a, s=None, axes=None, norm=None):
    if axes is not None and isinstance(axes, (list, tuple)):
        if len(tuple(axes)) == 0:
            return a
    (a, s, axes) = _get_params_nd(a, s, axes, _get_real_ndarray)

    if len(axes) == 1:
        return rfft(a, n=s[0], axis=axes[0], norm=norm)
    else:
        inv_norm=1
        if norm is not None and _unitary(norm):
            inv_norm = reduce(lambda x, y: x*y, list(map(lambda _s: sqrt(_s), s)))
        return _raw_fft_nd(a, s, axes, inv_norm, is_real=True)


def irfftn(a, s=None, axes=None, norm=None):
    if axes is not None and isinstance(axes, (list, tuple)):
        if len(tuple(axes)) == 0:
            return a
    (a, s, axes) = _get_params_nd(a, s, axes, _get_complex_ndarray_sub,
                                  invreal=True)
    if len(axes) == 1:
        return irfft(a, n=s[0], axis=axes[0], norm=norm)
    else:
        if norm is not None and _unitary(norm):
            inv_norm = reduce(lambda x, y: x*y,
                              list(map(lambda _s: sqrt(max(_s, 1)), s)))
        else:
            inv_norm = reduce(lambda x, y: x*y, s)
        return _raw_fft_nd(a, s, axes, inv_norm, inv=True, is_real=True)


def hfft(a, n=None, axis=-1, norm=None):
    (a, n, axis) = _get_params_1d(a, n, axis)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    unitary = _unitary(norm)
    return irfft(conjugate(a), n, axis) * (sqrt(n) if unitary else n)


def ihfft(a, n=None, axis=-1, norm=None):
    (a, n, axis) = _get_params_1d(a, n, axis, array_func=_get_real_ndarray)
    if n is None:
        n = a.shape[axis]
    unitary = _unitary(norm)
    output = conjugate(rfft(a, n, axis))
    return output * (1 / (sqrt(n) if unitary else n))


def fftfreq(n, d=1.0):
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    results = empty(n, int)
    N = (n-1)//2 + 1
    p1 = arange(0, N, dtype=int)
    results[:N] = p1
    p2 = arange(-(n//2), 0, dtype=int)
    results[N:] = p2
    return results * val


def rfftfreq(n, d=1.0):
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0/(n*d)
    N = n//2 + 1
    results = arange(0, N, dtype=int)
    return results * val


def fftshift(x, axes=None):
    x = numpy.asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, integer_types):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]

    return _nlcpy_roll(x, shift, axes)


def ifftshift(x, axes=None):
    x = numpy.asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, integer_types):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    return _nlcpy_roll(x, shift, axes)


def _nlcpy_roll(a, shift, axis=None):
    if axis is None:
        return _nlcpy_roll(a.ravel(), shift, 0).reshape(a.shape)
    else:
        axis = _nlcpy_normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        broadcasted, o_shape = _nlcpy_broadcast(shift, axis)

        shifts = {ax: 0 for ax in range(a.ndim)}
        i=0
        for mix in list(broadcasted):
            pal = mix[0]
            dat = mix[1]
            shifts[i] += pal
            i += 1

        rolls = [((slice(None), slice(None)),)] * a.ndim
        for ax, offset in shifts.items():
            offset %= a.shape[ax] or 1
            if offset:
                rolls[ax] = ((slice(None, -offset), slice(offset, None)),
                             (slice(-offset, None), slice(None, offset)))

        result = nlcpy.empty_like(a)
        for indices in itertools.product(*rolls):
            arr_index, res_index = zip(*indices)
            result[res_index] = a[arr_index]

        return result


def _nlcpy_normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    axis = tuple([_nlcpy_normalize_axis_index(ax, ndim, argname) for ax in axis])
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError('repeated axis in `{}` argument'.format(argname))
        else:
            raise ValueError('repeated axis')
    return axis


def _nlcpy_broadcast(shift, axis):
    out_shape = _nlcpy_broadcast_shape(shift, axis)

    bd_shift = _nlcpy_broadcast_to(nlcpy.array(shift), out_shape)
    bd_axis = _nlcpy_broadcast_to(nlcpy.array(axis), out_shape)

    ret = []
    for x, y in zip(bd_shift, bd_axis):
        c_x = x.astype(x.dtype)
        c_y = y.astype(y.dtype)

        ret.append((c_x, c_y))
    return ret, out_shape


def _nlcpy_broadcast_shape(shift, axis):
    sf = _nlcpy_cnv_tuple(shift)
    ax = _nlcpy_cnv_tuple(axis)
    cal_shape = _nlcpy_cal_shape(sf, ax)

    return cal_shape


def _nlcpy_cnv_tuple(lst, shape=()):
    if not isinstance(lst, Sequence):
        return shape

    if isinstance(lst[0], Sequence):
        _l = len(lst[0])
        if not all(len(item) == _l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )
    shape = _nlcpy_cnv_tuple(lst[0], shape)

    return shape


def _nlcpy_broadcast_to(array, shape, subok=False):
    return nlcpy.broadcast_to(array, shape, subok)


def _nlcpy_normalize_axis_index(axis, ndim, msg_prefix=None):
    ret = 0
    if axis<0:
        ret += ndim
    else:
        ret = axis

    return ret


def _nlcpy_is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def _nlcpy_cal_shape(shp1, shp2):
    ret = []
    if _nlcpy_is_broadcastable(shp1, shp2):
        if len(shp1) != len(shp2):
            one, tmp = _nlcpy_sort_order(shp1, shp2)
            l_tmp = list(tmp)
            l_tmp.insert(0, 1)
            two = tuple(l_tmp)
        else:
            one, two = shp1, shp2

        ans = []
        for a, b in zip(one, two):
            if a<b:
                ans.append(b)
            else:
                ans.append(a)

        ret = ans

    return tuple(ret)


def _nlcpy_sort_order(shp1, shp2):
    if len(shp1) < len(shp2):
        return shp2, shp1

    return shp1, shp2

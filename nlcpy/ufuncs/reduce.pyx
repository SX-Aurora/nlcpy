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

# distutils: language = c++
import numpy
import nlcpy
import numbers
import warnings
import ctypes
import sys

import nlcpy
from nlcpy import veo
from nlcpy.core cimport core
from nlcpy.core cimport broadcast
from nlcpy.core.core cimport *
from nlcpy.core.dtype cimport get_dtype
from nlcpy.manipulation.shape import reshape
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.request cimport request

cimport numpy as cnp

cdef get_minus_infinity(dtype, op_name, is_identity=False):
    if dtype == 'int32':
        return numpy.array(-2147483648, dtype=dtype)
    elif dtype == 'int64':
        return numpy.array(-9223372036854775808, dtype=dtype)
    elif dtype == 'uint32':
        return numpy.array(0, dtype=dtype)
    elif dtype == 'uint64':
        if is_identity and op_name != 'logaddexp2':
            return numpy.array(0, dtype=dtype)
        else:
            return numpy.array(9223372036854775808, dtype=dtype)
    elif dtype == 'bool':
        if op_name in ('maximum', 'fmax'):
            return numpy.array(False, dtype=dtype)
        else:
            return numpy.array(True, dtype=dtype)
    return numpy.array(-nlcpy.inf, dtype=dtype)

cdef get_plus_infinity(dtype, op_name, is_identity=False):
    if dtype == 'int32':
        if is_identity:
            return numpy.array(2147483647, dtype=dtype)
        else:
            return numpy.array(-2147483648, dtype=dtype)
    elif dtype == 'int64':
        if is_identity:
            return numpy.array(9223372036854775807, dtype=dtype)
        else:
            return numpy.array(-9223372036854775808, dtype=dtype)
    elif dtype == 'uint32':
        if is_identity:
            return numpy.array(4294967295, dtype=dtype)
        else:
            return numpy.array(0, dtype=dtype)
    elif dtype == 'uint64':
        if is_identity:
            return numpy.array(18446744073709551615, dtype=dtype)
        else:
            return numpy.array(0, dtype=dtype)
    elif dtype in ('bool',):
        return numpy.array(True, dtype=dtype)
    return numpy.array(nlcpy.inf, dtype=dtype)

cpdef reduce_core(name, a, axis=None, dtype=None, out=None, keepdims=False,
                  initial=nlcpy._NoValue, where=True):

    if a is not None:
        a = nlcpy.asarray(a)
    if a._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError('reduce on VH is not yet implemented.')

    axis_save = axis
    if isinstance(axis, int):
        axis = (axis,)
    elif isinstance(axis, numpy.ndarray) or isinstance(axis, nlcpy.ndarray):
        axis = (int(axis),)
    elif axis is not None:
        axis = [int(ax) if isinstance(ax, numpy.ndarray) or
                isinstance(ax, nlcpy.ndarray) else ax for ax in axis]
        axis = tuple(axis)

    op_name = (name.replace('nlcpy_', '')).replace('_reduce', '')

    if dtype is not None and out is not None and \
            (axis is None or len(axis) > 1) and \
            (a.dtype != dtype or a.dtype != out.dtype):
        ret = getattr(numpy, op_name).reduce(
            a, axis=axis, dtype=dtype, out=out.get(), keepdims=keepdims,
            initial=initial, where=where)
        nlcpy.copyto(out, ret)
        return out

    nlcpy_identity = {'add': 0, 'multiply': 1,
                      'logical_and': True, 'logical_or': False, 'logical_xor': False,
                      'bitwise_and': -1, 'bitwise_or': 0, 'bitwise_xor': 0,
                      'maximum': -nlcpy.inf, 'fmax': -nlcpy.inf,
                      'minimum': nlcpy.inf, 'fmin': nlcpy.inf,
                      'hypot': 0, 'logaddexp': -nlcpy.inf, 'logaddexp2': -nlcpy.inf}

    if dtype is not None:
        if len(nlcpy.dtype(dtype)) > 0:
            raise TypeError('cannot perform reduce with flexible type')
        dtype = get_dtype(dtype)

    if op_name == 'subtract':
        _msg = "numpy boolean subtract, the `-` operator, is deprecated," \
               " use the bitwise_xor, the `^` operator," \
               " or the logical_xor function instead."
        if dtype is not None and dtype.char == '?':
            raise TypeError(_msg)
        elif dtype is None and out is not None and out.dtype.char == '?':
            raise TypeError(_msg)
        elif dtype is None and out is None and a.dtype.char == '?':
            raise TypeError(_msg)

    op_has_identity = False
    op_is_reordable = False
    op_is_boolean = False
    if op_name in nlcpy_identity:
        op_is_reordable = True
        if op_name not in ('maximum', 'minimum', 'fmax', 'fmin'):
            op_has_identity = True
    if op_name in ('less', 'less_equal', 'greater', 'greater_equal',
                   'equal', 'not_equal', 'logical_and',
                   'logical_or', 'logical_xor'):
        op_is_boolean = True

    if axis is None or len(axis) > 1:
        if not op_is_reordable:
            raise ValueError("reduction operation '{}' is not reorderable,"
                             " so at most one axis may be specified".format(op_name))

    if type(where) is tuple:
        where = where[0]
    if axis_save is not None and len(axis) < a.ndim:
        axis = [ax + a.ndim if ax < 0 else ax for ax in axis]
        raveled = False
    else:
        axis = (0,)
        if hasattr(where, '__iter__'):
            where = nlcpy.asanyarray(where, dtype=bool)
            if a.shape != where.shape:
                raise NotImplementedError(
                    "broadcast of 'where' is not implemented yet.")
            where = where.ravel()
        shape_out = (1,) * a.ndim if keepdims else ()
        a = a.ravel()
        raveled = True

    if where is not True and (initial is None or
                              not op_has_identity and initial is nlcpy._NoValue):
        raise ValueError("reduction operation '{}' does not have an identity,"
                         " so to use a where mask one has to specify "
                         "'initial'".format(op_name))

    if out is not None:
        if out._memloc in {on_VH, on_VE_VH}:
            raise NotImplementedError('reduce on VH is not yet implemented.')

    if type(keepdims) is not int and (numpy.asarray(keepdims).dtype.char not in '?iIlL'):
        raise TypeError("an integer is required (got type " +
                        type(keepdims).__name__ + ")")
    if numpy.asarray(keepdims).size != 1:
        raise TypeError("only size-1 arrays can be converted to Python scalars")

    if (initial is None or (initial is nlcpy._NoValue and not op_has_identity)) and \
       a.ndim > 0 and any([a.shape[i] == 0 for i in axis]):
        _op_name = 'true_divide' if op_name == 'divide' \
            else 'remainder' if op_name == 'mod' else op_name
        raise ValueError("zero-size array to reduction operation "
                         + _op_name + " which has no identity")

    if keepdims == 0:
        keepdims = False
    else:
        keepdims = True
    if not raveled:
        if keepdims:
            if axis_save is None:
                if out is not None:
                    raise ValueError("output parameter for reduction operation "
                                     + op_name + " has too many dimensions")
                lst = [1] * a.ndim
            else:
                lst = list(a.shape)
                if a.ndim > 0:
                    for axis_i in axis:
                        lst[axis_i] = 1
        else:
            if axis_save is None:
                if out is not None and out.ndim > 0:
                    raise ValueError("output parameter for reduction operation "
                                     + op_name + " has too many dimensions")
                lst=[]
            else:
                lst = list(a.shape)
                if a.ndim > 0:
                    axis_desc = sorted(axis, reverse=True)
                    for i in axis_desc:
                        lst.pop(i)

        shape_out = tuple(lst)
    if out is not None and shape_out != out.shape:
        raise NotImplementedError(
            "out.shape must equal to " + str(shape_out) +
            "; implicit reduction or broadcasting with 'out' is not implemented yet.")

    if a.size < 1:
        is_identity = False
        if initial is nlcpy._NoValue and op_has_identity:
            initial = nlcpy_identity[op_name]
            is_identity = True
        if out is not None:
            _dtype = out.dtype
        elif op_is_boolean:
            _dtype = bool
        elif dtype is not None:
            _dtype = dtype
        elif dtype is None:
            _dtype = a.dtype
            if op_name in ('add', 'multiply'):
                if a.dtype in ('bool', 'int32'):
                    _dtype = 'int64'
                elif a.dtype == 'uint32':
                    _dtype = 'uint64'

        if axis != [] and initial is not nlcpy._NoValue:
            if type(initial) == complex and \
                    _dtype not in ('bool', 'complex64', 'complex128'):
                initial = initial.real
            if initial == nlcpy.inf:
                initial = get_plus_infinity(_dtype, op_name, is_identity)
            elif initial == -nlcpy.inf:
                initial = get_minus_infinity(_dtype, op_name, is_identity)
            if out is not None:
                out.fill(initial)
                return out
            ret = nlcpy.empty(shape_out, dtype=_dtype)
            ret.fill(initial)
            return ret
        else:
            if out is not None:
                out.fill(0)
                return out
            return nlcpy.zeros(shape_out, dtype=_dtype)

    _flag_initial = numpy.isscalar(initial)
    if op_name == "power" and (a.size + _flag_initial) > 1 and \
            (len(axis) >= 1 or _flag_initial) and not (where is None or where is False):
        _dtype = dtype if dtype is not None else \
            out.dtype if out is not None else a.dtype
        if ((_dtype == 'int32' and a.dtype.char not in '?') or
                (_dtype == 'int64' and a.dtype.char not in '?I')):
            if _flag_initial:
                if a.ndim > 0:
                    sl = [slice(0, None) for i in range(a.ndim)]
                else:
                    sl = ()
            else:
                sl = [slice(0, None) if i not in axis
                      else slice(1, None) for i in range(a.ndim)]
            if a[sl].size > 0 and \
                    nlcpy.any(nlcpy.logical_and(
                    where, nlcpy.array(a, dtype=_dtype)[sl] < 0)):
                raise ValueError(
                    "Integers to negative integer powers are not allowed.")

    flag_init_after = False
    where_for_init = True
    if where is True or numpy.isscalar(where) and where != 0:
        where = nlcpy.empty(1)
        flag_where = 0
        where_is_false = False
    elif where is False or where is None or \
            numpy.isscalar(where) and where == 0:
        where = nlcpy.zeros(a.shape, dtype=bool)
        flag_where = 1
        flag_init_after = 0 if initial is None else 1
        where_is_false = True
    else:
        where = nlcpy.asanyarray(where, dtype=bool)
        if len(axis) > 1 and op_name == 'hypot':
            where_for_init = nlcpy.logical_or.reduce(where, axis=tuple(axis))
            where_is_false = not nlcpy.any(where_for_init)
        else:
            where_is_false = not nlcpy.any(where)
        flag_where = 1
        if where_is_false:
            flag_init_after = 0 if initial is None else 1

    if flag_where and a.shape != where.shape:
        raise NotImplementedError(
            "broadcast of 'where' is not implemented yet.")

    if out is not None:
        odt = out.dtype
    elif dtype is not None:
        if op_name in ('divide', 'logaddexp', 'logaddexp2', 'true_divide',
                       'heaviside', 'arctan2', 'hypot', 'copysign', 'nextafter'):
            if dtype in ('int32', 'int64', 'uint32', 'uint64'):
                odt = 'float64'
            else:
                odt = dtype
        elif op_is_boolean:
            odt = 'bool'
        else:
            odt = dtype
    else:
        if op_name in ('add', 'multiply'):
            if a.dtype in ('bool', 'int32'):
                odt = 'int64'
            elif a.dtype in ('uint32',):
                odt = 'uint64'
            else:
                odt = a.dtype
        elif op_is_boolean:
            odt = 'bool'
        else:
            odt = a.dtype

    initial_save = initial
    if initial is nlcpy._NoValue and op_name in nlcpy_identity:
        if op_name not in ('maximum', 'minimum', 'fmax', 'fmin') and \
           not op_is_boolean or len(axis) == 0 or dtype is None:
            _dtype = odt
        else:
            _dtype = dtype
        initial = nlcpy_identity[op_name]
        if initial == nlcpy.inf:
            initial = get_plus_infinity(_dtype, op_name, True)
        elif initial == -nlcpy.inf:
            initial = get_minus_infinity(_dtype, op_name, True)
        else:
            initial = numpy.array(initial, dtype=_dtype)
        flag_init = 1
    elif initial in (None, nlcpy._NoValue):
        initial = numpy.array(0, dtype=odt)
        flag_init = 0
    elif numpy.isscalar(initial):
        if initial == nlcpy.inf:
            initial = get_plus_infinity(odt, op_name)
        elif initial == -nlcpy.inf:
            initial = get_minus_infinity(odt, op_name)
        else:
            if type(initial) == complex and \
                    odt not in ('bool', 'complex64', 'complex128'):
                initial = numpy.array(initial.real, dtype=odt)
            else:
                initial = numpy.array(initial, dtype=odt)
        flag_init = 1
    else:
        raise ValueError("Input object 'initial=%s' is not a scalar" % initial)

    if a._f_contiguous and not a._c_contiguous:
        order_out = 'F'
    else:
        order_out = 'C'

    x = a
    dtype = dtype if dtype is not None else odt
    if not op_is_boolean and not (len(axis) > 0 or flag_init):
        x = nlcpy.asarray(x, dtype=odt)
    else:
        if dtype != odt and (initial_save is None or
                             initial_save is nlcpy._NoValue and
                             not op_has_identity):
            x = nlcpy.array(x)
            for i in range(len(axis)):
                if x.ndim > axis[i] and x.shape[axis[i]] > 1:
                    sl = [slice(0, None) if j != axis[i] else
                          slice(0, 1) for j in range(x.ndim)]
                    x[sl] = x[sl].astype(odt)

        if op_is_boolean:
            dtype = bool
            x = nlcpy.asarray(x, dtype=dtype)
            initial = numpy.asarray(initial, dtype=dtype)
        elif (len(axis) > 0 or flag_init):
            x = nlcpy.asarray(x, dtype=dtype)
            initial = numpy.asarray(initial, dtype=dtype)

    if where_is_false:
        op_name = 'add'
        name = 'nlcpy_add_reduce'
    if len(axis) > 1 and flag_init == 1 or flag_init_after:
        flag_init = 0
        flag_init_after = True
        initial_after = initial.copy()
        if op_name in nlcpy_identity.keys() and flag_where == 1:
            flag_init = 1
            _identity = nlcpy_identity[op_name]
            if _identity == nlcpy.inf:
                initial = get_plus_infinity(initial.dtype, op_name, True)
            elif _identity == -nlcpy.inf:
                initial = get_minus_infinity(initial.dtype, op_name, True)
            else:
                initial = numpy.asanyarray(_identity, dtype=initial.dtype)

    do_copyto = 0
    dst = nlcpy.empty(1) if out is None else out
    first_loop = True
    for axis_i in axis:
        if first_loop and not flag_where and \
            (a.ndim <= axis_i or a.shape[axis_i] == 1) and \
            (not flag_init or op_is_boolean or
             op_name in ('maximum', 'minimum', 'fmax', 'fmin', 'hypot')):
            continue
        if not first_loop:
            flag_init = 0
        first_loop = False

        shape = []
        if x.ndim is not 0:
            for i in range(x.ndim):
                if i is not axis_i and axis_i is not -1:
                    shape.append(x.shape[i])
                else:
                    shape.append(1)
        else:
            shape = [1, ]
            axis_i = 0

        if out is not None and axis_i == axis[-1] and not flag_init_after:
            if shape_out != out.shape:
                do_copyto = 1
            if do_copyto:
                y = ndarray(shape=shape, dtype=odt, order=order_out)
            else:
                y = broadcast.broadcast_to(out.reshape(shape), shape)
            if y.dtype == dtype:
                w = y
            else:
                w = ndarray(shape=shape, dtype=dtype, order=order_out)
        else:
            y = ndarray(shape=shape, dtype=dtype, order=order_out)
            w = y

        request._push_request(
            name,
            'reduce_op',
            (x, y, w, axis_i, flag_init,
             initial, flag_where, where, dst, do_copyto),
        )
        if op_name in ('logaddexp', 'logaddexp2'):
            y = nlcpy.fmax(y, -nlcpy.inf)
        x = y
        flag_where = 0

    if first_loop:
        no_cast = initial_save is None
        if initial_save is nlcpy._NoValue and \
            op_name in ('maximum', 'minimum', 'fmax', 'fmin',
                        'less', 'less_equal', 'equal', 'not_equal',
                        'greater', 'greater_equal'):
            flag_init = 0
            flag_init_after = 0
            no_cast = 1
        if flag_init or flag_init_after:
            y = x
            flag_init_after = 1
            initial_after = initial
        elif out is None:
            if op_name == 'hypot' and initial_save is nlcpy._NoValue:
                y = abs(x)
            else:
                y = x
        else:
            do_copyto = 1
            if len(axis) > 0 or no_cast:
                if a.shape == dst.shape:
                    y = nlcpy.array(a)
                    w = nlcpy.array(a)
                else:
                    y = nlcpy.array(a).reshape(dst.shape)
                    w = nlcpy.array(y)
            else:
                y = nlcpy.array(x, dtype=odt, order=order_out)
                w = ndarray(shape=x.shape, dtype=dtype, order=order_out)
            request._push_request(
                name,
                'reduce_op',
                (x, y, w, -1, 0,
                 initial, flag_where, where, dst, do_copyto),
            )
    if flag_init_after:
        if op_name == 'hypot' and where_for_init is not True:
            x = y
            if out is not None:
                y = out
            y = y[None]
            x = x.reshape(y.shape)
            if isinstance(where_for_init, ndarray):
                where_for_init = where_for_init.reshape(y.shape)
            w = ndarray(x.shape, dtype=dtype)
            flag_where = 1 if where_for_init is not True else 0
            request._push_request(
                name,
                'reduce_op',
                (x, y, w, 0, 1, initial_after, flag_where, where_for_init, dst, 0)
            )
        else:
            z = y
            if out is not None:
                y = out
                z = z.reshape(y.shape)
            x = nlcpy.full_like(z, initial_after, dtype=initial_after.dtype)
            w = z
            w2 = y if y.dtype == dtype else nlcpy.empty_like(y, dtype=dtype)
            request._push_request(
                'nlcpy_'+op_name,
                'binary_op',
                (x, w, y, w2, 0, where)
            )

    return out if out is not None else y.reshape(shape_out)

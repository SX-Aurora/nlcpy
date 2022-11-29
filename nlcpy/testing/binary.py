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

from __future__ import absolute_import
from __future__ import print_function

import numpy
import itertools

from nlcpy.testing import ufunc


def _recreate_array_or_scalar(op, in1, in2, minval, maxval):
    if op in ('divide', 'true_divide', 'remainder', 'mod', 'floor_divide'):
        if isinstance(in2, numpy.ndarray):
            in2 = numpy.where(abs(in2) < 1, in2 + abs(minval) + 1, in2)
        else:
            in2 = in2 + abs(in2) + 1 if abs(in2) < 1 else in2
    elif op in ('power'):
        if isinstance(in1, numpy.ndarray):
            in1 = numpy.where(abs(in1) > 5, in1 * 5 / maxval, in1)
        else:
            in1 = in1 * 5 / maxval if abs(in1) > 5 else in1
        if isinstance(in2, numpy.ndarray):
            in2 = numpy.where(abs(in2) > 5, in2 * 5 / maxval, in2)
            in2 = numpy.where(in2 <= 0, 1, in2)
        else:
            in2 = in2 * 5 / maxval if abs(in2) > 5 else in2
            in2 = 1 if in2.real <= 0 else in2.real
    elif op in ('right_shift', 'left_shift'):
        if isinstance(in2, numpy.ndarray):
            in2 = numpy.where(in2 > 31, in2 * 31 / maxval, in2)
            in2 = numpy.where(in2 < 0, 0, in2)
        else:
            in2 = in2 * 31 / maxval if in2.real > 31 else in2
            in2 = 0 if in2.real < 0 else in2
    elif op == 'ldexp':
        if isinstance(in2, numpy.ndarray):
            in2 = numpy.where(in2 > 15, in2 * 15 / maxval, in2)
        else:
            in2 = in2 * 15 / maxval if in2.real > 15 else in2
    elif op == 'fmod':
        if isinstance(in2, numpy.ndarray):
            in2 = numpy.where(in2 == 0, 2, in2)
        else:
            in2 = 2 if in2 == 0 else in2

    return in1, in2


def _create_out_array(in1, in2, order, dtype, ufunc_name='', is_broadcast=False):
    if ufunc_name == 'outer':
        shape = numpy.asarray(in1).shape + numpy.asarray(in2).shape
    else:
        shape = numpy.broadcast(in1, in2).shape

    # expand shape for broadcast
    if is_broadcast:
        shape = (2,) + shape

    return numpy.zeros(shape, dtype=dtype, order=order)


def _check_binary_no_out_no_where_no_dtype(
        self, args, kw, impl, name_xp, name_in1, name_in2, name_order, name_casting,
        op, minval, maxval, shape, order_x, order_y, order_arg,
        dtype_x, dtype_y, mode, ufunc_name, casting):
    if mode == 'array_array':
        param = itertools.product(
            shape, order_x, order_y, dtype_x, dtype_y, order_arg, casting)
    elif mode == 'array_scalar':
        param = itertools.product(
            shape, order_x, dtype_x, dtype_y, order_arg, casting)
    elif mode == 'scalar_scalar':
        param = itertools.product(dtype_x, dtype_y, casting)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            order = p[5]
            casting = p[6]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[3], minval, maxval)
            in2, minval, maxval = ufunc._create_random_array(
                shape2, p[2], p[4], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, in2.dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            order = p[4]
            casting = p[5]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[2], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(
                p[3], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            dt_in2 = numpy.dtype(p[3])
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, dt_in2))
        elif mode == 'scalar_scalar':
            casting = p[2]
            order = 'K'
            in1, minval, maxval = ufunc._create_random_scalar(p[0], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(p[1], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            dt_in1 = numpy.dtype(p[0])
            dt_in2 = numpy.dtype(p[1])
            worst_dtype = ufunc._guess_worst_dtype((dt_in1, dt_in2))

        kw[name_in1] = in1
        kw[name_in2] = in2
        if ufunc_name == 'outer':
            kw[name_order] = order
            kw[name_casting] = casting

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True,
            (TypeError, ValueError, UnboundLocalError))
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1, in2=in2,
                    ufunc_name=ufunc_name)
                del nlcpy_r, numpy_r


def _check_binary_no_out_no_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_in2, name_order, name_casting,
        name_dtype, op, minval, maxval, shape, order_x, order_y, order_arg, dtype_x,
        dtype_y, dtype_arg, mode, ufunc_name, casting):
    if mode == 'array_array':
        param = itertools.product(
            shape, order_x, order_y, dtype_x, dtype_y, dtype_arg, order_arg, casting)
    elif mode == 'array_scalar':
        param = itertools.product(
            shape, order_x, dtype_x, dtype_y, dtype_arg, order_arg, casting)
    elif mode == 'scalar_scalar':
        param = itertools.product(dtype_x, dtype_y, dtype_arg, casting)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            order = p[6]
            casting = p[7]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[3], minval, maxval)
            in2, minval, maxval = ufunc._create_random_array(
                shape2, p[2], p[4], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            dtype = numpy.dtype(p[5])
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, in2.dtype, dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            order = p[5]
            casting = p[6]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[2], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(
                p[3], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            dt_in2 = numpy.dtype(p[3])
            dtype = numpy.dtype(p[4])
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, dt_in2, dtype))
        elif mode == 'scalar_scalar':
            order = 'K'
            casting = p[3]
            in1, minval, maxval = ufunc._create_random_scalar(p[0], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(p[1], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            dt_in1 = numpy.dtype(p[0])
            dt_in2 = numpy.dtype(p[1])
            dtype = numpy.dtype(p[2])
            worst_dtype = ufunc._guess_worst_dtype((dt_in1, dt_in2, dtype))

        kw[name_in1] = in1
        kw[name_in2] = in2
        kw[name_dtype] = dtype
        if ufunc_name == 'outer':
            kw[name_order] = order
            kw[name_casting] = casting

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True,
            (TypeError, ValueError, UnboundLocalError))
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1, in2=in2,
                    dtype=dtype, ufunc_name=ufunc_name)
                del nlcpy_r, numpy_r


def _check_binary_with_out_no_where_no_dtype(
        self, args, kw, impl, name_xp, name_in1, name_in2, name_order, name_casting,
        name_out, op, minval, maxval, shape, order_x, order_y, order_out, order_arg,
        dtype_x, dtype_y, dtype_out, mode, is_broadcast, ufunc_name, casting):
    if mode == 'array_array':
        param = itertools.product(
            shape, order_x, order_y, order_out,
            dtype_x, dtype_y, dtype_out, order_arg, casting)
    elif mode == 'array_scalar':
        param = itertools.product(
            shape, order_x, order_out, dtype_x,
            dtype_y, dtype_out, order_arg, casting)
    elif mode == 'scalar_scalar':
        param = itertools.product(
            order_out, dtype_x, dtype_y, dtype_out, casting)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            order = p[7]
            casting = p[8]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[4], minval, maxval)
            in2, minval, maxval = ufunc._create_random_array(
                shape2, p[2], p[5], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[3], p[6], ufunc_name, is_broadcast)
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, in2.dtype, out.dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            order = p[6]
            casting = p[7]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[3], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(
                p[4], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[2], p[5], ufunc_name, is_broadcast)
            dt_in2 = numpy.dtype(p[4])
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, dt_in2, out.dtype))
        elif mode == 'scalar_scalar':
            order = 'K'
            casting = p[4]
            in1, minval, maxval = ufunc._create_random_scalar(p[1], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(p[2], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[0], p[3], ufunc_name, is_broadcast)
            dt_in1 = numpy.dtype(p[1])
            dt_in2 = numpy.dtype(p[2])
            worst_dtype = ufunc._guess_worst_dtype((dt_in1, dt_in2, out.dtype))

        kw[name_in1] = in1
        kw[name_in2] = in2
        kw[name_out] = out
        if ufunc_name == 'outer':
            kw[name_order] = order
            kw[name_casting] = casting

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True,
            (TypeError, ValueError, UnboundLocalError))
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1, in2=in2,
                    out=out, ufunc_name=ufunc_name)
                del nlcpy_r, numpy_r


def _check_binary_with_out_no_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_in2, name_order,
        name_casting, name_out, name_dtype, op, minval, maxval, shape, order_x,
        order_y, order_out, order_arg, dtype_x, dtype_y, dtype_out, dtype_arg,
        mode, ufunc_name, casting):
    if mode == 'array_array':
        param = itertools.product(
            shape, order_x, order_y, order_out, dtype_x, dtype_y,
            dtype_out, dtype_arg, order_arg, casting)
    elif mode == 'array_scalar':
        param = itertools.product(
            shape, order_x, order_out, dtype_x, dtype_y, dtype_out,
            dtype_arg, order_arg, casting)
    elif mode == 'scalar_scalar':
        param = itertools.product(
            order_out, dtype_x, dtype_y, dtype_out, dtype_arg, casting)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            order = p[8]
            casting = p[9]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[4], minval, maxval)
            in2, minval, maxval = ufunc._create_random_array(
                shape2, p[2], p[5], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[3], p[6], ufunc_name)
            dtype = numpy.dtype(p[7])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, in2.dtype, out.dtype, dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            order = p[7]
            casting = p[8]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[3], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(
                p[4], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[2], p[5], ufunc_name)
            dt_in2 = numpy.dtype(p[4])
            dtype = numpy.dtype(p[6])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, dt_in2, out.dtype, dtype))
        elif mode == 'scalar_scalar':
            order = 'K'
            casting = p[5]
            in1, minval, maxval = ufunc._create_random_scalar(p[1], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(p[2], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[0], p[3], ufunc_name)
            dt_in1 = numpy.dtype(p[1])
            dt_in2 = numpy.dtype(p[2])
            dtype = numpy.dtype(p[4])
            worst_dtype = ufunc._guess_worst_dtype(
                (dt_in1, dt_in2, out.dtype, dtype))

        kw[name_in1] = in1
        kw[name_in2] = in2
        kw[name_out] = out
        kw[name_dtype] = dtype
        if ufunc_name == 'outer':
            kw[name_order] = order
            kw[name_casting] = casting

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True,
            (TypeError, ValueError, UnboundLocalError))
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op,
                    worst_dtype,
                    nlcpy_r,
                    numpy_r,
                    in1=in1,
                    in2=in2,
                    out=out,
                    dtype=dtype,
                    ufunc_name=ufunc_name)
                del nlcpy_r, numpy_r


def _check_binary_with_out_with_where_no_dtype(
        self,
        args,
        kw,
        impl,
        name_xp,
        name_in1,
        name_in2,
        name_order,
        name_casting,
        name_out,
        name_where,
        op,
        minval,
        maxval,
        shape,
        order_x,
        order_y,
        order_out,
        order_where,
        order_arg,
        dtype_x,
        dtype_y,
        dtype_out,
        mode,
        is_broadcast,
        ufunc_name,
        casting):
    if mode == 'array_array':
        param = itertools.product(
            shape, order_x, order_y, order_out, order_where,
            dtype_x, dtype_y, dtype_out, order_arg, casting)
    elif mode == 'array_scalar':
        param = itertools.product(
            shape, order_x, order_out, order_where, dtype_x,
            dtype_y, dtype_out, order_arg, casting)
    elif mode == 'scalar_scalar':
        param = itertools.product(
            order_out, order_where, dtype_x, dtype_y, dtype_out, casting)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            order = p[8]
            casting = p[9]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[5], minval, maxval)
            in2, minval, maxval = ufunc._create_random_array(
                shape2, p[2], p[6], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[3], p[7], ufunc_name, is_broadcast)
            where, _, _ = ufunc._create_random_array(
                out.shape, p[4], ufunc.DT_BOOL, minval, maxval)
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, in2.dtype, out.dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            order = p[7]
            casting = p[8]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[4], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(
                p[5], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[2], p[6], ufunc_name, is_broadcast)
            where, _, _ = ufunc._create_random_array(
                out.shape, p[3], ufunc.DT_BOOL, minval, maxval)
            dt_in2 = numpy.dtype(p[5])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, dt_in2, out.dtype))
        elif mode == 'scalar_scalar':
            order = 'K'
            casting = p[5]
            in1, minval, maxval = ufunc._create_random_scalar(p[2], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(p[3], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[0], p[4], ufunc_name, is_broadcast)
            where, _, _ = ufunc._create_random_array(
                out.shape, p[1], ufunc.DT_BOOL, minval, maxval)
            dt_in1 = numpy.dtype(p[2])
            dt_in2 = numpy.dtype(p[3])
            worst_dtype = ufunc._guess_worst_dtype(
                (dt_in1, dt_in2, out.dtype))

        # expand shape for broadcast
        if is_broadcast:
            where = numpy.resize(where, ((2,) + out.shape))

        kw[name_in1] = in1
        kw[name_in2] = in2
        kw[name_out] = out
        kw[name_where] = where
        if ufunc_name == 'outer':
            kw[name_order] = order
            kw[name_casting] = casting

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True,
            (TypeError, ValueError, UnboundLocalError))
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op,
                    worst_dtype,
                    nlcpy_r,
                    numpy_r,
                    in1=in1,
                    in2=in2,
                    out=out,
                    where=where,
                    ufunc_name=ufunc_name)
                del nlcpy_r, numpy_r


def _check_binary_with_out_with_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_in2, name_order, name_casting,
        name_out, name_where, name_dtype, op, minval, maxval, shape,
        order_x, order_y, order_out, order_where, order_arg, dtype_x, dtype_y,
        dtype_out, dtype_arg, mode, ufunc_name, casting):
    if mode == 'array_array':
        param = itertools.product(
            shape, order_x, order_y, order_out, order_where,
            dtype_x, dtype_y, dtype_out, dtype_arg, order_arg, casting)
    elif mode == 'array_scalar':
        param = itertools.product(
            shape, order_x, order_out, order_where, dtype_x,
            dtype_y, dtype_out, dtype_arg, order_arg, casting)
    elif mode == 'scalar_scalar':
        param = itertools.product(
            order_out, order_where, dtype_x, dtype_y, dtype_out,
            dtype_arg, casting)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            order = p[9]
            casting = p[10]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[5], minval, maxval)
            in2, minval, maxval = ufunc._create_random_array(
                shape2, p[2], p[6], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[3], p[7], ufunc_name)
            where, _, _ = ufunc._create_random_array(
                out.shape, p[4], ufunc.DT_BOOL, minval, maxval)
            dtype = numpy.dtype(p[8])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, in2.dtype, out.dtype, dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            order = p[8]
            casting = p[9]
            in1, minval, maxval = ufunc._create_random_array(
                shape1, p[1], p[4], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(
                p[5], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[2], p[6], ufunc_name)
            where, _, _ = ufunc._create_random_array(
                out.shape, p[3], ufunc.DT_BOOL, minval, maxval)
            dt_in2 = numpy.dtype(p[5])
            dtype = numpy.dtype(p[7])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, dt_in2, out.dtype, dtype))
        elif mode == 'scalar_scalar':
            order = 'K'
            casting = p[6]
            in1, minval, maxval = ufunc._create_random_scalar(p[2], minval, maxval)
            in2, minval, maxval = ufunc._create_random_scalar(p[3], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2, minval, maxval)
            out = _create_out_array(in1, in2, p[0], p[4], ufunc_name)
            where, _, _ = ufunc._create_random_array(
                out.shape, p[1], ufunc.DT_BOOL, minval, maxval)
            dt_in1 = numpy.dtype(p[2])
            dt_in2 = numpy.dtype(p[3])
            dtype = numpy.dtype(p[5])
            worst_dtype = ufunc._guess_worst_dtype(
                (dt_in1, dt_in2, out.dtype, dtype))

        kw[name_in1] = in1
        kw[name_in2] = in2
        kw[name_out] = out
        kw[name_where] = where
        kw[name_dtype] = dtype
        if ufunc_name == 'outer':
            kw[name_order] = order
            kw[name_casting] = casting

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True,
            (TypeError, ValueError, UnboundLocalError))
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op,
                    worst_dtype,
                    nlcpy_r,
                    numpy_r,
                    in1=in1,
                    in2=in2,
                    out=out,
                    where=where,
                    dtype=dtype,
                    ufunc_name=ufunc_name)
                del nlcpy_r, numpy_r

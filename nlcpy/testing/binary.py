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


def _recreate_array_or_scalar(op, in1, in2):
    if op in ('divide', 'true_divide', 'remainder', 'mod', 'floor_divide'):
        if isinstance(in2, numpy.ndarray):
            in2[in2 == 0] = 1
        else:
            in2 = 1 if in2 == 0 else in2
    elif op in ('power'):
        if isinstance(in1, numpy.ndarray):
            in1[in1 > 5] = 5
            in1[in1 < -5] = -5
        else:
            in1 = 5 if in1.real > 5 else in1.real
            in1 = -5 if in1.real < 5 else in1.real
        if isinstance(in2, numpy.ndarray):
            in2[in2 > 5] = 5
            # TODO: valid minus value
            in2[in2 <= 0] = 1
        else:
            in2 = 5 if in2.real > 5 else in2.real
            # TODO: valid minus value
            in2 = 1 if in2.real <= 0 else in2.real
    elif op in ('right_shift', 'left_shift'):
        if isinstance(in2, numpy.ndarray):
            in2[in2 > 31] = 31
            in2[in2 < 0] = 0
        else:
            in2 = 31 if in2.real > 31 else in2
            in2 = 0 if in2.real < 0 else in2
    elif op == 'ldexp':
        if isinstance(in2, numpy.ndarray):
            in2[in2 > 15] = 15
        else:
            in2 = 15 if in2.real > 15 else in2

    return in1, in2


def _check_binary_no_out_no_where_no_dtype(
        self, args, kw, impl, name_xp, name_in1, name_in2, op, minval, maxval,
        shape, order_x, order_y, dtype_x, dtype_y, mode):
    if mode == 'array_array':
        param = itertools.product(shape, order_x, order_y, dtype_x, dtype_y)
    elif mode == 'array_scalar':
        param = itertools.product(shape, order_x, dtype_x, dtype_y)
    elif mode == 'scalar_scalar':
        param = itertools.product(dtype_x, dtype_y)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            in1 = ufunc._create_random_array(shape1, p[1], p[3], minval, maxval)
            in2 = ufunc._create_random_array(shape2, p[2], p[4], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, in2.dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            in1 = ufunc._create_random_array(shape1, p[1], p[2], minval, maxval)
            in2 = ufunc._create_random_scalar(p[3], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            dt_in2 = numpy.dtype(p[3])
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, dt_in2))
        elif mode == 'scalar_scalar':
            in1 = ufunc._create_random_scalar(p[0], minval, maxval)
            in2 = ufunc._create_random_scalar(p[1], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            dt_in1 = numpy.dtype(p[0])
            dt_in2 = numpy.dtype(p[1])
            worst_dtype = ufunc._guess_worst_dtype((dt_in1, dt_in2))

        kw[name_in1] = in1
        kw[name_in2] = in2

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True, Exception)
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1, in2=in2)


def _check_binary_no_out_no_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_in2, name_dtype, op,
        minval, maxval, shape, order_x, order_y, dtype_x, dtype_y, dtype_arg, mode):
    if mode == 'array_array':
        param = itertools.product(shape, order_x, order_y, dtype_x, dtype_y, dtype_arg)
    elif mode == 'array_scalar':
        param = itertools.product(shape, order_x, dtype_x, dtype_y, dtype_arg)
    elif mode == 'scalar_scalar':
        param = itertools.product(dtype_x, dtype_y, dtype_arg)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            in1 = ufunc._create_random_array(shape1, p[1], p[3], minval, maxval)
            in2 = ufunc._create_random_array(shape2, p[2], p[4], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            dtype = numpy.dtype(p[5])
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, in2.dtype, dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            in1 = ufunc._create_random_array(shape1, p[1], p[2], minval, maxval)
            in2 = ufunc._create_random_scalar(p[3], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            dt_in2 = numpy.dtype(p[3])
            dtype = numpy.dtype(p[4])
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, dt_in2, dtype))
        elif mode == 'scalar_scalar':
            in1 = ufunc._create_random_scalar(p[0], minval, maxval)
            in2 = ufunc._create_random_scalar(p[1], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            dt_in1 = numpy.dtype(p[0])
            dt_in2 = numpy.dtype(p[1])
            dtype = numpy.dtype(p[2])
            worst_dtype = ufunc._guess_worst_dtype((dt_in1, dt_in2, dtype))

        kw[name_in1] = in1
        kw[name_in2] = in2
        kw[name_dtype] = dtype

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True, Exception)
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1, in2=in2,
                    dtype=dtype)


def _check_binary_with_out_no_where_no_dtype(
        self, args, kw, impl, name_xp, name_in1, name_in2, name_out, op,
        minval, maxval, shape, order_x, order_y, order_out, dtype_x, dtype_y, dtype_out,
        mode, is_broadcast):
    if mode == 'array_array':
        param = itertools.product(
            shape, order_x, order_y, order_out,
            dtype_x, dtype_y, dtype_out)
    elif mode == 'array_scalar':
        param = itertools.product(
            shape, order_x, order_out, dtype_x,
            dtype_y, dtype_out)
    elif mode == 'scalar_scalar':
        param = itertools.product(order_out, dtype_x, dtype_y, dtype_out)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            in1 = ufunc._create_random_array(shape1, p[1], p[4], minval, maxval)
            in2 = ufunc._create_random_array(shape2, p[2], p[5], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[3], dtype=p[6])
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, in2.dtype, out.dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            in1 = ufunc._create_random_array(shape1, p[1], p[3], minval, maxval)
            in2 = ufunc._create_random_scalar(p[4], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[2], dtype=p[5])
            dt_in2 = numpy.dtype(p[4])
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, dt_in2, out.dtype))
        elif mode == 'scalar_scalar':
            in1 = ufunc._create_random_scalar(p[1], minval, maxval)
            in2 = ufunc._create_random_scalar(p[2], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[0], dtype=p[3])
            dt_in1 = numpy.dtype(p[1])
            dt_in2 = numpy.dtype(p[2])
            worst_dtype = ufunc._guess_worst_dtype((dt_in1, dt_in2, out.dtype))

        # expand shape for broadcast
        if is_broadcast:
            out = numpy.resize(out, ((2,) + out.shape))

        kw[name_in1] = in1
        kw[name_in2] = in2
        kw[name_out] = out

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True, Exception)
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1, in2=in2,
                    out=out)


def _check_binary_with_out_no_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_in2, name_out,
        name_dtype, op, minval, maxval, shape, order_x, order_y, order_out,
        dtype_x, dtype_y, dtype_out, dtype_arg, mode):
    if mode == 'array_array':
        param = itertools.product(
            shape, order_x, order_y, order_out, dtype_x, dtype_y,
            dtype_out, dtype_arg)
    elif mode == 'array_scalar':
        param = itertools.product(
            shape, order_x, order_out, dtype_x, dtype_y, dtype_out,
            dtype_arg)
    elif mode == 'scalar_scalar':
        param = itertools.product(
            order_out, dtype_x, dtype_y, dtype_out, dtype_arg)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            in1 = ufunc._create_random_array(shape1, p[1], p[4], minval, maxval)
            in2 = ufunc._create_random_array(shape2, p[2], p[5], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[3], dtype=p[6])
            dtype = numpy.dtype(p[7])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, in2.dtype, out.dtype, dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            in1 = ufunc._create_random_array(shape1, p[1], p[3], minval, maxval)
            in2 = ufunc._create_random_scalar(p[4], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[2], dtype=p[5])
            dt_in2 = numpy.dtype(p[4])
            dtype = numpy.dtype(p[6])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, dt_in2, out.dtype, dtype))
        elif mode == 'scalar_scalar':
            in1 = ufunc._create_random_scalar(p[1], minval, maxval)
            in2 = ufunc._create_random_scalar(p[2], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[0], dtype=p[3])
            dt_in1 = numpy.dtype(p[1])
            dt_in2 = numpy.dtype(p[2])
            dtype = numpy.dtype(p[4])
            worst_dtype = ufunc._guess_worst_dtype(
                (dt_in1, dt_in2, out.dtype, dtype))

        kw[name_in1] = in1
        kw[name_in2] = in2
        kw[name_out] = out
        kw[name_dtype] = dtype

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True, Exception)
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
                    dtype=dtype)


def _check_binary_with_out_with_where_no_dtype(
        self,
        args,
        kw,
        impl,
        name_xp,
        name_in1,
        name_in2,
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
        dtype_x,
        dtype_y,
        dtype_out,
        mode,
        is_broadcast):
    if mode == 'array_array':
        param = itertools.product(
            shape, order_x, order_y, order_out, order_where,
            dtype_x, dtype_y, dtype_out)
    elif mode == 'array_scalar':
        param = itertools.product(
            shape, order_x, order_out, order_where, dtype_x,
            dtype_y, dtype_out)
    elif mode == 'scalar_scalar':
        param = itertools.product(
            order_out, order_where, dtype_x, dtype_y, dtype_out)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            in1 = ufunc._create_random_array(shape1, p[1], p[5], minval, maxval)
            in2 = ufunc._create_random_array(shape2, p[2], p[6], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[3], dtype=p[7])
            where = ufunc._create_random_array(
                out.shape, p[4], ufunc.DT_BOOL, minval, maxval)
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, in2.dtype, out.dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            in1 = ufunc._create_random_array(shape1, p[1], p[4], minval, maxval)
            in2 = ufunc._create_random_scalar(p[5], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[2], dtype=p[6])
            where = ufunc._create_random_array(
                out.shape, p[3], ufunc.DT_BOOL, minval, maxval)
            dt_in2 = numpy.dtype(p[5])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, dt_in2, out.dtype))
        elif mode == 'scalar_scalar':
            in1 = ufunc._create_random_scalar(p[2], minval, maxval)
            in2 = ufunc._create_random_scalar(p[3], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[0], dtype=p[4])
            where = ufunc._create_random_array(
                out.shape, p[1], ufunc.DT_BOOL, minval, maxval)
            dt_in1 = numpy.dtype(p[2])
            dt_in2 = numpy.dtype(p[3])
            worst_dtype = ufunc._guess_worst_dtype(
                (dt_in1, dt_in2, out.dtype))

        # expand shape for broadcast
        if is_broadcast:
            out = numpy.resize(out, ((2,) + out.shape))
            where = numpy.resize(where, ((2,) + where.shape))

        kw[name_in1] = in1
        kw[name_in2] = in2
        kw[name_out] = out
        kw[name_where] = where

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True, Exception)
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
                    where=where)


def _check_binary_with_out_with_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_in2, name_out,
        name_where, name_dtype, op, minval, maxval, shape, order_x, order_y,
        order_out, order_where, dtype_x, dtype_y, dtype_out, dtype_arg, mode):
    if mode == 'array_array':
        param = itertools.product(
            shape, order_x, order_y, order_out, order_where,
            dtype_x, dtype_y, dtype_out, dtype_arg)
    elif mode == 'array_scalar':
        param = itertools.product(
            shape, order_x, order_out, order_where, dtype_x,
            dtype_y, dtype_out, dtype_arg)
    elif mode == 'scalar_scalar':
        param = itertools.product(
            order_out, order_where, dtype_x, dtype_y, dtype_out,
            dtype_arg)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array_array':
            shape1 = p[0][0]
            shape2 = p[0][1]
            in1 = ufunc._create_random_array(shape1, p[1], p[5], minval, maxval)
            in2 = ufunc._create_random_array(shape2, p[2], p[6], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[3], dtype=p[7])
            where = ufunc._create_random_array(
                out.shape, p[4], ufunc.DT_BOOL, minval, maxval)
            dtype = numpy.dtype(p[8])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, in2.dtype, out.dtype, dtype))
        elif mode == 'array_scalar':
            shape1 = p[0][0]
            in1 = ufunc._create_random_array(shape1, p[1], p[4], minval, maxval)
            in2 = ufunc._create_random_scalar(p[5], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[2], dtype=p[6])
            where = ufunc._create_random_array(
                out.shape, p[3], ufunc.DT_BOOL, minval, maxval)
            dt_in2 = numpy.dtype(p[5])
            dtype = numpy.dtype(p[7])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, dt_in2, out.dtype, dtype))
        elif mode == 'scalar_scalar':
            in1 = ufunc._create_random_scalar(p[2], minval, maxval)
            in2 = ufunc._create_random_scalar(p[3], minval, maxval)
            in1, in2 = _recreate_array_or_scalar(op, in1, in2)
            out = numpy.zeros(
                numpy.broadcast(in1, in2).shape, order=p[0], dtype=p[4])
            where = ufunc._create_random_array(
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

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True, Exception)
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
                    dtype=dtype)

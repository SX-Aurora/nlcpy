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


def _recreate_array_or_scalar(op, in1):
    if op == 'reciprocal':
        if isinstance(in1, numpy.ndarray):
            in1[in1 == 0] = 1
        else:
            in1 = 1 if in1 == 0 else in1
    elif op in ('sin', 'cos'):
        if isinstance(in1, numpy.ndarray):
            if in1.dtype.kind == 'c':
                in1 = numpy.where(abs(in1.imag) > 88, in1.real + 88j, in1)
        else:
            if isinstance(in1, complex):
                if abs(in1.imag) > 88:
                    in1 = complex(in1.real, 88)
    elif op == 'exp':
        if isinstance(in1, numpy.ndarray):
            in1[in1 > 80] = 80
        else:
            in1 = 80 if in1.real > 80 else in1
    elif op == 'arctanh':
        if isinstance(in1, numpy.ndarray):
            if in1.dtype.kind == 'b':
                in1[...] = False
            else:
                in1[abs(in1) >= 1] = 0.9
        else:
            if isinstance(in1, bool):
                in1 = False
            elif isinstance(in1, int):
                in1 = 0
            elif isinstance(in1, float):
                in1 = 0.9 if in1 >= 1 else in1
            elif isinstance(in1, complex):
                in1 = complex(0.9, in1.imag) if abs(in1.real) >= 0 else in1

    return in1


def _check_unary_no_out_no_where_no_dtype(
        self, args, kw, impl, name_xp, name_in1, op, minval, maxval,
        shape, order_x, dtype_x, mode):
    if mode == 'array':
        param = itertools.product(shape, order_x, dtype_x)
    elif mode == 'scalar':
        param = itertools.product(dtype_x)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1 = ufunc._create_random_array(p[0], p[1], p[2], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            worst_dtype = in1.dtype
        elif mode == 'scalar':
            in1 = ufunc._create_random_scalar(p[0], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            worst_dtype = numpy.dtype(p[0])

        kw[name_in1] = in1

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True, Exception)
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1)


def _check_unary_no_out_no_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_dtype, op,
        minval, maxval, shape, order_x, dtype_x, dtype_arg, mode):
    if mode == 'array':
        param = itertools.product(shape, order_x, dtype_x, dtype_arg)
    elif mode == 'scalar':
        param = itertools.product(dtype_x, dtype_arg)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1 = ufunc._create_random_array(p[0], p[1], p[2], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            dtype = numpy.dtype(p[3])
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, dtype))
        elif mode == 'scalar':
            in1 = ufunc._create_random_scalar(p[0], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            dtype = numpy.dtype(p[1])
            worst_dtype = ufunc._guess_worst_dtype((numpy.dtype(p[0]), dtype))

        kw[name_in1] = in1
        kw[name_dtype] = dtype

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True, Exception)
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1, dtype=dtype)


def _check_unary_with_out_no_where_no_dtype(
        self, args, kw, impl, name_xp, name_in1, name_out, op,
        minval, maxval, shape, order_x, order_out, dtype_x, dtype_out,
        mode, is_broadcast):
    if mode == 'array':
        param = itertools.product(shape, order_x, order_out, dtype_x, dtype_out)
    elif mode == 'scalar':
        param = itertools.product(shape, order_out, dtype_x, dtype_out)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1 = ufunc._create_random_array(p[0], p[1], p[3], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            out = numpy.zeros(shape=p[0], order=p[2], dtype=p[4])
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, out.dtype))
        elif mode == 'scalar':
            in1 = ufunc._create_random_scalar(p[2], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            out = numpy.zeros(shape=p[0], order=p[1], dtype=p[3])
            worst_dtype = ufunc._guess_worst_dtype(
                (numpy.dtype(p[2]), out.dtype))

        # expand shape for broadcast
        if is_broadcast:
            out = numpy.resize(out, ((2,) + out.shape))

        kw[name_in1] = in1
        kw[name_out] = out

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True, Exception)
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1, out=out)


def _check_unary_with_out_no_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_out, name_dtype, op,
        minval, maxval, shape, order_x, order_out, dtype_x, dtype_out, dtype_arg, mode):
    if mode == 'array':
        param = itertools.product(
            shape, order_x, order_out, dtype_x, dtype_out, dtype_arg)
    elif mode == 'scalar':
        param = itertools.product(shape, order_out, dtype_x, dtype_out, dtype_arg)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1 = ufunc._create_random_array(p[0], p[1], p[3], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            out = numpy.zeros(shape=p[0], order=p[2], dtype=p[4])
            dtype = numpy.dtype(p[5])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, out.dtype, dtype))
        elif mode == 'scalar':
            in1 = ufunc._create_random_scalar(p[2], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            out = numpy.zeros(shape=p[0], order=p[1], dtype=p[3])
            dtype = numpy.dtype(p[4])
            worst_dtype = ufunc._guess_worst_dtype(
                (numpy.dtype(p[2]), out.dtype, dtype))

        kw[name_in1] = in1
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
                    out=out,
                    dtype=dtype)


def _check_unary_with_out_with_where_no_dtype(
        self,
        args,
        kw,
        impl,
        name_xp,
        name_in1,
        name_out,
        name_where,
        op,
        minval,
        maxval,
        shape,
        order_x,
        order_out,
        order_where,
        dtype_x,
        dtype_out,
        mode,
        is_broadcast):
    if mode == 'array':
        param = itertools.product(
            shape, order_x, order_out, order_where, dtype_x, dtype_out)
    elif mode == 'scalar':
        param = itertools.product(shape, order_out, order_where, dtype_x, dtype_out)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1 = ufunc._create_random_array(p[0], p[1], p[4], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            out = numpy.zeros(shape=p[0], order=p[2], dtype=p[5])
            where = ufunc._create_random_array(
                p[0], p[3], ufunc.DT_BOOL, minval, maxval)
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, out.dtype))
        elif mode == 'scalar':
            in1 = ufunc._create_random_scalar(p[3], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            out = numpy.zeros(shape=p[0], order=p[1], dtype=p[4])
            where = ufunc._create_random_array(
                p[0], p[2], ufunc.DT_BOOL, minval, maxval)
            worst_dtype = ufunc._guess_worst_dtype(
                (numpy.dtype(p[3]), out.dtype))

        # expand shape for broadcast
        if is_broadcast:
            out = numpy.resize(out, ((2,) + out.shape))
            where = numpy.resize(where, ((2,) + where.shape))

        kw[name_in1] = in1
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
                    out=out,
                    where=where)


def _check_unary_with_out_with_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_out, name_where,
        name_dtype, op, minval, maxval, shape, order_x, order_out, order_where,
        dtype_x, dtype_out, dtype_arg, mode):
    if mode == 'array':
        param = itertools.product(
            shape,
            order_x,
            order_out,
            order_where,
            dtype_x,
            dtype_out,
            dtype_arg)
    elif mode == 'scalar':
        param = itertools.product(
            shape, order_out, order_where, dtype_x, dtype_out, dtype_arg)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1 = ufunc._create_random_array(p[0], p[1], p[4], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            out = numpy.zeros(shape=p[0], order=p[2], dtype=p[5])
            where = ufunc._create_random_array(
                p[0], p[3], ufunc.DT_BOOL, minval, maxval)
            dtype = numpy.dtype(p[6])
            worst_dtype = ufunc._guess_worst_dtype(
                (in1.dtype, out.dtype, dtype))
        elif mode == 'scalar':
            in1 = ufunc._create_random_scalar(p[3], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1)
            out = numpy.zeros(shape=p[0], order=p[1], dtype=p[4])
            where = ufunc._create_random_array(
                p[0], p[2], ufunc.DT_BOOL, minval, maxval)
            dtype = numpy.dtype(p[5])
            worst_dtype = ufunc._guess_worst_dtype(
                (numpy.dtype(p[3]), out.dtype, dtype))

        kw[name_in1] = in1
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
                    out=out,
                    where=where,
                    dtype=dtype)

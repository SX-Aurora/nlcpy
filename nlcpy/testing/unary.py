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


float16_op = (
    'divide',
    'true_divide',
    'logaddexp',
    'logaddexp2',
    'arctan2',
    'hypot',
    'copysign',
    'heaviside',
    'nextafter',
)


def _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name=""):
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
    elif op == 'tan':
        if isinstance(in1, numpy.ndarray):
            in1 = numpy.where(in1 == 0, 1, in1)
        else:
            if in1 == 0:
                in1 = 1
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
                in1 = numpy.where(abs(in1) >= 1, in1 * .9 / maxval, in1)
        else:
            if isinstance(in1, bool):
                in1 = False
            elif isinstance(in1, int):
                in1 = 0
            elif isinstance(in1, float):
                in1 = in1 * .9 / maxval if abs(in1) >= 1 else in1
            elif isinstance(in1, complex):
                in1 = complex(in1.real * .9 / maxval, 0) \
                    if abs(in1.real) >= 1 else in1
    elif op in ('log', 'sqrt', 'cbrt'):
        if isinstance(in1, numpy.ndarray):
            in1 = numpy.where(in1 <= 0, -1 * in1 + 1, in1)
        else:
            if isinstance(in1, complex):
                in1 = -1 * in1.real + 1 + 1j if in1.real <= 0 else in1
            else:
                in1 = -1 * in1 + 1 if in1 <= 0 else in1
    elif op in ('sinh', 'cosh'):
        if isinstance(in1, numpy.ndarray):
            in1 = numpy.where(abs(in1) > 10, in1 * 9 / maxval, in1)
        else:
            in1 = in1 * 9 / maxval if abs(in1) > 10 else in1
    elif op in ('arcsin', 'arccos'):
        if isinstance(in1, numpy.ndarray):
            in1 = numpy.where(abs(in1) > 1, in1 * .9 / maxval, in1)
        else:
            in1 = in1 * .9 / maxval if abs(in1) > 1 else in1
    elif op in ('arccosh'):
        if isinstance(in1, numpy.ndarray):
            in1 = numpy.where(in1 < 1, in1 + abs(minval) + 1, in1)
        else:
            if not isinstance(in1, complex):
                in1 = in1 + abs(minval) + 1 if in1 < 1 else in1

    if ufunc_name in ('reduce', 'reduceat', 'accumulate'):
        if op in ('power', 'multiply', 'left_shift', 'right_shift'):
            if op == 'power':
                _maxval = 2
            elif op == 'multiply':
                _maxval = 10
            else:
                _maxval = 16
            if isinstance(in1, numpy.ndarray):
                if in1.dtype.kind == 'c':
                    in1 = numpy.where(
                        abs(in1.real) > _maxval, _maxval + in1.imag * 1.0j, in1)
                    in1 = numpy.where(
                        abs(in1.imag) > _maxval, in1.real + _maxval * 1.0j, in1)
                else:
                    in1[in1 > _maxval] = _maxval
            else:
                if isinstance(in1, complex):
                    if abs(in1.real) > _maxval:
                        in1 = complex(_maxval, in1.real)
                    if abs(in1.imag) > _maxval:
                        in1 = complex(in1.real, _maxval)
                else:
                    in1 = _maxval
        if op in ('power', 'divide', 'true_divide', 'floor_divide',
                  'mod', 'fmod', 'remainder', 'nextafter'):
            if isinstance(in1, numpy.ndarray):
                if in1.dtype.kind == 'c':
                    in1 = numpy.where(abs(in1.real) < 1, 1 + in1.imag * 1.0j, in1)
                    in1 = numpy.where(abs(in1.imag) < 1, in1.real + 1.0j, in1)
                else:
                    in1[in1 < 1] = 1
            else:
                if isinstance(in1, complex):
                    if abs(in1.real) < 1:
                        in1 = complex(1, in1.real)
                    if abs(in1.imag) < 1:
                        in1 = complex(in1.real, 1)
                else:
                    in1 = 1
    return in1


def _create_out_array(shape, order, dtype, ufunc_name, mode, axis=None, indices=None,
                      keepdims=False, is_broadcast=False):
    if ufunc_name in ('reduce', 'reduceat'):
        _shape = (shape,) if numpy.isscalar(shape) else shape
        if axis is None:
            _axis = range(len(shape))
        else:
            _axis = [axis, ] if numpy.isscalar(axis) else list(axis)
            for i in range(len(_axis)):
                if _axis[i] < 0:
                    _axis[i] += len(shape)
        out_shape = []
        for i in range(len(shape)):
            if i in _axis:
                if keepdims:
                    out_shape.append(1)
                elif ufunc_name == 'reduceat':
                    out_shape.append(len(indices))
            else:
                out_shape.append(_shape[i])
    else:
        out_shape = shape

    if is_broadcast:
        if ufunc_name in ('reduceat', 'accumulate'):
            out_shape = tuple(out_shape) + (2,)
        else:
            out_shape = (2,) + out_shape

    return numpy.zeros(shape=out_shape, order=order, dtype=dtype)


def _count_number_of_calculation(ufunc_name, shape, axis, indices=None):
    if ufunc_name == 'accumulate':
        _axis = axis + len(shape) if axis < 0 else axis
        return shape[axis] - 1

    if ufunc_name == 'reduce':
        if numpy.isscalar(axis):
            _axis = axis + len(shape) if axis < 0 else axis
            return shape[_axis]

        n_calc = 1
        if axis is None:
            axis = range(len(shape))
        for i in range(len(axis)):
            _axis = axis[i] + len(shape) if axis[i] < 0 else axis[i]
            n_calc *= shape[_axis]
        return n_calc

    if ufunc_name == 'reduceat':
        _axis = axis + len(shape) if axis < 0 else axis
        n_calc = shape[axis] - indices[-1] - 1
        for i in range(1, len(indices)):
            n_calc = max(n_calc, indices[i] - indices[i - 1] - 1)
        return n_calc

    return 1


def _check_unary_no_out_no_where_no_dtype(
        self, args, kw, impl, name_xp, name_in1, name_axis, name_indices, op,
        minval, maxval, shape, order_x, dtype_x, mode, ufunc_name, axes, indices):
    if mode == 'array':
        param = itertools.product(shape, order_x, dtype_x, axes, indices)
    elif mode == 'scalar':
        param = itertools.product(dtype_x)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1, minval, maxval = ufunc._create_random_array(
                p[0], p[1], p[2], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            worst_dtype = in1.dtype
            n_calc = _count_number_of_calculation(ufunc_name, p[0], p[3], p[4])
            if ufunc_name in ('reduce', 'accumulate', 'reduceat'):
                kw[name_axis] = p[3]
                if ufunc_name == 'reduceat':
                    kw[name_indices] = p[4]
        elif mode == 'scalar':
            in1, minval, maxval = ufunc._create_random_scalar(p[0], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            worst_dtype = numpy.dtype(p[0])
            n_calc = 1

        if p[0] == numpy.bool and op in float16_op:
            worst_dtype = ufunc._guess_worst_dtype((worst_dtype, numpy.dtype('f4')))

        kw[name_in1] = in1
        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True,
            (TypeError, ValueError, UnboundLocalError))
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1,
                    ufunc_name=ufunc_name, n_calc=n_calc)
                del nlcpy_r, numpy_r


def _check_unary_no_out_no_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_axis, name_indices,
        name_dtype, op, minval, maxval, shape, order_x, dtype_x, dtype_arg,
        mode, ufunc_name, axes, indices):
    if mode == 'array':
        param = itertools.product(shape, order_x, dtype_x, dtype_arg, axes, indices)
    elif mode == 'scalar':
        param = itertools.product(dtype_x, dtype_arg)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1, minval, maxval = ufunc._create_random_array(
                p[0], p[1], p[2], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            dtype = numpy.dtype(p[3])
            if dtype == numpy.bool and op in float16_op:
                worst_dtype = ufunc._guess_worst_dtype((in1.dtype, numpy.dtype('f4')))
            else:
                worst_dtype = ufunc._guess_worst_dtype((in1.dtype, dtype))
            n_calc = _count_number_of_calculation(ufunc_name, p[0], p[4], p[5])
            if ufunc_name in ('reduce', 'accumulate', 'reduceat'):
                kw[name_axis] = p[4]
                if ufunc_name == 'reduceat':
                    kw[name_indices] = p[5]
        elif mode == 'scalar':
            in1, minval, maxval = ufunc._create_random_scalar(p[0], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            dtype = numpy.dtype(p[1])
            if dtype == numpy.bool and op in float16_op:
                worst_dtype = ufunc._guess_worst_dtype((in1.dtype, numpy.dtype('f4')))
            else:
                worst_dtype = ufunc._guess_worst_dtype((numpy.dtype(p[0]), dtype))
            n_calc = 1

        kw[name_in1] = in1
        kw[name_dtype] = dtype

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True,
            (TypeError, ValueError, UnboundLocalError))
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1, dtype=dtype,
                    ufunc_name=ufunc_name, n_calc=n_calc)
                del nlcpy_r, numpy_r


def _check_unary_with_out_no_where_no_dtype(
        self, args, kw, impl, name_xp, name_in1, name_axis, name_indices,
        name_out, op, minval, maxval, shape, order_x, order_out, dtype_x,
        dtype_out, mode, is_broadcast, ufunc_name, axes, indices, keepdims):
    if mode == 'array':
        param = itertools.product(
            shape, order_x, order_out, dtype_x, dtype_out, axes, indices)
    elif mode == 'scalar':
        param = itertools.product(shape, order_out, dtype_x, dtype_out)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1, minval, maxval = ufunc._create_random_array(
                p[0], p[1], p[3], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            out = _create_out_array(
                p[0], p[2], p[4], ufunc_name, mode,
                axis=p[5], indices=p[6], keepdims=keepdims, is_broadcast=is_broadcast)
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, out.dtype))
            n_calc = _count_number_of_calculation(ufunc_name, p[0], p[5], p[6])
            if ufunc_name in ('reduce', 'accumulate', 'reduceat'):
                kw[name_axis] = p[5]
                if ufunc_name == 'reduceat':
                    kw[name_indices] = p[6]
        elif mode == 'scalar':
            in1, minval, maxval = ufunc._create_random_scalar(p[2], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            out = _create_out_array(
                p[0], p[1], p[3], ufunc_name, mode, is_broadcast=is_broadcast)
            worst_dtype = ufunc._guess_worst_dtype(
                (numpy.dtype(p[2]), out.dtype))
            n_calc = 1

        kw[name_in1] = in1
        kw[name_out] = out

        nlcpy_result, numpy_result = ufunc._precheck_func_for_ufunc(
            self, args, kw, impl, name_xp, op, True,
            (TypeError, ValueError, UnboundLocalError))
        # result check
        if nlcpy_result is not None and numpy_result is not None:
            for nlcpy_r, numpy_r in zip(nlcpy_result, numpy_result):
                ufunc._check_ufunc_result(
                    op, worst_dtype, nlcpy_r, numpy_r, in1=in1, out=out,
                    ufunc_name=ufunc_name, n_calc=n_calc)
                del nlcpy_r, numpy_r


def _check_unary_with_out_no_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_axis, name_indices, name_out,
        name_dtype, op, minval, maxval, shape, order_x, order_out, dtype_x, dtype_out,
        dtype_arg, mode, ufunc_name, axes, indices, keepdims):
    if mode == 'array':
        param = itertools.product(
            shape, order_x, order_out, dtype_x, dtype_out, dtype_arg, axes, indices)
    elif mode == 'scalar':
        param = itertools.product(shape, order_out, dtype_x, dtype_out, dtype_arg)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1, minval, maxval = ufunc._create_random_array(
                p[0], p[1], p[3], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            out = _create_out_array(
                p[0], p[2], p[4], ufunc_name, mode,
                axis=p[6], indices=p[7], keepdims=keepdims)
            dtype = numpy.dtype(p[5])
            if dtype == numpy.bool and op in float16_op:
                worst_dtype = ufunc._guess_worst_dtype(
                    (in1.dtype, out.dtype, numpy.dtype('f4')))
            else:
                worst_dtype = ufunc._guess_worst_dtype(
                    (in1.dtype, out.dtype, dtype))
            n_calc = _count_number_of_calculation(ufunc_name, p[0], p[6], p[7])
            if ufunc_name in ('reduce', 'accumulate', 'reduceat'):
                kw[name_axis] = p[6]
                if ufunc_name == 'reduceat':
                    kw[name_indices] = p[7]
        elif mode == 'scalar':
            in1, minval, maxval = ufunc._create_random_scalar(p[2], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            out = _create_out_array(p[0], p[1], p[3], ufunc_name, mode)
            dtype = numpy.dtype(p[4])
            if dtype == numpy.bool and op in float16_op:
                worst_dtype = ufunc._guess_worst_dtype(
                    (numpy.dtype(p[2]), out.dtype, numpy.dtype('f4')))
            else:
                worst_dtype = ufunc._guess_worst_dtype(
                    (numpy.dtype(p[2]), out.dtype, dtype))
            n_calc = 1

        kw[name_in1] = in1
        kw[name_out] = out
        kw[name_dtype] = dtype

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
                    out=out,
                    dtype=dtype,
                    ufunc_name=ufunc_name,
                    n_calc=n_calc
                )
                del nlcpy_r, numpy_r


def _check_unary_with_out_with_where_no_dtype(
        self,
        args,
        kw,
        impl,
        name_xp,
        name_in1,
        name_axis,
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
        is_broadcast,
        ufunc_name,
        axes,
        keepdims):
    if mode == 'array':
        param = itertools.product(
            shape, order_x, order_out, order_where, dtype_x, dtype_out, axes)
    elif mode == 'scalar':
        param = itertools.product(shape, order_out, order_where, dtype_x, dtype_out)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1, minval, maxval = ufunc._create_random_array(
                p[0], p[1], p[4], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            out = _create_out_array(
                p[0], p[2], p[5], ufunc_name, mode,
                axis=p[6], keepdims=keepdims, is_broadcast=is_broadcast)
            where, minval, maxval = ufunc._create_random_array(
                p[0], p[3], ufunc.DT_BOOL, minval, maxval)
            worst_dtype = ufunc._guess_worst_dtype((in1.dtype, out.dtype))
            n_calc = _count_number_of_calculation(ufunc_name, p[0], p[6])
            if ufunc_name == 'reduce':
                kw[name_axis] = p[6]
        elif mode == 'scalar':
            in1, minval, maxval = ufunc._create_random_scalar(p[3], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            out = _create_out_array(
                p[0], p[1], p[4], ufunc_name, mode, is_broadcast=is_broadcast)
            where, minval, maxval = ufunc._create_random_array(
                p[0], p[2], ufunc.DT_BOOL, minval, maxval)
            worst_dtype = ufunc._guess_worst_dtype(
                (numpy.dtype(p[3]), out.dtype))
            n_calc = 1

        # expand shape for broadcast
        if is_broadcast:
            where = numpy.resize(where, ((2,) + where.shape))

        kw[name_in1] = in1
        kw[name_out] = out
        kw[name_where] = where

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
                    out=out,
                    where=where,
                    ufunc_name=ufunc_name,
                    n_calc=n_calc
                )
                del nlcpy_r, numpy_r


def _check_unary_with_out_with_where_with_dtype(
        self, args, kw, impl, name_xp, name_in1, name_axis, name_out, name_where,
        name_dtype, op, minval, maxval, shape, order_x, order_out, order_where,
        dtype_x, dtype_out, dtype_arg, mode, ufunc_name, axes, keepdims):
    if mode == 'array':
        param = itertools.product(
            shape,
            order_x,
            order_out,
            order_where,
            dtype_x,
            dtype_out,
            dtype_arg,
            axes,
        )
    elif mode == 'scalar':
        param = itertools.product(
            shape, order_out, order_where, dtype_x, dtype_out, dtype_arg)
    else:
        raise TypeError('unknown mode was detected.')
    for p in param:
        if mode == 'array':
            in1, minval, maxval = ufunc._create_random_array(
                p[0], p[1], p[4], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            out = _create_out_array(
                p[0], p[2], p[5], ufunc_name, mode, axis=p[7], keepdims=keepdims)
            where, minval, maxval = ufunc._create_random_array(
                p[0], p[3], ufunc.DT_BOOL, minval, maxval)
            dtype = numpy.dtype(p[6])
            if dtype == numpy.bool and op in float16_op:
                worst_dtype = ufunc._guess_worst_dtype(
                    (in1.dtype, out.dtype, numpy.dtype('f4')))
            else:
                worst_dtype = ufunc._guess_worst_dtype(
                    (in1.dtype, out.dtype, dtype))
            n_calc = _count_number_of_calculation(ufunc_name, p[0], p[7])
            if ufunc_name == 'reduce':
                kw[name_axis] = p[7]
        elif mode == 'scalar':
            in1, minval, maxval = ufunc._create_random_scalar(p[3], minval, maxval)
            in1 = _recreate_array_or_scalar(op, in1, minval, maxval, ufunc_name)
            out = _create_out_array(p[0], p[1], p[4], ufunc_name, mode)
            where, minval, maxval = ufunc._create_random_array(
                p[0], p[2], ufunc.DT_BOOL, minval, maxval)
            dtype = numpy.dtype(p[5])
            if dtype == numpy.bool and op in float16_op:
                worst_dtype = ufunc._guess_worst_dtype(
                    (in1.dtype, out.dtype, numpy.dtype('f4')))
            else:
                worst_dtype = ufunc._guess_worst_dtype(
                    (numpy.dtype(p[3]), out.dtype, dtype))
            n_calc = 1

        kw[name_in1] = in1
        kw[name_out] = out
        kw[name_where] = where
        kw[name_dtype] = dtype

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
                    out=out,
                    where=where,
                    dtype=dtype,
                    ufunc_name=ufunc_name,
                    n_calc=n_calc
                )
                del nlcpy_r, numpy_r

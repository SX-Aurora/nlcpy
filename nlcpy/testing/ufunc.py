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

import pkg_resources
import numpy
import re

import nlcpy
from nlcpy.testing import array
from nlcpy.testing import helper
from nlcpy.testing import unary
from nlcpy.testing import binary

TOL_SINGLE = 1e-5
TOL_DOUBLE = 1e-12
TOL_SINGLE_EXCEPTION = 1e-3
TOL_DOUBLE_EXCEPTION = 1e-12

DT_BOOL = numpy.dtype('bool')
DT_I32 = numpy.dtype('i4')
DT_U32 = numpy.dtype('u4')
DT_F32 = numpy.dtype('f4')
DT_I64 = numpy.dtype('i8')
DT_U64 = numpy.dtype('u8')
DT_F64 = numpy.dtype('f8')
DT_C64 = numpy.dtype('c8')
DT_C128 = numpy.dtype('c16')

# operations to be checked with numpy.testing.assert_allclose()
check_close_op_set = (
    # *** unary operators ***
    'negative',
    'positive',
    'absolute',
    'fabs',
    'sign',
    'conj',
    'conjugate',
    'exp',
    'log',
    'sqrt',
    'square',
    'cbrt',
    'reciprocal',
    'sin',
    'cos',
    'tan',
    'arcsin',
    'arccos',
    'arctan',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    'arccosh',
    'arctanh',
    'deg2rad',
    'rad2deg',
    'degrees',
    'radians',
    'invert',
    'signbit',
    'spacing',
    'floor',
    'ceil',
    # *** binary operators ***
    'add',
    'subtract',
    'multiply',
    'divide',
    'logaddexp',
    'logaddexp2',
    'true_divide',
    'floor_divide',
    'power',
    'remainder',
    'mod',
    'fmod',
    'arctan2',
    'ldexp',
    'hypot',
    'nextafter',
)

# operations to be checked with numpy.testing.assert_equal()
check_equal_op_set = (
    # *** unary operators ***
    'logical_not',
    'isfinite',
    'isinf',
    'isnan',
    # *** binary operators ***
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'left_shift',
    'right_shift',
    'greater',
    'greater_equal',
    'less',
    'less_equal',
    'not_equal',
    'equal',
    'logical_and',
    'logical_or',
    'logical_xor',
    'fmax',
    'fmin',
    'minimum',
    'maximum',
    'heaviside',
    'copysign',
)


def _create_random_array(shape, order, dtype, minval, maxval):
    dtype = numpy.dtype(dtype)
    if dtype == '?':
        a = numpy.random.randint(2, size=shape)
        return a.astype(dtype=dtype, order=order), minval, maxval
    elif dtype.kind == 'c':
        a = numpy.random.uniform(minval, maxval, size=shape)
        b = numpy.random.uniform(minval, maxval, size=shape)
        return numpy.asarray(a + 1j * b).astype(dtype=dtype, order=order), \
            minval, maxval
    elif dtype.kind == 'u':
        if minval < 0:
            maxval += (-minval)
            minval = 0
        a = numpy.random.uniform(minval, maxval, size=shape)
        return a.astype(dtype=dtype, order=order), minval, maxval
    else:
        a = numpy.random.uniform(minval, maxval, size=shape)
        return a.astype(dtype=dtype, order=order), minval, maxval


def _create_random_scalar(dtype, minval, maxval):
    dtype = numpy.dtype(dtype)
    if dtype == '?':
        a = numpy.random.randint(2, size=1)
        return bool(a), minval, maxval
    elif dtype.kind == 'c':
        a = numpy.random.uniform(minval, maxval, size=1)
        b = numpy.random.uniform(minval, maxval, size=1)
        return complex(a, b), minval, maxval
    elif dtype.kind == 'i':
        a = numpy.random.randint(minval, maxval, size=1)
        return int(a), minval, maxval
    elif dtype.kind == 'u':
        if minval < 0:
            maxval += (-minval)
            minval = 0
        a = numpy.random.randint(minval, maxval, size=1)
        return int(a), minval, maxval
    else:
        a = numpy.random.uniform(minval, maxval, size=1)
        return float(a), minval, maxval


def _precheck_func_for_ufunc(
        self,
        args,
        kw,
        impl,
        name,
        op,
        type_check,
        accept_error):
    kw[name] = nlcpy
    nlcpy_result, nlcpy_error, nlcpy_msg, nlcpy_tb = \
        helper._call_func(self, impl, args, kw)

    kw[name] = numpy
    numpy_result, numpy_error, numpy_msg, numpy_tb = \
        helper._call_func(self, impl, args, kw)

    if nlcpy_msg is not None:
        nlcpy_msg = re.sub(r'nlcpy', "numpy", nlcpy_msg)
    if nlcpy_error or numpy_error:
        helper._check_nlcpy_numpy_error(self, nlcpy_error, nlcpy_msg,
                                        nlcpy_tb, numpy_error, numpy_msg,
                                        numpy_tb, accept_error=accept_error)
        return None, None

    if not isinstance(nlcpy_result, (tuple, list)):
        nlcpy_result = nlcpy_result,
    if not isinstance(numpy_result, (tuple, list)):
        numpy_result = numpy_result,

    # shape check
    for numpy_r, nlcpy_r in zip(numpy_result, nlcpy_result):
        assert numpy.asarray(numpy_r).shape == nlcpy.asarray(nlcpy_r).shape

    # type check
    if type_check:
        for numpy_r, nlcpy_r in zip(numpy_result, nlcpy_result):
            if type(numpy_r) is not numpy.ndarray:
                numpy_r = numpy.array(numpy_r)
            if type(nlcpy_r) is not nlcpy.ndarray:
                nlcpy_r = nlcpy.array(numpy_r)
            if numpy_r.dtype != nlcpy_r.dtype:
                msg = ['\n']
                msg.append(' numpy.dtype: {}'.format(numpy_r.dtype))
                msg.append(' nlcpy.dtype: {}'.format(nlcpy_r.dtype))
                raise AssertionError('\n'.join(msg))
    return nlcpy_result, numpy_result


def _check_for_unary_with_create_param(
        self,
        args,
        kw,
        op,
        shape,
        order1,
        order2,
        order3,
        dtype1,
        dtype2,
        dtype3,
        minval,
        maxval,
        mode,
        is_out,
        is_where,
        is_dtype,
        is_broadcast,
        name_xp,
        name_in1,
        name_axis,
        name_indices,
        name_out,
        name_where,
        name_op,
        name_dtype,
        impl,
        ufunc_name,
        axes,
        indices,
        keepdims):
    kw[name_op] = op
    shape = (shape,)
    # no out, no where, no dtype
    if not is_out and not is_where and not is_dtype:
        unary._check_unary_no_out_no_where_no_dtype(
            self, args, kw, impl, name_xp, name_in1, name_axis, name_indices, op,
            minval, maxval, shape, order1, dtype1, mode, ufunc_name, axes, indices)
    # no out, no where, with dtype
    elif not is_out and not is_where and is_dtype:
        unary._check_unary_no_out_no_where_with_dtype(
            self, args, kw, impl, name_xp, name_in1, name_axis, name_indices,
            name_dtype, op, minval, maxval, shape, order1, dtype1, dtype3, mode,
            ufunc_name, axes, indices)
    # with out, no where, no dtype
    elif is_out and not is_where and not is_dtype:
        unary._check_unary_with_out_no_where_no_dtype(
            self, args, kw, impl, name_xp, name_in1, name_axis, name_indices, name_out,
            op, minval, maxval, shape, order1, order2, dtype1, dtype2,
            mode, is_broadcast, ufunc_name, axes, indices, keepdims)
    # with out, no where, with dtype
    elif is_out and not is_where and is_dtype:
        unary._check_unary_with_out_no_where_with_dtype(
            self, args, kw, impl, name_xp, name_in1, name_axis, name_indices, name_out,
            name_dtype, op, minval, maxval, shape, order1, order2, dtype1, dtype2,
            dtype3, mode, ufunc_name, axes, indices, keepdims)
    # with out, with where, no dtype
    elif is_out and is_where and not is_dtype:
        unary._check_unary_with_out_with_where_no_dtype(
            self, args, kw, impl, name_xp, name_in1, name_axis, name_out,
            name_where, op, minval, maxval, shape, order1, order2,
            order3, dtype1, dtype2, mode, is_broadcast, ufunc_name, axes, keepdims)
    # with out, with where, with dtype
    elif is_out and is_where and is_dtype:
        unary._check_unary_with_out_with_where_with_dtype(
            self, args, kw, impl, name_xp, name_in1, name_axis, name_out,
            name_where, name_dtype, op, minval, maxval, shape,
            order1, order2, order3, dtype1, dtype2, dtype3, mode,
            ufunc_name, axes, keepdims)
    else:
        raise TypeError(
            'out={}, where={}, dtype={} tests is not supported.'.format(
                True if is_out is True else False,
                True if is_where is True else False,
                True if is_dtype is True else False))


def _check_for_binary_with_create_param(
        self,
        args,
        kw,
        op,
        shape,
        order1,
        order2,
        order3,
        order4,
        order5,
        dtype1,
        dtype2,
        dtype3,
        dtype4,
        minval,
        maxval,
        mode,
        is_out,
        is_where,
        is_dtype,
        is_broadcast,
        name_xp,
        name_in1,
        name_in2,
        name_order,
        name_casting,
        name_out,
        name_where,
        name_op,
        name_dtype,
        ufunc_name,
        casting,
        impl):
    kw[name_op] = op
    shape = (shape,)
    if type(casting) == str:
        casting = (casting, )

    # skip check
    ws = pkg_resources.WorkingSet()
    try:
        ws.require('numpy<1.18')
    except pkg_resources.ResolutionError:
        if op == 'floor_divide':
            return

    # no out, no where, no dtype
    if not is_out and not is_where and not is_dtype:
        binary._check_binary_no_out_no_where_no_dtype(
            self, args, kw, impl, name_xp, name_in1, name_in2, name_order, name_casting,
            op, minval, maxval, shape, order1, order2, order5, dtype1,
            dtype2, mode, ufunc_name, casting)
    # no out, no where, with dtype
    elif not is_out and not is_where and is_dtype:
        binary._check_binary_no_out_no_where_with_dtype(
            self, args, kw, impl, name_xp, name_in1, name_in2, name_order, name_casting,
            name_dtype, op, minval, maxval, shape, order1, order2, order5,
            dtype1, dtype2, dtype4, mode, ufunc_name, casting)
    # with out, no where, no dtype
    elif is_out and not is_where and not is_dtype:
        binary._check_binary_with_out_no_where_no_dtype(
            self, args, kw, impl, name_xp, name_in1, name_in2, name_order, name_casting,
            name_out, op, minval, maxval, shape, order1, order2, order3,
            order5, dtype1, dtype2, dtype3, mode, is_broadcast, ufunc_name, casting)
    # with out, no where, with dtype
    elif is_out and not is_where and is_dtype:
        binary._check_binary_with_out_no_where_with_dtype(
            self, args, kw, impl, name_xp, name_in1, name_in2, name_order, name_casting,
            name_out, name_dtype, op, minval, maxval, shape, order1, order2, order3,
            order5, dtype1, dtype2, dtype3, dtype4, mode, ufunc_name, casting)
    # with out, with where, no dtype
    elif is_out and is_where and not is_dtype:
        binary._check_binary_with_out_with_where_no_dtype(
            self, args, kw, impl, name_xp, name_in1, name_in2, name_order, name_casting,
            name_out, name_where, op, minval, maxval, shape, order1, order2, order3,
            order4, order5, dtype1, dtype2, dtype3, mode, is_broadcast, ufunc_name,
            casting)
    # with out, with where, with dtype
    elif is_out and is_where and is_dtype:
        binary._check_binary_with_out_with_where_with_dtype(
            self, args, kw, impl, name_xp, name_in1, name_in2, name_order, name_casting,
            name_out, name_where, name_dtype, op, minval, maxval, shape,
            order1, order2, order3, order4, order5, dtype1, dtype2, dtype3, dtype4,
            mode, ufunc_name, casting)
    else:
        raise TypeError(
            'out={}, where={}, dtype={} tests is not supported.'.format(
                True if is_out is True else False,
                True if is_where is True else False,
                True if is_dtype is True else False))


def _guess_tolerance(op, worst_dtype, ufunc_name):
    if op in ('power', 'multiply') and \
            ufunc_name in ('reduce', 'reduceat', 'accumulate', 'outer'):
        return 1e-3, 1e-3
    elif op in check_close_op_set:
        if worst_dtype in (DT_BOOL, DT_I32, DT_U32, DT_F32, DT_C64):
            if op in ('exp', 'tan', 'remainder', 'mod', 'fmod'):
                return TOL_SINGLE_EXCEPTION, TOL_SINGLE_EXCEPTION
            else:
                return TOL_SINGLE, TOL_SINGLE
        elif worst_dtype in (DT_I64, DT_U64, DT_F64, DT_C128):
            return TOL_DOUBLE, TOL_DOUBLE
    elif op in check_equal_op_set:
        return 0, 0
    else:
        raise TypeError('unknown op \'{}\' was detected.'.format(op))


def _guess_worst_dtype(dtypes):
    dt_itemsizes = []
    for dt in dtypes:
        if dt == DT_C64:
            dt_itemsizes.append(DT_F32.itemsize)
        else:
            dt_itemsizes.append(dt.itemsize)
    min_idx = numpy.argmin(dt_itemsizes)
    return dtypes[min_idx]


def _nan_inf_care(v, n):
    v_tmp = v.flatten()
    n_tmp = n.flatten()
    for i, (ve, ne) in enumerate(zip(v_tmp, n_tmp)):
        if numpy.isinf(ne) and nlcpy.isinf(ve):
            v_tmp[i] = n_tmp[i]
        elif numpy.isnan(ne) and nlcpy.isnan(ve):
            v_tmp[i] = n_tmp[i]
    return nlcpy.reshape(v_tmp, v.shape), numpy.reshape(n_tmp, n.shape)


def _check_ufunc_result(op, worst_dtype, v, n, in1=None, in2=None,
                        out=None, where=None, dtype=None, ufunc_name='', n_calc=1):
    # nan/inf care
    if numpy.isscalar(n):
        if numpy.isinf(n) and nlcpy.isinf(v) or numpy.isnan(n) and nlcpy.isnan(v):
            return
    else:
        v, n = _nan_inf_care(v, n)

    v_array = nlcpy.asarray(v)
    n_array = numpy.asarray(n)
    atol, rtol = _guess_tolerance(op, worst_dtype, ufunc_name)
    if ufunc_name in ('reduce', 'accumulate', 'reduceat'):
        atol *= n_calc
        rtol *= n_calc
        if n_array.dtype.char in '?ilIL' and \
                (numpy.asarray(in1).dtype.char not in '?ilIL' or
                 numpy.dtype(dtype).char not in '?ilIL' or
                 op in ('logaddexp', 'logaddexp2', 'arctan2', 'hypot')):
            atol = 1

    # prepare error message
    msg = "\n"
    msg += "***** parameters when pytest raised an error *****"
    msg += "\nout={}, where={}, dtype={}".format(
        True if out is not None else False,
        True if where is not None else False, dtype)
    msg += "\nop: {}".format(op)
    if in1 is not None:
        if isinstance(in1, numpy.ndarray):
            msg += "\nin1: dtype={}\n{}".format(in1.dtype, in1)
        else:
            msg += "\nin1: dtype={}\n{}".format(type(in1), in1)
    if in2 is not None:
        if isinstance(in2, numpy.ndarray):
            msg += "\nin2: dtype={}\n{}".format(in2.dtype, in2)
        else:
            msg += "\nin2: dtype={}\n{}".format(type(in2), in2)
    if out is not None:
        msg += "\nout: dtype={}\n{}".format(out.dtype, out)
    if where is not None:
        msg += "\nwhere: dtype={}\n{}".format(where.dtype, where)
    msg += "\n\nnlcpy_result: dtype={}\n{}".format(v_array.dtype, v_array)
    msg += "\nnumpy_result: dtype={}\n{}".format(n_array.dtype, n_array)
    msg += "\n"

    # compare results
    try:
        with numpy.errstate(invalid='ignore'):
            if v_array.dtype == DT_BOOL and n_array.dtype == DT_BOOL:
                array.assert_array_equal(v_array, n_array, verbose=True, err_msg=msg)
            elif atol == 0 and rtol == 0:
                array.assert_array_equal(v_array, n_array, verbose=True, err_msg=msg)
            else:
                array.assert_allclose(
                    v_array, n_array, rtol, atol, verbose=True, err_msg=msg)
    except Exception:
        raise

    # if contiguous_check and isinstance(n, numpy.ndarray):
    if isinstance(n, numpy.ndarray):
        if n.flags.c_contiguous and not v.flags.c_contiguous:
            raise AssertionError(
                'The state of c_contiguous flag is false. \n\n'
                'nlcpy_flags:\n{} \n\nnumpy_flags:\n{})'.format(
                    v.flags, n.flags))
        if n.flags.f_contiguous and not v.flags.f_contiguous:
            raise AssertionError(
                'The state of f_contiguous flag is false. \n\n'
                'nlcpy_flags:\n{} \n\nnumpy_flags:\n{})'.format(
                    v.flags, n.flags))

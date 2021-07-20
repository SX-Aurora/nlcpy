#
# * The source code in this file is based on the soure code of CuPy.
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
# # CuPy License #
#
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

# distutils: language = c++

import string
import os
import numpy
import time
import pickle

import nlcpy

from libcpp.vector cimport vector
from libc.stdint cimport *

from nlcpy import veo
from nlcpy.core cimport core
from nlcpy.core cimport broadcast
from nlcpy.core cimport vememory
from nlcpy.core cimport dtype as _dtype
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport internal
from nlcpy.core cimport scalar
from nlcpy.request cimport request
from nlcpy.ufuncs cimport reduce as _reduce
from nlcpy.ufuncs cimport outer as _outer
from nlcpy.ufuncs cimport reduceat as _reduceat
from nlcpy.ufuncs cimport accumulate as _accumulate
from nlcpy.linalg cimport cblas_wrapper
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.core.core cimport on_VH, on_VE_VH

cdef _ufunc_boolean_name_set = (
    'greater',
    'greater_equal',
    'less',
    'less_equal',
    'not_equal',
    'equal',
    'logical_and',
    'logical_or',
    'logical_xor',
    'logical_not',
)


# @name Universal Functions(ufunc)
# @{


cdef inline int get_kind_score(int kind):
    if b'b' == kind:
        return 0
    if b'u' == kind or b'i' == kind:
        return 1
    if b'f' == kind or b'c' == kind:
        return 2
    return -1


cpdef list _preprocess_args(args):
    """Preprocesses arguments for kernel invocation

    - Converts Python scalars into NumPy scalars
    """
    cdef list ret = []
    for arg in args:
        if type(arg) in (list, tuple):
            arg = nlcpy.array(arg)
        elif type(arg) is not ndarray:
            s = scalar.convert_scalar(arg)
            if s is None:
                raise TypeError('Unsupported type %s' % type(arg))
            ret.append(s)
            continue
            # arg = nlcpy.asanyarray(arg)
        ret.append(arg)
    return ret


cdef _make_copy_if_needed(in_args, out_args):
    """
    Make copy if in_args and out_args share memory and
    there's strides is different.
    """
    out_a = out_args
    in_args_new = []
    for in_a in in_args:
        if out_a.base is not None and in_a.base is not None and \
                id(out_a.base) == id(in_a.base):
            if not internal.vector_equal(
                    out_a._strides, in_a._strides):
                in_args_new.append(in_a.copy())
                continue
        if out_a.base is None and in_a.base is not None and \
                id(out_a) == id(in_a.base):
            if not internal.vector_equal(
                    out_a._strides, in_a._strides):
                in_args_new.append(in_a.copy())
                continue
        in_args_new.append(in_a)
    return in_args_new, [out_args]


cpdef tuple _get_args_info(list args):
    ret = []
    for a in args:
        t = type(a)
        dtype = a.dtype.type
        ret.append((t, dtype, a.ndim))
    return tuple(ret)


cdef _guess_dtypes_from_in_types(list types, tuple in_types):
    cdef Py_ssize_t n = len(in_types)
    cdef Py_ssize_t i
    can_cast = numpy.can_cast
    for op_types in types:
        _op_types = op_types[0]  # get input types
        for i in range(n):
            it = in_types[i]
            ot = _op_types[i]
            if isinstance(it, tuple):
                if not can_cast(it[0], ot) and not can_cast(it[1], ot):
                    break
            elif not can_cast(it, ot):
                break
        else:
            return op_types
    return None


cdef tuple _guess_dtypes_from_dtype(list types, object dtype):
    cdef tuple _types, t
    for t in types:
        _types = t[1]
        for _t in _types:
            if _t != dtype:
                break
        else:
            if numpy.dtype(_types[-1]).kind != 'V':
                return t
    return None


cdef inline bint _check_should_use_min_scalar(list in_args) except? -1:
    cdef int kind, max_array_kind, max_scalar_kind
    cdef bint all_scalars
    all_scalars = True
    max_array_kind = -1
    max_scalar_kind = -1
    for i in in_args:
        kind = get_kind_score(ord(i.dtype.kind))
        if isinstance(i, ndarray):
            all_scalars = False
            max_array_kind = max(max_array_kind, kind)
        else:
            max_scalar_kind = max(max_scalar_kind, kind)
    return (max_scalar_kind != -1
            and not all_scalars
            and max_array_kind >= max_scalar_kind)


cdef tuple _guess_dtypes(str name, dict cache, list types, list in_args, dtype):
    if dtype is None:
        use_raw_value = _check_should_use_min_scalar(in_args)
        if use_raw_value:
            in_types = tuple([
                i.dtype.type if isinstance(i, ndarray)
                else scalar._min_scalar_type(i)
                for i in in_args])
        else:
            in_types = tuple([i.dtype.type for i in in_args])
        ret = cache.get(in_types, ())
        if ret is ():
            ret = _guess_dtypes_from_in_types(types, in_types)
            cache[in_types] = ret
    else:
        ret = cache.get(dtype, ())
        if ret is ():
            ret = _guess_dtypes_from_dtype(types, dtype)
            cache[dtype] = ret
    return ret


cdef class ufunc:
    """Function that operate element by element on whole arrays.

    Calling ufuncs:
        op(*x, out, where=True, casting ='same_kind', order='K', dtype=None,
           subok=False)
        Apply `op` to the arguments `*x` elementwise, broadcasting the arguments.

        The broadcasting rules are:

            * Dimensions of length 1 may be prepended to either array.
            * Arrays may be repeated along dimensions of length 1.

    Args:
        *x : array_like
            Input arrays.
        out : ndarray, None, or tuple of ndarray and None, optional
            Alternate array object(s) in which to put the result; if provided, it
            must have a shape that the inputs broadcast to. A tuple of arrays
            (possible only as a keyword argument) must have length equal to the
            number of outputs; use `None` for uninitialized outputs to be
            allocated by the ufunc.
        where : array_like, optional
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value.
            Note that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        casting : {'no', 'equiv', 'safe', 'same_kind'}, optional
            Controls what kind of data casting may occur.
              * 'no' means the data types should not be cast at all.
              * 'equiv' means only byte-order changes are allowed.
              * 'safe' means only casts which can preserve values are allowed.
              * 'same_kind' means only safe casts or casts within a kind,
                like float64 to float32, are allowed.
            NLCPy does NOT support 'unsafe', which is supported in NumPy.
        order : character, optional
            Specifies the calculation iteration order/memory layout of the output
            array. Defaults to 'K'. 'C' means the output should be C-contiguous,
            'F' means F-contiguous, 'A' means F-contiguous if the inputs are
            F-contiguous and not also not C-contiguous, C-contiguous otherwise,
            and 'K' means to match the element ordering of the inputs as closely
            as possible.
        dtype : dtype, optional
            Overrides the dtype of the calculation and output arrays.
        subok : bool, optional
            Not implemented in NLCPy.

    Returns:
        r : ndarray or tuple of ndarray
            `r` will have the shape that the arrays in `x` broadcast to; if `out` is
            provided, it will be returned. If not, `r` will be allocated and
            may contain uninitialized values. In NLCPy, if the input parameters are
            all scalars, `r` returns the result as a 0-dimension ndarray.
            If the function has more than one output, then the result will be a
            tuple of arrays.
    """
    cdef:
        readonly Py_ssize_t nin
        readonly Py_ssize_t nout
        readonly Py_ssize_t nargs
        readonly object name
        readonly list types
        readonly object _default_casting
        readonly dict _cache
        readonly object _err_func
        readonly object __doc__
        readonly object __name__
        readonly object __module__

    def __init__(self, name, nin, nout, types,
                 err_func, default_casting=None, doc=''):
        self.name = name
        self.__name__ = name
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self.types = types
        self.__doc__ = doc
        if default_casting is None:
            self._default_casting = 'same_kind'
        else:
            self._default_casting = default_casting
        self._cache = {}
        self._err_func = err_func

    def __repr__(self):
        return '<ufunc \'%s\'>' % self.name

    def __call__(self, *args, **kwargs):
        """Call self as a function.

        """
        cdef list broad_values
        cdef tuple shape
        name = self.name.replace('nlcpy_', '')
        casting = self._default_casting
        out, where, casting, order, dtype, subok = parse_argument(self, kwargs)

        if where is True:
            where = <int32_t>0    # set dummy value
            where_flag = <int32_t>0
        else:
            if isinstance(where, (ndarray, numpy.ndarray)):
                if not numpy.can_cast(where, 'bool'):
                    raise TypeError("Cannot cast array data from dtype('%s') "
                                    "to dtype('%s') according to the rule 'safe'"
                                    % (where.dtype, 'bool'))
            where = nlcpy.asanyarray(where, dtype='bool')
            where_flag = <int32_t>1

        n_args = len(args)
        if n_args != self.nin and n_args != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % name)
        if args[self.nin:] is not ():
            if not isinstance(args[self.nin:][0], nlcpy.ndarray):
                raise TypeError('return arrays must be of ArrayType')

        args = _preprocess_args(args)
        if out is None:
            in_args = args[:self.nin]
            out_args = args[self.nin:]
            if len(out_args) == 1:
                out_args = out_args[0]
            elif len(out_args) == 0:
                out_args = None
        else:
            if not isinstance(out, nlcpy.ndarray):
                raise TypeError('return arrays must be of nlcpy.ndarray')
            if self.nout != 1:
                raise ValueError('Cannot use \'out\' in %s' % name)
            if n_args != self.nin:
                raise ValueError('Cannot specify \'out\' as both '
                                 'a positional and keyword argument')
            in_args = list(args)
            out_args = _preprocess_args((out,))[0]
            in_args_tmp, out_args_tmp = \
                _make_copy_if_needed(in_args, out_args)
            args = in_args_tmp + out_args_tmp

        _types = _guess_dtypes(
            self.name, self._cache, self.types, in_args, dtype)

        if _types is None:
            in_types = None
            out_types = None
            dtype_out = numpy.dtype(dtype)
            valid_dtype = False
        else:
            in_types, out_types = _types
            dtype_out = numpy.dtype(out_types[0])
            valid_dtype = True
        if dtype is None:
            check_cast = False
            ari_dtype = dtype_out
        else:
            check_cast = True
            ari_dtype = dtype
        if out is None:
            check_out = False
        else:
            check_out = True

        ari_dtype = _dtype.promote_dtype_to_supported(numpy.dtype(ari_dtype))

        if _types is None and dtype is None:
            raise TypeError("ufunc '%s' not supported for the input types, and the "
                            "inputs could not be safely coerced to any supported types "
                            "according to the casting rule ''safe''" % name)

        if callable(self._err_func):
            self._err_func(dtype_out, in_args, name,
                           casting, valid_dtype, check_cast,
                           check_out, out_args, ari_dtype)

        if name in _ufunc_boolean_name_set:
            dtype_out = numpy.dtype('bool')
        elif out is not None:
            dtype_out = out_args.dtype

        if dtype_out in _dtype._nlcpy_not_supported_type_set:
            raise TypeError('\'{}\' is not supported as an output dtype '
                            'of ufunc \'{}\''
                            .format(dtype_out, name))

        args = convert_args_to_ndarray(args)

        if self.name is 'nlcpy_matmul':
            return cblas_wrapper.cblas_gemm(args[0], args[1], out=out_args,
                                            order=order, dtype=dtype_out)

        order = guess_out_order(order, args)

        if out_args is None:
            ret = <int32_t>0    # set dummy value
        else:
            ret = out_args

        if len(args) < self.nargs:
            args.append(ret)
        args.append(where)
        values, shape = broadcast._broadcast_core(args)
        where = values.pop(-1)
        ret = values.pop(-1)

        if 0 in shape:
            return ndarray(shape, dtype=dtype_out, order=order)

        if out_args is None:
            ret = ndarray(shape, dtype=dtype_out, order=order)
        elif not internal.vector_equal(ret._shape, out_args._shape):
            raise ValueError("non-broadcastable output operand with shape "
                             + str(out_args.shape).replace(" ", "") + " "
                             + "doesn't match the broadcast shape "
                             + str(ret.shape).replace(" ", "").rstrip())

        if ari_dtype == ret.dtype:
            work = ret
        else:
            work = ndarray(shape, dtype=ari_dtype, order=order)

        values.append(ret)
        values.append(work)
        values.append(where_flag)
        values.append(where)
        push_ufunc(self.name, values, self.nin)
        return ret

    def reduce(self, array, axis=0, dtype=None, out=None, keepdims=False,
               initial=nlcpy._NoValue, where=True):
        """Reduces one of the dimension of the input array, by applying ufunc along one
        axis.

        Let :math:`a.shape = (N_{0}, ..., N_{i}, ..., N_{M-1})`. Then
        :math:`ufunc.reduce(a, axis = i)[k_{0}, ..., k_{i-1}, k_{i+1}, ..., k_{M-1}] =`
        the result of iterating :math:`j` over :math:`range(N_{i})` , cumulatively
        applying ufunc to each :math:`a[k_{0}, ..., k_{i-1}, j, k_{i+1}, ..., k_{M-1}]`.

        For example, ``nlcpy.add.reduce()`` is equivalent to :func:`nlcpy.sum()`.

        Parameters
        ----------
        array : array_like
            The array to act on.
        axis : None or int or tuple of ints, optional
            Axis or axes along which a reduction is performed. The default (*axis* = 0)
            is perform a reduction over the first dimension of the input array. *axis*
            may be negative, in which case it counts from the last to the first axis. If
            this is None, a reduction is performed over all the axes. If this is a tuple
            of ints, a reduction is performed on multiple axes, instead of a single axis
            or all the axes as before. For operations which are either not commutative or
            not associative, doing a reduction over multiple axes is not well-defined.
            The ufuncs do no currently raise an exception in this case, but will likely
            do so in the future.
        dtype : dtype, optional
            The type used to represent the intermediate results. Defaults to the
            data-type of the output array if this is provided, or the data-type of the
            input array if no output array is provided.
        out : ndarray, optional
            A location into which the result is stored. If not provided or None, a
            freshly-allocated array is returned.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the original *array*.
        initial : scalar, optional
            The value with which to start the reduction. If the ufunc has no identity or
            the dtype is object, this defaults to None - otherwise it defaults to
            ufunc.identity. If None is given, the first element of the reduction is
            used, and an error is thrown if the reduction is empty.
        where : array_like of bool, optional
            A boolean array which shape is same as shape of *array*, and selects elements
            to include in the reduction. Note that for ufuncs like ``minimum`` that do
            not have an identity defined, one has to pass in also ``initial``.

        Returns
        -------
        r : ndarray
            The reduced array. If out was supplied, *r* is a reference to it.

        Restriction
        -----------
        - If an ndarray is passed to ``where`` and ``where.shape != a.shape``,
          *NotImplementedError* occurs.
        - If an ndarray is passed to ``out`` and ``out.shape != r.shape``,
          *NotImplementedError* occurs.

        Examples
        --------
        >>> import nlcpy as vp
        >>> vp.multiply.reduce([2,3,5])
        array(30)

        A multi-dimensional array example:

        >>> X = vp.arange(8).reshape((2,2,2))
        >>> X
        array([[[0, 1],
                [2, 3]],
        <BLANKLINE>
               [[4, 5],
                [6, 7]]])
        >>> vp.add.reduce(X, 0)
        array([[ 4,  6],
               [ 8, 10]])
        >>> vp.add.reduce(X) # confirm: default axis value is 0
        array([[ 4,  6],
               [ 8, 10]])
        >>> vp.add.reduce(X, 1)
        array([[ 2,  4],
               [10, 12]])
        >>> vp.add.reduce(X, 2)
        array([[ 1,  5],
               [ 9, 13]])

        You can use the ``initial`` keyword argument to initialize the reduction with a
        different value, and ``where`` to select specific elements to include:

        >>> vp.add.reduce([10], initial=5)
        array(15)
        >>> vp.add.reduce(vp.ones((2, 2, 2)), axis=(0, 2), initial=10)
        array([14., 14.])
        >>> a = vp.array([10., vp.nan, 10])
        >>> vp.add.reduce(a, where=~vp.isnan(a))
        array(20.)

        Allows reductions of empty arrays where they would normally fail, i.e. for ufuncs
        without an identity.

        >>> vp.minimum.reduce([], initial=vp.inf)
        array(inf)
        >>> vp.minimum.reduce([[1., 2.], [3., 4.]], initial=10.)
        array([1., 2.])
        >>> vp.minimum.reduce([])  # doctest: +SKIP
        Traceback (most recent call last):
            ...
            ValueError: zero-size array to reduction operation minimum
                        which has no identity

        """
        if self.nin != 2:
            raise ValueError('reduce only supported for binary functions')

        op_name = self.name.replace("nlcpy_", "")
        dtype = None if dtype is None else numpy.dtype(dtype)

        if out is not None:
            if type(out) != nlcpy.ndarray and type(out) != tuple:
                raise TypeError("output must be an array")

        if isinstance(out, tuple):
            if len(out) != 1:
                raise ValueError("The 'out' tuple must have exactly one entry")
            elif not isinstance(out[0], nlcpy.ndarray):
                raise TypeError("output must be an array")
            else:
                if out[0]._memloc in {on_VH, on_VE_VH}:
                    raise NotImplementedError(
                        "reduce_core on VH is not yet implemented.")
                out = out[0]

        axis_chk = axis

        if isinstance(axis_chk, int):
            axis_chk = (axis_chk,)
        elif isinstance(axis_chk, list):
            raise TypeError("'list' object cannot be interpreted as an integer")
        elif isinstance(axis_chk, tuple):
            for ax in axis_chk:
                if isinstance(ax, numpy.ndarray) or isinstance(ax, nlcpy.ndarray):
                    if ax.ndim > 0 or ax.dtype.char not in 'ilIL':
                        raise TypeError(
                            'only integer scalar arrays can be converted to '
                            'a scalar index')
                elif not isinstance(ax, int):
                    raise TypeError(
                        "'" + type(ax).__name__
                        + "' object cannot be interpreted as an integer")
        elif isinstance(axis_chk, numpy.ndarray) or isinstance(axis_chk, nlcpy.ndarray):
            if axis_chk.ndim > 0 or axis_chk.dtype.char not in 'ilIL':
                raise TypeError(
                    'only integer scalar arrays can be converted to a scalar index')
            axis_chk = (axis_chk,)

        if axis_chk is not None:
            axis_chk = list(axis_chk)

        if axis_chk is not None and array is not None:
            array_chk = nlcpy.asanyarray(array)
            for i in range(len(axis_chk)):
                if axis_chk[i] < 0:
                    axis_chk[i] = array_chk.ndim + axis_chk[i]

            for i in range(len(axis_chk)):
                if array_chk.ndim > 0:
                    if axis_chk[i] < 0 or axis_chk[i] > array_chk.ndim-1:
                        raise AxisError(
                            'axis ' + str(axis_chk[i])
                            + ' is out of bounds for array of dimension '
                            + str(array_chk.ndim))

        if axis_chk is not None:
            for ax in axis_chk:
                if axis_chk.count(ax) > 1:
                    raise ValueError("duplicate value in 'axis'")

        if isinstance(where, nlcpy.ndarray) or isinstance(where, numpy.ndarray):
            if where.dtype != bool:
                raise TypeError("Cannot cast array data from dtype('{}') to "
                                .format(where.dtype.name)
                                + "dtype('bool') according to the rule 'safe'")

        _msg = "No loop matching the specified signature" \
            + " and casting was found for ufunc " + op_name
        _ngcharlist = ""  # defaults NO-Exception
        if op_name in ("arctan2", "hypot", "logaddexp", "logaddexp2",
                       "heaviside", "copysign", "nextafter"):
            _ngcharlist = "?ilILFD"
        if op_name in ("floor_divide", "power"):
            _ngcharlist = "?"
        if op_name in ("divide", "true_divide"):
            _ngcharlist = "?ilIL"
        if op_name.startswith("bitwise_"):
            _ngcharlist = "fdFD"
        if op_name in ("mod", "remainder", "fmod"):
            _ngcharlist = "?FD"
        if op_name.endswith("_shift"):
            _ngcharlist = "?fdFD"
        if dtype is not None and dtype.char in _ngcharlist:
            raise TypeError(_msg)
        elif dtype is None and out is not None and out.dtype.char in _ngcharlist:
            raise TypeError(_msg)
        elif dtype is None and out is None and array is None:
            if op_name in ("logaddexp", "logaddexp2", "heaviside", "copysign",
                           "nextafter"):
                raise TypeError(_msg)
        elif dtype is None and out is None \
                and nlcpy.asanyarray(array).dtype.char in _ngcharlist:
            raise TypeError(_msg)

        if self.name == "nlcpy_subtract" and (numpy.dtype(dtype) == bool or (
                dtype is None and out is not None and out.dtype == bool)):
            raise TypeError("nlcpy boolean subtract, the `-` operator,"
                            " is deprecated, use the bitwise_xor, the `^` operator,"
                            " or the logical_xor function instead.")
        if array is None:
            dtype = bool if (
                op_name in ("less", "greater", "less_equal", "greater_equal",
                            "equal", "not_equal")
            ) else dtype
            if out is None and dtype is None:
                return None
            elif out is None:
                dtype = bool if self.name.startswith("nlcpy_logical_") else dtype
                return nlcpy.array(numpy.cast[dtype](None))
            else:
                out[()] = nlcpy.array(numpy.cast[out.dtype](None))
                return out

        array = nlcpy.asanyarray(array)

        if isinstance(out, nlcpy.ndarray):
            result_ndim = array.ndim if keepdims else 0 if axis is None \
                else array.ndim - (1 if numpy.isscalar(axis) else len(axis))

            if keepdims:
                if result_ndim != out.ndim:
                    raise ValueError("output parameter for reduction operation "
                                     + op_name + " has the wrong number of dimensions "
                                     + "(must match the operand's when keepdims=True)")
                else:
                    if isinstance(axis, int):
                        axis = (axis,)
                    elif axis is None:
                        axis = [i for i in range(array.ndim)]
                    if out.ndim > 0:
                        for i in axis:
                            if out.shape[i] != 1:
                                raise ValueError(
                                    "output parameter for reduction operation "
                                    + op_name + " has a reduction dimension "
                                    + "not equal to one "
                                    + "(required when keepdims=True)")

            if result_ndim > out.ndim:
                raise ValueError("output parameter for reduction operation "
                                 + op_name + " does not have enough dimensions")
            if 0 <= result_ndim < out.ndim:
                raise ValueError("output parameter for reduction operation "
                                 + op_name + " has too many dimensions")
        elif out is None:
            pass
        else:
            raise TypeError("output must be an array")

        return _reduce.reduce_core(self.name + '_reduce', array, axis=axis,
                                   dtype=dtype, out=out,
                                   keepdims=keepdims, initial=initial, where=where)

    def reduceat(self, array, indices, axis=0, dtype=None, out=None):
        """Performs a (local) reduce with specified slices over a single axis.

        For i in ``range(len(indices))``, reduceat computes
        ``ufunc.reduce(a[indices[i]:indices[i+1]])``,
        which becomes the i-th generalized "row" parallel to *axis* in the final result
        (i.e., in a 2-D array, for example, axis = 0, it becomes the i-th row, but if
        *axis = 1*, it becomes the i-th column). There are three exceptions to this:

        - when ``i = len(indices) - 1`` (so for the last index),
          ``indices[i+1] = a.shape[axis]``.
        - if ``indices[i] >= indices[i + 1]``, the i-th generalized "row" is simply
          ``a[indices[i]]``.
        - if ``indices[i] >= len(a)`` or ``indices[i] < 0`` , an error is raised.
          The shape of the output depends on the size of indices, and may be larger
          than *a* (this happens if ``len(indices) > a.shape[axis]``).

        Parameters
        ----------
        array : array_like
            The array to act on.
        indices : array_like
            Paired indices, comma separated (not colon), specifying slices to reduce.
        axis : int, optional
            The axis along which to apply the reduceat.
        dtype : dtype, optional
            The type used to represent the intermediate results. Defaults to the data
            type of the output array if this is provided, or the data type of the input
            array if no output array is provided.
        out : ndarray, None, or tuple of ndarray and None, optional
            A location into which the result is stored. If not provided or *None*, a
            freshly-allocated array is returned.

        Returns
        -------
        r : ndarray
            The reduced array. If *out* was supplied, *r* is a reference to *out*.

        Restriction
        -----------
        - If a list is passed to the parameter *a* of `power.reduceat()`,
          there are cases where *ValueError* occurs.

        Note
        ----
        A descriptive example:

        If *a* is 1-D, the function `ufunc.accumulate(a)` is the same as
        ``ufunc.reduceat(a, indices)[::2]`` where indices is ``range(len(array) - 1)``
        with a zero placed in every other element: ``indices = zeros(2 * len(a) - 1)``,
        ``indices[1::2] = range(1, len(a))``.

        Don't be fooled by this attribute's name: `reduceat(a)` is not necessarily
        smaller than *a*.

        Examples
        --------

        To take the running sum of four successive values:

        >>> import nlcpy as vp
        >>> vp.add.reduceat(vp.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
        array([ 6, 10, 14, 18])

        A 2-D example:

        >>> x = vp.linspace(0, 15, 16).reshape(4,4)
        >>> x    # doctest: +SKIP
        array([[ 0.,  1.,  2.,  3.],
               [ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.],
               [12., 13., 14., 15.]])
        # reduce such that the result has the following five rows:
        # [row1 + row2 + row3]
        # [row4]
        # [row2]
        # [row3]
        # [row1 + row2 + row3 + row4]
        >>> vp.add.reduceat(x, [0, 3, 1, 2, 0])   # doctest: +SKIP
        array([[12., 15., 18., 21.],
               [12., 13., 14., 15.],
               [ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.],
               [24., 28., 32., 36.]])
        # reduce such that result has the following two columns:
        # [col1 * col2 * col3, col4]
        >>> vp.multiply.reduceat(x, [0, 3], 1)  # doctest: +SKIP
        array([[   0.,    3.],
               [ 120.,    7.],
               [ 720.,   11.],
               [2184.,   15.]])
        """
        return _reduceat.reduceat_core(self.name + '_reduceat', array, indices=indices,
                                       axis=axis, dtype=dtype, out=out)

    def accumulate(self, array, axis=0, dtype=None, out=None):
        """Accumulates the result of applying the operator to all elements.

        For example, ``nlcpy.add.accumulate()`` is equivalent to
        :func:`nlcpy.cumsum()`.
        For a multi-dimensional array, accumulate is applied along only one axis (axis
        zero by default; see Examples below) so repeated use is necessary if one wants to
        accumulate over multiple axes.

        Parameters
        ----------
        array : array_like
            The array to act on.
        axis : int, optional
            The axis along which to apply the accumulation; default is zero.
        dtype : dtype, optional
            The type used to represent the intermediate results. Defaults to the data
            type of the output array if such is provided, or the data type of the input
            array if no output array is provided.
        out : ndarray, None, or tuple of ndarray and None, optional
            A location into which the result is stored. If not provided or *None*, a
            freshly-allocated array is returned.

        Returns
        -------
        r : ndarray
            The accumulated array. If *out* was supplied, *r* is a reference to *out*.

        Restriction
        -----------
        - If a list is passed to the parameter *a* of `power.reduceat()`,
          there are cases where *ValueError* occurs.

        Examples
        --------

        1-D array examples:

        >>> import nlcpy as vp
        >>> vp.add.accumulate([2, 3, 5])
        array([ 2,  5, 10])
        >>> vp.multiply.accumulate([2, 3, 5])
        array([ 2,  6, 30])

        2-D array examples:

        >>> I = vp.eye(2)
        >>> I
        array([[1., 0.],
               [0., 1.]])

        Accumulate along axis 0 (rows), down columns:

        >>> vp.add.accumulate(I, 0)
        array([[1., 0.],
               [1., 1.]])
        >>> vp.add.accumulate(I) # no axis specified = axis zero
        array([[1., 0.],
               [1., 1.]])

        Accumulate along axis 1 (columns), through rows:

        >>> vp.add.accumulate(I, 1)
        array([[1., 1.],
               [0., 1.]])

        """
        return _accumulate.accumulate_core(self.name + '_accumulate', array, axis=axis,
                                           dtype=dtype, out=out)

    def outer(self, A, B, **kwargs):
        """Applies the ufunc *op* to all pairs (a, b) with a in *A* and b in *B*.

        Let ``M = A.ndim``, ``N = B.ndim``. Then the result, *C*, of ``op.outer(A, B)``
        is an array of dimension M + N such that:

        .. math::

            C[i_{0}, ..., i_{M-1}, j_{0}, ..., j_{N-1}] = op(A[i_{0}, ...,
            i_{M-1}],B[j_{0}, ..., j_{N-1}])

        Parameters
        ----------
        A : (M,) array_like
            First array
        B : (N,) array_like
            Second array
        kwargs : any
            Arguments to pass on to the ufunc. Typically dtype or out.

        Returns
        -------
        r : ndarray
            Output array

        Examples
        --------
        >>> import nlcpy as vp
        >>> vp.multiply.outer([1, 2, 3], [4, 5, 6])
        array([[ 4,  5,  6],
               [ 8, 10, 12],
               [12, 15, 18]])

        A multi-dimensional example:

        >>> A = vp.array([[1, 2, 3], [4, 5, 6]])
        >>> A.shape
        (2, 3)
        >>> B = vp.array([[1, 2, 3, 4]])
        >>> B.shape
        (1, 4)
        >>> C = vp.multiply.outer(A, B)
        >>> C.shape; C
        (2, 3, 1, 4)
        array([[[[ 1,  2,  3,  4]],
        <BLANKLINE>
                [[ 2,  4,  6,  8]],
        <BLANKLINE>
                [[ 3,  6,  9, 12]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 4,  8, 12, 16]],
        <BLANKLINE>
                [[ 5, 10, 15, 20]],
        <BLANKLINE>
                [[ 6, 12, 18, 24]]]])

        """

        out = kwargs.pop('out', None)
        where = kwargs.pop('where', True)
        casting = kwargs.pop('casting', self._default_casting)
        order = kwargs.pop('order', 'K')
        dtype = kwargs.pop('dtype', None)
        if dtype is not None:
            dtype = _dtype.get_dtype(dtype).type
        subok = kwargs.pop('subok', True)  # changed from parse_argument
        if subok is False:
            raise NotImplementedError
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        return _outer.outer_core(self.name + '_outer', A, B, out=out, where=where,
                                 casting=casting, order=order, dtype=dtype, subok=subok)

    def at(a, indices, b=None):
        raise NotImplementedError


cpdef convert_args_to_ndarray(list args):
    ret = []
    for i in args:
        if isinstance(i, ndarray):
            ret.append(i)
        elif numpy.isscalar(i):
            ret.append(i)
        else:
            ret.append(nlcpy.asarray(i))
    return ret


cpdef parse_argument(self, kwargs):
    out = kwargs.pop('out', None)
    where = kwargs.pop('where', True)
    casting = kwargs.pop('casting', self._default_casting)
    order = kwargs.pop('order', 'K')
    dtype = kwargs.pop('dtype', None)
    if dtype is not None:
        dtype = _dtype.get_dtype(dtype).type
    subok = kwargs.pop('subok', False)
    if subok:
        raise NotImplementedError
    if kwargs:
        raise TypeError('Wrong arguments %s' % kwargs)
    if where is None:
        where = False
    return out, where, casting, order, dtype, subok


cpdef guess_out_order(order, args):
    if order is None:
        order = 'K'
    order_char = internal._normalize_order(order)
    if order_char == b'F':
        order_out = 'F'
    elif order_char == b'C':
        order_out = 'C'
    elif order_char == b'K' or order_char == b'A':
        for x in args:
            if x.flags.f_contiguous and not x.flags.c_contiguous:
                order_out = 'F'
                break
        else:
            order_out = 'C'
    else:
        raise ValueError('unknown order was detected.')
    return order_out


cpdef push_ufunc(name, args, nin):
    request._push_request(
        name,
        "binary_op" if nin == 2 else "unary_op",
        args,
    )


cpdef create_ufunc(name, types, err_func,
                   default_casting=None, doc=''):
    _types = []
    for t in types:
        typ = t

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple([_dtype.get_dtype(t).type for t in in_types])
        out_types = tuple([_dtype.get_dtype(t).type for t in out_types])
        if None not in (in_types) and None not in (out_types):
            _types.append((in_types, out_types))

    ret = ufunc(name, len(_types[0][0]), len(_types[0][1]), _types,
                err_func, default_casting=default_casting, doc=doc)
    return ret

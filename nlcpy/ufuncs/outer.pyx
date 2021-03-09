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
import numpy as np
import ctypes
import sys

import nlcpy
from nlcpy import veo
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport manipulation
from nlcpy.core cimport core
from nlcpy.core cimport broadcast
from nlcpy.core.core cimport *
from nlcpy.manipulation.shape import reshape
from nlcpy.core cimport internal
from nlcpy.request cimport request
from nlcpy.core.dtype cimport get_dtype_number, get_dtype

cimport numpy as np


def get_contiguity(x):
    if x._c_contiguous or x._f_contiguous:
        return True

    sorted_set = sorted(zip(x.strides, x.shape), reverse=True)
    sorted_strides = [i[0] for i in sorted_set]
    sorted_shape = [i[1] for i in sorted_set]

    return internal.get_c_contiguity(sorted_shape, sorted_strides, x.itemsize)


def get_calc_dtype_bool_bool(ufunc_name):
    if ufunc_name in ("add", "subtract", "multiply", "maximum", "minimum",
                      "logical_and", "logical_or", "logical_xor",
                      "bitwise_and", "bitwise_or", "bitwise_xor",
                      "less", "greater", "less_equal", "greater_equal",
                      "equal", "not_equal",
                      "fmax", "fmin"):
        return np.dtype("bool")

    elif ufunc_name in ("floor_divide", "mod", "remainder", "power",
                        "right_shift", "left_shift", "fmod"):
        return np.dtype("int8")

    elif ufunc_name in ("arctan2", "hypot", "logaddexp", "logaddexp2",
                        "heaviside", "copysign", "nextafter", "ldexp"):
        return np.dtype("float16")

    elif ufunc_name in ("divide", "true_divide"):
        return np.dtype("float64")

    else:
        raise AttributeError("module 'nlcpy' has no attribute '{}'".format(ufunc_name))


def equiv_scalar(x):
    if np.isscalar(x):
        return True
    elif isinstance(x, np.ndarray):
        if x.ndim == 0:
            return True
        else:
            return False
    else:  # nlcpy.ndarray, list, tuple
        # regard nlcpy.ndarray with ndim=0 as NOT scalar
        return False


def get_how_treat_dtype(ufunc_name, x, arg_x, is_scalar_x):
    if ufunc_name in ("add", "subtract", "multiply", "maximum", "minimum",
                      "logical_and", "logical_or", "logical_xor",
                      "bitwise_and", "bitwise_or", "bitwise_xor",
                      "less", "greater", "less_equal", "greater_equal",
                      "equal", "not_equal",
                      "fmax", "fmin",
                      "floor_divide", "mod", "remainder", "power",
                      "right_shift", "left_shift", "fmod"):
        return arg_x if is_scalar_x else x.dtype

    elif ufunc_name in ("arctan2", "hypot", "logaddexp", "logaddexp2",
                        "heaviside", "copysign", "nextafter",
                        "divide", "true_divide"):
        if x.dtype.kind == 'i' or x.dtype.kind == 'u':
            return float(arg_x) if is_scalar_x else np.dtype("float64")
        else:
            return arg_x if is_scalar_x else x.dtype

    elif ufunc_name == "ldexp":
        if x.dtype == "bool":
            return np.dtype("float16")
        elif x.dtype.kind == 'i' or x.dtype.kind == 'u':
            return np.dtype("float64")
        else:
            return x.dtype

    else:
        raise AttributeError(
            "module 'nlcpy' has no attribute '{}'".format(ufunc_name))


def get_calc_dtype(ufunc_name, A, arg_A, is_scalar_A, B, arg_B, is_scalar_B):
    # quick return for bool * bool
    if A.dtype == 'bool' and B.dtype == 'bool':
        return get_calc_dtype_bool_bool(ufunc_name)

    treat_A = get_how_treat_dtype(ufunc_name, A, arg_A, is_scalar_A)
    if ufunc_name != "ldexp":
        treat_B = get_how_treat_dtype(ufunc_name, B, arg_B, is_scalar_B)
        return np.result_type(treat_A, treat_B)
    else:  # ldexp
        if B.dtype == "uint64" or B.dtype.kind == 'f' or B.dtype.kind == 'c':
            raise TypeError("ufunc '{}' not supported for the "
                            "input types,".format(ufunc_name),
                            "and the inputs could not be safely coerced to any "
                            "supported types according to the casting rule ''safe''")
        return treat_A


def check_calc_type(ufunc_name, calc_dtype):

    if ufunc_name == 'subtract' and calc_dtype=='bool':
        raise TypeError('numpy boolean subtract, the `-` operator, is deprecated, '
                        'use the bitwise_xor, the `^` operator, '
                        'or the logical_xor function instead.')

    calc_dtype_name = calc_dtype.name
    if calc_dtype_name=='bool':
        if ufunc_name in ("divide", "floor_divide", "true_divide", "mod", "remainder",
                          "power", "right_shift", "left_shift", "fmod",
                          "arctan2", "hypot", "logaddexp", "logaddexp2",
                          "heaviside", "copysign", "nextafter", "ldexp"):
            raise TypeError("No loop matching the specified signature and casting\n"
                            + "was found for ufunc " + ufunc_name)

    elif 'int' in calc_dtype_name:
        if ufunc_name in ("divide", "true_divide",
                          "arctan2", "hypot", "logaddexp", "logaddexp2",
                          "heaviside", "copysign", "nextafter", "ldexp"):
            raise TypeError("No loop matching the specified signature and casting\n"
                            + "was found for ufunc " + ufunc_name)

    elif 'float' in calc_dtype_name:
        if ufunc_name in ("bitwise_and", "bitwise_or", "bitwise_xor",
                          "right_shift", "left_shift"):
            raise TypeError("ufunc '{}' not supported for the input types, and "
                            "the inputs could not be safely coerced to any supported "
                            "types according to the casting "
                            "rule ''safe''".format(ufunc_name))

    elif 'complex' in calc_dtype_name:
        if ufunc_name in ("mod", "remainder",
                          "bitwise_and", "bitwise_or", "bitwise_xor",
                          "right_shift", "left_shift", "fmod",
                          "arctan2", "hypot", "logaddexp", "logaddexp2",
                          "heaviside", "copysign", "nextafter", "ldexp"):
            ufunc_name = "remainder" if ufunc_name == "mod" else ufunc_name
            raise TypeError("ufunc '{}' not supported for the input types, and "
                            "the inputs could not be safely coerced to any supported "
                            "types according to the casting "
                            "rule ''safe''".format(ufunc_name))


def promote_dtype(dtype):
    if dtype in ("int8", "int16"):
        return np.dtype("int32")
    elif dtype == "float16":
        return np.dtype("float32")
    else:
        return dtype


def str_shape(shape):
    return str(shape).replace(" ", "") + " "


def raise_binary_broadcast_err(a_shape, b_shape, out_shape):
    work_shape = a_shape + b_shape
    cmp_range = -min(len(work_shape), len(out_shape))
    for i in range(-1, cmp_range-1, -1):
        if work_shape[i] != out_shape[i] and work_shape[i] != 1:
            raise ValueError("operands could not be broadcast together with shapes "
                             + str_shape(a_shape + (1,) * len(b_shape))
                             + str_shape(b_shape) + str_shape(out_shape))
    else:
        raise ValueError("non-broadcastable output operand with shape "
                         + str_shape(out_shape)
                         + "doesn't match the broadcast shape "
                         + str_shape(work_shape).rstrip())


def check_can_cast(from_dtype, to_dtype, casting, ufunc_name, str_where):
    if np.can_cast(from_dtype, to_dtype, casting=casting):
        return None
    else:
        return "Cannot cast ufunc '{}' {} from dtype('{}') to dtype('{}') with "\
            "casting rule '{}'".format(ufunc_name, str_where,
                                       from_dtype.name, to_dtype.name, casting)


def check_can_cast2(from_obj, from_dtype, to_dtype, casting, ufunc_name, str_where):
    if np.can_cast(from_obj, to_dtype, casting=casting):
        return None
    else:
        return "Cannot cast ufunc '{}' {} from dtype('{}') to dtype('{}') with "\
            "casting rule '{}'".format(ufunc_name, str_where,
                                       str(from_dtype), str(to_dtype), casting)


##################################
#
#     Main method
#
##################################
cpdef outer_core(name, A, B, out=None, where=True,
                 casting='same_kind', order='K', dtype=None, subok=True):

    if A is None:
        raise NotImplementedError("not supported for A is None.")

    if B is None:
        raise NotImplementedError("not supported for B is None.")

    if casting == 'unsafe':
        raise NotImplementedError("not supported for casting == 'unsafe'.")

    # keep the arguments A and B as they are not to use get() when decide calc_dtype
    arg_A = A
    arg_B = B

    is_scalar_B = equiv_scalar(B)
    # A is scalar equivalent only if both A and B are scalar equivalent, for outer
    is_scalar_A = equiv_scalar(A) and is_scalar_B

    # convert to nlcpy.ndarray
    A = core.argument_conversion(A)
    B = core.argument_conversion(B)

    ufunc_name = name[6:-6]  # use for error message

    # strip tuple
    if isinstance(out, tuple):
        if len(out) == 1:
            out = out[0]
        else:
            raise ValueError(
                "The 'out' tuple must have exactly one entry per ufunc output")

    # check whether out is valid
    if not(out is None or isinstance(out, ndarray)):
        raise TypeError("return arrays must be of ArrayType")

    # decide calc_dtype
    ###
    if casting in ('no', 'equiv'):
        if A.dtype != B.dtype:
            raise TypeError(
                "Cannot cast ufunc '{}' input 0 from dtype('{}') to dtype('{}') with "
                "casting rule '{}'".format(ufunc_name, A.dtype, B.dtype, casting))
        calc_dtype = A.dtype
    else:
        calc_dtype = get_dtype(dtype) if dtype is not None else \
            get_calc_dtype(ufunc_name, A, arg_A, is_scalar_A, B, arg_B, is_scalar_B)
    check_calc_type(ufunc_name, calc_dtype)
    # extra check for 2nd argument of ldexp
    if ufunc_name == "ldexp":
        if dtype is not None:
            if B.dtype.kind == 'f' or B.dtype.kind == 'c':  # uint64 is allowed
                raise TypeError("No loop matching the specified signature and "
                                "casting was found for ufunc ldexp")

    # decide work_dtype(dtype of workspace)
    promoted_dtype = False
    if ufunc_name in ("logical_and", "logical_or", "logical_xor",
                      "less", "greater", "less_equal", "greater_equal",
                      "equal", "not_equal"):
        work_dtype = np.dtype("bool")
    else:
        if calc_dtype in ("int8", "int16", "float16"):
            if out is None:  # out_dtype is clac_dtype
                raise TypeError("not support for int8, int16, and float16.")
            elif out.dtype == "bool":
                # rasie TypeError before promoted_dtype
                err_msg = check_can_cast(calc_dtype, out.dtype, casting,
                                         ufunc_name, "output")
                if err_msg is not None:
                    raise TypeError(err_msg)
            calc_dtype = promote_dtype(calc_dtype)
            promoted_dtype = True

        work_dtype = calc_dtype

    # decide out_dtype
    out_dtype = out.dtype if out is not None else work_dtype

    if dtype is not None:
        temp_A = arg_A if is_scalar_A else A
        err_msg = check_can_cast2(temp_A, A.dtype, calc_dtype, casting,
                                  ufunc_name, "input 0")
        if err_msg is not None:
            raise TypeError(err_msg)
        if ufunc_name != "ldexp":
            temp_B = arg_B if is_scalar_B else B
            err_msg = check_can_cast2(temp_B, B.dtype, calc_dtype, casting,
                                      ufunc_name, "input 1")
            if err_msg is not None:
                raise TypeError(err_msg)

    should_check_cast = True
    # where following two cases, numpy doesn't raise TypeError although
    # check_can_cast will return err then not check can cast exceptionally
    if promoted_dtype and work_dtype == "int32" and out_dtype.kind == 'u':
        should_check_cast = False

    elif ufunc_name == "floor_divide":
        # when all of the following three conditions are satisfied,
        # can unsafe cast from int to uint on output
        # and calc_dtype and work_dtype are "int64"
        #   1. (A.dtype, B.dtype) is pair of ("int32" or "int64") and "uint64"
        #      (thier result_type = "float64")
        #   2. dtype is None
        #   3. out_dtype.kind is 'i' or 'u'
        int_kinds = ('i', 'u')
        if dtype is None and out_dtype.kind == 'i':
            if A.dtype.kind in int_kinds and B.dtype.kind in int_kinds and \
               calc_dtype == "float64":
                calc_dtype = work_dtype = np.dtype("int64")
                should_check_cast = False

    if should_check_cast:
        err_msg = check_can_cast(work_dtype, out_dtype, casting, ufunc_name, "output")
        if err_msg is not None:
            raise TypeError(err_msg)

    # check out.shape for broadcast
    work_shape = A.shape + B.shape
    if out is not None:
        out_shape = out.shape
        if out_shape == work_shape:
            bcast_dim = nlcpy.empty(0, dtype="bool")
        else:
            diff_dim = out.ndim - (A.ndim + B.ndim)
            if diff_dim < 0:  # out.ndim < A.ndim + B.ndim
                raise_binary_broadcast_err(A.shape, B.shape, out.shape)
            else:  # diff_dim >= 0
                bcast_dim = nlcpy.empty(out.ndim, dtype='bool')
                # all of dimensions that is less than diff_dim need braodcast
                bcast_dim[0:diff_dim] = True
                for i in range(A.ndim + B.ndim):
                    if out.shape[diff_dim + i] == work_shape[i]:
                        bcast_dim[diff_dim + i] = False
                    else:
                        if work_shape[i] == 1:
                            bcast_dim[diff_dim + i] = True
                        else:
                            raise_binary_broadcast_err(A.shape, B.shape, out.shape)

    ########################################################################
    # make workspace whose strides is good for vectorize

    # get the shape of workspace when transposed to be c_contiguous
    soted_set_A = sorted(zip(A.strides, A.shape, range(A.ndim)), reverse=True)
    soted_set_B = sorted(zip(B.strides, B.shape, range(B.ndim)), reverse=True)
    if A.size > B.size:
        where_vectorize = 0  # vectorize for input 0 (=A)

        work_c_contiguous_shape = [i[1] for i in soted_set_B] \
            + [i[1] for i in soted_set_A]

        # get the list to pass to nlcpy.transpose from c_contiguous shape
        # to original shape
        trans_A = [-1, ] * A.ndim  # make list whose length is A.ndim
        for i in range(A.ndim):
            trans_A[soted_set_A[i][2]] = i  # soted_set_A[i][2]=j => trans_A[j]=i
        trans_A =[i + B.ndim for i in trans_A]

        trans_B = [-1, ] * B.ndim
        for i in range(B.ndim):
            trans_B[soted_set_B[i][2]] = i

        trans_out = trans_A + trans_B

        workspace = nlcpy.empty(work_c_contiguous_shape, dtype=work_dtype,
                                order='C').transpose(trans_out)
    else:  # A.size ≦ B.size
        where_vectorize = 1  # vectorize for input 1 (=B)

        work_c_contiguous_shape = [i[1] for i in soted_set_A] \
            + [i[1] for i in soted_set_B]

        # get the list to pass to nlcpy.transpose from c_contiguous shape
        # to original shape
        trans_A = [-1, ] * A.ndim  # make list whose length is A.ndim
        for i in range(A.ndim):
            trans_A[soted_set_A[i][2]] = i  # soted_set_A[i][2]=j => trans_A[j]=i

        trans_B = [-1, ] * B.ndim
        for i in range(B.ndim):
            trans_B[soted_set_B[i][2]] = i
        trans_B =[i + A.ndim for i in trans_B]

        trans_out = trans_A + trans_B

        workspace = nlcpy.empty(work_c_contiguous_shape, dtype=work_dtype,
                                order='C').transpose(trans_out)

    if out is None:
        out_shape = A.shape+B.shape
        bcast_dim = nlcpy.empty(0, dtype="bool")
        if order == 'C' or order == 'c':
            out = nlcpy.empty(out_shape, order='C', dtype=out_dtype)

        elif order == 'F' or order == 'f':
            out = nlcpy.empty(out_shape, order='F', dtype=out_dtype)

        elif order == 'A' or order == 'a':
            if A._f_contiguous and B._f_contiguous:
                out = nlcpy.empty(out_shape, order='F', dtype=out_dtype)
            else:
                out = nlcpy.empty(out_shape, order='C', dtype=out_dtype)

        elif order == 'K' or order == 'k':
            out = nlcpy.empty(work_c_contiguous_shape, dtype=out_dtype,
                              order='C').transpose(trans_out)

        else:
            raise ValueError("unknown order was detected.")

    flag_bcast = 1 if out is not None and out_shape != work_shape else 0

    ########################################################################
    # make where to pass VE

    # check where can cast to bool with casting="safe".
    # if where is not ndarray (e.g. 99, [0,1,2,3]), we don't check.
    if isinstance(where, ndarray):
        if not np.can_cast(where.dtype, bool, casting='safe'):
            raise TypeError("Cannot cast array data from dtype('" + str(where.dtype)
                            + "') to dtype('bool') according to the rule 'safe'")

    if nlcpy.all(where):
        # do not use where
        flag_where = 0
        where = nlcpy.asanyarray([True], dtype="bool")

    elif not nlcpy.any(where):
        # do not any calculate
        return out

    else:
        # use where
        flag_where = 1
        where = nlcpy.asanyarray(where, dtype="bool")
        try:
            where = broadcast.broadcast_to(where, out.shape)
        except ValueError:
            raise_binary_broadcast_err(A.shape, B.shape, where.shape)

    # size of A and/or B is zero
    # this quick return occurs when any error above was not raised
    if A.size == 0 or B.size == 0:
        return out

    if ufunc_name == "power":
        if np.dtype(calc_dtype).kind in (ord('i'), ord('u')):
            if nlcpy.any(nlcpy.less(B, 0)):
                raise ValueError("Integers to negative integer powers are not allowed.")

    # make copy to increase vector efficiency if need
    x = A if get_contiguity(A) else A.copy(order='K')
    y = B if get_contiguity(B) else B.copy(order='K')

    #######################################################################
    # TODO: VE-VH collaboration
    if A._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError('amax on VH is not yet implemented.')

    if out is not None:
        if out._memloc in {on_VH, on_VE_VH}:
            raise NotImplementedError('amax on VH is not yet implemented.')

    if flag_bcast:
        bcast_src = nlcpy.empty(work_shape, order='C', dtype=out_dtype)
    else:
        bcast_src = nlcpy.empty(1)

    request._push_request(
        name,
        "outer_op",
        (out, x, y, where_vectorize,
         flag_where, where,
         flag_bcast, bcast_dim,
         workspace, bcast_src, ),
    )
    return out

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
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.core.dtype cimport get_dtype_number, get_dtype
from nlcpy.request cimport request

cimport numpy as cnp

cpdef accumulate_core(name, a, axis=0, dtype=None, out=None):

    if dtype is not None:
        dtype = nlcpy.dtype(dtype)
    if name in ('nlcpy_invert_accumulate', 'nlcpy_logical_not_accumualte'):
        raise ValueError("accumulate only supported for binary functions")

    input_arr = core.argument_conversion(a)

    if input_arr is None or input_arr.ndim == 0:
        raise TypeError("cannot accumulate on a scalar")

    ########################################################################
    # TODO: VE-VH collaboration
    if input_arr._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError(
            "accumulate_core on VH is not yet impremented")

    ########################################################################
    # check order
    if input_arr._f_contiguous and not input_arr._c_contiguous:
        order_out = 'F'
    else:
        order_out = 'C'

    ########################################################################
    # error check for "out"
    output_flg = False
    if out is not None:
        if type(out) != nlcpy.ndarray and type(out) != tuple:
            raise TypeError("output must be an array")

        if isinstance(out, tuple):
            if len(out) != 1:
                raise ValueError("The 'out' tuple must have exactly one entry")
            elif not isinstance(out[0], nlcpy.ndarray):
                raise TypeError("output must be an array")
            else:
                # TODO: VE-VH collaboration
                if out[0]._memloc in {on_VH, on_VE_VH}:
                    raise NotImplementedError(
                        "accumulate_core on VH is not yet implemented.")
                out = out[0]

        if isinstance(out, nlcpy.ndarray):
            if input_arr.ndim == 1:
                if name in ('nlcpy_divide_accumulate',
                            'nlcpy_true_divide_accumulate',
                            'nlcpy_logaddexp_accumulate',
                            'nlcpy_logaddexp2_accumulate',
                            'nlcpy_logical_and_accumulate',
                            'nlcpy_logical_or_accumulate',
                            'nlcpy_logical_xor_accumulate',
                            'nlcpy_less_accumulate',
                            'nlcpy_greater_accumulate',
                            'nlcpy_less_equal_accumulate',
                            'nlcpy_greater_equal_accumulate',
                            'nlcpy_equal_accumulate',
                            'nlcpy_not_equal_accumulate',
                            'nlcpy_arctan2_accumulate',
                            'nlcpy_hypot_accumulate',
                            'nlcpy_heaviside_accumulate',
                            'nlcpy_copysign_accumulate',
                            'nlcpy_nextafter_accumulate',
                            ):

                    if input_arr.shape[0] != out.shape[0]:
                        raise ValueError("operands could not be broadcast together with "
                                         +"remapped shapes [original->remapped]: "
                                         +str(out.shape).replace(" ", "")+"->"
                                         +str(out.shape[0:input_arr.ndim])
                                         .replace(" ", "")+" "
                                         +str(input_arr.shape).replace(" ", "")+"->"
                                         +str(input_arr.shape).replace(" ", "")+" ")
                    else:
                        y = out
                        output_flg = True

                else:
                    if out.ndim > 1 or input_arr.shape[0] != out.shape[0]:
                        raise ValueError(
                            "provided out is the wrong size for the reduction")
                    else:
                        y = out
                        output_flg = True

            else:
                if input_arr.ndim <= out.ndim:
                    for i in range(input_arr.ndim):
                        if input_arr.shape[i] != out.shape[i]:
                            raise ValueError("operands could not be broadcast "
                                             +"together with remapped shapes "
                                             +"[original->remapped]: "
                                             +str(out.shape).replace(" ", "")+"->"
                                             +str(out.shape[0:input_arr.ndim])
                                             .replace(" ", "")+" "
                                             +str(input_arr.shape).replace(" ", "")
                                             +"->"+str(input_arr.shape).replace(" ", "")
                                             +" ")
                        # TODO: VE-VH collaboration
                        if out._memloc in {on_VH, on_VE_VH}:
                            raise NotImplementedError(
                                "accumulate_core on VH is not yet implemented.")

                    y = out
                    output_flg = True
                else:
                    raise ValueError("Iterator input op_axes[0]["
                                     +str(input_arr.ndim-out.ndim-1)+"] "
                                     +"(=="+str(out.ndim)
                                     +") is not a valid axis of op[0], "
                                     +"which has "+str(out.ndim)+" dimensions ")

        if output_flg and dtype is None:
            if name in ('nlcpy_logaddexp_accumulate',
                        'nlcpy_logaddexp2_accumulate',
                        'nlcpy_remainder_accumulate',
                        'nlcpy_mod_accumulate',
                        'nlcpy_fmod_accumulate',
                        'nlcpy_heaviside_accumulate',
                        'nlcpy_arctan2_accumulate',
                        'nlcpy_hypot_accumulate',
                        'nlcpy_copysign_accumulate',
                        'nlcpy_nextafter_accumulate',
                        ):
                if out.dtype == 'complex64':
                    raise ValueError("could not find a matching type for "+name
                                     +", requested type has type code 'F'")
                elif out.dtype == 'complex128':
                    raise ValueError("could not find a matching type for "+name
                                     +", requested type has type code 'D'")

            elif name in ('nlcpy_bitwise_and_accumulate',
                          'nlcpy_bitwise_or_accumulate',
                          'nlcpy_bitwise_xor_accumulate',
                          'nlcpy_right_shift_accumulate',
                          'nlcpy_left_shift_accumulate',
                          ):
                if out.dtype == 'float32':
                    raise ValueError("could not find a matching type for "+name
                                     +", requested type has type code 'f'")
                elif out.dtype == 'complex64':
                    raise ValueError("could not find a matching type for "+name
                                     +", requested type has type code 'F'")
                elif out.dtype == 'float64':
                    raise ValueError("could not find a matching type for "+name
                                     +", requested type has type code 'd'")
                elif out.dtype == 'complex128':
                    raise ValueError("could not find a matching type for "+name
                                     +", requested type has type code 'D'")

    ########################################################################
    # dtype check
    if dtype is None and out is None:
        if name in ('nlcpy_logaddexp_accumulate',
                    'nlcpy_logaddexp2_accumulate',
                    'nlcpy_remainder_accumulate',
                    'nlcpy_mod_accumulate',
                    'nlcpy_fmod_accumulate',
                    'nlcpy_heaviside_accumulate',
                    'nlcpy_arctan2_accumulate',
                    'nlcpy_hypot_accumulate',
                    'nlcpy_copysign_accumulate',
                    'nlcpy_nextafter_accumulate',
                    ):
            if input_arr.dtype.name in ('complex64'):
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'F'")
            elif input_arr.dtype.name in ('complex128'):
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'D'")

        elif name in ('nlcpy_bitwise_and_accumulate',
                      'nlcpy_bitwise_or_accumulate',
                      'nlcpy_bitwise_xor_accumulate',
                      'nlcpy_right_shift_accumulate',
                      'nlcpy_left_shift_accumulate',
                      ):
            if input_arr.dtype.name in ('float32'):
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'f'")
            elif input_arr.dtype.name in ('complex64'):
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'F'")
            elif input_arr.dtype.name in ('float64'):
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'd'")
            elif input_arr.dtype.name in ('complex128'):
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'D'")

        if type(dtype) == str and dtype.find(',') > 0:
            raise TypeError('cannot perform accumulate with flexible type')

    if dtype is None:
        dt = input_arr.dtype if out is None else out.dtype
        if name in ('nlcpy_add_accumulate', 'nlcpy_multiply_accumulate'):
            if input_arr.dtype.name in ('bool', 'int32'):
                dt = 'int64'
            elif input_arr.dtype.name in ('uint32'):
                dt = 'uint64'

        elif name in ('nlcpy_greater_accumulate',
                      'nlcpy_less_accumulate',
                      'nlcpy_greater_equal_accumulate',
                      'nlcpy_less_equal_accumulate',
                      'nlcpy_not_equal_accumulate',
                      'nlcpy_equal_accumulate',
                      'nlcpy_logical_and_accumulate',
                      'nlcpy_logical_or_accumulate',
                      'nlcpy_logical_xor_accumulate',
                      ):
            dt = 'bool'
            input_arr = nlcpy.array(input_arr, dtype='bool')

        elif name in ('nlcpy_logaddexp_accumulate',
                      'nlcpy_logaddexp2_accumulate',
                      'nlcpy_divide_accumulate',
                      'nlcpy_true_divide_accumulate',
                      'nlcpy_heaviside_accumulate',
                      'nlcpy_arctan2_accumulate',
                      'nlcpy_hypot_accumulate',
                      'nlcpy_copysign_accumulate',
                      'nlcpy_nextafter_accumulate',
                      ):
            if dt.name == 'bool':
                if out is None:
                    raise TypeError("not support for float16.")
                else:
                    dt = 'float32'

            elif dt.name not in ('float32', 'float64', 'complex64', 'complex128'):
                dt = 'float64'

        elif name in ('nlcpy_floor_divide_accumulate',
                      'nlcpy_power_accumulate',
                      'nlcpy_remainder_accumulate',
                      'nlcpy_mod_accumulate',
                      'nlcpy_fmod_accumulate',
                      'nlcpy_left_shift_accumulate',
                      'nlcpy_right_shift_accumulate',
                      'nlcpy_subtract_accumulate',
                      ):
            if dt.name == 'bool':
                # dt = 'int8' nlcpy can't use
                if out is None:
                    raise TypeError("not support for int8.")
                else:
                    dt = 'int32'

    else:
        # argument dtype = int8/int16/float16 -> not supported error
        if dtype in ('int8', 'int16', 'float16'):
            raise TypeError("not support for %s." % (dtype))

        if name in ('nlcpy_logaddexp_accumulate',
                    'nlcpy_logaddexp2_accumulate',
                    'nlcpy_remainder_accumulate',
                    'nlcpy_mod_accumulate',
                    'nlcpy_fmod_accumulate',
                    'nlcpy_heaviside_accumulate',
                    'nlcpy_arctan2_accumulate',
                    'nlcpy_hypot_accumulate',
                    'nlcpy_copysign_accumulate',
                    'nlcpy_nextafter_accumulate',
                    ):
            if dtype == 'complex64':
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'F'")
            elif dtype == 'complex128':
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'D'")

        elif name in ('nlcpy_bitwise_and_accumulate',
                      'nlcpy_bitwise_or_accumulate',
                      'nlcpy_bitwise_xor_accumulate',
                      'nlcpy_right_shift_accumulate',
                      'nlcpy_left_shift_accumulate',
                      ):
            if dtype == 'float32':
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'f'")
            elif dtype == 'complex64':
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'F'")
            elif dtype == 'float64':
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'd'")
            elif dtype == 'complex128':
                raise ValueError("could not find a matching type for "+name
                                 +", requested type has type code 'D'")

        if type(dtype) == str and dtype.find(',') > 0:
            raise TypeError('cannot perform accumulate with flexible type')

        if name in ('nlcpy_greater_accumulate',
                    'nlcpy_less_accumulate',
                    'nlcpy_greater_equal_accumulate',
                    'nlcpy_less_equal_accumulate',
                    'nlcpy_not_equal_accumulate',
                    'nlcpy_equal_accumulate',
                    'nlcpy_logical_and_accumulate',
                    'nlcpy_logical_or_accumulate',
                    'nlcpy_logical_xor_accumulate',
                    ):
            dt = 'bool'
            input_arr = nlcpy.array(input_arr, dtype='bool')

        elif name in ('nlcpy_divide_accumulate',
                      'nlcpy_logaddexp_accumulate',
                      'nlcpy_logaddexp2_accumulate',
                      'nlcpy_true_divide_accumulate',
                      'nlcpy_heaviside_accumulate',
                      'nlcpy_arctan2_accumulate',
                      'nlcpy_hypot_accumulate',
                      'nlcpy_copysign_accumulate',
                      'nlcpy_nextafter_accumulate',
                      ):
            if dtype == 'bool':
                # dt = 'float16' nlcpy can't use
                if out is None:
                    raise TypeError("not support for float16.")
                else:
                    dt = 'float32'

            elif dtype in ('float32', 'float64', 'complex', 'complex64', 'complex128'):
                dt = dtype
            else:
                dt = 'float64'

        elif name in ('nlcpy_floor_divide_accumulate',
                      'nlcpy_power_accumulate',
                      'nlcpy_remainder_accumulate',
                      'nlcpy_mod_accumulate',
                      'nlcpy_fmod_accumulate',
                      'nlcpy_left_shift_accumulate',
                      'nlcpy_right_shift_accumulate',
                      'nlcpy_subtract_accumulate',
                      ):
            if dtype == 'bool' and out is None:
                # dt = 'int8' nlcpy can't use
                raise TypeError("not support for int8.")
            if dtype == 'bool':
                dt = 'int32'
            elif input_arr.dtype == 'bool' and name == 'nlcpy_subtract_accumulate':
                dt = 'int32'
            else:
                dt = dtype
        else:
            dt = dtype

    ########################################################################
    # create work array and casting input array
    if out is None:
        y = core.ndarray(shape=input_arr.shape, dtype=dt, order=order_out)
        w = y
    else:
        if out.dtype.name == dt:
            if name in ('nlcpy_divide_accumulate',
                        'nlcpy_true_divide_accumulate',
                        'nlcpy_heaviside_accumulate',
                        'nlcpy_arctan2_accumulate',
                        'nlcpy_hypot_accumulate',
                        'nlcpy_logaddexp_accumulate',
                        'nlcpy_logaddexp2_accumulate',
                        'nlcpy_copysign_accumulate',
                        'nlcpy_nextafter_accumulate',
                        ):
                if out.dtype.name in ('bool'):
                    w = nlcpy.array(y, dtype='float32')
                elif out.dtype.name in ('int32', 'int64', 'uint32', 'uint64'):
                    w = nlcpy.array(y, dtype='float64')
                else:
                    w = y

            elif name in ('nlcpy_greater_accumulate',
                          'nlcpy_less_accumulate',
                          'nlcpy_greater_equal_accumulate',
                          'nlcpy_less_equal_accumulate',
                          'nlcpy_not_equal_accumulate',
                          'nlcpy_equal_accumulate',
                          'nlcpy_logical_and_accumulate',
                          'nlcpy_logical_or_accumulate',
                          'nlcpy_logical_xor_accumulate',
                          ):
                w = nlcpy.array(y, dtype='bool')
            elif name in ('nlcpy_floor_divide_accumulate',
                          'nlcpy_power_accumulate',
                          'nlcpy_left_shift_accumulate',
                          'nlcpy_right_shift_accumulate',
                          ):
                if out.dtype.name in ('bool'):
                    w = nlcpy.array(y, dtype='int32')
                else:
                    w = y
            elif name in ('nlcpy_remainder_accumulate',
                          'nlcpy_mod_accumulate',
                          'nlcpy_fmod_accumulate',
                          ):
                if out.dtype.name in ('bool', 'complex64', 'complex128'):
                    w = nlcpy.array(y, dtype='int32')
                else:
                    w = y
            else:
                w = y
        else:
            w = nlcpy.array(y, dtype=dt)

    if name in ('nlcpy_maximum_accumulate', 'nlcpy_minimum_accumulate',
                'nlcpy_fmax_accumulate', 'nlcpy_fmin_accumulate',
                'nlcpy_right_shift_accumulate', 'nlcpy_left_shift_accumulate',
                'nlcpy_floor_divide_accumulate', 'nlcpy_mod_accumulate',
                'nlcpy_remainder_accumulate', 'nlcpy_fmod_accumulate',
                'nlcpy_bitwise_and_accumulate', 'nlcpy_bitwise_or_accumulate',
                'nlcpy_bitwise_xor_accumulate', 'nlcpy_power_accumulate',
                'nlcpy_arctan2_accumulate', 'nlcpy_logaddexp_accumulate',
                'nlcpy_logaddexp2_accumulate', 'nlcpy_heaviside_accumulate',
                'nlcpy_copysign_accumulate', 'nlcpy_nextafter_accumulate'
                ):
        input_arr = nlcpy.asarray(input_arr, dtype=w.dtype)

    elif name in ('nlcpy_add_accumulate', 'nlcpy_subtract_accumulate',
                  'nlcpy_multiply_accumulate',
                  ):
        if w.dtype.name in ('bool'):
            input_arr = nlcpy.asarray(input_arr, dtype=w.dtype)

    ########################################################################
    # axis check
    if isinstance(axis, list):
        raise TypeError("'list' object cannot be interpreted as an integer")
    elif isinstance(axis, nlcpy.ndarray) or isinstance(axis, numpy.ndarray):
        if axis.ndim > 0 or axis.dtype.char not in 'ilIL':
            raise TypeError(
                "only integer scalar arrays can be converted to a scalar index")
        axis = int(axis)
    elif isinstance(axis, tuple):
        badVal = False
        for ax in axis:
            if isinstance(ax, nlcpy.ndarray) or isinstance(ax, numpy.ndarray):
                if ax.ndim > 0 or ax.dtype.char not in 'ilIL':
                    raise TypeError(
                        "only integer scalar arrays can be converted to a scalar index")
            if ax < 0 or ax > input_arr.ndim-1:
                badVal = True
                break
        if badVal:
            raise AxisError(
                'axis '+str(ax)+' is out of bounds for array of dimension '
                +str(input_arr.ndim))
        else:
            raise ValueError("accumulate does not allow multiple axes")
    elif axis is None and input_arr.ndim > 0:
        raise ValueError("accumulate does not allow multiple axes")

    if axis is None:
        axis_tmp = 0
    elif axis < 0:
        axis_tmp = input_arr.ndim + axis
    else:
        axis_tmp = axis

    if axis_tmp < 0 or axis_tmp > input_arr.ndim-1:
        raise AxisError('axis '+str(axis)+' is out of bounds for array of dimension '
                        +str(input_arr.ndim))

    axis = axis_tmp

    ########################################################################
    # input "a" check
    if name in ("nlcpy_power_accumulate",):
        sl = [slice(0, None) if i != axis else slice(1, None)
              for i in range(input_arr.ndim)]
        if numpy.any(input_arr[sl] < 0):
            raise ValueError("Integers to negative integer powers are not allowed.")

    ########################################################################
    # call accumulate function on VE
    request._push_request(
        name,
        "accumulate_op",
        (input_arr, y, w, axis, get_dtype_number(get_dtype(dt))),
    )
    return y

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
from nlcpy.manipulation.add_remove import resize
from nlcpy.core.error import _AxisError as AxisError
from nlcpy.request.ve_kernel cimport *
from nlcpy.request cimport request

cimport numpy as cnp

cpdef reduceat_core(name, a, indices, axis=0, dtype=None, out=None):
    # TODO: remove after nlcpy.ndarray(None) is implemented. #####
    if a is None:
        raise TypeError('cannot reduceat on a scalar')

    if indices is None:
        raise ValueError('object of too small depth for desired array')
    ##############################################################
    # convert to nlcpy.ndarray
    if not isinstance(a, ndarray):
        a = array(a)
    if a.ndim == 0:
        raise TypeError('cannot reduceat on a scalar')

    indices = nlcpy.asarray(indices, dtype='int64')

    if indices.ndim > 1:
        raise ValueError('object too deep for desired array')
    if indices.ndim < 1:
        raise ValueError('object of too small depth for desired array')

    if a.size == 0 and indices.size != 0:
        raise IndexError('index '+str(indices[0])+' out-of-bounds in '+name+' [0, 0)')

    # TODO: VE-VH collaboration #############################################
    if a._memloc in {on_VH, on_VE_VH}:
        raise NotImplementedError('reduceat on VH is not yet implemented.')

    if isinstance(out, ndarray):
        if out._memloc in {on_VH, on_VE_VH}:
            raise NotImplementedError('reduceat on VH is not yet implemented.')
    ########################################################################
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
            if ax < 0 or ax > a.ndim-1:
                badVal = True
                break
        if badVal:
            raise AxisError(
                'axis '+str(ax)+' is out of bounds for array of dimension '
                +str(a.ndim))
        else:
            raise ValueError("reduceat does not allow multiple axes")
    elif axis is None and a.ndim > 0:
        raise ValueError("reduceat does not allow multiple axes")

    # check axis range
    axis_save = axis
    if axis < 0:
        axis = a.ndim + axis

    if axis < 0 or axis > a.ndim - 1:
        raise AxisError(
            'axis '+str(axis_save)+' is out of bounds for array of dimension '
            +str(a.ndim))

    # determine output shape
    shape = list(a.shape)
    shape[axis] = len(indices)

    # error check for "out"
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
                        "reduceat_core on VH is not yet implemented.")
                out = out[0]
        if isinstance(out, nlcpy.ndarray):
            if a.ndim > out.ndim:
                raise ValueError("Iterator input op_axes[0]["+str(a.ndim-out.ndim-1)
                                 +"] (=="+str(out.ndim)
                                 +") is not a valid axis of op[0], "
                                 +"which has "+str(out.ndim)+" dimensions ")
            for i in range(a.ndim):
                if out.shape[i] != shape[i]:
                    del shape[axis]
                    indices_out_shape = [str(len(indices))
                                         if i == 0 else 'newaxis' for i in range(a.ndim)]
                    raise ValueError("operands could not be broadcast together with "
                                     +"remapped shapes [original->remapped]: "
                                     +str(out.shape).replace(" ", "")+"->"
                                     +str(out.shape[:a.ndim]).replace(" ", "")+" "
                                     +str(tuple(a.shape)).replace(" ", "")+"->"
                                     +str(tuple(shape)).replace(" ", "")
                                     .replace(",)", ")")+" "
                                     +str(tuple([len(indices)])).replace(" ", "")+"->"
                                     +str(tuple(indices_out_shape)).replace(" ", "")
                                     .replace("'", "")+" ")

    if name == 'nlcpy_power_reduceat' \
       and (
            out is None and a.dtype in ('int32', 'int64')
            or out is not None and out.dtype in ('int32', 'int64')):
        sl = [slice(0, None) if i != axis else slice(1, None) for i in range(a.ndim)]

        if nlcpy.any(a[sl] < 0):
            raise ValueError("Integers to negative integer powers are not allowed.")

    # check order
    if a._f_contiguous and not a._c_contiguous:
        order_out = 'F'
    else:
        order_out = 'C'

    # check dtype
    dtype = None if dtype is None else nlcpy.dtype(dtype)
    if a.dtype == bool and dtype is None and out is None:
        if name in ('nlcpy_divide_reduceat',
                    'nlcpy_true_divide_reduceat',
                    'nlcpy_nextafter_reduceat',
                    'nlcpy_arctan2_reduceat',
                    'nlcpy_hypot_reduceat',
                    'nlcpy_logaddexp_reduceat',
                    'nlcpy_logaddexp2_reduceat',
                    'nlcpy_heaviside_reduceat',
                    'nlcpy_fmod_reduceat',
                    ):
            raise TypeError("not support for float16.")
        elif name in ('nlcpy_floor_divide_reduceat',
                      'nlcpy_mod_reduceat',
                      'nlcpy_remainder_reduceat',
                      'nlcpy_power_reduceat'
                      ):
            raise TypeError("not support for int8.")

    if name == 'nlcpy_subtract_reduceat' and a.dtype == 'bool':
        dtype = 'int32'

    if name in ('nlcpy_logical_or_reduceat',
                'nlcpy_logical_and_reduceat',
                'nlcpy_logical_xor_reduceat',
                'nlcpy_equal_reduceat',
                'nlcpy_not_equal_reduceat',
                'nlcpy_less_reduceat',
                'nlcpy_greater_reduceat',
                'nlcpy_less_equal_reduceat',
                'nlcpy_greater_equal_reduceat',
                ):
        dt = 'bool'
        a = nlcpy.array(a, dtype='bool')
    elif dtype is not None:
        if type(dtype) == str and dtype.find(',') > 0:
            raise TypeError('cannot perform reduceat with flexible type')
        elif name == 'nlcpy_multiply_reduceat' \
                and a.dtype not in ('float32', 'float64', 'complex64', 'complex128') \
                and out is not None and out.dtype == 'bool':
            dt = 'int64'
        elif name in (
                'nlcpy_nextafter_reduceat',
                'nlcpy_copysign_reduceat',
                'nlcpy_heaviside_reduceat') and dtype == 'bool':
            dt = 'float32'
        else:
            dt = dtype
    elif out is not None:
        dt = out.dtype
    elif name in ('nlcpy_add_reduceat',
                  'nlcpy_multiply_reduceat'
                  ):
        if a.dtype in ('bool', 'int32'):
            dt = 'int64'
        elif a.dtype == 'uint32':
            dt = 'uint64'
        else:
            dt = a.dtype
    else:
        dt = a.dtype

    if name in ('nlcpy_bitwise_and_reduceat',
                'nlcpy_bitwise_or_reduceat',
                'nlcpy_bitwise_xor_reduceat',
                'nlcpy_right_shift_reduceat',
                'nlcpy_left_shift_reduceat',
                ):
        if dt in ('float32', ):
            raise ValueError("could not find a matching type for "+name
                             +", requested type has type code 'f'")
        if dt in ('float64', ):
            raise ValueError("could not find a matching type for "+name
                             +", requested type has type code 'd'")

    if name in ('nlcpy_mod_reduceat',
                'nlcpy_remainder_reduceat',
                'nlcpy_bitwise_and_reduceat',
                'nlcpy_bitwise_or_reduceat',
                'nlcpy_bitwise_xor_reduceat',
                'nlcpy_right_shift_reduceat',
                'nlcpy_left_shift_reduceat',
                'nlcpy_arctan2_reduceat',
                'nlcpy_hypot_reduceat',
                'nlcpy_logaddexp_reduceat',
                'nlcpy_logaddexp2_reduceat',
                'nlcpy_heaviside_reduceat',
                'nlcpy_copysign_reduceat',
                'nlcpy_nextafter_reduceat',
                'nlcpy_fmod_reduceat',
                ):
        if dt == 'complex64':
            raise ValueError("could not find a matching type for "+name
                             +", requested type has type code 'F'")
        if dt == 'complex128':
            raise ValueError("could not find a matching type for "+name
                             +", requested type has type code 'D'")

    if name in ('nlcpy_right_shift_reduceat',
                'nlcpy_left_shift_reduceat',
                'nlcpy_power_reduceat',
                'nlcpy_mod_reduceat',
                'nlcpy_fmod_reduceat',
                'nlcpy_remainder_reduceat',
                'nlcpy_subtract_reduceat',
                ):
        if dt == 'bool':
            dt = 'int32'
        if dt != a.dtype:
            if dt in ('uint32', 'uint64'):
                a = nlcpy.array(a, dtype=dt)
            elif a.dtype in ('float32', 'float64', 'complex64', 'complex128'):
                a = nlcpy.array(a, dtype=dt)

    elif name in ('nlcpy_bitwise_and_reduceat',
                  'nlcpy_bitwise_or_reduceat',
                  'nlcpy_bitwise_xor_reduceat',
                  ):
        if dt != a.dtype:
            if dt in ('uint32', 'uint64'):
                a = nlcpy.array(a, dtype=dt)
            elif a.dtype in ('float32', 'float64', 'complex64', 'complex128'):
                a = nlcpy.array(a, dtype=dt)
    elif name in ('nlcpy_divide_reduceat',
                  'nlcpy_true_divide_reduceat',
                  ):
        if dt not in ('float32', 'complex64', 'complex128'):
            dt = 'float64'

    elif name in ('nlcpy_arctan2_reduceat',
                  'nlcpy_hypot_reduceat',
                  'nlcpy_heaviside_reduceat',
                  'nlcpy_logaddexp_reduceat',
                  'nlcpy_logaddexp2_reduceat',
                  'nlcpy_copysign_reduceat',
                  'nlcpy_nextafter_reduceat',
                  ):
        if dt != 'float32':
            dt = 'float64'
        if dt != a.dtype:
            a = nlcpy.array(a, dtype=dt)

    # create return object
    if out is not None:
        y = out
    else:
        y = ndarray(shape=shape, dtype=dt, order=order_out)
    if y.ve_adr == 0:
        raise MemoryError()

    if name == 'nlcpy_fmod_reduceat' and dt == 'bool':
        w = array(y, dtype='float64')
    elif name == 'nlcpy_floor_divide_reduceat' and dt == 'bool':
        w = array(y, dtype='int64')
    elif out is None or out.dtype == dt:
        w = y
    else:
        w = array(out, dtype=dt)

    # call reduceat function on VE
    if y.size > 0:
        bad_index = numpy.empty(1, dtype=numpy.int32)
        fpe_flags = request._get_fpe_flag()
        args = (
            a._ve_array,
            indices._ve_array,
            y._ve_array,
            w._ve_array,
            <int32_t>axis,
            veo.OnStack(bad_index, inout=veo.INTENT_OUT),
            veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),
        )

        request._push_and_flush_request(
            name,
            args,
            callback=_reduceat_chkerr(
                bad_index, indices, name, a, axis)
        )

    return y


def _reduceat_chkerr(bad_index, indices, name, a, axis):
    def _chkerr(*args):
        if bad_index[0] >= 0:
            raise IndexError('index '+str(indices.get()[bad_index[0]])
                             +' out-of-bounds in '+name+' [0, '
                             +str(a.shape[axis])+')')
    return _chkerr

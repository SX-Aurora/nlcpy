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
import nlcpy
import numpy

from nlcpy import veo
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport internal as core_internal
from nlcpy.sca.description cimport elements_per_array
from nlcpy.sca.description cimport description

from libcpp.vector cimport vector
from libc.stdint cimport *


cdef class descriptor:

    def __init__(self, ndarray arr):
        if arr.dtype not in (nlcpy.dtype('f4'), nlcpy.dtype('f8')):
            raise TypeError('input array\'s dtype must be `float32` or `float64`')
        if arr.ndim == 0 or arr.ndim > 4:
            raise ValueError('input array has invalid dimension: '
                             'got `{}`, expected `1 <= ndim <= 4`.'
                             .format(arr.ndim))

        # strides check
        std_pre = arr._strides[arr.ndim - 1]
        for std in reversed(arr._strides[0:arr.ndim - 1]):
            if std < std_pre:
                raise ValueError('input array\'s strides must be monotonously '
                                 'increasing.')
            std_pre = std

        if arr.ndim >= 2:
            if arr._strides[arr.ndim - 2] % arr._strides[arr.ndim - 1] != 0:
                raise ValueError('input array has invalid strides: '
                                 '`strides[{}]` is indivisible by `strides[{}]`'
                                 .format(arr.ndim - 2, arr.ndim - 1))
        if arr.ndim >= 3:
            if arr._strides[arr.ndim - 3] % arr._strides[arr.ndim - 2] != 0:
                raise ValueError('input array has invalid strides: '
                                 '`strides[{}]` is indivisible by `strides[{}]`'
                                 .format(arr.ndim - 3, arr.ndim - 2))
        if arr.ndim >= 4:
            if arr._strides[arr.ndim - 4] % arr._strides[arr.ndim - 3] != 0:
                raise ValueError('input array has invalid strides: '
                                 '`strides[{}]` is indivisible by `strides[{}]`'
                                 .format(arr.ndim - 4, arr.ndim - 3))

        self.arr = arr

    def __getitem__(self, slices):
        if type(slices) not in (tuple, list, int, slice, type(Ellipsis)):
            raise TypeError('instance of `{}` is not supported.'
                            .format(type(slices)))

        if type(slices) in (tuple, list):
            slice_list = list(slices)
        else:
            slice_list = [slices]

        slice_list, n_newaxes = core_internal.complete_slice_list(
            slice_list, self.arr.ndim)

        if len(slice_list) != self.arr.ndim:
            raise IndexError('wrong number of indices: got `{}`, expected `{}`.'
                             .format(len(slices), self.arr.ndim))

        # convert slice(None) to 0
        for i, s in enumerate(slice_list):
            if s == slice(None):
                slice_list[i] = 0
            elif not isinstance(s, int):
                raise IndexError(
                    'only integers, slices (\':\' or \'slice(None)\'), '
                    'ellipsis (\'...\'), are valid indices')

        desc = description(self.arr, tuple(slice_list))
        return desc

    def __repr__(self):
        msg = ''
        msg += 'assigned array : {}-dim {}\n'.format(self.arr.ndim, self.arr.shape)
        return msg

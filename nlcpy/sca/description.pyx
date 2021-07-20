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
import threading
import copy

import numpy

from nlcpy import veo
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport internal

from libcpp.vector cimport vector
from libc.stdint cimport *

cdef SCA_NDIM = 4


cdef class elements_per_array:

    def __init__(self, ndarray array, tuple location=(0, 0, 0, 0),
                 double factor=1.0, ndarray coef=None):
        cdef vector[Py_ssize_t] _loc
        _loc.assign(SCA_NDIM, 0)
        for i in range(1, len(location)+1):
            _loc[SCA_NDIM - i] = location[-i]

        self.array = array
        self.location.push_back(_loc)
        self.factor.push_back(factor)
        if coef is None:
            self.coef = ()
        else:
            raise RuntimeError

        self.lxm_max = 0 if _loc[SCA_NDIM - 1] >= 0 else abs(_loc[SCA_NDIM - 1])
        self.lxp_max = 0 if _loc[SCA_NDIM - 1] <= 0 else abs(_loc[SCA_NDIM - 1])
        self.lym_max = 0 if _loc[SCA_NDIM - 2] >= 0 else abs(_loc[SCA_NDIM - 2])
        self.lyp_max = 0 if _loc[SCA_NDIM - 2] <= 0 else abs(_loc[SCA_NDIM - 2])
        self.lzm_max = 0 if _loc[SCA_NDIM - 3] >= 0 else abs(_loc[SCA_NDIM - 3])
        self.lzp_max = 0 if _loc[SCA_NDIM - 3] <= 0 else abs(_loc[SCA_NDIM - 3])
        self.lwm_max = 0 if _loc[SCA_NDIM - 4] >= 0 else abs(_loc[SCA_NDIM - 4])
        self.lwp_max = 0 if _loc[SCA_NDIM - 4] <= 0 else abs(_loc[SCA_NDIM - 4])
        # offset will be updated in description class
        self.offset = 0
        self.nelm = 1

    cdef _set_offset(self, int64_t offset):
        self.offset = offset

    cdef _set_nelm(self, int64_t nelm):
        self.nelm = nelm

    cdef _set_max_location_value(self, int64_t lxm_max, int64_t lxp_max,
                                 int64_t lym_max, int64_t lyp_max,
                                 int64_t lzm_max, int64_t lzp_max,
                                 int64_t lwm_max, int64_t lwp_max):
        self.lxm_max = lxm_max
        self.lxp_max = lxp_max
        self.lym_max = lym_max
        self.lyp_max = lyp_max
        self.lzm_max = lzm_max
        self.lzp_max = lzp_max
        self.lwm_max = lwm_max
        self.lwp_max = lwp_max

    cdef _apply_factor(self, double factor):
        for i in range(self.nelm):
            self.factor[i] *= factor

    cdef _append_coef(self, ndarray coef, int64_t idx):
        self.coef += ((idx, coef),)

    cdef _copy(self):
        res = elements_per_array(self.array)
        res.location = self.location
        res.factor = self.factor
        res.coef = self.coef
        res.lxm_max = self.lxm_max
        res.lxp_max = self.lxp_max
        res.lym_max = self.lym_max
        res.lyp_max = self.lyp_max
        res.lzm_max = self.lzm_max
        res.lzp_max = self.lzp_max
        res.lwm_max = self.lwm_max
        res.lwp_max = self.lwp_max
        res.offset = self.offset
        res.nelm = self.nelm
        return res

    def __repr__(self):
        msg = ''
        # ndarray
        msg += 'array\n'
        msg += ('  shape={}, dtype={} array\n'
                .format(self.array.shape, self.array.dtype))
        # location
        msg += 'location\n'
        msg += ('  {}\n'.format(self.location))
        # factor
        msg += 'factor\n'
        msg += ('  {}\n'.format(self.factor))
        # coefficient
        msg += 'coefficient\n'
        msg += ('  {}\n'.format(self.coef))
        # offset
        msg += 'offset: {}\n'.format(self.offset)
        # max/min of locations
        msg += ''
        msg += ('lxm_max: {}\n'.format(self.lxm_max))
        msg += ('lxp_max: {}\n'.format(self.lxp_max))
        msg += ('lym_max: {}\n'.format(self.lym_max))
        msg += ('lyp_max: {}\n'.format(self.lyp_max))
        msg += ('lzm_max: {}\n'.format(self.lzm_max))
        msg += ('lzp_max: {}\n'.format(self.lzp_max))
        msg += ('lwm_max: {}\n'.format(self.lwm_max))
        msg += ('lwp_max: {}\n'.format(self.lwp_max))
        # nelm
        msg += 'nelm: {}\n'.format(self.nelm)
        return msg


cdef class description:

    __array_priority__ = 2.0

    def __init__(self, ndarray arr=None, tuple location=(0, 0, 0, 0), double factor=1.0):
        self.elems = ()
        self.nx = 1
        self.ny = 1
        self.nz = 1
        self.nw = 1
        self.lxm_max2 = 0
        self.lxp_max2 = 0
        self.lym_max2 = 0
        self.lyp_max2 = 0
        self.lzm_max2 = 0
        self.lzp_max2 = 0
        self.lwm_max2 = 0
        self.lwp_max2 = 0
        self.ndim = 0
        self.dtype = None
        self.nelm_total = 0
        if arr is not None:
            self.ndim = arr.ndim
            self.dtype = arr.dtype
            self.elems += (elements_per_array(arr, location, factor),)
            self._update_all_attributes()

    def __add__(self, description other):
        res = description()

        if self.nelm_total == 0 and other.nelm_total > 0:
            res._copy_description(other)
            return res
        if self.nelm_total > 0 and other.nelm_total == 0:
            res._copy_description(self)
            return res

        if self.ndim != other.ndim:
            raise ValueError('array must have same dimension.')
        if self.dtype != other.dtype:
            raise TypeError('array must have same dtype.')

        oi = [0 for _ in range(len(other.elems))]
        is_fused = False

        # self
        for se in self.elems:
            sx = se.array
            is_fused = False
            for i, oe in enumerate(other.elems):
                if id(sx) == id(oe.array):
                    res._append_elem(_fuse_elems(se, oe))
                    oi[i] = 1
                    is_fused = True
            if not is_fused:
                res._append_elem(se)

        # other
        for i, oe in enumerate(other.elems):
            if oi[i] == 0:
                res._append_elem(oe)

        res.ndim = self.ndim
        res.dtype = self.dtype
        res._update_all_attributes()

        return res

    def __sub__(self, description other):
        return self + (-1) * other

    def __mul__(self, other):
        res = description()
        if type(self) is description:
            res._append_all_elem(self)
            res.ndim = self.ndim
            res.dtype = self.dtype
            if type(other) is ndarray:
                res._apply_coef(other)
            else:
                res._apply_factor(other)
        elif type(other) is description:
            res._append_all_elem(other)
            res.ndim = other.ndim
            res.dtype = other.dtype
            if type(other) is ndarray:
                res._apply_coef(self)
            else:
                res._apply_factor(self)
        else:
            raise RuntimeError

        res._update_all_attributes()

        return res

    def __truediv__(self, other):
        if type(self) is not description:
            raise TypeError('numerator object must be `description`.')
        if not isinstance(other, ndarray):
            try:
                other = float(other)
            except Exception:
                raise TypeError('instance of `{}` cannot devide with `{}`.'
                                .format(type(self), type(other)))
        return self * (1 / other)

    # not support __floordiv__

    cdef _copy_description(self, description other):
        assert self.nelm_total == 0
        for oe in other.elems:
            self._append_elem(oe)
        self.ndim = other.ndim
        self.dtype = other.dtype
        self._update_all_attributes()

    cdef _apply_factor(self, double factor):
        cdef elements_per_array se
        for i in range(len(self.elems)):
            se = self.elems[i]
            se._apply_factor(factor)

    cdef _apply_coef(self, ndarray coef):
        if coef.size != 1 and coef.ndim != self.ndim:
            raise ValueError('coefficient ndarray has invalid ndim: '
                             'got `{}`, expected `{}`.'
                             .format(coef.ndim, self.ndim))
        if self.dtype != coef.dtype:
            raise TypeError('coefficient ndarray has invalid dtype: '
                            'got `{}`, expected `{}`.'
                            .format(coef.dtype, self.dtype))

        cdef elements_per_array se
        for i in range(len(self.elems)):
            se = self.elems[i]
            for i in range(se.nelm):
                for sec in se.coef:
                    if i == sec[0]:
                        raise TypeError(
                            'cannot assign multiple coefficient ndarray '
                            'for a single stencil element.')
                se._append_coef(coef, i)

    cdef _border_check(self):
        cdef int64_t dx_m = 0
        cdef int64_t dx_p = 0
        cdef int64_t dy_m = 0
        cdef int64_t dy_p = 0
        cdef int64_t dz_m = 0
        cdef int64_t dz_p = 0
        cdef int64_t dw_m = 0
        cdef int64_t dw_p = 0

        for se in self.elems:
            dx_m = max(dx_m, se.lxm_max)
            dx_p = max(dx_p, se.lxp_max)
            dy_m = max(dy_m, se.lym_max)
            dy_p = max(dy_p, se.lyp_max)
            dz_m = max(dz_m, se.lzm_max)
            dz_p = max(dz_p, se.lzp_max)
            dw_m = max(dw_m, se.lwm_max)
            dw_p = max(dw_p, se.lwp_max)

        if self.shape[self.ndim-1] < (dx_m + dx_p + 1):
            raise IndexError('out of ranges for relative index of '
                             'last-axis.')
        if self.ndim >= 2:
            if self.shape[self.ndim-2] < (dy_m + dy_p + 1):
                raise IndexError('out of ranges for relative index of '
                                 '(last-1)-axis.')
        if self.ndim >= 3:
            if self.shape[self.ndim-3] < (dz_m + dz_p + 1):
                raise IndexError('out of ranges for relative index of '
                                 '(last-2)-axis.')
        if self.ndim >= 4:
            if self.shape[self.ndim-4] < (dw_m + dw_p + 1):
                raise IndexError('out of ranges for relative index of '
                                 '(last-3)-axis.')

        self.lxm_max2 = dx_m
        self.lxp_max2 = dx_p
        self.lym_max2 = dy_m
        self.lyp_max2 = dy_p
        self.lzm_max2 = dz_m
        self.lzp_max2 = dz_p
        self.lwm_max2 = dw_m
        self.lwp_max2 = dw_p

    cdef _shape_check_for_out(self):
        assert self.nelm_total == 1
        arr = self.elems[0].array
        if self.shape[self.ndim-1] > arr._shape[self.ndim-1]:
            raise ValueError('last-axis of output array must be larger than '
                             'or equal to {}.'.format(arr._shape[self.ndim-1]))
        if self.ndim >= 2:
            if self.shape[self.ndim-2] > arr._shape[self.ndim-2]:
                raise ValueError('(last-1)-axis of output array must be larger than '
                                 'or equal to {}.'.format(arr._shape[self.ndim-2]))
        if self.ndim >= 3:
            if self.shape[self.ndim-3] > arr._shape[self.ndim-3]:
                raise ValueError('(last-2)-axis of output array must be larger than '
                                 'or equal to {}.'.format(arr._shape[self.ndim-3]))
        if self.ndim >= 4:
            if self.shape[self.ndim-4] > arr._shape[self.ndim-4]:
                raise ValueError('(last-3-axis of output array must be larger than '
                                 'or equal to {}.'.format(arr._shape[self.ndim-4]))

    cdef _location_check_for_out(self):
        assert self.nelm_total == 1
        arr = self.elems[0].array
        cdef vector[Py_ssize_t] location = self.elems[0].location[0]

        if location[SCA_NDIM - 1] < 0 and abs(location[SCA_NDIM - 1]) > self.lxm_max2:
            raise IndexError('out of ranges for relative index of output description.')
        if location[SCA_NDIM - 1] > 0 and \
                (self.lxm_max2 + location[SCA_NDIM - 1] + self.nx) > \
                arr.shape[self.ndim-1]:
            raise IndexError('out of ranges for relative index of output description.')

        if self.ndim >= 2:
            if location[SCA_NDIM - 2] < 0 and abs(location[SCA_NDIM - 2]) > \
                    self.lym_max2:
                raise IndexError('out of ranges for relative index of '
                                 'output description.')
            if location[SCA_NDIM - 2] > 0 and \
                    (self.lym_max2 + location[SCA_NDIM - 2] + self.ny) > \
                    arr.shape[self.ndim-2]:
                raise IndexError('out of ranges for relative index '
                                 'of output description.')

        if self.ndim >= 3:
            if location[SCA_NDIM - 3] < 0 and \
                    abs(location[SCA_NDIM - 3]) > self.lzm_max2:
                raise IndexError('out of ranges for relative index of '
                                 'output description.')
            if location[SCA_NDIM - 3] > 0 and \
                    (self.lzm_max2 + location[SCA_NDIM - 3] + self.nz) > \
                    arr.shape[self.ndim-3]:
                raise IndexError('out of ranges for relative index '
                                 'of output description.')

        if self.ndim >= 4:
            if location[SCA_NDIM - 4] < 0 and abs(location[SCA_NDIM - 4]) > \
                    self.lwm_max2:
                raise IndexError('out of ranges for relative index of '
                                 'output description.')
            if location[SCA_NDIM - 4] > 0 and \
                    (self.lwm_max2 + location[SCA_NDIM - 4] + self.nw) > \
                    arr.shape[self.ndim-4]:
                raise IndexError('out of ranges for relative index '
                                 'of output description.')

    cdef _update_all_attributes(self):
        if len(self.elems) == 0:
            return
        self._update_shape()
        self._border_check()
        self._update_computation_size()
        self._update_offset()
        self._update_nelm_total()

    cdef _update_shape(self):
        self.shape = self.elems[0].array._shape
        cdef vector[Py_ssize_t] sh
        for i in range(1, len(self.elems)):
            sh = self.elems[i].array._shape
            for j in range(self.ndim):
                self.shape[j] = min(self.shape[j], sh[j])

    cdef _update_nelm_total(self):
        cdef int64_t nelm_total = 0
        for se in self.elems:
            nelm_total += se.nelm
        self.nelm_total = nelm_total

    cdef _update_computation_size(self):
        cdef int64_t nx = 2305843009213693952  # int64_max
        cdef int64_t ny = 2305843009213693952  # int64_max
        cdef int64_t nz = 2305843009213693952  # int64_max
        cdef int64_t nw = 2305843009213693952  # int64_max

        for se in self.elems:
            arr = se.array
            nx = min(
                nx,
                arr._shape[self.ndim-1] - (self.lxm_max2 + self.lxp_max2)
            )
            if self.ndim >= 2:
                ny = min(
                    ny,
                    arr._shape[self.ndim-2] - (self.lym_max2 + self.lyp_max2)
                )
            if self.ndim >= 3:
                nz = min(
                    nz,
                    arr._shape[self.ndim-3] - (self.lzm_max2 + self.lzp_max2)
                )
            if self.ndim >= 4:
                nw = min(
                    nw,
                    arr._shape[self.ndim-4] - (self.lwm_max2 + self.lwp_max2)
                )

        self.nx = nx
        self.ny = ny if self.ndim >= 2 else 1
        self.nz = nz if self.ndim >= 3 else 1
        self.nw = nw if self.ndim >= 4 else 1

    cdef _update_offset(self):
        cdef int64_t offset
        cdef elements_per_array se
        for i in range(len(self.elems)):
            se = self.elems[i]
            arr = se.array
            offset = 0
            offset += <int64_t>(arr._strides[self.ndim-1] / arr.itemsize) * \
                self.lxm_max2
            if self.ndim >= 1:
                offset += <int64_t>(arr._strides[self.ndim-2] / arr.itemsize) * \
                    self.lym_max2
            if self.ndim >= 2:
                offset += <int64_t>(arr._strides[self.ndim-3] / arr.itemsize) * \
                    self.lzm_max2
            if self.ndim >= 3:
                offset += <int64_t>(arr._strides[self.ndim-4] / arr.itemsize) * \
                    self.lwm_max2
            se._set_offset(offset)

    cdef _update_offset_for_out(self):
        assert self.nelm_total == 1
        cdef int64_t offset
        cdef elements_per_array se = self.elems[0]
        arr = se.array
        offset = se.offset
        cdef vector[Py_ssize_t] location = se.location[0]
        offset += <int64_t>(arr._strides[self.ndim-1] / arr.itemsize) * \
            location[SCA_NDIM - 1]
        if self.ndim >= 1:
            offset += <int64_t>(arr._strides[self.ndim-2] / arr.itemsize) * \
                location[SCA_NDIM - 2]
        if self.ndim >= 2:
            offset += <int64_t>(arr._strides[self.ndim-3] / arr.itemsize) * \
                location[SCA_NDIM - 3]
        if self.ndim >= 3:
            offset += <int64_t>(arr._strides[self.ndim-4] / arr.itemsize) * \
                location[SCA_NDIM - 4]
        if offset < 0:
            raise IndexError('output description has invalid indices.')
        se._set_offset(offset)

    cdef _append_elem(self, elements_per_array elem):
        self.elems += (elem,)

    cdef _append_all_elem(self, description other):
        for e in other.elems:
            self._append_elem(e)

    cdef _set_param_for_out(self, description src):
        assert self.nelm_total == 1
        self.lxm_max2 = src.lxm_max2
        self.lxp_max2 = src.lxp_max2
        self.lym_max2 = src.lym_max2
        self.lyp_max2 = src.lyp_max2
        self.lzm_max2 = src.lzm_max2
        self.lzp_max2 = src.lzp_max2
        self.lwm_max2 = src.lwm_max2
        self.lwp_max2 = src.lwp_max2
        self.nx = src.nx
        self.ny = src.ny
        self.nz = src.nz
        self.nw = src.nw
        self.shape = src.shape
        self._shape_check_for_out()
        self._location_check_for_out()
        self._update_offset()
        self._update_offset_for_out()

    def __repr__(self):
        msg = ''
        msg += 'stencil description\n'

        is_first = 1
        # stencil description
        for i, se in enumerate(self.elems):
            for j, l in enumerate(se.location):
                if not is_first:
                    msg += ' +\n'
                msg += '  '
                # coefficient
                msg_coef = ''
                is_coef_valid = False
                if len(se.coef) > 0:
                    msg_coef += 'coef('
                    is_first_coef = 1
                    for c in se.coef:
                        if c[0] != j:
                            continue
                        if not is_first_coef:
                            msg_coef += '*'
                        msg_coef += '{}'.format(c[1])
                        is_first_coef = 0
                        is_coef_valid = True
                    msg_coef += ')*'
                if not is_coef_valid:
                    msg_coef = ''

                # factor
                msg_factor = ''
                if se.factor[j] != 1.0:
                    msg_factor += 'factor({})*'.format(se.factor[j])
                msg += '{}{}in_{}{}'.format(
                    msg_factor, msg_coef, i, se.location[j])
                is_first = 0
        msg += '\n'

        # assignment arrays
        msg += '\nassigned arrays\n'
        for i, se in enumerate(self.elems):
            msg += ('  in_{}: shape={}, dtype={} array\n'
                    .format(i, se.array.shape, se.array.dtype))

        msg += '\ncomputation size\n'
        msg += ('  nx = {}, ny = {}, nz = {}, nw = {}'
                .format(self.nx, self.ny, self.nz, self.nw))
        return msg


cdef _fuse_elems(elements_per_array x, elements_per_array y):
    assert id(x.array) == id(y.array)
    cdef elements_per_array res = x._copy()
    for i in range(y.factor.size()):
        res.location.push_back(y.location[i])
        res.factor.push_back(y.factor[i])
    for yc in y.coef:
        # for xc in x.coef:
        #     if x.location[xc[0]] == y.location[yc[0]]:
        #         raise TypeError('cannot assign multiple coefficient ndarray '
        #                         'for a single stencil element.')
        res._append_coef(yc[1], x.nelm + yc[0])
    res._set_nelm(x.nelm + y.nelm)

    # updates max values of relative indices
    cdef int64_t lxm_max
    cdef int64_t lxp_max
    cdef int64_t lym_max
    cdef int64_t lyp_max
    cdef int64_t lzm_max
    cdef int64_t lzp_max
    cdef int64_t lwm_max
    cdef int64_t lwp_max
    lxm_max = x.lxm_max if x.lxm_max >= y.lxm_max else y.lxm_max
    lxp_max = x.lxp_max if x.lxp_max >= y.lxp_max else y.lxp_max
    lym_max = x.lym_max if x.lym_max >= y.lym_max else y.lym_max
    lyp_max = x.lyp_max if x.lyp_max >= y.lyp_max else y.lyp_max
    lzm_max = x.lzm_max if x.lzm_max >= y.lzm_max else y.lzm_max
    lzp_max = x.lzp_max if x.lzp_max >= y.lzp_max else y.lzp_max
    lwm_max = x.lwm_max if x.lwm_max >= y.lwm_max else y.lwm_max
    lwp_max = x.lwp_max if x.lwp_max >= y.lwp_max else y.lwp_max
    res._set_max_location_value(
        lxm_max, lxp_max, lym_max, lyp_max,
        lzm_max, lzp_max, lwm_max, lwp_max
    )

    return res

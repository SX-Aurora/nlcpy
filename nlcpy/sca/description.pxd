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

from nlcpy.core.core cimport ndarray
from libcpp.vector cimport vector
from libc.stdint cimport *


cdef class elements_per_array:
    cdef:
        readonly ndarray array
        readonly vector[vector[Py_ssize_t]] location
        readonly vector[double] factor
        readonly tuple coef
        readonly int64_t lxm_max
        readonly int64_t lxp_max
        readonly int64_t lym_max
        readonly int64_t lyp_max
        readonly int64_t lzm_max
        readonly int64_t lzp_max
        readonly int64_t lwm_max
        readonly int64_t lwp_max
        readonly int64_t offset
        readonly int64_t nelm

    cdef _set_offset(self, int64_t offset)
    cdef _set_nelm(self, int64_t nelm)
    cdef _set_max_location_value(self, int64_t lxm_max, int64_t lxp_max,
                                 int64_t lym_max, int64_t lyp_max,
                                 int64_t lzm_max, int64_t lzp_max,
                                 int64_t lwm_max, int64_t lwp_max)
    cdef _apply_factor(self, double factor)
    cdef _append_coef(self, ndarray coef, int64_t idx)
    cdef _copy(self)


cdef class description:
    cdef:
        readonly tuple elems
        readonly vector[Py_ssize_t] shape
        readonly int64_t nx
        readonly int64_t ny
        readonly int64_t nz
        readonly int64_t nw
        readonly int64_t lxm_max2
        readonly int64_t lxp_max2
        readonly int64_t lym_max2
        readonly int64_t lyp_max2
        readonly int64_t lzm_max2
        readonly int64_t lzp_max2
        readonly int64_t lwm_max2
        readonly int64_t lwp_max2
        readonly int64_t ndim
        readonly object dtype
        readonly int64_t nelm_total

    cdef _copy_description(self, description other)
    cdef _apply_factor(self, double factor)
    cdef _apply_coef(self, ndarray coef)
    cdef _border_check(self)
    cdef _shape_check_for_out(self)
    cdef _location_check_for_out(self)
    cdef _update_all_attributes(self)
    cdef _update_shape(self)
    cdef _update_nelm_total(self)
    cdef _update_computation_size(self)
    cdef _update_offset(self)
    cdef _update_offset_for_out(self)
    cdef _append_elem(self, elements_per_array elem)
    cdef _append_all_elem(self, description other)
    cdef _set_param_for_out(self, description src)


cdef _fuse_elems(elements_per_array x, elements_per_array y)

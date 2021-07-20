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

from nlcpy import veo
from nlcpy.request import request
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport core
from nlcpy.sca cimport kernel
from nlcpy.sca cimport internal as sca_internal
from nlcpy.sca.description cimport description

from libc.stdint cimport *

import numpy


cdef class sca_handle:

    def __init__(self, dtype='float64'):
        self.destroyed = False

        dt = numpy.dtype(dtype)
        ve_adr = numpy.empty(1, dtype='u8')
        fpe_flags = request._get_fpe_flag()
        args = (
            veo.OnStack(ve_adr, inout=veo.INTENT_OUT),
            veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),
        )
        if dt == numpy.dtype('float32'):
            func_name = 'nlcpy_sca_stencil_create_s'
        elif dt == numpy.dtype('float64'):
            func_name = 'nlcpy_sca_stencil_create_d'
        else:
            raise TypeError('dtype is only acceptable `float32` or `float64`')
        self.dtype = dt

        request._push_and_flush_request(
            func_name,
            args,
            sync=True
        )

        self.hnd_adr = <uint64_t>ve_adr

    def set_elements(self, description desc_i, description desc_o):
        cdef int64_t offset = 0
        cdef int64_t elem_offset = 0
        cdef int64_t nelm
        cdef int64_t sx, mx, my, mz

        self.desc_i = desc_i
        self.desc_o = desc_o

        if self.dtype == numpy.dtype('float32'):
            func_name = 'nlcpy_sca_set_elements_s'
        elif self.dtype == numpy.dtype('float64'):
            func_name = 'nlcpy_sca_set_elements_d'
        else:
            raise RuntimeError

        fpe_flags = request._get_fpe_flag()

        # set stencil elements
        for elem in desc_i.elems:
            nelm = elem.nelm
            arr = elem.array
            offset = elem.offset
            sx = (arr.strides[-1] / arr.itemsize)
            mx, my, mz = sca_internal._get_leading_dimensions(arr)
            if len(elem.coef) == 0:
                coef_idx = nlcpy.empty((), dtype='i8')
                coef = nlcpy.empty((), dtype='u8')
                coef_leading = nlcpy.empty((), dtype='i8')
            else:
                c_idx_tmp = []
                c_tmp = []
                c_leading_tmp = []
                for c in elem.coef:
                    c_idx_tmp.append(c[0])
                    c_tmp.append(c[1].ve_adr)
                    if c[1].size > 1:
                        c_leading_tmp.append(
                            (c[1].size,) +
                            (int(c[1].strides[-1] / arr.itemsize),) +
                            sca_internal._get_leading_dimensions(c[1])
                        )
                    else:
                        # below indicates (size, sx_c, mx_c, my_c, mz_c)
                        c_leading_tmp.append((1, 0, 0, 0, 0))
                    if c[1].size > 1:
                        if c[1].shape[-1] != desc_i.nx or \
                                (desc_i.ndim > 1 and c[1].shape[-2] != desc_i.ny) or \
                                (desc_i.ndim > 2 and c[1].shape[-3] != desc_i.nz) or \
                                (desc_i.ndim > 3 and c[1].shape[-4] != desc_i.nw):
                            expected = (desc_i.nw, desc_i.nz, desc_i.ny, desc_i.nx)
                            raise ValueError(
                                'coefficient ndarray set in desc_i has invalid shape: '
                                'got `{}`, expected `{}`.'.format(
                                    c[1].shape, expected[-c[1].ndim:]
                                )
                            )
                coef_idx = nlcpy.array(c_idx_tmp, dtype='i8')
                coef = nlcpy.array(c_tmp, dtype='u8')
                coef_leading = nlcpy.array(c_leading_tmp, dtype='i8')
            location = nlcpy.array(elem.location, dtype='i8')
            factor = nlcpy.array(elem.factor, dtype=self.dtype)
            args = (
                self.hnd_adr,
                arr._ve_array,
                location._ve_array,
                factor._ve_array,
                coef._ve_array,
                coef_idx._ve_array,
                coef_leading._ve_array,
                <int64_t>elem.offset,
                <int64_t>len(elem.factor),
                <int64_t>elem_offset,
                <int64_t>sx,
                <int64_t>mx,
                <int64_t>my,
                <int64_t>mz,
                veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),
            )
            request._push_and_flush_request(
                func_name,
                args,
            )
            elem_offset += nelm

        if self.dtype == numpy.dtype('float32'):
            func_name = 'nlcpy_sca_set_output_array_s'
        elif self.dtype == numpy.dtype('float64'):
            func_name = 'nlcpy_sca_set_output_array_d'
        else:
            raise RuntimeError

        # set output array
        desc_o._set_param_for_out(desc_i)
        elem = desc_o.elems[0]
        out = elem.array
        mx, my, mz = sca_internal._get_leading_dimensions(out)
        args = (
            self.hnd_adr,
            out._ve_array,
            <int64_t>(out.strides[-1] / out.itemsize),
            <int64_t>mx,
            <int64_t>my,
            <int64_t>mz,
            <int64_t>elem.offset,
            veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),
        )
        request._push_and_flush_request(
            func_name,
            args
        )

    def create_kernel(self):
        if self.destroyed:
            raise RuntimeError('this handle has already been destroyed.')

        cdef int64_t nx = self.desc_o.nx
        cdef int64_t ny = self.desc_o.ny
        cdef int64_t nz = self.desc_o.nz
        cdef int64_t nw = self.desc_o.nw
        cdef ndarray data_o = self.desc_o.elems[0].array

        ve_adr = numpy.empty(1, dtype='u8')
        fpe_flags = request._get_fpe_flag()
        args = (
            veo.OnStack(ve_adr, inout=veo.INTENT_OUT),
            self.hnd_adr,
            nx,
            ny,
            nz,
            nw,
            veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),
        )
        request._push_and_flush_request(
            'nlcpy_sca_code_create',
            args,
            sync=True
        )
        return kernel.kernel(<uint64_t>ve_adr, self.desc_i, self.desc_o)

    def reset_stencil_elements(self):
        if self.destroyed:
            raise RuntimeError('this handle has already been destroyed.')
        fpe_flags = request._get_fpe_flag()
        name = 'nlcpy_sca_stencil_reset_elements'
        args = (
            self.hnd_adr,
            veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),
        )
        request._push_and_flush_request(
            name,
            args
        )

    def _destroy(self):
        if self.destroyed:
            raise RuntimeError('this handle has already been destroyed.')
        fpe_flags = request._get_fpe_flag()
        args = (
            self.hnd_adr,
            veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),
        )
        request._push_and_flush_request(
            'nlcpy_sca_stencil_destroy',
            args
        )
        self.hnd_adr = 0
        self.dtype = None
        self.desc_i = None
        self.desc_o = None
        self.destroyed = True

    def __dealloc__(self):
        if self.destroyed:
            return
        self._destroy()

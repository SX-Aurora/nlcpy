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

from libc.stdint cimport *
from nlcpy import veo
from nlcpy.core.core cimport ndarray
from nlcpy.core.core cimport MemoryLocation
from nlcpy.core cimport dtype as _dtype
from nlcpy.request.ve_kernel cimport *
from nlcpy.request import request

from libcpp.vector cimport vector

import numpy
cimport numpy as cnp


cdef _write_array(a_cpu, ndarray a_ve):
    # TODO: check asynchronous request timing
    proc = veo._get_veo_proc()
    ctx = veo._get_veo_ctx()
    rm = request._get_request_manager()
    # rm._flush(sync=False)
    if rm.timing == 'on-the-fly':
        proc.write_mem(
            veo.VEMemPtr(proc, a_ve.ve_adr, a_ve.nbytes),
            a_cpu.data,
            a_cpu.nbytes
        )
    else:
        req = ctx.async_write_mem(
            veo.VEMemPtr(proc, a_ve.ve_adr, a_ve.nbytes),
            a_cpu.data,
            a_cpu.nbytes
        )
        vr = request._get_veo_requests()
        vr._push_req(req, callback='nothing')

cdef _destroy_array(uint64_t ve_adr, uint64_t nbytes):
    proc = veo._get_veo_proc()
    proc.free_mem(veo.VEMemPtr(proc, ve_adr, nbytes))

cdef _alloc_array(ndarray a):
    proc = veo._get_veo_proc()
    a.ve_adr = proc.alloc_mem(a.nbytes).addr

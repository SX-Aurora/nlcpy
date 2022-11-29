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

# distutils: language = c++

from libc.stdint cimport *
from nlcpy.veo cimport _veo
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport dtype as _dtype
from nlcpy.venode._venode cimport VENode
from nlcpy.venode._venode cimport VE

import gc


cdef _alloc_mem_core(size_t nbytes, VENode venode):
    cdef uint64_t veo_hmem
    cdef bint is_pool = True
    if not venode.connected:
        raise ValueError('VE process is not created on {}'.format(venode))
    veo_hmem = venode.pool.reserve(nbytes)
    if veo_hmem == 0LU:
        veo_hmem = venode.proc.alloc_hmem(nbytes)
        is_pool = False
    return (is_pool, veo_hmem)


cpdef _alloc_mem(size_t nbytes, VENode venode):
    cdef bint retry = False
    cdef uint64_t veo_hmem = 0
    cdef uint64_t ve_adr = 0
    cdef bint is_pool = True
    if not venode.connected:
        raise ValueError('VE process is not created on {}'.format(venode))
    try:
        is_pool, veo_hmem = _alloc_mem_core(nbytes, venode)
    except MemoryError:
        retry = True
    if retry:
        venode.request_manager._flush(sync=True)
        gc.collect()
        is_pool, veo_hmem = _alloc_mem_core(nbytes, venode)
    ve_adr = _veo.VEO_HMEM.get_hmem_addr(veo_hmem)

    return is_pool, veo_hmem, ve_adr


cpdef _write_mem(a_cpu, uint64_t ve_adr, size_t nbytes, VENode venode):
    assert ve_adr != 0
    if not venode.connected:
        raise ValueError('VE process is not created on {}'.format(venode))
    proc = venode.proc
    ctx = venode.ctx
    rm = venode.request_manager
    if rm.timing == 'on-the-fly':
        proc.write_mem(
            ve_adr,
            a_cpu.data,
            nbytes
        )
    else:
        req = ctx.async_write_mem(
            ve_adr,
            a_cpu.data,
            nbytes
        )
        venode.request_manager.veo_reqs._push_req(req, callback='nothing')


cpdef _free_mem(uint64_t veo_hmem, bint is_pool, VENode venode):
    assert veo_hmem != 0
    if not venode.connected:
        raise ValueError('VE process is not created on {}'.format(venode))
    if is_pool:
        venode.pool.release(veo_hmem)
    else:
        venode.proc.free_hmem(veo_hmem)


cpdef _hmemcpy(uint64_t dst, uint64_t src, size_t size):
    _veo.VEO_HMEM.hmemcpy(dst, src, size)

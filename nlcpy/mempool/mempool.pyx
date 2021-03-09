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

# distutils: sources=["nlcpy/mempool/nlcpy_mempool.c"]
import nlcpy
from nlcpy.mempool cimport mempool

cdef class MemPool(object):

    def __init__(self, VeoProc proc, VeoCtxt ctxt):
        self._hnd = proc.proc_handle
        cdef uint64_t lib = proc.lib_handle
        cdef veo_thr_ctxt *ctx = ctxt.thr_ctxt
        self._pool = nlcpy_mempool_alloc(self._hnd, lib, ctx)
        if self._pool == NULL:
            raise MemoryError("Out of memory on VE")

    def reserve(self, size_t size):
        cdef bint have_room = nlcpy_mempool_is_available(self._pool, size)
        cdef uint64_t ve_adr
        if have_room is True:
            iret = nlcpy_mempool_reserve(self._pool, size, &ve_adr)
            # print("VH ve_adr1=",hex(ve_adr))
            # print("VH size  1=",size)
            # print("VH iret  =",iret)
            if iret == NLCPY_OUT_OF_MEMORY:
                nlcpy.request.flush()
                iret = nlcpy_mempool_reserve(self._pool, size, &ve_adr)
                if iret == NLCPY_OUT_OF_MEMORY:
                    raise MemoryError("Out of memory on VE")
            elif iret == NLCPY_POOL_NOT_USED:
                pass
            elif iret == NLCPY_INTERNAL_ERROR:
                raise RuntimeError("An unknown error occured in nlcpy.mempool.reserve")
        else:
            # Not use the memory pool
            if veo_alloc_mem(self._hnd, &ve_adr, size):
                raise MemoryError("Out of memory on VE")
            # print("VH ve_adr2=",hex(ve_adr))
            # print("VH size  2=",size)
        return ve_adr

    def release(self, ve_adr):
        iret = nlcpy_mempool_release(self._pool, ve_adr)
        # print("VH release ve_adr=",hex(ve_adr))
        if iret == NLCPY_OUT_OF_MEMORY:
            raise MemoryError("Out of memory on VE")
        elif iret == NLCPY_POOL_NOT_USED:
            pass
        elif iret == NLCPY_INTERNAL_ERROR:
            raise RuntimeError("An unknown error occured in nlcpy.mempool.release")

    def __del__(self):
        if self._pool != NULL:
            nlcpy_mempool_free(self._pool)
            self._pool = NULL

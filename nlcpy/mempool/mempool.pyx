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

# distutils: sources=["nlcpy/mempool/nlcpy_mempool.c"]
import nlcpy
cimport mempool

cdef class MemPool(object):

    def __init__(self, VeoProc proc, VeoCtxt ctxt):
        cdef veo_proc_handle *hnd = proc.proc_handle
        cdef uint64_t lib = proc.lib_handle
        cdef veo_thr_ctxt *ctx = ctxt.thr_ctxt
        self._pool = nlcpy_mempool_alloc(hnd, lib, ctx)
        if self._pool == NULL:
            raise MemoryError("Out of memory on VE")

    def reserve(self, size_t size):
        cdef uint64_t ve_adr
        iret = nlcpy_mempool_reserve(self._pool, size, &ve_adr)
        # print("VH ve_adr=",hex(ve_adr))
        # print("VH size  =",size)
        # print("VH iret  =",iret)
        if iret is NLCPY_OUT_OF_MEMORY:
            nlcpy.request.flush()
            iret = nlcpy_mempool_reserve(self._pool, size, &ve_adr)
            if iret is NLCPY_OUT_OF_MEMORY:
                raise MemoryError("Out of memory on VE")
        elif iret is NLCPY_INTERNAL_ERROR:
            raise RuntimeError("An unknown error occured in nlcpy.mempool.reserve")
        return ve_adr

    def release(self, ve_adr):
        iret = nlcpy_mempool_release(self._pool, ve_adr)
        # print("VH release ve_adr=",hex(ve_adr))
        if iret is NLCPY_OUT_OF_MEMORY:
            raise MemoryError("Out of memory on VE")
        elif iret is NLCPY_INTERNAL_ERROR:
            raise RuntimeError("An unknown error occured in nlcpy.mempool.release")

    def __del__(self):
        if self._pool != NULL:
            nlcpy_mempool_free(self._pool)
            self._pool = NULL

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
# distutils: sources=["nlcpy/mempool/nlcpy_mempool.c"]

from nlcpy import _environment
from nlcpy.veosinfo import mem_info
from nlcpy.logging import _vp_logging
from nlcpy.mempool cimport mempool
from nlcpy.venode._venode cimport VENode
from nlcpy.veo cimport _nlcpy_veo_hook
cimport cython


@cython.no_gc
cdef class MemPool(object):

    def __init__(self, VENode venode):
        self._venode = venode
        cdef veo_proc_handle *proc_hnd = self._venode.proc.proc_handle

        # set pool size
        pool_size = _environment._get_pool_size()
        if pool_size is None:
            pool_size = DEFAULT_POOL_SIZE
        nlcpy_mempool_set_size(int(pool_size))

        # total memsize
        cdef size_t tot_memsize = mem_info(venode.pid)['kb_main_total'] * 1024

        # set hooked veo symbol
        cdef void *_hooked_veo_alloc_hmem = _nlcpy_veo_hook._get_hooked_veo_alloc_hmem()
        cdef void *_hooked_veo_free_hmem = _nlcpy_veo_hook._get_hooked_veo_free_hmem()
        nlcpy_mempool_set_hooked_veo_sym(_hooked_veo_alloc_hmem, _hooked_veo_free_hmem)

        self._pool = nlcpy_mempool_alloc(proc_hnd, tot_memsize)
        if self._pool == NULL:
            raise MemoryError("Out of memory on VE")

    def reserve(self, size_t size):
        cdef bint have_room = nlcpy_mempool_is_available(self._pool, size)
        cdef int iret
        cdef uint64_t ve_adr
        if have_room is True:
            iret = nlcpy_mempool_reserve(self._pool, size, &ve_adr)
            if iret == NLCPY_RESULT_OK:
                if _vp_logging._is_enable(_vp_logging.MEMPOOL):
                    _vp_logging.info(
                        _vp_logging.MEMPOOL,
                        'nlcpy_mempool_reserve used: nodeid=%d, addr=%x, size=%d',
                        self._venode.lid, ve_adr, size)
            elif iret == NLCPY_OUT_OF_MEMORY:
                raise MemoryError("Out of memory on VE")
            elif iret == NLCPY_POOL_NOT_USED:
                if _vp_logging._is_enable(_vp_logging.MEMPOOL):
                    _vp_logging.info(
                        _vp_logging.MEMPOOL,
                        'nlcpy_mempool_reserve not used: nodeid=%d, addr=%x, size=%d',
                        self._venode.lid, ve_adr, size)
            elif iret == NLCPY_INTERNAL_ERROR:
                raise RuntimeError("An unknown error occured in nlcpy.mempool.reserve")
        else:
            # Not use the memory pool
            ve_adr = 0LU
        return ve_adr

    def release(self, uint64_t ve_adr):
        iret = nlcpy_mempool_release(self._pool, ve_adr)
        if iret == NLCPY_RESULT_OK:
            if _vp_logging._is_enable(_vp_logging.MEMPOOL):
                _vp_logging.info(
                    _vp_logging.MEMPOOL,
                    'nlcpy_mempool_release: nodeid=%d, addr=%x',
                    self._venode.lid, ve_adr)
        elif iret == NLCPY_OUT_OF_MEMORY:
            raise MemoryError("Out of memory on VE")
        elif iret == NLCPY_POOL_NOT_USED:
            pass
        elif iret == NLCPY_INTERNAL_ERROR:
            raise RuntimeError("An unknown error occured in nlcpy.mempool.release")

    def get_status(self):
        cdef mempool_mng_t *mng = nlcpy_mempool_get_mng(self._pool)
        return {
            'base': hex(mng.base),
            'total_ve_memsize': mng.tot_memsize,
            'pool_capacity': mng.capa,
            'pool_used': mng.used,
            'pool_remainder': mng.remainder,
            'p': hex(mng.p),
            'maxp': hex(mng.maxp),
            'id': mng.id,
            'maxid': mng.maxid,
            'merged': mng.merged
        }

    def _finalize(self):
        if self._pool != NULL:
            nlcpy_mempool_free(self._pool)
            self._pool = NULL


cpdef _get_default_pool_size():
    return DEFAULT_POOL_SIZE

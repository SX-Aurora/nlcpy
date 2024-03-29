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

from libc.stdint cimport *
from nlcpy.veo.libveo cimport *
from nlcpy.veo._veo cimport *
from nlcpy.venode._venode cimport VENode
cimport cython

cdef extern from '../mempool/nlcpy_mempool.h':

    const size_t DEFAULT_POOL_SIZE

    const int NLCPY_RESULT_OK
    const int NLCPY_OUT_OF_MEMORY
    const int NLCPY_INTERNAL_ERROR
    const int NLCPY_POOL_NOT_USED

    ctypedef struct link_t:
        uint64_t *next
        uint64_t *prev
        uint64_t  head
        uint64_t  tail

    ctypedef struct sort_t:
        uint64_t *ids
        uint64_t *bytes
        uint64_t  num
        uint64_t  maxnum
        uint64_t *buff

    ctypedef struct hash_t:
        uint64_t *next
        uint64_t *prev
        uint64_t *head
        uint64_t  tail
        uint64_t  num
        uint64_t  maxnum
        uint64_t *buff

    ctypedef struct mempool_mng_t:
        veo_proc_handle *hnd
        uint64_t         base
        size_t           tot_memsize
        uint64_t         p
        uint64_t         maxp
        size_t           capa
        size_t           used
        size_t           remainder
        uint64_t         id
        uint64_t         maxid
        uint64_t *ptrs
        uint64_t *bytes
        uint64_t *esegs
        link_t *blocks
        sort_t *sort
        uint64_t *buff
        bint *dora
        bint             merged

    ctypedef struct mempool_t:
        veo_proc_handle *hnd
        uint64_t         base
        mempool_mng_t *mng
        hash_t *hash

    mempool_t *nlcpy_mempool_alloc(veo_proc_handle *hnd, size_t tot_memsize)
    int  nlcpy_mempool_reserve(mempool_t *pool, const size_t size, uint64_t *ve_adr)
    int  nlcpy_mempool_release(mempool_t *pool, const uint64_t ve_adr)
    void nlcpy_mempool_free(mempool_t *pool)
    bint nlcpy_mempool_is_available(const mempool_t *pool, const size_t size)
    void nlcpy_mempool_set_size(const size_t pool_size)
    void nlcpy_mempool_set_hooked_veo_sym(const void * const _hooked_veo_alloc_hmem,
                                          const void * const _hooked_veo_free_hmem)
    mempool_mng_t *nlcpy_mempool_get_mng(const mempool_t * const pool)


@cython.no_gc
cdef class MemPool(object):
    cdef mempool_t *_pool
    cdef mempool_mng_t *_mng
    cdef VENode _venode


cpdef _get_default_pool_size()

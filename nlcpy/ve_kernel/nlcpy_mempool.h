/*
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
*/

#ifndef MEMPOOL_H_INCLUDED
#define MEMPOOL_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include <inttypes.h>

#if !defined(__ve__)
#include <ve_offload.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/sysinfo.h>
#define _GNU_SOURCE
#include <sys/mman.h>
#endif

#define POOL_THREASHOLD        (size_t)         64*1024*1024 //  64MByte
#define SMALL_POOL_SIZE        (size_t)        256*1024*1024 // 256MByte
#define LARGE_POOL_SIZE        (size_t)     1*1024*1024*1024 //   1GByte
#define EXPAND_RATIO           2
#define THREASH_RATIO          0.4

#define DEFAULT_MAXID          (size_t)1024*1024
#define MAXID                  (size_t)0x7fffffffffffffff
#define HASH_SIZE              (size_t)1024*1024
#define ALIGN                  (size_t)8
#define ALIGNED_SIZE(size)     ((((size_t)(size)-1)/ALIGN + 1) * ALIGN )
#define HASH(x)                (uint64_t)((x/ALIGN)%HASH_SIZE)
#define END                    UINT64_MAX

// Error Indicator
const static int NLCPY_RESULT_OK      = 0;
const static int NLCPY_OUT_OF_MEMORY  = -1;
const static int NLCPY_INTERNAL_ERROR = -2;
const static int NLCPY_POOL_NOT_USED  = -3;

typedef struct link_tag {
    uint64_t *next;
    uint64_t *prev;
    uint64_t  head;
    uint64_t  tail;
    uint64_t  num;
} link_t;

typedef struct sort_tag {
    uint64_t *ids;
    uint64_t *bytes;
    uint64_t  num;
    uint64_t  maxnum;
    uint64_t *buff;
} sort_t;

typedef struct hash_tag {
    uint64_t *next;
    uint64_t *prev;
    uint64_t *head;
    uint64_t  tail;
    uint64_t  maxnum;
    uint64_t *buff;
} hash_t;

typedef struct mempool_mng_tag {
    struct veo_proc_handle *hnd;
    uint64_t                lib;
    struct veo_thr_ctxt    *ctx;
    uint64_t                base;
    size_t                  tot_memsize;
    uint64_t                p;
    uint64_t                maxp;
    size_t                  capa;
    uint64_t                id;
    uint64_t                maxid;
    uint64_t               *ptrs;
    uint64_t               *bytes;
    uint64_t               *esegs;
    link_t                 *blocks;
    sort_t                 *sort;
    uint64_t               *buff;
    bool                   *dora;
    bool                    merged;
} mempool_mng_t;


typedef struct mempool_tag {
    struct veo_proc_handle *hnd;
    uint64_t                lib;
    struct veo_thr_ctxt    *ctx;
    uint64_t                base;
    mempool_mng_t          *small;
    mempool_mng_t          *large;
    hash_t                 *hash;
} mempool_t;


#if !defined(__ve__)
// Entry functions
mempool_t *nlcpy_mempool_alloc(struct veo_proc_handle *hnd, uint64_t lib, struct veo_thr_ctxt *ctx);
int        nlcpy_mempool_reserve(mempool_t *pool, const size_t size, uint64_t *ve_adr);
int        nlcpy_mempool_release(mempool_t *pool, const int64_t ve_adr);
void       nlcpy_mempool_free(mempool_t *pool);
bool       nlcpy_mempool_is_available(const mempool_t *pool, const size_t size);

// Driver functions
mempool_mng_t *nlcpy__mempool_alloc_mng(struct veo_proc_handle *hnd, uint64_t lib, struct veo_thr_ctxt *ctx, const uint64_t base, const uint64_t offset, const size_t default_poolsize, const size_t tot_memsize);
int            nlcpy__mempool_reserve(mempool_mng_t *mng, const size_t size, uint64_t *id, uint64_t *ve_adr);
int            nlcpy__mempool_release(mempool_mng_t *mng, const uint64_t id);
void           nlcpy__mempool_free_mng(mempool_mng_t *mng);
// Allocation
sort_t *nlcpy__mempool_create_sort(const size_t size);
hash_t *nlcpy__mempool_create_hash(const size_t size);
// Reallocation
int nlcpy__mempool_extend_pool(const size_t n, mempool_mng_t *mng);
int nlcpy__mempool_extend_mnglist(const size_t n, mempool_mng_t *mng);
int nlcpy__mempool_extend_sort(const size_t n, sort_t *hp);
int nlcpy__mempool_extend_hash(const size_t n, hash_t *hh);
// Deallocation
void nlcpy__mempool_free_sort(sort_t *hp);
void nlcpy__mempool_free_hash(hash_t *hh);
// Update
int nlcpy__mempool_append_to_link(const uint64_t id, link_t *block);
int nlcpy__mempool_remove_from_link(const uint64_t id, link_t *block);
int nlcpy__mempool_register_to_sort(const uint64_t id, const uint64_t size, sort_t *sort);
int nlcpy__mempool_extract_from_sort(sort_t *sort);
int nlcpy__mempool_append_to_hash(const uint64_t ve_adr, const uint64_t gid, hash_t *hh);
int nlcpy__mempool_remove_from_hash(const uint64_t ve_adr, mempool_t *mng, uint64_t *gid);
int nlcpy__mempool_split_dead_blocks(const uint64_t id, const uint64_t new_id, link_t *ll);
int nlcpy__mempool_merge_dead_blocks(mempool_mng_t *mng);
// Auxiliary
bool nlcpy__mempool_is_available(mempool_mng_t *mng, const size_t size);
#if defined(DEBUG)
size_t nlcpy__mempool_get_capasity(const mempool_mng_t *mng);
#endif

#else /* __ve__ */

uint64_t nlcpy__mempool_alloc_ve(const uint64_t size);
uint64_t nlcpy__mempool_extend_ve(const uint64_t size);
uint64_t nlcpy__mempool_free_ve(const uint64_t base);

#endif /* __ve__ */

#define FREE(p)  {if(p!=NULL) {free(p); p=NULL;} }

#endif /* MEMPOOL_H_INCLUDED */

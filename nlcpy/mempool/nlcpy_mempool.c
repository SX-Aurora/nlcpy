/*
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
*/

#include "nlcpy_mempool.h"

static size_t POOL_SIZE = DEFAULT_POOL_SIZE;

// *************************************************************************
//  Entry Functions
// *************************************************************************
mempool_t *nlcpy_mempool_alloc(struct veo_proc_handle *hnd, size_t tot_memsize)
{
#if defined(DEBUG)
    fprintf(stderr,"mempool_alloc: hnd=%p, tot_memsize=%ld\n", hnd, tot_memsize);
#endif
    mempool_t *pool    = (mempool_t *)malloc(sizeof(mempool_t));
    if (pool==NULL) return NULL;
    // initialize
    pool->base          = 0;
    pool->mng           = NULL;
    pool->hash          = NULL;
    pool->hnd           = hnd;

    // allocate pool
    int iret;
    void *vemem;
    iret = veo_alloc_hmem(hnd, &vemem, POOL_SIZE);
    if (iret != VEO_COMMAND_OK) return NULL;
    pool->base = (uint64_t)vemem;

    //
    mempool_mng_t *mng;
    pool->mng = nlcpy__mempool_alloc_mng(hnd, pool->base, (uint64_t)0, (size_t)POOL_SIZE, tot_memsize);
    if (pool->mng == NULL) {
        nlcpy_mempool_free(pool);
        return NULL;
    }
    mng = pool->mng;

    // create a dictionary (hash table) to get a block id from VE address
    const uint64_t maxid = mng->maxid;
    pool->hash = nlcpy__mempool_create_hash(maxid);
    if (pool->hash == NULL) {
        nlcpy_mempool_free(pool);
        return NULL;
    }

    return pool;
}


int nlcpy_mempool_reserve(mempool_t *pool, const size_t size, uint64_t *ve_adr)
{
#if defined(DEBUG)
    fprintf(stderr,"mempool_reserve\n");
#endif
    mempool_mng_t *mng = pool->mng;
    int iret;
    uint64_t lid, gid;

    // If 0 byte reservation is ordered, this function internally reserves 8 bytes.
    size_t asize = size;
    if(size==0) asize=8;

    iret = nlcpy__mempool_reserve(mng, asize, &lid, ve_adr);
    gid = lid;
#if defined(DEBUG)
    fprintf(stderr,"  ve_adr=%lx size=%ld, base=%lx, endptr=%lx, gid=%ld, lid=%ld\n", *ve_adr, asize, pool->base, mng->maxp, gid, lid);
#endif
    if (iret!=NLCPY_RESULT_OK && iret!=NLCPY_POOL_NOT_USED) return iret;

    if (iret==NLCPY_RESULT_OK) {
        // register ve_adr and gid to the hash table.
        iret = nlcpy__mempool_append_to_hash(*ve_adr, gid, pool->hash);
    } else {
        // overflow
        iret = NLCPY_POOL_NOT_USED;
    }

    return iret;
}


int nlcpy_mempool_release(mempool_t *pool, const int64_t ve_adr)
{
#if defined(DEBUG)
    fprintf(stderr,"mempool_release\n");
#endif
    mempool_mng_t *mng = pool->mng;
    uint64_t gid, lid;

    int iret = nlcpy__mempool_remove_from_hash(ve_adr, pool, &gid);
    if (iret) return iret;

    lid = gid;
    iret = nlcpy__mempool_release(mng, lid); if (iret) return iret;
#if defined(DEBUG)
    fprintf(stderr,"  ve_adr=%lx gid=%ld, lid=%ld\n", ve_adr, gid, lid);
#endif
    return NLCPY_RESULT_OK;
}


void nlcpy_mempool_free(mempool_t *pool)
{
#if defined(DEBUG)
    fprintf(stderr,"mempool_free\n");
#endif
    // The memory pool is allocated in heap area.
    // So, nothing to do on VE side.

    nlcpy__mempool_free_mng(pool->mng);
    nlcpy__mempool_free_hash(pool->hash);
    //
    FREE(pool);
}


bool nlcpy_mempool_is_available(const mempool_t *pool, const size_t size)
{
    mempool_mng_t *mng;
    mng = pool->mng;
#if defined(DEBUG)
    bool iret = nlcpy__mempool_is_available(mng, size);
//    size_t capa = nlcpy__mempool_get_capasity(mng);
//    printf("capa = %zu\n",capa);
    return iret;
#endif
    return nlcpy__mempool_is_available(mng, size);
}


void nlcpy_mempool_set_size(const size_t pool_size) {
#if defined(DEBUG)
    printf("set pool size: %lu\n", pool_size);
#endif
    POOL_SIZE = pool_size;
}


mempool_mng_t *nlcpy_mempool_get_mng(const mempool_t * const pool) {
    return pool->mng;
}


// *************************************************************************
//  Driver Functions
// *************************************************************************
mempool_mng_t *nlcpy__mempool_alloc_mng(struct veo_proc_handle *hnd, const uint64_t base, const uint64_t offset, const size_t default_poolsize, const size_t tot_memsize)
{
    mempool_mng_t *mng     = (mempool_mng_t *)malloc(sizeof(mempool_mng_t));
    if (mng==NULL) return mng;
    //
    mng->hnd         = hnd;
    mng->base        = base;
    mng->p           = base + offset;
    mng->id          = 0;
    mng->tot_memsize = tot_memsize;
    mng->maxp        = mng->p + default_poolsize;
    mng->capa        = default_poolsize;
    mng->used        = 0;
    mng->remainder   = mng->capa;
    mng->maxid       = DEFAULT_MAXID;
    // initialization for the deallocation
    mng->buff        = NULL;
    mng->blocks      = NULL;
    mng->sort        = NULL;
    mng->alive       = NULL;
    mng->merged      = true;

    //
    link_t *ll = mng->blocks = (link_t *)malloc(sizeof(link_t));
    if (ll == NULL) {
        nlcpy__mempool_free_mng(mng);
        return NULL;
    }

    //
#if ! defined(POOL_EXTENTON)
    mng->buff        = (uint64_t *)malloc(sizeof(uint64_t)*mng->maxid*4);
#else
    mng->buff        = (uint64_t *)malloc(sizeof(uint64_t)*mng->maxid*5);
#endif
    if (mng->buff == NULL) {
        nlcpy__mempool_free_mng(mng);
        return NULL;
    }
    mng->ptrs        = mng->buff;
    mng->bytes       = mng->buff + mng->maxid;
#if ! defined(POOL_EXTENTON)
    ll->next         = mng->buff + mng->maxid * 2;
    ll->prev         = mng->buff + mng->maxid * 3;
#else
    mng->esegs       = mng->buff + mng->maxid * 2;
    ll->next         = mng->buff + mng->maxid * 3;
    ll->prev         = mng->buff + mng->maxid * 4;
#endif
    ll->head         = END;
    ll->tail         = END;
    //
    mng->sort        = nlcpy__mempool_create_sort(mng->maxid);
    if (mng->sort == NULL) {
        nlcpy__mempool_free_mng(mng);
        return NULL;
    }
    //
    mng->alive        = (bool*)malloc(sizeof(bool)*mng->maxid);
    if (mng->alive == NULL) {
        nlcpy__mempool_free_mng(mng);
        return NULL;
    }
    uint64_t i;
    for (i=0; i< mng->maxid; i++) mng->alive[i] = false;
    //
    return mng;
}


int nlcpy__mempool_reserve(mempool_mng_t *mng, const size_t size, uint64_t *id, uint64_t *ve_adr)
{
    int iret;
    link_t *blocks = mng->blocks;
    sort_t *sort   = mng->sort;
    // initialize
    *id =0;
    *ve_adr =0;

    uint64_t asize  = ALIGNED_SIZE(size);
    size_t   new_p = mng->p + asize;
    if (new_p <= mng->maxp) {
        uint64_t new_id = mng->id++;
        if (new_id >= mng->maxid) {
            // extend memory because the number of ids is overflow
            iret = nlcpy__mempool_extend_mnglist((size_t)mng->maxid*2, mng);
            if (iret) return NLCPY_OUT_OF_MEMORY;
        } else if (new_id >= MAXID) {
            *id = mng->id = MAXID;
            return NLCPY_POOL_NOT_USED;
        }
        iret = nlcpy__mempool_append_to_link(new_id, blocks);  if (iret) return iret;

        // save pointer and size of id-th
        mng->ptrs[new_id]  = mng->p;
        mng->bytes[new_id] = asize;
#if defined(POOL_EXTENTON)
        mng->esegs[new_id] = mng->maxp;
#endif
        mng->alive[new_id]  = true; // the id-th block is newly assigned.

        *id     = new_id;
        *ve_adr = (uint64_t)mng->p;

        // update pointer;
        mng->p = new_p;

    } else {
        // Next, check the size of sort blocks
        if (sort->num>0 && sort->bytes[0] >= asize) {
            // Fortunately, we found a room whose size is enough.
            // The room can be reused.
            *id        = sort->ids[0];
            uint64_t s = sort->bytes[0];
            if (mng->alive[*id]) {
                fprintf(stderr,"NLCPy internal error: the %ld-th block has already been reserved.\n",*id);
                return NLCPY_INTERNAL_ERROR;
            }
            iret = nlcpy__mempool_extract_from_sort(sort);    if (iret) return iret;
            //
            if (s>asize) {
                uint64_t new_id = mng->id++;
                if (new_id >= mng->maxid) {
                    // extend memory because the number of id is overflow
                    iret = nlcpy__mempool_extend_mnglist((size_t)mng->maxid*2, mng);
                    if (iret) return NLCPY_OUT_OF_MEMORY;
                } else if (new_id >= MAXID) {
                    *id = mng->id = MAXID;
                    return NLCPY_POOL_NOT_USED;
                }
                //
                iret = nlcpy__mempool_split_dead_blocks(*id, new_id, blocks); if (iret) return iret;
                mng->bytes[*id]    = asize;
                // register as a new block
                mng->ptrs[new_id]  = mng->ptrs[*id] + asize;
                mng->alive[new_id]  = false;
#if defined(POOL_EXTENTON)
                mng->esegs[new_id] = mng->esegs[*id];
#endif
                mng->bytes[new_id] = s - asize;
                iret = nlcpy__mempool_register_to_sort(new_id, mng->bytes[new_id], mng->sort); if (iret) return iret;
            }
            *ve_adr = mng->ptrs[*id];
            mng->alive[*id] = true; // revive the id-th blocks

        } else {
            new_p = mng->p + asize;
            if (new_p > mng->maxp) {
#if ! defined(POOL_EXTENTON)
                // fprintf(stderr,"NLCPy internal error: the memory pool size is not enogh.\n");
                // return NLCPY_INTERNAL_ERROR;
                return NLCPY_POOL_NOT_USED;
#else
                // Unfortunately, we could not find a room, have to extend the pool.
                // extend memory because mng size is overflow
                size_t extended_size = ( asize > mng->capa ) ?  asize*EXPAND_RATIO : mng->capa;
                       // Is the extened size bigger than total_memsize * THREASH_RATIO?
                       // If so, heap is extended just by asize.
                       extended_size = ( mng->capa > mng->tot_memsize * THREASH_RATIO) ? asize : extended_size;
                       //extended_size = ( extended_size > 1*1024*1024*1024 ) ? asize : extended_size;
                iret = nlcpy__mempool_extend_pool(extended_size, mng);
                if (iret) return iret;
                new_p = mng->p + asize;
#endif
            }

            // a new block is assigned.
            uint64_t new_id = mng->id++;
            if (new_id >= mng->maxid) {
                // extend memory because the number of id is overflow
                iret = nlcpy__mempool_extend_mnglist((size_t)mng->maxid*2, mng);
                if (iret) return NLCPY_OUT_OF_MEMORY;
            } else if (new_id >= MAXID) {
                *id = mng->id = MAXID;
                return NLCPY_POOL_NOT_USED;
            }
            iret = nlcpy__mempool_append_to_link(new_id, blocks);  if (iret) return iret;

            // save pointer and size of id-th
            mng->ptrs[new_id]  = mng->p;
            mng->bytes[new_id] = asize;
#if defined(POOL_EXTENTON)
            mng->esegs[new_id] = mng->maxp;
#endif
            mng->alive[new_id]  = true; // the id-th block is newly assigned.

            *id     = new_id;
            *ve_adr = mng->p;

            // update pointer;
            mng->p = new_p;
        }
    }
    mng->used += asize;
    mng->remainder -= asize;
    return NLCPY_RESULT_OK;
}


int nlcpy__mempool_release(mempool_mng_t *mng, const uint64_t id)
{
    uint64_t size = mng->bytes[id];

    if ( id==END ) {
        fprintf(stderr,"NLCPy internal error: Invalid ID is detected. ( ID = %ld )\n",id);
        return NLCPY_INTERNAL_ERROR;
    }
    if ( !mng->alive[id] ) {
        fprintf(stderr,"NLCPy internal error: the %ld-th block is not reserved.\n",id);
        return NLCPY_INTERNAL_ERROR;
    }

    int iret = nlcpy__mempool_register_to_sort(id, size, mng->sort);
    if (iret) return iret;

    mng->alive[id] = false; // the id-th block is died.

    const uint64_t *next = mng->blocks->next;
    const uint64_t *prev = mng->blocks->prev;
    if ( next[id]!=END && !mng->alive[next[id]] ) mng->merged = false; // it is possible to merge dead blocks.
    if ( prev[id]!=END && !mng->alive[prev[id]] ) mng->merged = false; // it is possible to merge dead blocks.

    mng->used -= size;
    mng->remainder += size;
    return iret;
}


void nlcpy__mempool_free_mng(mempool_mng_t *mng)
{
    //
    FREE(mng->buff);
    FREE(mng->alive);
    FREE(mng->blocks);
    //
    nlcpy__mempool_free_sort(mng->sort);
    //
    FREE(mng);
    //
    return;
}


// *************************************************************************
//  Allocation Functions (work routine)
// *************************************************************************
sort_t *nlcpy__mempool_create_sort(const size_t n)
{
    sort_t *st = (sort_t *)malloc(sizeof(sort_t));
    if (st == NULL) {
        return NULL;
    }

    st->buff = (uint64_t *)malloc(sizeof(uint64_t)*n*2);
    if (st->buff == NULL) {
        FREE(st);
        return NULL;
    }

    st->ids   = st->buff;
    st->bytes = st->buff + n;

    st->num   = 0;
    st->maxnum = n;
    return st;
}


hash_t *nlcpy__mempool_create_hash(const size_t n)
{
    hash_t *hh = (hash_t *)malloc(sizeof(hash_t));
    if (hh == NULL) {
        return NULL;
    }

    hh->buff = (uint64_t *)malloc(sizeof(uint64_t)*((size_t)n*2));
    if (hh->buff == NULL) {
        FREE(hh);
        return NULL;
    }
    hh->next = hh->buff;
    hh->prev = hh->buff + n;

    hh->head = (uint64_t *)malloc(sizeof(uint64_t)*((size_t)HASH_SIZE));
    hh->tail = END;
    uint64_t i;
    for (i=0;i<HASH_SIZE;i++) hh->head[i] = hh->tail;

    hh->maxnum = n;
    return hh;
}


// *************************************************************************
//  Update Functions (work routine)
// *************************************************************************
#if defined(POOL_EXTENTON)
int nlcpy__mempool_extend_pool(const size_t n, mempool_mng_t *mng)
{
    int iret;
    uint64_t ireq, retval;
    struct veo_args *args = veo_args_alloc();

    iret = veo_alloc_hmem(mng->hnd, &retval, n);
    if (iret != VEO_COMMAND_OK) return NLCPY_INTERNAL_ERROR;

    if (retval==0) return NLCPY_OUT_OF_MEMORY;

    // the pointer of the extended pool
    mng->p = retval;
    // the end of the extended pool
    mng->maxp = retval + n;
    mng->capa+= n;

    link_t *blocks = mng->blocks;
    uint64_t id = blocks->tail;
    while ( id !=END ) {
        // If the end of the previous segment equals to the current statging pointer,
        // the previous segment can be merged.
        if ( mng->esegs[id] == mng->p ){
            mng->esegs[id] = mng->maxp;
        } else {
            break;
        }
        id = blocks->prev[id];
    }
#if defined(DEBUG)
    fprintf(stderr,"extend_pool, head=%lx, tail=%lx n=%ld, capa=%ld\n",retval,retval+n,n,mng->capa);
#endif

    return NLCPY_RESULT_OK;
}
#endif


int nlcpy__mempool_extend_mnglist(const size_t n, mempool_mng_t *mng)
{
    assert(n > mng->maxid);

#if ! defined(POOL_EXTENTON)
    uint64_t *buff = (uint64_t *)malloc(sizeof(uint64_t)*n*4);
#else
    uint64_t *buff = (uint64_t *)malloc(sizeof(uint64_t)*n*5);
#endif
    if (buff == NULL) {
        return NLCPY_OUT_OF_MEMORY;
    }
    bool *alive = (bool*)malloc(sizeof(bool)*n);
    if (alive == NULL) {
        return NLCPY_OUT_OF_MEMORY;
    }

    uint64_t *ptrs  = buff;
    uint64_t *bytes = buff + n;
#if ! defined(POOL_EXTENTON)
    uint64_t *next  = buff + n * 2;
    uint64_t *prev  = buff + n * 3;
#else
    uint64_t *esegs = buff + n * 2;
    uint64_t *next  = buff + n * 3;
    uint64_t *prev  = buff + n * 4;
#endif

    link_t *ll = mng->blocks;
    memcpy(ptrs,  mng->ptrs,  sizeof(uint64_t)*(size_t)mng->maxid);
    memcpy(bytes, mng->bytes, sizeof(uint64_t)*(size_t)mng->maxid);
#if defined(POOL_EXTENTON)
    memcpy(esegs, mng->esegs, sizeof(uint64_t)*(size_t)mng->maxid);
#endif
    memcpy(next,  ll->next,   sizeof(uint64_t)*(size_t)mng->maxid);
    memcpy(prev,  ll->prev,   sizeof(uint64_t)*(size_t)mng->maxid);
    memcpy(alive, mng->alive, sizeof(bool)*(size_t)mng->maxid);
    uint64_t i;
    for (i=mng->maxid; i<n; i++) alive[i] = false; //initialization

    FREE(mng->buff);
    mng->buff  = buff;
    mng->ptrs  = ptrs;
    mng->bytes = bytes;
#if defined(POOL_EXTENTON)
    mng->esegs = esegs;
#endif
    ll->next   = next;
    ll->prev   = prev;

    FREE(mng->alive);
    mng->alive  = alive;

    mng->maxid = n;

    // extend area of mng->sort because the number of id is overflow
    return nlcpy__mempool_extend_sort(n, mng->sort);
}


int nlcpy__mempool_extend_sort(const size_t n, sort_t *st)
{
    assert(n > st->maxnum);

    uint64_t *buff = (uint64_t *)malloc(sizeof(uint64_t)*n*2);
    if (buff == NULL) {
        return NLCPY_OUT_OF_MEMORY;
    }
    uint64_t *ids   = buff;
    uint64_t *bytes = buff + n;

    memcpy(ids, st->ids, sizeof(uint64_t)*(st->maxnum));
    memcpy(bytes, st->bytes, sizeof(uint64_t)*(st->maxnum));

    FREE(st->buff);
    st->buff  = buff;
    st->ids   = ids;
    st->bytes = bytes;

    st->maxnum = n;
    return NLCPY_RESULT_OK;
}


int nlcpy__mempool_extend_hash(const size_t n, hash_t *hh)
{
    assert(n > hh->maxnum);

    uint64_t *buff = (uint64_t *)malloc(sizeof(uint64_t)*(n*2));
    if (buff == NULL) {
        return NLCPY_OUT_OF_MEMORY;
    }
    uint64_t *next = buff;
    uint64_t *prev = buff + n;

    memcpy(next, hh->next, sizeof(uint64_t)*hh->maxnum);
    memcpy(prev, hh->prev, sizeof(uint64_t)*hh->maxnum);

    FREE(hh->buff);
    hh->buff = buff;
    hh->next = next;
    hh->prev = prev;

    hh->maxnum = n;
    return NLCPY_RESULT_OK;
}


// *************************************************************************
//  Deallocation Functions (work routine)
// *************************************************************************
void nlcpy__mempool_free_sort(sort_t *st)
{
    FREE(st->buff);
    FREE(st);
    return;
}


void nlcpy__mempool_free_hash(hash_t *hh)
{
    FREE(hh->buff);
    FREE(hh->head);
    FREE(hh);
    return;
}


// *************************************************************************
//  Updating Functions (work routine)
// *************************************************************************
int nlcpy__mempool_append_to_link(const uint64_t id, link_t *ll)
{
    uint64_t *prev = ll->prev;
    uint64_t *next = ll->next;
    uint64_t *head = &(ll->head);
    uint64_t *tail = &(ll->tail);

    //
    if ( *tail != END )
        next[*tail] = id;
    else
        *head = id;

    prev[id] = *tail;
    next[id] = END;
    *tail = id;
#if defined(DEBUG)
    fprintf(stderr, "nlcpy__mempool_append_to_link: id=%lu, prev=%lu, next=%lu, head=%ld, tail=%lu\n",
            id, prev[id], next[id], *head, *tail);
#endif
    return NLCPY_RESULT_OK;
}


int nlcpy__mempool_split_dead_blocks(const uint64_t id, const uint64_t new_id, link_t *ll)
{
    uint64_t *prev = ll->prev;
    uint64_t *next = ll->next;
    uint64_t *tail = &(ll->tail);
    //
    prev[new_id] = id;
    next[new_id] = next[id];
    if ( next[id] != END ){
        prev[next[id]] = new_id;
    }else{
        *tail = new_id;
    }
    next[id] = new_id;
    return NLCPY_RESULT_OK;
}


int nlcpy__mempool_remove_from_link(const uint64_t id, link_t *ll)
{
    uint64_t *prev = ll->prev;
    uint64_t *next = ll->next;
    uint64_t *head = &(ll->head);
    uint64_t *tail = &(ll->tail);
    //
    if ( next[id] != END ){
        prev[next[id]] = prev[id];
    }else{
        *tail = prev[id];
        if(*tail!=END) next[*tail] = END;
    }

    if( prev[id] != END ){
        next[prev[id]] = next[id];
    }else{
        *head = next[id];
        if( *head != END ) prev[*head] = END;
    }
    return NLCPY_RESULT_OK;
}


int nlcpy__mempool_register_to_sort(const uint64_t id, const uint64_t size, sort_t *st)
{
    uint64_t *indx = st->ids;
    uint64_t *sort = st->bytes;
    uint64_t *num  = &(st->num);

    //
    // descending order
    (*num)++;
    uint64_t p1 = *num;
    uint64_t p2 = p1/2;
    indx[p1-1] = id;
    sort[p1-1] = size;
    if(p2!=0){
        while(p1>1 && sort[p1-1]>sort[p2-1]) {
            uint64_t id = indx[p2-1];
            indx[p2-1]  = indx[p1-1];
            indx[p1-1]  = id;
            uint64_t t  = sort[p2-1];
            sort[p2-1]  = sort[p1-1];
            sort[p1-1]  = t;
            p1=p2;
            p2=p1/2;
            if (p2==0) return NLCPY_RESULT_OK;
        }
    }
    return NLCPY_RESULT_OK;
}


int nlcpy__mempool_extract_from_sort(sort_t *st)
{
    uint64_t *indx = st->ids;
    uint64_t *sort = st->bytes;
    uint64_t *num  = &(st->num);
    //
    if (*num<=0) {
        fprintf(stderr,"NLCPy: double free in nlcpy__mempool_extract_from_sort()\n");
        return NLCPY_INTERNAL_ERROR;
    }
    //
    uint64_t p1 = 1;
    uint64_t p2 = p1*2;
    (*num)--;
    indx[0] = indx[*num];
    sort[0] = sort[*num];
    // descending order
    while(p2<=*num){
        if (p2+1<=*num){
            if (sort[p2-1]<sort[p2]) p2++;
        }
        if (sort[p1-1]<sort[p2-1]) {
            uint64_t id = indx[p2-1];
            indx[p2-1]  = indx[p1-1];
            indx[p1-1]  = id;
            uint64_t t  = sort[p2-1];
            sort[p2-1]  = sort[p1-1];
            sort[p1-1]  = t;
        }
        p1=p2;
        p2=p1*2;
    }
    return NLCPY_RESULT_OK;
}


int nlcpy__mempool_append_to_hash(const uint64_t ve_adr, const uint64_t gid, hash_t *hash)
{
    while (gid>=hash->maxnum) {
        // the number of gid is overflow
        const int iret = nlcpy__mempool_extend_hash((size_t)hash->maxnum*2, hash);
        if (iret) return iret;
    }
    //
    uint64_t *next = hash->next;
    uint64_t *prev = hash->prev;
    uint64_t *head = hash->head;
    uint64_t  tail = hash->tail;
    uint64_t  h    = HASH(ve_adr);
    //

#ifdef DEBUG
    fprintf(stderr, "nlcpy__mempool_append_to_hash: ve_adr=%lu, gid=%lu, h=%lu, head[h]=%lu\n",
            ve_adr, gid, h, head[h]);
#endif

    if ( head[h] != tail ) {
        next[gid] = head[h];
        prev[head[h]] = gid;
    } else {
        next[gid] = tail;
        prev[gid] = tail;
    }

    if (next[gid]==gid) {
        fprintf(stderr,"NLCPy internal error: allocated doubly.\n");
        return NLCPY_INTERNAL_ERROR;
    }

    head[h] = gid;
    return NLCPY_RESULT_OK;
}


int nlcpy__mempool_remove_from_hash(const uint64_t ve_adr, mempool_t *pool, uint64_t *gid)
{
    hash_t   *hash = pool->hash;
    uint64_t *next = hash->next;
    uint64_t *prev = hash->prev;
    uint64_t *head = hash->head;
    uint64_t  tail = hash->tail;
    uint64_t  h    = HASH(ve_adr);
    uint64_t  id   = head[h];
#if defined(DEBUG)
    fprintf(stderr, "nlcpy__mempool_remove_from_hash: prev=%lu, next=%lu, head=%lu, tail=%lu, h=%lu, id=%lu\n",
            *prev, *next, *head, tail, h, id);
#endif
    //
    while ( id != tail ) {
        uint64_t  lid = id;
        mempool_mng_t *mng;
        mng = pool->mng;
        if ( ve_adr==mng->ptrs[lid] ) {
            // found the-id corresponding to ve_adr
            if ( head[h] != id ){
                if ( next[id] != tail ){
                    prev[next[id]] = prev[id];
                }
                if ( prev[id] != tail ){
                    next[prev[id]] = next[id];
                } else {
                    head[h] = tail;
                }
            } else {
                head[h]  = next[id];
                prev[id] = tail;
            }
            next[id] = tail;
            prev[id] = tail;
            break;
        }
        id = next[id];
    }
    if ( id == tail ) {
        // not found. the case may be allocated by veo_alloc_hmem.
        return NLCPY_INTERNAL_ERROR;
    }
    *gid = id; // return
    return NLCPY_RESULT_OK;
}


int nlcpy__mempool_merge_dead_blocks(mempool_mng_t *mng)
{
    if ( mng->sort->num < 2 ) return NLCPY_RESULT_OK;
    if ( mng->blocks->head < END ) return NLCPY_RESULT_OK;
    // clear sort blocks
    mng->sort->num = 0;

    link_t *blocks = mng->blocks;
    uint64_t id = blocks->head;
    while ( id != blocks->tail ) {
        if ( !mng->alive[id] ) {
            // found sort blocks
            const uint64_t id0 = id;  // head of the sort blocks
#if defined(POOL_EXTENTON)
            const uint64_t p0  = mng->ptrs[id0];
            const uint64_t e0  = mng->esegs[id0];
#endif
            uint64_t size = mng->bytes[id];
            id = blocks->next[id];
            while ( id != END && !mng->alive[id] ) {
                const uint64_t s = size+mng->bytes[id];
#if defined(POOL_EXTENTON)
                if ( p0+s > e0 ) break;
#endif
                size = s;
                mng->bytes[id] = 0;
                const uint64_t k = id;
                id = blocks->next[id];  // go to next blocks
                nlcpy__mempool_remove_from_link(k, blocks);
            }
            mng->bytes[id0] = size;
            int iret = nlcpy__mempool_register_to_sort(id0, size, mng->sort);
            if (iret) return iret;
        } else {
            id = blocks->next[id];  // go to next blocks
        }
    }
#if defined(DEBUG)
    fprintf(stderr,"merged, mng->sort->num=%ld\n",mng->sort->num);
#endif
    return NLCPY_RESULT_OK;
}


bool nlcpy__mempool_is_available(mempool_mng_t *mng, const size_t size)
{
    // Check the size of unused area
    size_t unused = mng->maxp - mng->p;
    if ( unused >= size ) return true;

    // Check the size of dead area
    if ( mng->sort->num > 0 ) {
        if ( mng->sort->bytes[0] >= size ) {
            return true;
        } else if ( !mng->merged ) {
            // Merge dead blocks
            int iret = nlcpy__mempool_merge_dead_blocks(mng);
            if (iret) return false;
            mng->merged = true; // sort blocks are merged

            return ( mng->sort->bytes[0] >= size );
        }
    }
    return false;
}


#if defined(DEBUG)
size_t nlcpy__mempool_get_capasity(const mempool_mng_t *mng)
{
    size_t ucapa = mng->maxp - mng->p;
    size_t dcapa=0;
    uint64_t i;
    for ( i = 0; i < mng->sort->num; i++ ) dcapa +=  mng->sort->bytes[i];
#if 0
    if ( mng->sort->num > 0 ) {
        dcapa = ucapa;
        if ( ucapa < mng->sort->bytes[0] ) {
            ucapa = mng->sort->bytes[0];
        }
    }
#else
    uint64_t id = mng->blocks->head;
    size_t acapa=0;
    while ( id != END ) {
        if ( mng->alive[id] ) {
            acapa+=mng->bytes[id];
        }
        id = mng->blocks->next[id];  // go to next blocks
    }
#endif
    size_t tcapa = ucapa + dcapa + acapa;
    printf("unused = %zu, dead = %zu, alive = %zu, total=%zu, num=%" PRIu64 "\n",ucapa, dcapa, acapa, tcapa, mng->sort->num);
    return (ucapa>dcapa) ? ucapa : dcapa;
}
#endif

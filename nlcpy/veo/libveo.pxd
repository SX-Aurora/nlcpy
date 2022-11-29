#
# * The source code in this file is based on the soure code of PyVEO.
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
#
# Interface to libveo
#

from libc.stdint cimport *

cdef extern from "<ve_offload.h>" nogil:

    # uncomment until API_VERSION comes back into mainline VEO
    # enum: VEO_API_VERSION

    # maximum number of arguments to VEO calls (32)
    cdef enum: VEO_MAX_NUM_ARGS

    # invalid request ID
    cdef enum: VEO_REQUEST_ID_INVALID

    cdef enum veo_context_state:
        VEO_STATE_UNKNOWN
        VEO_STATE_RUNNING
        VEO_STATE_SYSCALL
        VEO_STATE_BLOCKED
        VEO_STATE_EXIT

    cdef enum veo_command_state:
        VEO_COMMAND_OK
        VEO_COMMAND_EXCEPTION
        VEO_COMMAND_ERROR
        VEO_COMMAND_UNFINISHED

    cdef enum veo_args_intent:
        VEO_INTENT_IN
        VEO_INTENT_INOUT
        VEO_INTENT_OUT

    cdef struct veo_args:
        pass

    cdef struct veo_proc_handle:
        pass

    cdef struct veo_thr_ctxt:
        pass

    # int veo_api_version()
    veo_proc_handle *veo_proc_create(int)
    veo_proc_handle *veo_proc_create_static(int, char *)
    int veo_proc_destroy(veo_proc_handle *)
    int veo_proc_identifier(veo_proc_handle *)
    void *veo_set_proc_identifier(void *addr, int proc_ident)
    uint64_t veo_load_library(veo_proc_handle *, const char *)
    int veo_unload_library(veo_proc_handle *, const uint64_t)
    uint64_t veo_get_sym(veo_proc_handle *, uint64_t, const char *)
    veo_thr_ctxt *veo_context_open(veo_proc_handle *)
    int veo_context_close(veo_thr_ctxt *)
    int veo_get_context_state(veo_thr_ctxt *)
    void veo_context_sync(veo_thr_ctxt *ctx)
    veo_args *veo_args_alloc()
    int veo_args_set_u64(veo_args *, int, uint64_t)
    int veo_args_set_i64(veo_args *, int, int64_t)
    int veo_args_set_u32(veo_args *, int, uint32_t)
    int veo_args_set_i32(veo_args *, int, int32_t)
    int veo_args_set_float(veo_args *, int, float)
    int veo_args_set_double(veo_args *, int, double)
    int veo_args_set_stack(veo_args *, int, int, void *, size_t)
    void veo_args_clear(veo_args *)
    void veo_args_free(veo_args *)
    uint64_t veo_call_async(veo_thr_ctxt *, uint64_t, veo_args *)
    int veo_call_peek_result(veo_thr_ctxt *, uint64_t, uint64_t *)
    int veo_call_wait_result(veo_thr_ctxt *, uint64_t, uint64_t *)
    int veo_alloc_mem(veo_proc_handle *, uint64_t *, size_t)
    int veo_alloc_hmem(veo_proc_handle *, void **, size_t)
    int veo_free_mem(veo_proc_handle *, uint64_t)
    int veo_read_mem(veo_proc_handle *, void *, uint64_t, size_t)
    int veo_write_mem(veo_proc_handle *, uint64_t, void *, size_t)
    int veo_hmemcpy(void *dst, const void *src, size_t size)
    uint64_t veo_async_read_mem(veo_thr_ctxt *, void *, uint64_t, size_t)
    uint64_t veo_async_write_mem(veo_thr_ctxt *, uint64_t, void *, size_t)

    ##############################################
    # VEO HMEM API
    #############################################
    int veo_is_ve_addr(const void *)
    void *veo_get_hmem_addr(void *)
    int veo_get_max_proc_identifier()
    int veo_get_proc_identifier_from_hmem(const void *)

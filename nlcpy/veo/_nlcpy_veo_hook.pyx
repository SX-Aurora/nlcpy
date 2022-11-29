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
from nlcpy import _environment
from nlcpy.veo.libveo cimport *

cdef int (*hooked_veo_alloc_hmem)(veo_proc_handle *, void **, const size_t)
cdef int (*hooked_veo_free_hmem)(void *)


cdef _get_veo_sym():
    global hooked_veo_alloc_hmem, hooked_veo_free_hmem
    cdef void *hdl_veo = NULL
    cdef void *hdl_mpi = NULL
    cdef char *err = NULL

    cdef bint is_mpi = _environment._is_mpi()
    if is_mpi:
        hdl_veo = <void *>dlopen('libmpi_veo.so.1', RTLD_NOW)
    else:
        hdl_veo = NULL
    err = dlerror()

    hooked_veo_alloc_hmem = \
        <int (*)(veo_proc_handle *, void **, const size_t)>dlsym(
            hdl_veo, 'veo_alloc_hmem')
    err = dlerror()
    if err != NULL:
        raise RuntimeError(err)

    hooked_veo_free_hmem = \
        <int (*)(void *)>dlsym(hdl_veo, 'veo_free_hmem')
    err = dlerror()
    if err != NULL:
        raise RuntimeError(err)

cdef int _hooked_alloc_hmem(veo_proc_handle *proc, uint64_t *addr, const size_t size):
    global hooked_veo_alloc_hmem
    if hooked_veo_alloc_hmem == NULL:
        _get_veo_sym()
    cdef void *vemem
    cdef int ret
    ret = hooked_veo_alloc_hmem(proc, &vemem, size)
    addr[0] = <uint64_t>vemem
    return ret

cdef int _hooked_free_hmem(uint64_t addr):
    global hooked_veo_free_hmem
    if hooked_veo_free_hmem == NULL:
        _get_veo_sym()
    cdef int ret
    ret = hooked_veo_free_hmem(<void *>addr)
    return ret

cdef void *_get_hooked_veo_alloc_hmem():
    global hooked_veo_alloc_hmem
    if hooked_veo_alloc_hmem == NULL:
        _get_veo_sym()
    return hooked_veo_alloc_hmem

cdef void *_get_hooked_veo_free_hmem():
    global hooked_veo_free_hmem
    if hooked_veo_free_hmem == NULL:
        _get_veo_sym()
    return hooked_veo_free_hmem

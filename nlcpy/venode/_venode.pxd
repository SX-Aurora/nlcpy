#
# * The source code in this file is based on the soure code of CuPy.
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

from nlcpy.core.core cimport ndarray
from nlcpy.veo cimport _veo
from nlcpy.mempool cimport mempool
from nlcpy.request cimport request
from libcpp cimport bool
from libc.stdint cimport *
cimport cython


@cython.no_gc
cdef class VENodePool:
    cdef:
        readonly list venode_list


@cython.no_gc
cdef class VENode:
    cdef:
        readonly int pid  # physical id
        readonly int lid  # logical id
        readonly int serial_id  # serial id
        readonly int arch  # VE architecture
        readonly object libpath
        readonly bool _phys
        readonly bool _connected
        readonly bool _is_fast_math
        readonly _veo.VeoProc proc
        readonly _veo.VeoLibrary lib
        readonly _veo.VeoLibrary lib_prof
        readonly _veo.VeoCtxt ctx
        readonly mempool.MemPool pool
        readonly request.RequestManager request_manager


cdef bool _is_multive
cpdef _create_venode_pool()
cpdef VENode _find_venode_from_proc_handle(uint64_t key_proc)
cpdef VENode VE(int veid=*)
cpdef transfer_array(ndarray src, VENode target_ve, ndarray dst=*)

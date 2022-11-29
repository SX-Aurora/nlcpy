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

from nlcpy.veo._veo cimport VeoFunction
from nlcpy.core.core cimport ndarray
from nlcpy.venode._venode cimport VENode
from libc.stdint cimport *
from libcpp.vector cimport vector
cimport numpy as cnp


cdef class _ReqNames:
    cdef:
        readonly list _reqnames
        readonly Py_ssize_t _nreq

cdef class VeoReqs:
    cdef:
        readonly list veo_reqs
        readonly list callbacks
        readonly Py_ssize_t cnt
        readonly object fpe
        readonly list reqnamess

cdef class RequestManager:
    cdef:
        readonly cnp.ndarray reqs
        readonly Py_ssize_t nreq
        readonly list refs_queing
        readonly object reqnames
        readonly str timing
        readonly uint64_t head
        readonly uint64_t tail
        readonly uint64_t reqs_ve_ptr
        readonly VENode venode
        readonly VeoReqs veo_reqs

    cdef clear(self)
    cdef increment_head(self, int num)
    cdef increment_tail(self, int num)
    cdef increment_nreq(self)
    cdef flush_if_needed(self)
    cdef _set_request(self, int func_num, int func_type, args)
    cpdef _push_request_core(self, str name, str typ, args)
    cpdef _push_and_flush_request_core(
        self, VeoFunction func, tuple args, callback=*, sync=*)


cpdef _get_request_manager()
cpdef _get_veo_requests()
cpdef _get_fpe_flag(VENode venode=*)
cpdef _push_request(str name, str typ, args)
cpdef _push_and_flush_request(
    str name, tuple args, callback=*, sync=*)
cpdef flush(VENode venode=*, bint sync=*)
cpdef set_max_request(int num, VENode venode=*)

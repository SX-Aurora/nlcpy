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
import cython
import string
import os
import six
import time
import nlcpy

from nlcpy import veo
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport core
from nlcpy.request.ve_kernel cimport *

from libcpp.vector cimport vector
from libc.stdint cimport *

import numpy
cimport numpy as cnp

cdef MAX_REQUEST = 100

cdef _rm = RequestManager()

cdef ve_runtime = numpy.empty(1, dtype='f8')
cdef fpe_flags = numpy.empty(1, dtype='i4')

cdef class RequestManager:
    cdef:
        readonly cnp.ndarray reqs
        readonly Py_ssize_t nreq
        readonly list ref
        readonly str timing
        readonly uint64_t head
        readonly uint64_t tail
        readonly object reqs_ptr

    def __init__(self):
        self.reqs = numpy.zeros(N_REQUEST_PACKAGE * MAX_REQUEST, dtype='uint64')
        self.nreq = 0
        self.ref = []
        self.timing = 'lazy'
        self.head = 0
        self.tail = 0
        v = veo.VeoAlloc()
        self.reqs_ptr = v.proc.alloc_mem(self.reqs.nbytes)

    def __repr__(self):
        print "size = ", self.reqs.size
        return repr(self.reqs)

    def __str__(self):
        print "size = ", self.reqs.size
        return str(self.reqs)

    def flush(self):
        v = veo.VeoAlloc()
        reqs = self.reqs[self.head:self.tail]
        v.proc.write_mem(self.reqs_ptr, reqs, reqs.nbytes)
        args = (
            # veo.OnStack(reqs, inout=veo.INTENT_IN),
            self.reqs_ptr.addr,
            self.nreq,
            veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),
            # veo.OnStack(ve_runtime, inout=veo.INTENT_OUT),
        )
        req = v.lib.func[b"kernel_launcher"](v.ctx, *args)
        err = req.wait_result()
        self.clear()
        check_error(err)
        core.check_fpe_flags(fpe_flags[0])
        fpe_flags.fill(0)
        # print "ve_runtime = ", ve_runtime

    def clear(self):
        self.reqs.fill(0)
        self.nreq = 0
        self.ref.clear()
        self.head = 0
        self.tail = 0

    def set_timing(self, timing='lazy'):
        self.timing = timing

    def increment_tail(self, num):
        self.tail = self.tail + num

    def increment_nreq(self):
        self.nreq = self.nreq + 1

    def flush_if_needed(self):
        if self.timing == 'on-the-fly':
            flush()
        elif self.nreq >= MAX_REQUEST:
            flush()

    def keep_refs(self, refs):
        self.ref.append(refs)

    def __del__(self):
        self.reqs = None
        self.ref.clear()
        v = veo.VeoAlloc()
        v.proc.free_mem(self.reqs_ptr)

cpdef _push_request(name, typ, args):
    func_num = funcNumList.get(name, -1)
    func_type = funcTypeList.get(typ, -1)
    if func_num == -1:
        raise RuntimeError('unknown function number was detected.')
    if func_type == -1:
        raise RuntimeError('unknown function type was detected.')

    _set_request(func_num, func_type, args)

    _rm.increment_nreq()
    _rm.keep_refs(args)  # to avoid deallocation before request is executed.
    _rm.flush_if_needed()


cpdef _clear_requests():
    _rm.clear()

cpdef _print_requests():
    print _rm

cpdef flush():
    """Flushes stacked requests on VH to VE, and waits until VE exectuion is completed.

    """
    if _rm.nreq > 0:
        _rm.flush()
        # _rm.clear()


cpdef set_offload_timing_onthefly():
    """Sets kernel offload timing on-the-fly.

    After calling this function, `nlcpy.request.flush` will be called every time requests
    are stacked on VH.

    Note:
        This function does not flush stacked requests, so we recommend to execute
        `nlcpy.request.flush` before calling this function.

    """
    _rm.set_timing('on-the-fly')


cpdef set_offload_timing_lazy():
    """Sets kernel offload timing lazy.

    After calling this function, requests will be packaged and flush together to VE.

    Note:
        Default offload setting is lazy, so you don't need to call this function
        normally.

    """
    _rm.set_timing('lazy')


cpdef get_offload_timing():
    """Gets kernel offload timing.

    Returns:
        out : str

    Examples:
        >>> import nlcpy as vp
        >>> vp.request.get_offload_timing()
        "current offload timing is 'lazy'"
        >>> vp.request.set_offload_timing_onthefly()
        >>> vp.request.get_offload_timing()
        "current offload timing is 'on-the-fly'"

    """
    return str("current offload timing is \'" + _rm.timing + "\'")


cdef _set_request(func_num, func_type, args):
    cdef int ini_tail = _rm.tail
    cdef int n, diff_tail
    _rm.reqs[_rm.tail:_rm.tail+2] = (func_num, func_type)
    _rm.increment_tail(2)
    for x in args:
        if isinstance(x, ndarray):
            n = _set_ve_array_without_scalar(x, _rm.reqs, _rm.tail)
            _rm.increment_tail(n)
        elif isinstance(x, numpy.number) or \
            isinstance(x, numpy.bool_) or \
                isinstance(x, numpy.bool):
            n = _set_ve_array_with_scalar(numpy.array(x), _rm.reqs, _rm.tail)
            _rm.increment_tail(n)
        elif isinstance(x, numpy.ndarray) and x.ndim == 0:
            n = _set_ve_array_with_scalar(numpy.array(x), _rm.reqs, _rm.tail)
            _rm.increment_tail(n)
        elif type(x) in (int, float):
            n = _set_scalar(numpy.array(x), _rm.reqs, _rm.tail)
            _rm.increment_tail(n)
        else:
            raise RuntimeError(
                'only nlcpy.ndarray or numpy scalar or Python '
                'scalar(not complex) can be accepted.')
    diff_tail = _rm.tail - ini_tail
    if <uint64_t>diff_tail < N_REQUEST_PACKAGE:
        _rm.increment_tail(N_REQUEST_PACKAGE - diff_tail)


cdef int _set_ve_array_without_scalar(ndarray a, cnp.ndarray dst, int offset):
    dst[offset + VE_ADR_OFFSET] = a.ve_adr
    dst[offset + NDIM_OFFSET] = a.ndim
    dst[offset + SIZE_OFFSET] = a.size
    dst[offset + DTYPE_OFFSET] = a.dtype.num
    dst[offset + ITEMSIZE_OFFSET] = a.itemsize
    dst[offset + C_CONTIGUOUS_OFFSET] = a._c_contiguous
    dst[offset + F_CONTIGUOUS_OFFSET] = a._f_contiguous
    cdef int i
    for i in range(a.ndim):
        dst[offset + SHAPE_OFFSET + i] = a._shape[i]
        dst[offset + STRIDES_OFFSET + i] = a._strides[i]
    return N_VE_ARRAY_ELEMENTS

cdef int _set_ve_array_with_scalar(cnp.ndarray a, cnp.ndarray dst, int offset):
    dst[offset + VE_ADR_OFFSET] = 0L
    dst[offset + NDIM_OFFSET] = 0L
    dst[offset + SIZE_OFFSET] = 1
    dst[offset + DTYPE_OFFSET] = a.dtype.num
    dst[offset + ITEMSIZE_OFFSET] = a.itemsize
    dst[offset + C_CONTIGUOUS_OFFSET] = 1L
    dst[offset + F_CONTIGUOUS_OFFSET] = 1L
    cdef int i
    for i in range(a.ndim):
        dst[offset + SHAPE_OFFSET + i] = 0L
        dst[offset + STRIDES_OFFSET + i] = 0L
    _set_scalar(a, dst, offset+SCALAR_OFFSET)
    return N_VE_ARRAY_ELEMENTS

cdef int _set_scalar(cnp.ndarray val, cnp.ndarray dst, int offset):
    # cast scalar values to 64bit data
    if val.dtype == numpy.dtype('bool'):
        val = val.astype('i8')
    elif val.dtype == numpy.dtype('i4'):
        val = val.astype('i8')
    elif val.dtype == numpy.dtype('u4'):
        val = val.astype('u8')
    elif val.dtype == numpy.dtype('f4'):
        val = val.astype('f8')

    cdef uint64_t *s_tmp = <uint64_t *>val.data
    cdef int n
    if val.dtype == numpy.dtype('c16'):
        dst[offset] = s_tmp[0]
        dst[offset + 1] = s_tmp[1]
        n = 2
    else:
        dst[offset] = s_tmp[0]
        n = 1
    return n


cdef cnp.ndarray[cnp.uint64_t, ndim=1] _create_ve_array_buffer(ndarray a):
    cdef vector[uint64_t] tmp
    tmp.resize(N_VE_ARRAY_ELEMENTS)
    tmp[VE_ADR_OFFSET] = a.ve_adr
    tmp[NDIM_OFFSET] = a.ndim
    tmp[SIZE_OFFSET] = a.size
    tmp[DTYPE_OFFSET] = a.dtype.num
    tmp[ITEMSIZE_OFFSET] = a.itemsize
    tmp[C_CONTIGUOUS_OFFSET] = a._c_contiguous
    tmp[F_CONTIGUOUS_OFFSET] = a._f_contiguous
    cdef int i
    for i in range(a.ndim):
        tmp[SHAPE_OFFSET + i] = a._shape[i]
        tmp[STRIDES_OFFSET + i] = a._strides[i]
    ve_array_buf = numpy.array(tmp, dtype='uint64')
    return ve_array_buf

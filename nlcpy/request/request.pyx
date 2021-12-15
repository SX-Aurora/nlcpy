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

# distutils: language = c++

import cython
import string
import os
import time
import nlcpy

from nlcpy import veo
from nlcpy.veo._veo cimport VeoFunction
from nlcpy.veo._veo cimport VeoRequest
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport core
from nlcpy.request.ve_kernel cimport *
from nlcpy.request.ve_kernel import check_error
from nlcpy.core cimport dtype as _dtype

from libcpp.vector cimport vector
from libc.stdint cimport *

import numpy
cimport numpy as cnp

cdef MAX_LAZY_REQUEST = 100
cdef MAX_ASYNC_REQUEST = 1000
cdef object _request_manager = None
cdef object _veo_requests = None
cdef object _fpe_flag = None


# TODO: Thread lock
cpdef _get_request_manager():
    global _request_manager
    if _request_manager is None:
        _request_manager = RequestManager()
    return _request_manager

cpdef _get_veo_requests():
    global _veo_requests
    if _veo_requests is None:
        _veo_requests = VeoReqs()
    return _veo_requests


cpdef _get_fpe_flag():
    global _fpe_flag
    if _fpe_flag is None:
        _fpe_flag = numpy.zeros(1, dtype=numpy.int32)
    return _fpe_flag


def _nothing(*arg):
    return


cdef class VeoReqs:
    cdef:
        readonly list veo_reqs
        readonly list callbacks
        readonly Py_ssize_t cnt

    def __init__(self):
        self.veo_reqs = [None for _ in range(MAX_ASYNC_REQUEST)]
        self.callbacks = [None for _ in range(MAX_ASYNC_REQUEST)]
        self.cnt = 0

    def _push_req(self, VeoRequest req, callback=None):
        self.veo_reqs[self.cnt] = req
        if callback is None:
            callback = check_error
        elif callback == 'nothing':
            callback = _nothing
        if not callable(callback):
            raise RuntimeError
        self.callbacks[self.cnt] = callback
        self.cnt += 1
        if self.cnt >= MAX_ASYNC_REQUEST:
            self._wait_result_all()

    def _wait_result_all(self):
        err = None
        try:
            for i in range(self.cnt):
                err = self.veo_reqs[i].wait_result()
                self.callbacks[i](err)
        finally:
            self.cnt = 0
        core.check_fpe_flags(_get_fpe_flag())
        return err


cdef class RequestManager:
    cdef:
        readonly cnp.ndarray reqs
        readonly Py_ssize_t nreq
        readonly list refs_queing
        readonly str timing
        readonly uint64_t head
        readonly uint64_t tail
        readonly object reqs_ve_ptr

    def __init__(self):
        self.reqs = numpy.zeros(N_REQUEST_PACKAGE * MAX_LAZY_REQUEST, dtype='uint64')
        self.nreq = 0
        self.refs_queing = []
        self.timing = 'lazy'
        self.head = 0
        self.tail = 0
        proc = veo._get_veo_proc()
        self.reqs_ve_ptr = proc.alloc_mem(self.reqs.nbytes)

    def __repr__(self):
        return repr(self.reqs)

    def __str__(self):
        return str(self.reqs)

    def _update_reqs(self):
        flush()
        self.reqs = numpy.zeros(N_REQUEST_PACKAGE * MAX_LAZY_REQUEST, dtype='uint64')
        proc = veo._get_veo_proc()
        proc.free_mem(self.reqs_ve_ptr)
        self.reqs_ve_ptr = proc.alloc_mem(self.reqs.nbytes)

    def _flush(self, sync=False):
        fpe = _get_fpe_flag()
        ctx = veo._get_veo_ctx()
        lib = veo._get_veo_lib()
        vr = _get_veo_requests()
        reqs = self.reqs[self.head:self.tail].copy()
        if self.timing == 'on-the-fly':
            proc = veo._get_veo_proc()
            proc.write_mem(self.reqs_ve_ptr, reqs, reqs.nbytes)
        else:
            wreq = ctx.async_write_mem(self.reqs_ve_ptr, reqs, reqs.nbytes)
            vr._push_req(wreq, callback='nothing')
        args = (
            self.reqs_ve_ptr.addr,
            self.nreq,
            veo.OnStack(fpe, inout=veo.INTENT_OUT),
        )
        req = lib.func[b"kernel_launcher"](ctx, *args)
        vr._push_req(req)
        self.clear()
        if sync:
            vr._wait_result_all()

    def clear(self):
        self.reqs.fill(0)
        self.nreq = 0
        self.refs_queing.clear()
        self.head = 0
        self.tail = 0

    def set_timing(self, timing='lazy'):
        self.timing = timing

    def increment_head(self, int num):
        self.head = self.head + num

    def increment_tail(self, int num):
        self.tail = self.tail + num

    def increment_nreq(self):
        self.nreq = self.nreq + 1

    def flush_if_needed(self):
        if self.timing == 'on-the-fly':
            self._flush(sync=True)
        elif self.nreq >= MAX_LAZY_REQUEST:
            self._flush(sync=False)
            # self._flush(sync=True)

    def keep_refs(self, refs):
        self.refs_queing.append(refs)


# for lazy
cpdef _push_request(str name, str typ, args):
    func_num = funcNumList.get(name, -1)
    func_type = funcTypeList.get(typ, -1)
    if func_num == -1:
        raise RuntimeError('unknown function number was detected.')
    if func_type == -1:
        raise RuntimeError('unknown function type was detected.')

    _set_request(func_num, func_type, args)

    _rm = _get_request_manager()
    _rm.increment_nreq()
    _rm.keep_refs(args)  # to avoid deallocation before request is executed.

    _rm.flush_if_needed()


# for not lazy
cpdef _push_and_flush_request(str name, tuple args, callback='default', sync=False):
    _rm = _get_request_manager()
    if _rm.nreq > 0:
        _rm._flush(sync=False)
    lib = veo._get_veo_lib()
    ctx = veo._get_veo_ctx()
    req = lib.func[name.encode('utf-8')](ctx, *args)
    if callback == 'default':
        callback = check_error
    elif callback is None:
        callback = _nothing
    vr = _get_veo_requests()
    vr._push_req(req, callback=callback)
    if sync or _rm.timing == 'on-the-fly':
        return vr._wait_result_all()

# for not lazy and JIT
cpdef _push_and_flush_request_with_JIT(
        VeoFunction func, tuple args, callback='default', sync=False):
    _rm = _get_request_manager()
    if _rm.nreq > 0:
        _rm._flush(sync=False)
    ctx = veo._get_veo_ctx()
    req = func(ctx, *args)
    if callback == 'default':
        callback = check_error
    elif callback is None:
        callback = _nothing
    vr = _get_veo_requests()
    vr._push_req(req, callback=callback)
    if sync or _rm.timing == 'on-the-fly':
        return vr._wait_result_all()


cpdef _clear_requests():
    _rm = _get_request_manager()
    _rm.clear()

cpdef _print_requests():
    _rm = _get_request_manager()
    print(_rm)

cpdef flush():
    """Flushes stacked requests on VH to VE, and waits until VE exectuion is completed.

    """
    _rm = _get_request_manager()
    if _rm.nreq > 0:
        _rm._flush(sync=True)
    else:
        vr = _get_veo_requests()
        vr._wait_result_all()

cpdef set_max_request(int num):
    global MAX_LAZY_REQUEST
    MAX_LAZY_REQUEST = num
    _rm = _get_request_manager()
    _rm._update_reqs()

cpdef set_offload_timing_onthefly():
    """Sets kernel offload timing on-the-fly.

    After calling this function, :func:`nlcpy.request.flush` will be called
    every time requests are stacked on VH.

    Note
    ----
    This function does not flush stacked requests, so we recommend to execute
    :func:`nlcpy.request.flush` before calling this function.
    """
    _rm = _get_request_manager()
    _rm.set_timing('on-the-fly')


cpdef set_offload_timing_lazy():
    """Sets kernel offload timing lazy.

    After calling this function, requests will be packaged and flush together to VE.

    Note
    ----
    Default offload setting is 'lazy', so you don't need to call this function normally.
    """
    _rm = _get_request_manager()
    _rm.set_timing('lazy')


cpdef get_offload_timing():
    """Gets kernel offload timing.

    Returns
    -------
    out : str

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.request.get_offload_timing()
    "current offload timing is 'lazy'"
    >>> vp.request.set_offload_timing_onthefly()
    >>> vp.request.get_offload_timing()
    "current offload timing is 'on-the-fly'"
    """
    _rm = _get_request_manager()
    return str("current offload timing is \'" + _rm.timing + "\'")


cdef _set_request(func_num, func_type, args):
    _rm = _get_request_manager()
    cdef int ini_tail = _rm.tail
    cdef int n
    cdef int diff_tail = 0
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] _reqs = _rm.reqs
    # _rm.reqs[_rm.tail:_rm.tail+2] = (func_num, func_type)
    _reqs[_rm.tail] = func_num
    _reqs[_rm.tail + 1] = func_type
    _rm.increment_tail(2)
    for x in args:
        if isinstance(x, ndarray):
            n = _set_ve_array_without_scalar(x, _reqs, _rm.tail)
            _rm.increment_tail(n)
        elif isinstance(x, (numpy.number, numpy.bool_)):
            n = _set_ve_array_with_scalar(numpy.array(x), _reqs, _rm.tail)
            _rm.increment_tail(n)
        elif isinstance(x, numpy.ndarray) and x.ndim == 0:
            n = _set_ve_array_with_scalar(numpy.array(x), _reqs, _rm.tail)
            _rm.increment_tail(n)
        elif isinstance(x, (int, float, bool)):
            n = _set_scalar(numpy.array(x), _reqs, _rm.tail)
            _rm.increment_tail(n)
        else:
            raise RuntimeError(
                'only nlcpy.ndarray or numpy scalar or Python '
                'scalar(not complex) can be accepted.')
    diff_tail = <int64_t>N_REQUEST_PACKAGE - (_rm.tail - ini_tail)
    if diff_tail > 0:
        _rm.increment_tail(diff_tail)
    # _rm.increment_tail(N_REQUEST_PACKAGE)

cdef int _set_ve_array_without_scalar(
        ndarray a, cnp.ndarray[cnp.uint64_t, ndim=1] dst, int offset):
    # cdef uint64_t[:] c_dst = dst
    dst[offset + VE_ADR_OFFSET] = a.ve_adr
    dst[offset + NDIM_OFFSET] = a.ndim
    dst[offset + SIZE_OFFSET] = a.size
    cdef int i
    # cdef vector[Py_ssize_t] _shape = a._shape
    # cdef vector[Py_ssize_t] _strides = a._strides
    for i in range(a.ndim):
        dst[offset + SHAPE_OFFSET + i] = a._shape[i]
        dst[offset + STRIDES_OFFSET + i] = a._strides[i]
    dst[offset + DTYPE_OFFSET] = a.dtype.num
    dst[offset + ITEMSIZE_OFFSET] = a.itemsize
    dst[offset + C_CONTIGUOUS_OFFSET] = a._c_contiguous
    dst[offset + F_CONTIGUOUS_OFFSET] = a._f_contiguous
    return N_VE_ARRAY_ELEMENTS

cdef int _set_ve_array_with_scalar(
        cnp.ndarray a, cnp.ndarray[cnp.uint64_t, ndim=1] dst, int offset):
    # cdef uint64_t[:] c_dst = dst
    dst[offset + VE_ADR_OFFSET] = 0L
    dst[offset + NDIM_OFFSET] = 0L
    dst[offset + SIZE_OFFSET] = 1
    cdef int i
    for i in range(a.ndim):
        dst[offset + SHAPE_OFFSET + i] = 0L
        dst[offset + STRIDES_OFFSET + i] = 0L
    dst[offset + DTYPE_OFFSET] = a.dtype.num
    dst[offset + ITEMSIZE_OFFSET] = a.itemsize
    dst[offset + C_CONTIGUOUS_OFFSET] = 1L
    dst[offset + F_CONTIGUOUS_OFFSET] = 1L
    _set_scalar(a, dst, offset+SCALAR_OFFSET)
    return N_VE_ARRAY_ELEMENTS

cdef int _set_scalar(
        cnp.ndarray val, cnp.ndarray[cnp.uint64_t, ndim=1] dst, int offset):
    # cdef uint64_t[:] c_dst = dst
    # cast bool to int32
    cdef int n = 1

    if val.dtype == _dtype.DT_BOOL:
        val = val.astype('i4')

    cdef uint64_t *s_tmp = <uint64_t *>val.data

    if val.dtype in (_dtype.DT_U64, _dtype.DT_I64,
                     _dtype.DT_F64, _dtype.DT_C64):
        dst[offset] = s_tmp[0]
        return 1

    # fill zero for upper bits
    if val.dtype in (_dtype.DT_I32, _dtype.DT_U32, _dtype.DT_F32):
        s_tmp[0] = 0x00000000ffffffff & s_tmp[0]

    if val.dtype is _dtype.DT_C128:
        dst[offset] = s_tmp[0]
        dst[offset + 1] = s_tmp[1]
        n = 2
    else:
        dst[offset] = s_tmp[0]
        n = 1
    return n

cpdef vector[uint64_t] _create_ve_array_buffer(ndarray a):
    cdef vector[uint64_t] buf
    buf.resize(N_VE_ARRAY_ELEMENTS)
    buf[VE_ADR_OFFSET] = a.ve_adr
    buf[NDIM_OFFSET] = a.ndim
    buf[SIZE_OFFSET] = a.size
    buf[DTYPE_OFFSET] = a.dtype.num
    buf[ITEMSIZE_OFFSET] = a.itemsize
    buf[C_CONTIGUOUS_OFFSET] = a._c_contiguous
    buf[F_CONTIGUOUS_OFFSET] = a._f_contiguous
    cdef int i
    # cdef vector[Py_ssize_t] _shape = a._shape
    # cdef vector[Py_ssize_t] _strides = a._strides
    for i in range(a.ndim):
        buf[SHAPE_OFFSET + i] = a._shape[i]
        buf[STRIDES_OFFSET + i] = a._strides[i]
    return buf

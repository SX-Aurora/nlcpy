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
import time
import nlcpy

from nlcpy.veo._veo cimport OnStack
from nlcpy.veo._veo cimport INTENT_IN
from nlcpy.veo._veo cimport INTENT_INOUT
from nlcpy.veo._veo cimport INTENT_OUT
from nlcpy.veo._veo cimport VeoFunction
from nlcpy.veo._veo cimport VeoRequest
from nlcpy.venode._venode cimport VENode
from nlcpy.venode._venode cimport VE
from nlcpy.venode._venode cimport transfer_array
from nlcpy.venode._venode cimport _is_multive
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport core
from nlcpy.request.ve_kernel cimport *
from nlcpy.request.ve_kernel import check_error
from nlcpy.core cimport dtype as _dtype
from nlcpy.logging import _vp_logging

from libcpp.vector cimport vector
from libc.stdint cimport *

import numpy
cimport numpy as cnp

cdef MAX_LAZY_REQUEST = 100
cdef MAX_ASYNC_REQUEST = 1000
cdef LAZY = 'lazy'
cdef ON_THE_FLY = 'on-the-fly'

cpdef _get_request_manager():
    return VE().request_manager

cpdef _get_veo_requests():
    return VE().request_manager.veo_reqs


cpdef _get_fpe_flag(VENode venode=None):
    if venode is None:
        venode = VE()
    if not venode.connected:
        venode.connect()
    return venode.request_manager.veo_reqs._get_fpe_flag()


def _nothing(*arg):
    return


cdef class _ReqNames:
    def __init__(self, n):
        self._reqnames = [None for _ in range(n)]
        self._nreq = 0

    def _set(self, n, name):
        self._reqnames[n] = name
        self._nreq = n + 1

    def _uniq_str(self):
        return ', '.join(set(self._reqnames[:self._nreq]))

    def __str__(self):
        return ', '.join(self._reqnames[:self._nreq])


cdef class VeoReqs:

    def __init__(self):
        self.veo_reqs = [None for _ in range(MAX_ASYNC_REQUEST)]
        self.callbacks = [None for _ in range(MAX_ASYNC_REQUEST)]
        self.cnt = 0
        self.fpe = numpy.zeros((MAX_ASYNC_REQUEST, 1), dtype=numpy.int32)
        self.reqnamess = [None for _ in range(MAX_ASYNC_REQUEST)]

    def _push_req(self, VeoRequest req, _ReqNames reqnames=None, callback=None):
        self.veo_reqs[self.cnt] = req
        if callback is None:
            callback = check_error
        elif callback == 'nothing':
            callback = _nothing
        if reqnames is not None:
            self.reqnamess[self.cnt] = reqnames
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
                if self.reqnamess[i] is not None:
                    core.check_fpe_flags(self.fpe[i], self.reqnamess[i])
        finally:
            self.fpe[:self.cnt, :] = 0
            for i in range(self.cnt):
                self.reqnamess[i] = None
            self.cnt = 0
        return err

    def _get_fpe_flag(self):
        return self.fpe[self.cnt]

    def __repr__(self):
        return '<VeoReqs cnt={}>'.format(self.cnt)


cdef class RequestManager:

    def __init__(self, VENode venode, VeoReqs veo_reqs):
        self.reqs = numpy.zeros(N_REQUEST_PACKAGE * MAX_LAZY_REQUEST, dtype='uint64')
        self.nreq = 0
        self.refs_queing = []
        self.reqnames = _ReqNames(MAX_LAZY_REQUEST)
        self.timing = LAZY
        self.head = 0
        self.tail = 0
        self.venode = venode
        self.veo_reqs = veo_reqs
        self.reqs_ve_ptr = self.venode.proc.alloc_mem(self.reqs.nbytes)

    def __repr__(self):
        return repr(self.reqs)

    def __str__(self):
        return str(self.reqs)

    def _update_reqs(self):
        self._flush(sync=True)
        self.reqs = numpy.zeros(N_REQUEST_PACKAGE * MAX_LAZY_REQUEST, dtype='uint64')
        proc = self.venode.proc
        proc.free_mem(self.reqs_ve_ptr)
        self.reqs_ve_ptr = proc.alloc_mem(self.reqs.nbytes)
        self.reqnames = _ReqNames(MAX_LAZY_REQUEST)

    def _flush(self, sync=False):
        if self.nreq > 0:
            ctx = self.venode.ctx
            lib = self.venode.lib
            reqs = self.reqs[self.head:self.tail].copy()
            if self.timing == ON_THE_FLY:
                if _vp_logging._is_enable(_vp_logging.REQUEST):
                    _vp_logging.info(
                        _vp_logging.REQUEST,
                        'veo_write_mem to send VE arguments (nodeid=%d)',
                        self.venode.lid)
                proc = self.venode.proc
                proc.write_mem(self.reqs_ve_ptr, reqs, reqs.nbytes)
            else:
                if _vp_logging._is_enable(_vp_logging.REQUEST):
                    _vp_logging.info(
                        _vp_logging.REQUEST,
                        'veo_async_write_mem to send VE arguments (nodeid=%d)',
                        self.venode.lid)
                wreq = ctx.async_write_mem(self.reqs_ve_ptr, reqs, reqs.nbytes)
                self.veo_reqs._push_req(wreq, reqnames=None, callback='nothing')
            args = (
                self.reqs_ve_ptr,
                self.nreq,
                OnStack(self.veo_reqs._get_fpe_flag(), inout=INTENT_OUT),
            )
            if _vp_logging._is_enable(_vp_logging.REQUEST):
                _vp_logging.info(
                    _vp_logging.REQUEST,
                    'veo_call_async to flush stacked requests (nodeid=%d): '
                    'requests <%s>',
                    self.venode.lid, self.reqnames
                )
            req = lib.func[b"kernel_launcher"](ctx, *args)
            self.veo_reqs._push_req(req, reqnames=self.reqnames)
            self.clear()
        if sync or self.timing == ON_THE_FLY:
            self.veo_reqs._wait_result_all()

    cdef clear(self):
        self.reqs[self.head:self.tail] = 0
        self.nreq = 0
        self.refs_queing.clear()
        self.head = 0
        self.tail = 0
        self.reqnames = _ReqNames(MAX_LAZY_REQUEST)

    def set_timing(self, timing=LAZY):
        self.timing = timing

    cdef increment_head(self, int num):
        self.head = self.head + num

    cdef increment_tail(self, int num):
        self.tail = self.tail + num

    cdef increment_nreq(self):
        self.nreq = self.nreq + 1

    cdef flush_if_needed(self):
        if self.timing == ON_THE_FLY:
            self._flush(sync=True)
        elif self.nreq >= MAX_LAZY_REQUEST:
            self._flush(sync=False)

    def keep_refs(self, refs):
        self.refs_queing.append(refs)

    def record_reqname(self, name):
        self.reqnames._set(self.nreq, name)

    cdef _set_request(self, int func_num, int func_type, args):
        cdef VENode venode = self.venode
        cdef RequestManager _rm = self
        cdef int _rm_tail = self.tail
        cdef int ini_tail = self.tail
        cdef int n
        cdef int diff_tail = 0
        cdef cnp.ndarray[cnp.uint64_t, ndim=1] _reqs = self.reqs
        _reqs[_rm_tail] = func_num
        _reqs[_rm_tail + 1] = func_type
        _rm_tail += 2

        for x in args:
            if isinstance(x, ndarray):
                n = _set_ve_array_without_scalar(x, _reqs, _rm_tail)
                _rm_tail += n
            elif isinstance(x, (numpy.number, numpy.bool_)):
                n = _set_ve_array_with_scalar(numpy.array(x), _reqs, _rm_tail)
                _rm_tail += n
            elif isinstance(x, numpy.ndarray) and x.ndim == 0:
                n = _set_ve_array_with_scalar(numpy.array(x), _reqs, _rm_tail)
                _rm_tail += n
            elif isinstance(x, (int, float, bool)):
                n = _set_scalar(numpy.array(x), _reqs, _rm_tail)
                _rm_tail += n
            else:
                raise RuntimeError(
                    'only nlcpy.ndarray or numpy scalar or Python '
                    'scalar(not complex) can be accepted.')
        diff_tail = <int64_t>N_REQUEST_PACKAGE - (_rm_tail - ini_tail)
        if <uint64_t>diff_tail > <uint64_t>N_REQUEST_PACKAGE:
            raise RuntimeError
        self.increment_tail(N_REQUEST_PACKAGE)

    cpdef _push_request_core(self, str name, str typ, args):
        cdef int func_num = funcNumList.get(name, -1)
        cdef int func_type = funcTypeList.get(typ, -1)
        if func_num == -1:
            raise RuntimeError('unknown function number was detected.')
        if func_type == -1:
            raise RuntimeError('unknown function type was detected.')

        if _is_multive:
            _check_ndarray_on_venode(args, self.venode)

        if _vp_logging._is_enable(_vp_logging.REQUEST):
            _vp_logging.info(
                _vp_logging.REQUEST,
                'push VE request `%s` (nodeid=%d) ',
                name, self.venode.lid)

        self._set_request(func_num, func_type, args)
        self.record_reqname(name)
        self.increment_nreq()
        self.keep_refs(args)  # to avoid deallocation before request is executed.
        self.flush_if_needed()

    # for lazy
    def _push_request(self, str name, str typ, args):
        self._push_request_core(name, typ, args)

    cpdef _push_and_flush_request_core(
            self, VeoFunction func, tuple args, callback='default', sync=False):
        venode = self.venode
        if _is_multive:
            _check_ndarray_on_venode(args, venode)
        if self.nreq > 0:
            self._flush(sync=False)

        if _vp_logging._is_enable(_vp_logging.REQUEST):
            _vp_logging.info(
                _vp_logging.REQUEST,
                'push and flush VE request `%s` (nodeid=%d)', func.name, self.venode.lid)

        self.record_reqname(str(func.name))
        req = func(venode.ctx, *args)
        if callback == 'default':
            callback = check_error
        elif callback is None:
            callback = _nothing
        vr = self.veo_reqs
        vr._push_req(req, reqnames=self.reqnames, callback=callback)
        if sync or self.timing == ON_THE_FLY:
            return vr._wait_result_all()

    # for not lazy
    def _push_and_flush_request(self, str name, tuple args,
                                callback='default', sync=False):
        func = self.venode.lib.func[name.encode('utf-8')]
        return self._push_and_flush_request_core(
            func, args, callback=callback, sync=sync)

    @staticmethod
    def _create_ve_array_buffer(ndarray a):
        cdef cnp.ndarray[uint64_t, ndim=1] buf = cnp.ndarray(
            N_VE_ARRAY_ELEMENTS, dtype='u8')
        buf[VE_ADR_OFFSET] = a.ve_adr
        buf[NDIM_OFFSET] = a.ndim
        buf[SIZE_OFFSET] = a.size
        buf[DTYPE_OFFSET] = a.dtype.num
        buf[ITEMSIZE_OFFSET] = a.itemsize
        buf[C_CONTIGUOUS_OFFSET] = a._c_contiguous
        buf[F_CONTIGUOUS_OFFSET] = a._f_contiguous
        cdef int i
        for i in range(a.ndim):
            buf[SHAPE_OFFSET + i] = a._shape[i]
            buf[STRIDES_OFFSET + i] = a._strides[i]
        return buf


cdef _check_ndarray_on_venode(args, VENode venode):
    for a in args:
        if isinstance(a, ndarray):
            if a.venode != venode:
                # a_new = transfer_array(a, venode)
                # new_args.append(a_new)
                # continue
                raise ValueError('ndarray(id={}) does not exist on {}'
                                 .format(id(a), venode))

# entry routine to access instance method
cpdef _push_request(str name, str typ, args):
    VE().request_manager._push_request_core(name, typ, args)

# entry routine to access instance method
cpdef _push_and_flush_request(str name, tuple args, callback='default', sync=False):
    return VE().request_manager._push_and_flush_request(
        name, args, callback=callback, sync=sync)

cpdef flush(VENode venode=None, bint sync=True):
    """Flushes stacked requests on VH to specified VE.

    Parameters
    ----------
    venode : VENode
        Targe VE node. If set to 'None', the current active VE node is set.
    sync : bool
        Whether synchronize completion of VE execution.
        Defaults to ``True``.

    Examples
    --------
    >>> import nlcpy as vp
    >>> ve0 = vp.venode.VE(0)
    >>> # First, none request on VH and VE
    >>> ve0.status['stacked_request_on_VH'], ve0.status['running_request_on_VE']
    (0, 0)
    >>> # Execute some function
    >>> _ = vp.arange(10)
    >>> # A request is stacked on VH
    >>> ve0.status['stacked_request_on_VH'], ve0.status['running_request_on_VE']
    (1, 0)
    >>> # Flush requests into VE#0 without synchronize
    >>> vp.request.flush(venode=vp.venode.VE(0), sync=False)
    >>> # Requests are running on VE
    >>> # (write_mem request for function arguments and arange kernel request)
    >>> ve0.status['stacked_request_on_VH'], ve0.status['running_request_on_VE']
    (0, 2)
    >>> # Flush requests into VE#0 with synchronize
    >>> vp.request.flush(venode=vp.venode.VE(0), sync=True)
    >>> # None request on VH and VE
    >>> ve0.status['stacked_request_on_VH'], ve0.status['running_request_on_VE']
    (0, 0)

    """
    if venode is None:
        venode = VE()
    if not venode.connected:
        venode.connect()
    _rm = venode.request_manager
    _rm._flush(sync=sync)

cpdef set_max_request(int num, VENode venode=None):
    global MAX_LAZY_REQUEST
    MAX_LAZY_REQUEST = num
    if venode is None:
        venode = VE()
    if not venode.connected:
        venode.connect()
    _rm = venode.request_manager
    _rm.flush(sync=True)
    _rm._update_reqs()

cpdef set_offload_timing_onthefly(VENode venode=None):
    """Sets kernel offload timing on-the-fly.

    After calling this function, :func:`nlcpy.request.flush` will be called
    every time requests are stacked on VH.

    Parameters
    ----------
    venode : VENode
        Targe VE node. If set to 'None', the current active VE node is set.

    Note
    ----
    This function does not flush stacked requests, so we recommend to execute
    :func:`nlcpy.request.flush` before calling this function.
    """
    if venode is None:
        venode = VE()
    if not venode.connected:
        venode.connect()
    venode.request_manager.set_timing(ON_THE_FLY)


cpdef set_offload_timing_lazy(VENode venode=None):
    """Sets kernel offload timing lazy.

    After calling this function, requests will be packaged and flush together to VE.

    Parameters
    ----------
    venode : VENode
        Targe VE node. If set to 'None', the current active VE node is set.

    Note
    ----
    Default offload setting is 'lazy', so you don't need to call this function normally.
    """
    if venode is None:
        venode = VE()
    if not venode.connected:
        venode.connect()
    venode.request_manager.set_timing(LAZY)


cpdef get_offload_timing(VENode venode=None):
    """Gets kernel offload timing.

    Parameters
    ----------
    venode : VENode
        Targe VE node. If set to 'None', the current active VE node is set.

    Returns
    -------
    out : str

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.request.get_offload_timing()
    'lazy'
    >>> vp.request.set_offload_timing_onthefly()
    >>> vp.request.get_offload_timing()
    'on-the-fly'
    >>> # Note that offload timing for VE#1 is not updated
    >>> vp.request.get_offload_timing(venode=vp.venode.VE(1))
    'lazy'
    """
    if venode is None:
        venode = VE()
    if not venode.connected:
        venode.connect()
    _rm = venode.request_manager
    return _rm.timing


cdef int _set_ve_array_without_scalar(
        ndarray a, cnp.ndarray[cnp.uint64_t, ndim=1] dst, int offset):
    dst[offset + VE_ADR_OFFSET] = a.ve_adr
    dst[offset + NDIM_OFFSET] = a.ndim
    dst[offset + SIZE_OFFSET] = a.size
    cdef int i
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
    cdef int n = 1
    # cast bool to int32
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

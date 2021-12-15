#
# * The source code in this file is based on the soure code of PyVEO.
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
from nlcpy.veo.libveo cimport *

#
# EF: commented out veo_api_version until it finds its way into the VEO mainline.
#
# _veo_api_version = veo_api_version()
# if _veo_api_version < 3:
#    raise ImportError("VEO API Version must be at least 3! The system uses version %d."
#                      % _veo_api_version)

import os
import numbers
import atexit
import sys
import nlcpy
from nlcpy.kernel_register import ve_kernel_register as register
from nlcpy.mempool.mempool cimport *
from nlcpy.prof import prof
from nlcpy import _path
from cpython.buffer cimport \
    PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS, Py_buffer, PyObject_GetBuffer, \
    PyObject_CheckBuffer, PyBuffer_Release
import numpy as np
cimport numpy as np

include "conv_i64.pxi"

cdef _proc_init_hook
_proc_init_hook = list()

#
# Re-declaring the enums here because they're otherwise not visible in Python
#
cpdef enum _veo_args_intent:
    INTENT_IN = VEO_INTENT_IN
    INTENT_INOUT = VEO_INTENT_INOUT
    INTENT_OUT = VEO_INTENT_OUT

cpdef enum _veo_context_state:
    STATE_UNKNOWN = VEO_STATE_UNKNOWN
    STATE_RUNNING = VEO_STATE_RUNNING
    STATE_SYSCALL = VEO_STATE_SYSCALL
    STATE_BLOCKED = VEO_STATE_BLOCKED
    STATE_EXIT = VEO_STATE_EXIT

cpdef enum _veo_command_state:
    COMMAND_OK = VEO_COMMAND_OK
    COMMAND_EXCEPTION = VEO_COMMAND_EXCEPTION
    COMMAND_ERROR = VEO_COMMAND_ERROR
    COMMAND_UNFINISHED = VEO_COMMAND_UNFINISHED

cpdef set_proc_init_hook(v):
    """
    Hook for a function that should be called as last in the
    initialization of a VeoProc.

    Usefull eg. for loading functions from a statically linked library.
    """
    global _proc_init_hook
    _proc_init_hook.append(v)

cpdef del_proc_init_hook(v):
    """
    Delete hook for a function that should be called as last in the
    initialization of a VeoProc.

    Usefull eg. for loading functions from a statically linked library.
    """
    global _proc_init_hook
    _proc_init_hook.remove(v)


cdef union U64:
    uint64_t u64
    int64_t i64
    uint32_t u32[2]
    int32_t i32[2]
    uint16_t u16[4]
    int16_t i16[4]
    uint8_t u8[8]
    int8_t i8[8]
    float f32[2]
    double d64


cdef inline zsign(x):
    return 0 if x >= 0 else -1


cdef class VeoFunction(object):
    """
    VE Offloaded function
    """
    def __init__(self, lib, uint64_t addr, name):
        self.lib = lib
        self.addr = addr
        self.name = name
        self.args_conv = None
        self.ret_conv = conv_from_i64_func(self.lib.proc, "int")

    def __repr__(self):
        out = "<%s object VE function %s%r at %s>" % \
              (self.__class__.__name__, self.name, self._args_type, hex(id(self)))
        return out

    def args_type(self, *args):
        self._args_type = args
        self.args_conv = list()
        for t in args:
            if t == "void":
                continue
            self.args_conv.append(conv_to_i64_func(self.lib.proc, t))

    def ret_type(self, rettype):
        self._ret_type = rettype
        self.ret_conv = conv_from_i64_func(self.lib.proc, rettype)

    def __call__(self, VeoCtxt ctx, *args):
        """
        @brief Asynchrounously call a function on VE.
        @param ctx VEO context in which the function shall run
        @param *args arguments of the called function

        The function's argument types must be registered with f.args_type(), as well
        as the return type with f.ret_type(). Passed arguments are converted to the
        appropriate C types and used to prepare the VeoArgs object.

        When things like structs or unions or arrays have to be passed by reference,
        the selected argument type should be "void *" (or some other pointer) and
        the passed argument should be wrapped into OnStack(buff, size), where buff
        is a memoryview of the object and size its length. Look at
        examples/pass_on_stack.py for an example.

        Returns: VeoRequest instance, None in case of error.
        """
        if self._args_type is None:
            raise RuntimeError("VeoFunction needs arguments format info before call()")
        if len(args) > VEO_MAX_NUM_ARGS:
            raise ValueError("call_async: too many arguments (%d)" % len(args))
        if len(args) != len(self.args_conv):
            raise ValueError("invalid number of arguments, expected `{}`, got `{}`"
                             .format(len(self.args_conv), len(args)))

        a = VeoArgs()
        for i in xrange(len(self.args_conv)):
            x = args[i]
            if isinstance(x, nlcpy.ndarray):
                x = x._ve_array
            if isinstance(x, OnStack):
                try:
                    # a.set_stack(x.scope(), i, x.c_pointer(), x.size())
                    a.set_stack(x, i)
                except Exception as e:
                    raise ValueError("%r : arg on stack: c_pointer = %r, size = %r" %
                                     (e, x.c_pointer(), x.size()))
            else:
                f = self.args_conv[i]
                try:
                    a.set_i64(i, f(x))
                except Exception as e:
                    raise ValueError("%r : args conversion: f = %r, x = %r" % (e, f, x))

        cdef uint64_t res = veo_call_async(ctx.thr_ctxt, self.addr, a.args)
        if res == VEO_REQUEST_ID_INVALID:
            return None
            # raise RuntimeError("veo_call_async failed")
        #
        # We need to pass the args over because they are needed when
        # collecting the result. If they go out of scope and are freed,
        # we'll get a SIGSEGV!
        #
        return VeoRequest(ctx, a, res, self.ret_conv)


cdef class VeoRequest(object):
    """
    VE offload call request
    """
#    cdef readonly uint64_t req
#    cdef readonly VeoCtxt ctx
#    cdef ret_conv
#    cdef VeoArgs args

    def __init__(self, ctx, args, req, ret_conv):
        self.ctx = ctx
        self.req = req
        self.args = args
        self.ret_conv = ret_conv

    def __repr__(self):
        out = "<%s object req %d in context %r>" % \
              (self.__class__.__name__, self.req, self.ctx)
        return out

    @prof.profile_wait_result
    def wait_result(self):
        cdef uint64_t res
        cdef int rc = veo_call_wait_result(self.ctx.thr_ctxt, self.req, &res)
        if rc == VEO_COMMAND_EXCEPTION:
            raise ArithmeticError("wait_result command exception on VE")
        elif rc == VEO_COMMAND_ERROR:
            raise RuntimeError("wait_result command handling error")
        elif rc < 0:
            raise RuntimeError("wait_result command exception on VH")
        return self.ret_conv(<int64_t>res)

    def peek_result(self):
        cdef uint64_t res
        cdef int rc = veo_call_peek_result(self.ctx.thr_ctxt, self.req, &res)
        if rc == VEO_COMMAND_EXCEPTION:
            raise ArithmeticError("peek_result command exception")
        elif rc == VEO_COMMAND_ERROR:
            raise RuntimeError("peek_result command error on VE")
        elif rc == VEO_COMMAND_UNFINISHED:
            raise NameError("peek_result command unfinished")
        elif rc < 0:
            raise RuntimeError("peek_result command exception on VH")
        return self.ret_conv(res)


cdef class VeoMemRequest(VeoRequest):

    @staticmethod
    cdef create(VeoCtxt ctx, req, Py_buffer data):
        vmr = VeoMemRequest(ctx, VeoArgs(), req, conv_from_i64_func(ctx.proc, "int"))
        vmr.data = data
        return vmr

    def wait_result(self):
        try:
            res = super(VeoMemRequest, self).wait_result()
        except Exception as e:
            raise e
        finally:
            PyBuffer_Release(&self.data)
        return res

    def peek_result(self):
        try:
            res = super(VeoMemRequest, self).peek_result()
            PyBuffer_Release(&self.data)
            return res
        except NameError as e:
            raise e
        except Exception as e:
            PyBuffer_Release(&self.data)
            raise e


cdef class VeoLibrary(object):
    """
    Library loaded in a VE Proc.

    The library object can be one loaded with dlopen
    or correspond to the static symbols in the veorun VE binary.
    """
#    cdef readonly VeoProc proc
#    cdef name
#    cdef uint64_t lib_handle
#    cdef readonly dict func
#    cdef readonly dict symbol

    def __getattr__(self, name):
        name = name.encode('utf-8')
        if name in self.func:
            return self.func[name]
        if name in self.symbol:
            return self.symbol[name]
        return self.find_function(name)

    def __init__(self, veo_proc, name, uint64_t handle):
        self.proc = veo_proc
        self.name = name
        self.lib_handle = handle
        self.func = dict()
        self.symbol = dict()

    def get_symbol(self, char *symname):
        cdef uint64_t res
        res = veo_get_sym(self.proc.proc_handle, self.lib_handle, symname)
        if res == 0UL:
            raise RuntimeError("veo_get_sym '%s' failed" % symname)
        memptr = VEMemPtr(self.proc, res, 0)
        self.symbol[<bytes>symname] = memptr
        return memptr

    def find_function(self, char *symname):
        cdef uint64_t res
        res = veo_get_sym(self.proc.proc_handle, self.lib_handle, symname)
        if res == 0UL:
            raise RuntimeError("veo_get_sym '%s' failed" % symname)
        func = VeoFunction(self, res, <bytes>symname)
        self.func[<bytes>symname] = func
        return func


cdef class OnStack(object):

    def __init__(self, buff, size=None, inout=VEO_INTENT_IN):
        #
        if not PyObject_CheckBuffer(buff):
            raise TypeError("OnStack buff must implement the buffer protocol!")

        PyObject_GetBuffer(buff, &self.data, PyBUF_ANY_CONTIGUOUS)

        if size is not None and self.data.len < size:
            PyBuffer_Release(&self.data)
            raise ValueError("OnStack buffer is smaller than expected size (%d < %d)"
                             % (self.data.len, size))
        #
        self._c_pointer = <uint64_t>self.data.buf
        if size is not None:
            self._size = size
        else:
            self._size = self.data.len
        self._inout = inout

    def __dealloc__(self):
        PyBuffer_Release(&self.data)

    def buff(self):
        return self._buff

    def c_pointer(self):
        return <uint64_t>self._c_pointer

    def scope(self):
        return self._inout

    def size(self):
        return self._size


cdef class VeoArgs(object):

    def __init__(self):
        self.args = veo_args_alloc()
        if self.args == NULL:
            raise RuntimeError("Failed to alloc veo_args")
        self.stacks = []

    def __dealloc__(self):
        veo_args_free(self.args)
        self.stacks.clear()

    def set_i32(self, int argnum, int32_t val):
        veo_args_set_i32(self.args, argnum, val)

    def set_i64(self, int argnum, int64_t val):
        veo_args_set_i64(self.args, argnum, val)

    def set_u32(self, int argnum, uint32_t val):
        veo_args_set_u64(self.args, argnum, val)

    def set_u64(self, int argnum, uint64_t val):
        veo_args_set_u64(self.args, argnum, val)

    def set_float(self, int argnum, float val):
        veo_args_set_float(self.args, argnum, val)

    def set_double(self, int argnum, double val):
        veo_args_set_double(self.args, argnum, val)

    # def set_stack(self, veo_args_intent inout, int argnum,
    #               uint64_t buff, size_t len):
    def set_stack(self, OnStack x, int argnum):
        cdef uint64_t buff = x.c_pointer()
        veo_args_set_stack(
            self.args, x.scope(), argnum, <char *>buff, x.size())
        self.stacks.append(x)

    def clear(self):
        veo_args_clear(self.args)


cdef class VeoCtxt(object):
    """
    VE Offloading thread context.

    This is corresponding to one VE worker thread. Technically
    it is cloned from the control thread started by the VeoProc
    therefore all VeoCtxt instances share the same memory and
    are controlled by their parent VeoProc.
    """

    def __init__(self, VeoProc proc):
        self.proc = proc
        self.thr_ctxt = veo_context_open(proc.proc_handle)
        if self.thr_ctxt == NULL:
            raise RuntimeError("veo_context_open failed")

    def __dealloc__(self):
        veo_context_close(self.thr_ctxt)

    def async_read_mem(self, dst, VEMemPtr src, Py_ssize_t size):
        if src.proc != self.proc:
            raise ValueError("src memptr not owned by this proc!")
        cdef Py_buffer data
        cdef uint64_t req
        if not PyObject_CheckBuffer(dst):
            raise TypeError("dst must implement the buffer protocol!")

        PyObject_GetBuffer(dst, &data, PyBUF_ANY_CONTIGUOUS)

        if data.len < size:
            PyBuffer_Release(&data)
            raise ValueError(
                "read_mem dst buffer is smaller than required size (%d < %d)"
                % (data.len, size)
            )

        req = veo_async_read_mem(self.thr_ctxt, data.buf, src.addr, size)
        if req == VEO_REQUEST_ID_INVALID:
            PyBuffer_Release(&data)
            raise RuntimeError("veo_async_read_mem failed")
        return VeoMemRequest.create(self, req, data)

    def async_write_mem(self, VEMemPtr dst, src, Py_ssize_t size):
        if dst.proc != self.proc:
            raise ValueError("dst memptr not owned by this proc!")
        cdef Py_buffer data
        cdef uint64_t req
        if not PyObject_CheckBuffer(src):
            raise TypeError("src must implement the buffer protocol!")

        PyObject_GetBuffer(src, &data, PyBUF_ANY_CONTIGUOUS)

        if data.len < size:
            PyBuffer_Release(&data)
            raise ValueError(
                "write_mem src buffer is smaller than required size (%d < %d)"
                % (data.len, size)
            )

        req = veo_async_write_mem(self.thr_ctxt, dst.addr, data.buf, size)
        if req == VEO_REQUEST_ID_INVALID:
            PyBuffer_Release(&data)
            raise RuntimeError("veo_write_mem failed")
        return VeoMemRequest.create(self, req, data)

    def context_sync(self):
        veo_context_sync(self.thr_ctxt)


# Memory Pool Threashold
# If the allocated memory size on VE is larger than pool_threashold,
# the memory pool implementation is not used.
# DEF pool_threashold = 0x0000000000000000
# 512 Mbyte
DEF pool_threashold = 0x0000000020000000
# 1 Gbyte
# DEF pool_threashold = 0x0000000040000000

cdef class VeoProc(object):

    def __init__(self, int nodeid, veorun_bin=None):
        global _proc_init_hook
        self.nodeid = nodeid
        self.context = list()
        self.lib = dict()
        if veorun_bin is not None:
            self.proc_handle = veo_proc_create_static(nodeid, veorun_bin)
            if self.proc_handle == NULL:
                raise RuntimeError("veo_proc_create_static(%d, %s) failed" %
                                   (nodeid, veorun_bin))
        else:
            self.proc_handle = veo_proc_create(nodeid)
            if self.proc_handle == NULL:
                raise RuntimeError("veo_proc_create(%d) failed" % nodeid)
        if len(_proc_init_hook) > 0:
            for init_func in _proc_init_hook:
                init_func(self)

    def __dealloc__(self):
        if self.proc_handle == NULL:
            return  # to avoid segmentation fault when ve node is offline.
        if veo_proc_destroy(self.proc_handle):
            raise RuntimeError("veo_proc_destroy failed")

    def get_function(self, name):
        """
        Return a VeoFunction for function 'name' which
        has been previously found in one of the loaded libraries.
        Return None if the function wasn't found.
        """
        for lib in self.lib:
            if name in lib.func.keys():
                return lib.func[name]
        return None

    def i64_to_memptr(self, int64_t x):
        return VEMemPtr(self, ConvFromI64.to_ulong(x), 0)

    def load_library(self, char *libname):
        cdef uint64_t res = veo_load_library(self.proc_handle, libname)
        self.lib_handle = res
        if res == 0UL:
            raise RuntimeError("veo_load_library '%s' failed" % libname)
        lib = VeoLibrary(self, <bytes> libname, res)
        self.lib[<bytes>libname] = lib
        return lib

    def unload_library(self, VeoLibrary lib):
        cdef int res = veo_unload_library(self.proc_handle, lib.lib_handle)
        if res != 0:
            raise RuntimeError("veo_unload_library '%s' failed" % lib.name)
        self.lib_handle = 0LU
        del self.lib[<bytes>lib.name]

    def static_library(self):
        lib = VeoLibrary(self, "__static__", 0UL)
        self.lib["__static__"] = lib
        return lib

    @prof.profile_alloc_mem
    def alloc_mem(self, size_t size):
        cdef uint64_t addr
        if size > pool_threashold:
            if veo_alloc_mem(self.proc_handle, &addr, size):
                nlcpy.request.flush()
                if veo_alloc_mem(self.proc_handle, &addr, size):
                    raise MemoryError("Out of memory on VE")
        else:
            pool = _get_veo_pool()
            addr = pool.reserve(<size_t>size)
        return VEMemPtr(self, addr, size)

    def realloc_mem(self, src, size_t size):
        ctx = _get_veo_ctx()
        lib = _get_veo_lib()
        dst = self.alloc_mem(size)
        args = (dst.addr, src.addr, size)
        req = lib.func[b"nlcpy_memcpy"](ctx, *args)
        req.wait_result()
        return dst

    @prof.profile_free_mem
    def free_mem(self, memptr):
        if memptr.proc != self:
            raise ValueError("memptr not owned by this proc!")
        if memptr.size > pool_threashold:
            if veo_free_mem(self.proc_handle, memptr.addr):
                raise RuntimeError("veo_free_mem failed")
        else:
            pool = _get_veo_pool()
            if pool is not None:
                pool.release(memptr.addr)

    @prof.profile_read_mem
    def read_mem(self, dst, VEMemPtr src, Py_ssize_t size):
        if src.proc != self:
            raise ValueError("src memptr not owned by this proc!")
        cdef Py_buffer data
        if not PyObject_CheckBuffer(dst):
            raise TypeError("dst must implement the buffer protocol!")
        try:
            PyObject_GetBuffer(dst, &data, PyBUF_ANY_CONTIGUOUS)

            if data.len < size:
                raise ValueError(
                    "read_mem dst buffer is smaller than required size (%d < %d)"
                    % (data.len, size)
                )

            if veo_read_mem(self.proc_handle, data.buf, src.addr, size):
                raise RuntimeError("veo_read_mem failed")
        finally:
            PyBuffer_Release(&data)

    @prof.profile_write_mem
    def write_mem(self, VEMemPtr dst, src, Py_ssize_t size):
        if dst.proc != self:
            raise ValueError("dst memptr not owned by this proc!")
        cdef Py_buffer data
        if not PyObject_CheckBuffer(src):
            raise TypeError("src must implement the buffer protocol!")
        try:
            PyObject_GetBuffer(src, &data, PyBUF_ANY_CONTIGUOUS)

            if data.len < size:
                raise ValueError(
                    "write_mem src buffer is smaller than required size (%d < %d)"
                    % (data.len, size)
                )

            if veo_write_mem(self.proc_handle, dst.addr, data.buf, size):
                raise RuntimeError("veo_write_mem failed")
        finally:
            PyBuffer_Release(&data)

    def alloc_mempool(self):
        return MemPool(self, self.context[0])

    def reserve_mempool(self, pool, size):
        return pool.reserve(size)

    def release_mempool(self, pool, ve_adr):
        return pool.release(ve_adr)

    def free_mempool(self, pool):
        pool.__del__()

    def open_context(self):
        cdef VeoCtxt c
        c = VeoCtxt(self)
        self.context.append(c)
        return c

    def close_context(self, VeoCtxt c):
        self.context.remove(c)
        del c


cdef class VEMemPtr(object):

    def __init__(self, proc, uint64_t addr, size_t size):
        """
        Initialize a VE memory pointer object

        Arguments:
        proc: VeoProc who owns the memory
        addr: the VEMVA virtual address of the memory pointer
        size: size of the memory, if known. Known if allocated.
        """
        self.addr = addr
        self.size = size
        self.proc = proc

    def __repr__(self):
        out = "<%s object VE addr: %s %s owner %r>" % \
              (self.__class__.__name__,
               hex(self.addr), ", size: %dbytes," % self.size if self.size != 0 else "",
               self.proc)
        return out


_here = _path._here

cdef object _proc = None
cdef object _lib = None
cdef object _ctx = None
cdef object _pool = None


# TODO: thread lock
cpdef _get_veo_proc():
    global _proc
    return _proc

cdef _set_veo_proc(proc):
    global _proc
    _proc = proc

cpdef _get_veo_lib():
    global _lib
    return _lib

cdef _set_veo_lib(lib):
    global _lib
    _lib = lib

cpdef _get_veo_ctx():
    global _ctx
    return _ctx

cdef _set_veo_ctx(ctx):
    global _ctx
    _ctx = ctx

cpdef _get_veo_pool():
    global _pool
    return _pool

cdef _set_veo_pool(pool):
    global _pool
    _pool = pool


cpdef _initialize(node=-1):
    set_proc_init_hook(register._register_ve_kernel)
    proc = VeoProc(node)
    lib = None

    fast_math = os.environ.get('VE_NLCPY_FAST_MATH', 'no')
    if fast_math in ('yes', 'YES'):
        lib = proc.lib[
            (
                _here + "/lib/libnlcpy_ve_kernel_fast_math.so"
            ).encode('utf-8')
        ]
    else:
        lib = proc.lib[
            (
                _here + "/lib/libnlcpy_ve_kernel_no_fast_math.so"
            ).encode('utf-8')
        ]

    if lib is None:
        raise RuntimeError("cannot detect ve kernel")
    ctx = proc.open_context()
    pool = proc.alloc_mempool()

    _set_veo_proc(proc)
    _set_veo_lib(lib)
    _set_veo_ctx(ctx)
    _set_veo_pool(pool)


@atexit.register
def finalize():
    proc = _get_veo_proc()
    pool = _get_veo_pool()
    try:
        proc.free_mempool(pool)
    except AttributeError as e:
        pass

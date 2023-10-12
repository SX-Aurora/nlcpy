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

import unittest
import pytest
import numpy
import nlcpy
from nlcpy import testing
from nlcpy import venode
from nlcpy import veo


class TestVeoFunction(unittest.TestCase):

    def test_veo_function_above_max_args(self):
        ve = venode.VE()
        args = [0 for _ in range(veo._veo._veo_max_num_args + 1)]
        with pytest.raises(ValueError) as ex:
            ve.lib.func[b'nlcpy_arange'](ve.ctx, *args)
        assert ('too many arguments' in str(ex.value))

    def test_veo_function_invalid_num_args(self):
        ve = venode.VE()
        args = [0 for _ in range(10)]
        with pytest.raises(ValueError) as ex:
            ve.lib.func[b'nlcpy_arange'](ve.ctx, *args)
        assert ('invalid number of arguments' in str(ex.value))

    def test_veo_function_not_registered(self):
        ve = venode.VE()
        args = [0 for _ in range(10)]
        with pytest.raises(RuntimeError):
            ve.lib.find_function(b'nlcpy_add')(ve.ctx, *args)


class TestVeoOnStack(unittest.TestCase):

    def test_on_stack_invalid_pointer(self):
        with pytest.raises(TypeError):
            veo.OnStack(0)

    def test_on_stack_size_mismatch(self):
        buf = numpy.empty(10)
        with pytest.raises(ValueError):
            veo.OnStack(buf, size=buf.nbytes + 1)

    def test_on_stack_with_size(self):
        buf = numpy.empty(10)
        veo.OnStack(buf, size=buf.nbytes)


class TestVeoRequest(unittest.TestCase):

    def test_veo_request_peek_result(self):
        nlcpy.random.rand(10)  # to initialize random handle
        ve = venode.VE()
        func = ve.lib.func[b'nlcpy_random_get_state_size']
        req = func(ve.ctx, *())
        ret = None
        while ret is None:
            try:
                ret = req.peek_result()
            except NameError:  # not finished
                ret = None
                continue
        assert ret > 0


class TestMemVeoRequest(unittest.TestCase):

    def test_veo_mem_request_peek_result(self):
        dat = numpy.arange(10, dtype='f8')
        mem = nlcpy.empty(dat.shape, dtype=dat.dtype)
        ve = nlcpy.venode.VE()
        req = ve.ctx.async_write_mem(
            mem.ve_adr, dat.data, dat.nbytes)
        ret = None
        while ret is None:
            try:
                ret = req.peek_result()
            except NameError:  # not finished
                ret = None
                continue
        assert ret == 0
        testing.assert_array_equal(dat, mem)


class TestStrRepr(unittest.TestCase):

    def test_str_repr(self):
        nlcpy.random.rand(10)  # to initialize random handle
        ve = venode.VE()
        func = ve.lib.func[b'nlcpy_random_get_state_size']
        assert func.__repr__() is not None
        req = func(ve.ctx, *())
        assert req.__repr__() is not None


class TestVeoLibrary(unittest.TestCase):

    def test_veo_library_getattr(self):
        ve = venode.VE()
        func = ve.lib.nlcpy_arange
        assert isinstance(func, veo._veo.VeoFunction)

    def test_veo_library_get_sym(self):
        ve = venode.VE()
        sym = ve.lib.get_symbol(b'nlcpy_set_constant')
        assert sym != 0

    def test_veo_library_get_sym_err(self):
        ve = venode.VE()
        with pytest.raises(RuntimeError):
            _ = ve.lib.get_symbol(b'nlcpy_foo_bar')


class TestVeoArgs(unittest.TestCase):

    def test_set_arg(self):
        arg = veo.VeoArgs()
        assert isinstance(arg, veo.VeoArgs)
        arg.set_i32(0, 1)
        arg.set_i64(1, 1)
        arg.set_u32(2, 1)
        arg.set_u64(3, 1)
        arg.set_float(4, 1)
        arg.set_double(5, 1)
        arg.clear()


class TestVeoCtxt(unittest.TestCase):

    def test_async_read_mem(self):
        src = nlcpy.arange(10, dtype='f8')
        dst = numpy.empty(src.shape, dtype=src.dtype)
        ve = venode.VE()
        ve.synchronize()
        req = ve.ctx.async_read_mem(dst.data, src.ve_adr, src.nbytes)
        ret = req.wait_result()
        assert ret == 0
        testing.assert_array_equal(src, dst)

    def test_async_read_mem_not_buffer(self):
        src = nlcpy.arange(10, dtype='f8')
        dst = 0
        ve = venode.VE()
        ve.synchronize()
        with pytest.raises(TypeError):
            _ = ve.ctx.async_read_mem(dst, src.ve_adr, src.nbytes)

    def test_async_read_mem_size_mismatch(self):
        src = nlcpy.arange(10, dtype='f8')
        dst = numpy.empty(src.shape, dtype=src.dtype)
        ve = venode.VE()
        ve.synchronize()
        with pytest.raises(ValueError):
            _ = ve.ctx.async_read_mem(dst.data, src.ve_adr, src.nbytes + 1)

    def test_async_write_mem_not_buffer(self):
        dst = nlcpy.empty(10, dtype='f8')
        ve = venode.VE()
        with pytest.raises(TypeError):
            _ = ve.ctx.async_write_mem(dst.ve_adr, 0, dst.nbytes)

    def test_async_write_mem_size_mismatch(self):
        src = numpy.arange(10, dtype='f8')
        dst = nlcpy.empty(src.shape, dtype=src.dtype)
        ve = venode.VE()
        with pytest.raises(ValueError):
            _ = ve.ctx.async_write_mem(dst.ve_adr, src.data, dst.nbytes + 1)

    def test_contrext_sync(self):
        venode.VE().ctx.context_sync()


class TestVeoProc(unittest.TestCase):

    def test_i64_to_addr(self):
        proc = venode.VE().proc
        ref = -1
        ret = proc.i64_to_addr(ref)
        assert ret == 0xffffffffffffffff

    def test_load_library_failed(self):
        proc = venode.VE().proc
        with pytest.raises(RuntimeError):
            _ = proc.load_library(b'lib_foo_bar.so')

    def test_alloc_mem_out_of_memory(self):
        ve = venode.VE()
        tot_memsize = ve.status['main_total_memsize']
        with pytest.raises(MemoryError):
            _ = ve.proc.alloc_mem(tot_memsize + 8)

    def test_alloc_hmem_out_of_memory(self):
        ve = venode.VE()
        tot_memsize = ve.status['main_total_memsize']
        with pytest.raises(MemoryError):
            _ = ve.proc.alloc_hmem(tot_memsize + 8)

    def test_read_mem_not_buffer(self):
        src = nlcpy.arange(10, dtype='f8')
        ve = venode.VE()
        with pytest.raises(TypeError):
            ve.proc.read_mem(0, src.ve_adr, src.nbytes)

    def test_read_mem_size_mismatch(self):
        src = nlcpy.arange(10, dtype='f8')
        dst = numpy.empty(src.shape, dtype=src.dtype)
        ve = venode.VE()
        with pytest.raises(ValueError):
            ve.proc.read_mem(dst.data, src.ve_adr, src.nbytes + 1)

    def test_write_mem_not_buffer(self):
        dst = nlcpy.arange(10, dtype='f8')
        ve = venode.VE()
        with pytest.raises(TypeError):
            ve.proc.write_mem(dst.ve_adr, 0, dst.nbytes)

    def test_write_mem_size_mismatch(self):
        src = numpy.arange(10, dtype='f8')
        dst = nlcpy.empty(src.shape, dtype=src.dtype)
        ve = venode.VE()
        with pytest.raises(ValueError):
            ve.proc.write_mem(dst.ve_adr, src.data, dst.nbytes + 1)

    def test_proc_identifier(self):
        assert venode.VE().proc.proc_identifier() >= 0

    def test_set_proc_identifier(self):
        mem = nlcpy.empty(10)
        iden = mem.venode.proc.proc_identifier()
        hmem = mem.venode.proc.set_proc_identifier(mem.ve_adr, iden)
        assert hmem == mem.veo_hmem

    def test_set_proc_identifier_failed(self):
        mem = nlcpy.empty(10)
        with pytest.raises(RuntimeError):
            mem.venode.proc.set_proc_identifier(mem.ve_adr, -1)


class TestVeoHmem(unittest.TestCase):

    def test_get_proc_identifier_from_hmem(self):
        mem = nlcpy.empty(23)
        iden = veo.VEO_HMEM.get_proc_identifier_from_hmem(mem.veo_hmem)
        assert iden == mem.venode.proc.proc_identifier()

    def test_get_proc_handle_from_hmem(self):
        mem = nlcpy.empty(33)
        proc_handle = veo.VEO_HMEM.get_proc_handle_from_hmem(mem.veo_hmem)
        assert proc_handle == mem.venode.proc._proc_handle

    def test_get_proc_handle_from_hmem_failed(self):
        with pytest.raises(RuntimeError):
            veo.VEO_HMEM.get_proc_handle_from_hmem(0)

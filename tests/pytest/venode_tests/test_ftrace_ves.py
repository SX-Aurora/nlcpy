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
import string
import os
import glob

import nlcpy
from nlcpy import venode
from nlcpy import testing
from nlcpy.ve_types import uint64


nve = nlcpy.venode.get_num_available_venodes()

##########################
# source for using ve_adr
##########################

_test_ve_adr_c = r'''
#include <stdint.h>
uint64_t test_sum_c(uint64_t xadr, uint64_t yadr, uint64_t zadr, uint64_t n) {
    ${dtype} *px = (${dtype} *)xadr;
    ${dtype} *py = (${dtype} *)yadr;
    ${dtype} *pz = (${dtype} *)zadr;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        pz[i] = px[i] + py[i];
    }
    return 0;
}
'''

_test_ve_adr_cpp = r'''
#include <stdint.h>
#include <complex>
void cpptest_sum(uint64_t xadr, uint64_t yadr, uint64_t zadr, uint64_t n) {
    ${dtype} *px = (${dtype} *)xadr;
    ${dtype} *py = (${dtype} *)yadr;
    ${dtype} *pz = (${dtype} *)zadr;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        pz[i] = px[i] + py[i];
    }
}

extern "C"{
uint64_t test_sum_cpp(uint64_t xadr, uint64_t yadr, uint64_t zadr, uint64_t n) {
    cpptest_sum(xadr, yadr, zadr, n);
    return 0;
}
}
'''

_test_ve_adr_f = r'''
function test_sum_f(px, py, pz, n)
    implicit none
    integer(kind=8), value :: n
    integer(kind=8) :: test_sum_f, i
    ${dtype} :: px(n), py(n), pz(n)
!$$omp parallel do
    do i=1, n
        pz(i) = px(i) + py(i)
    end do
    test_sum_f = 0
end
'''


@testing.multi_ve(nve)
@testing.parameterize(*testing.product({
    'veid': [i for i in range(nve)],
}))
@pytest.mark.ftrace_gen
class TestJitFtraceGenVEs(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)
        venode.VE(self.veid).apply()

    def tearDown(self):
        self._prev_ve.apply()

    def test_ftrace_c(self):
        ftrace_files = glob.glob("./ftrace.out*")
        for f in ftrace_files:
            os.remove(f)

        code = string.Template(_test_ve_adr_c).substitute(dtype='double')
        ve_lib = nlcpy.jit.CustomVELibrary(
            code=code,
            compiler='/opt/nec/ve/bin/ncc',
            ftrace=True
        )
        ve_sum = ve_lib.get_function(
            'test_sum_c',
            args_type=(uint64, uint64, uint64, uint64),
            ret_type=uint64
        )
        x = nlcpy.arange(10, dtype='f8')
        y = nlcpy.arange(10, dtype='f8')
        res_jit = nlcpy.empty(10, dtype='f8')
        err = ve_sum(x.ve_adr, y.ve_adr, res_jit.ve_adr, 10, sync=True)
        assert err == 0
        res_naive = x + y
        testing.assert_allclose(res_jit, res_naive, atol=1e-12, rtol=1e-12)
        assert res_jit.venode == venode.VE(self.veid)
        # TODO: unload VE kernel

    def test_ftrace_cpp(self):
        ftrace_files = glob.glob("./ftrace.out.veo.*")
        for f in ftrace_files:
            os.remove(f)

        code = string.Template(_test_ve_adr_cpp).substitute(dtype='double')
        ve_lib = nlcpy.jit.CustomVELibrary(
            code=code,
            compiler='/opt/nec/ve/bin/nc++',
            ftrace=True
        )
        ve_sum = ve_lib.get_function(
            'test_sum_cpp',
            args_type=(uint64, uint64, uint64, uint64),
            ret_type=uint64
        )
        x = nlcpy.arange(10, dtype='f8')
        y = nlcpy.arange(10, dtype='f8')
        res_jit = nlcpy.empty(10, dtype='f8')
        err = ve_sum(x.ve_adr, y.ve_adr, res_jit.ve_adr, 10, sync=True)
        assert err == 0
        res_naive = x + y
        testing.assert_allclose(res_jit, res_naive, atol=1e-12, rtol=1e-12)
        assert res_jit.venode == venode.VE(self.veid)
        # TODO: unload VE kernel

    def test_ftrace_f(self):
        ftrace_files = glob.glob("./ftrace.out*")
        for f in ftrace_files:
            os.remove(f)

        code = string.Template(_test_ve_adr_f).substitute(dtype='double')
        ve_lib = nlcpy.jit.CustomVELibrary(
            code=code,
            compiler='/opt/nec/ve/bin/nfort',
            ftrace=True
        )
        ve_sum = ve_lib.get_function(
            'test_sum_f_',
            args_type=(uint64, uint64, uint64, uint64),
            ret_type=uint64
        )
        x = nlcpy.arange(10, dtype='f8')
        y = nlcpy.arange(10, dtype='f8')
        res_jit = nlcpy.empty(10, dtype='f8')
        err = ve_sum(x.ve_adr, y.ve_adr, res_jit.ve_adr, 10, sync=True)
        assert err == 0
        res_naive = x + y
        testing.assert_allclose(res_jit, res_naive, atol=1e-12, rtol=1e-12)
        assert res_jit.venode == venode.VE(self.veid)
        # TODO: unload VE kernel


@testing.multi_ve(nve)
@pytest.mark.ftrace_chk
class TestJitFtraceChkVEs(unittest.TestCase):

    def test_ftrace_chk(self):
        ftrace_files = glob.glob("./ftrace.out.veo.*")

        assert len(ftrace_files) == nve
        for i in range(nve):
            veid = venode.VE(i).pid
            expected = 'ftrace.out.veo.{}'.format(veid)
            matched = False
            for f in ftrace_files:
                if expected in f:
                    matched = True
                    break
            assert matched

        for f in ftrace_files:
            msg = ''
            with open(f, 'rb') as fh:
                msg = str(fh.read())
            assert 'test_sum_c' in msg
            assert 'test_sum_cpp' in msg
            assert 'TEST_SUM_F' in msg
            os.remove(f)

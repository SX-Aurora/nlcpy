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

import unittest
import nlcpy
from nlcpy import testing
from nlcpy.ve_types import (int64, uint64)

DIST_DIR = None
# DIST_DIR = 'jit_cache'

LOG_STREAM = None
# import sys
# LOG_STREAM = sys.stdout

_test_asl_native_c = r'''
#include <asl.h>
int64_t test_dbgmsm(double *ab, int64_t *ipvt, int64_t lna,
                    int64_t n, int64_t m) {
    int64_t ierr;
    ierr = ASL_dbgmsm(ab, lna, n, m, ipvt);
    return ierr;
}
'''

_test_asl_native_cpp = r'''
#include <asl.h>
int64_t cpptest_dbgmsm(double *ab, int64_t *ipvt, int64_t lna,
                       int64_t n, int64_t m) {
    int64_t ierr;
    ierr = ASL_dbgmsm(ab, lna, n, m, ipvt);
    return ierr;
}

extern "C" {
int64_t test_dbgmsm(double *ab, int64_t *ipvt, int64_t lna,
                    int64_t n, int64_t m) {
    return cpptest_dbgmsm(ab, ipvt, lna, n, m);
}}
'''

_test_asl_native_f = r'''
function test_dbgmsm(ab, ipvt, lna, n, m)
    implicit none
    integer(kind=8), value :: lna, n, m
    real(kind=8) :: ab(n+m, lna)
    integer(kind=8) :: ipvt(n), test_dbgmsm, ierr
    test_dbgmsm = -1
    call dbgmsm(ab, lna, n, m, ipvt, ierr)
    test_dbgmsm = ierr
end
'''


def _callback(err):
    if err != 0:
        raise RuntimeError


@testing.parameterize(*testing.product({
    'openmp': [True, False],
    'sync': [True, False],
    'callback': [_callback, None],
}))
class TestAslNative(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        nlcpy.jit.unload_library(self.lib)

    def _helper(self, code, compiler, name, args_type, ret_type):
        self.lib = nlcpy.jit.CustomVELibrary(
            code=code,
            compiler=compiler,
            cflags=nlcpy.jit.get_default_cflags(openmp=self.openmp),
            ldflags=nlcpy.jit.get_default_ldflags(openmp=self.openmp),
            dist_dir=DIST_DIR,
            log_stream=LOG_STREAM,
            use_nlc=True
        )
        self.kern = self.lib.get_function(
            name,
            args_type=args_type,
            ret_type=ret_type
        )

    def _prep(self):
        lna = 11
        n = 4
        m = 2
        ab = nlcpy.empty((lna, n + m), dtype='f8', order='F')
        ipvt = nlcpy.empty(n, dtype='i8')
        # coefficient matrix
        ab[:n, :n] = [
            [2, 4, -1, 6],
            [-1, -5, 4, 2],
            [1, 2, 3, 1],
            [3, 5, -1, -3]
        ]
        # constant vectors
        ab[:n, n:n + m] = [
            [36, 11],
            [15, 0],
            [22, 7],
            [-6, 4]
        ]
        return lna, n, m, ab, ipvt

    def _make_ref(self):
        return [
            [1., 1.],
            [2., 1.],
            [4., 1.],
            [5., 1.]
        ]

    def test_asl_native_c(self):
        self._helper(_test_asl_native_c, '/opt/nec/ve/bin/ncc', 'test_dbgmsm',
                     (uint64, uint64, int64, int64, int64), int64)
        lna, n, m, ab, ipvt = self._prep()
        err = self.kern(ab.ve_adr, ipvt.ve_adr, lna, n, m,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(ab[:n, n:n + m], self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_asl_native_cpp(self):
        self._helper(_test_asl_native_cpp, '/opt/nec/ve/bin/nc++', 'test_dbgmsm',
                     (uint64, uint64, int64, int64, int64), int64)
        lna, n, m, ab, ipvt = self._prep()
        err = self.kern(ab.ve_adr, ipvt.ve_adr, lna, n, m,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(ab[:n, n:n + m], self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_asl_native_f(self):
        self._helper(_test_asl_native_f, '/opt/nec/ve/bin/nfort', 'test_dbgmsm_',
                     (uint64, uint64, int64, int64, int64), int64)
        lna, n, m, ab, ipvt = self._prep()
        err = self.kern(ab.ve_adr, ipvt.ve_adr, lna, n, m,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(ab[:n, n:n + m], self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

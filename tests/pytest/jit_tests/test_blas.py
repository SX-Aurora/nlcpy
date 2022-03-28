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
import nlcpy
from nlcpy import testing
from nlcpy.ve_types import (uint64, int64)

DIST_DIR = None
# DIST_DIR = 'jit_cache'

LOG_STREAM = None
# import sys
# LOG_STREAM = sys.stdout

_test_cblas_c = r'''
#include <cblas.h>
cblas_int_t test_cblas_dgemm(cblas_int_t M, cblas_int_t N, cblas_int_t K,
                             double *a, double *b, double *c) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, a, K, b, N, 0., c, N);
    return 0;
}
'''

_test_cblas_cpp = r'''
#include <cblas.h>
cblas_int_t cpptest_cblas_dgemm(cblas_int_t M, cblas_int_t N, cblas_int_t K,
                                double *a, double *b, double *c) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, a, K, b, N, 0., c, N);
    return 0;
}

extern "C" {
cblas_int_t test_cblas_dgemm(cblas_int_t M, cblas_int_t N, cblas_int_t K,
                             double *a, double *b, double *c) {
    return cpptest_cblas_dgemm(M, N, K, a, b, c);
}}
'''

_test_blas_f = r'''
function test_blas_dgemm(M, N, K, a, b, c)
    implicit none
    integer(kind=8), value :: M, N, K
    integer(kind=8) :: test_blas_dgemm
    real(kind=8) :: a(K, M), b(N, K), c(N, M)
    call DGEMM('N', 'N', M, N, K, 1.0d0, a, M, b, K, 0.d0, c, M)
    test_blas_dgemm = 0
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
class TestBlas(unittest.TestCase):

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

    def _prep(self, order='C'):
        M, N, K = 2, 3, 4
        rng = nlcpy.random.default_rng(0)
        a = nlcpy.asarray(rng.random((M, K), dtype='f8'), order=order)
        b = nlcpy.asarray(rng.random((K, N), dtype='f8'), order=order)
        c = nlcpy.empty((M, N), dtype='f8', order=order)
        return M, N, K, a, b, c

    def _exec_matmul(self, a, b):
        return a @ b

    def test_cblas_c(self):
        self._helper(_test_cblas_c, '/opt/nec/ve/bin/ncc', 'test_cblas_dgemm',
                     (int64, int64, int64, uint64, uint64, uint64), int64)
        M, N, K, a, b, c = self._prep(order='C')
        err = self.kern(M, N, K, a.ve_adr, b.ve_adr, c.ve_adr,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(c, self._exec_matmul(a, b),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_cblas_cpp(self):
        self._helper(_test_cblas_cpp, '/opt/nec/ve/bin/nc++', 'test_cblas_dgemm',
                     (int64, int64, int64, uint64, uint64, uint64), int64)
        M, N, K, a, b, c = self._prep(order='C')
        err = self.kern(M, N, K, a.ve_adr, b.ve_adr, c.ve_adr,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(c, self._exec_matmul(a, b),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_cblas_f(self):
        self._helper(_test_blas_f, '/opt/nec/ve/bin/nfort', 'test_blas_dgemm_',
                     (int64, int64, int64, uint64, uint64, uint64), int64)
        M, N, K, a, b, c = self._prep(order='F')
        err = self.kern(M, N, K, a.ve_adr, b.ve_adr, c.ve_adr,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(c, self._exec_matmul(a, b),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

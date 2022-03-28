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
from nlcpy.ve_types import (int64, uint64)

DIST_DIR = None
# DIST_DIR = 'jit_cache'

LOG_STREAM = None
# import sys
# LOG_STREAM = sys.stdout

_test_lapack_c = r'''
#include <stdint.h>
int64_t test_dgesv(int64_t n, int64_t nrhs, double *a, int64_t lda,
                   int64_t *ipiv, double *b, int64_t ldb) {
    int64_t info = -1;
    dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}
'''

_test_lapack_cpp = r'''
#include <stdint.h>

extern "C" void dgesv_(int64_t *, int64_t *, double *, int64_t *,
                       int64_t *, double *, int64_t *, int64_t *);

int64_t cpptest_dgesv(int64_t n, int64_t nrhs, double *a, int64_t lda,
                   int64_t *ipiv, double *b, int64_t ldb) {
    int64_t info = -1;
    dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

extern "C" {
int64_t test_dgesv(int64_t n, int64_t nrhs, double *a, int64_t lda,
                   int64_t *ipiv, double *b, int64_t ldb) {
    return cpptest_dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}}
'''

_test_lapack_f = r'''
function test_dgesv(n, nrhs, a, lda, ipiv, b, ldb)
    implicit none
    integer(kind=8), value :: n, nrhs, lda, ldb
    integer(kind=8) :: test_dgesv, info
    real(kind=8) :: a(n, lda), b(nrhs, ldb), ipiv(n)
    info = -1
    call dgesv(n, nrhs, a, lda, ipiv, b, ldb, info)
    test_dgesv = info
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
class TestLapack(unittest.TestCase):

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
        lnb = 9
        m = 2
        a = nlcpy.empty((lna, n), dtype='f8', order='F')
        b = nlcpy.empty((lnb, m), dtype='f8', order='F')
        ipvt = nlcpy.empty(n, dtype='i8')
        # coefficient matrix
        a[:n, :] = [
            [2, 4, -1, 6],
            [-1, -5, 4, 2],
            [1, 2, 3, 1],
            [3, 5, -1, -3]
        ]
        # constant vectors
        b[:n, :] = [
            [36, 11],
            [15, 0],
            [22, 7],
            [-6, 4]
        ]
        return lna, n, lnb, m, a, b, ipvt

    def _make_ref(self):
        return [
            [1., 1.],
            [2., 1.],
            [4., 1.],
            [5., 1.]
        ]

    def test_lapack_c(self):
        self._helper(_test_lapack_c, '/opt/nec/ve/bin/ncc', 'test_dgesv',
                     (int64, int64, uint64, int64, uint64, uint64, int64), int64)
        lna, n, lnb, m, a, b, ipvt = self._prep()
        err = self.kern(n, m, a.ve_adr, lna, ipvt.ve_adr, b.ve_adr, lnb,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(b[:n, :], self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_lapack_cpp(self):
        self._helper(_test_lapack_cpp, '/opt/nec/ve/bin/nc++', 'test_dgesv',
                     (int64, int64, uint64, int64, uint64, uint64, int64), int64)
        lna, n, lnb, m, a, b, ipvt = self._prep()
        err = self.kern(n, m, a.ve_adr, lna, ipvt.ve_adr, b.ve_adr, lnb,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(b[:n, :], self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_lapack_f(self):
        self._helper(_test_lapack_f, '/opt/nec/ve/bin/nfort', 'test_dgesv_',
                     (int64, int64, uint64, int64, uint64, uint64, int64), int64)
        lna, n, lnb, m, a, b, ipvt = self._prep()
        err = self.kern(n, m, a.ve_adr, lna, ipvt.ve_adr, b.ve_adr, lnb,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(b[:n, :], self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

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
import numpy
from nlcpy import testing
from nlcpy import veo
from nlcpy.ve_types import (void_p, int32, int64, uint64, float64)

DIST_DIR = None
# DIST_DIR = 'jit_cache'

LOG_STREAM = None
# import sys
# LOG_STREAM = sys.stdout

_test_asl_random_c = r'''
#include <stdio.h>
#include <asl.h>

int random_normal_d(int64_t n, const uint32_t *seed,
                    double d_m, double d_s, double *val) {
    asl_random_t rng;
    /* Generator Preparation */
    asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
    asl_random_distribute_normal(rng, d_m, d_s);
    asl_random_initialize(rng, 1, seed);
    /* Generation */
    asl_random_generate_d(rng, n, val);
    /* Generator Finalization */
    asl_random_destroy(rng);
    /* Library Finalization */
    return 0;
}
'''

_test_asl_random_cpp = r'''
#include <stdio.h>
#include <asl.h>

void cpprandom_normal_d(int64_t n, const uint32_t *seed,
                       double d_m, double d_s, double *val) {
    asl_random_t rng;
    /* Generator Preparation */
    asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
    asl_random_distribute_normal(rng, d_m, d_s);
    asl_random_initialize(rng, 1, seed);
    /* Generation */
    asl_random_generate_d(rng, n, val);
    /* Generator Finalization */
    asl_random_destroy(rng);
    /* Library Finalization */
}

extern "C" {
int random_normal_d(int64_t n, const uint32_t *seed,
                    double d_m, double d_s, double *val) {
    cpprandom_normal_d(n, seed, d_m, d_s, val);
    return 0;
}}
'''

_test_asl_random_f = r'''
function random_normal_d(n, seed, d_m, d_s, val)
   use asl_unified
   implicit none
   integer(kind=8), value :: n
   real(8), value :: d_m, d_s
   real(8) :: val(n)
   integer(kind=8) :: rng
   integer(kind=4) :: seed(1), random_normal_d
   ! Generator Preparation
   call asl_random_create(rng, ASL_RANDOMMETHOD_MT19937_64)
   call asl_random_distribute_normal(rng, d_m, d_s)
   call asl_random_initialize(rng, 1, seed)
   ! Generation
   call asl_random_generate_d(rng, n, val)
   ! Generator Finalization
   call asl_random_destroy(rng)
   random_normal_d = 0
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
class TestAslRandom(unittest.TestCase):

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
        N = 100
        D_M = 1.0
        D_S = 0.5
        val = nlcpy.empty(N, dtype='f8')
        seed = numpy.array(0, dtype='u4')
        return N, seed, D_M, D_S, val

    def _exec_random(self, seed, mu, gamma, N):
        nlcpy.random.seed(seed)
        return nlcpy.random.normal(loc=mu, scale=gamma, size=N)

    def test_asl_random_c(self):
        self._helper(_test_asl_random_c, '/opt/nec/ve/bin/ncc', 'random_normal_d',
                     (int64, void_p, float64, float64, uint64), int32)
        N, seed, D_M, D_S, val = self._prep()
        err = self.kern(N, veo.OnStack(seed, inout=veo.INTENT_IN), D_M, D_S, val.ve_adr,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(val, self._exec_random(seed, D_M, D_S, N),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_asl_random_cpp(self):
        self._helper(_test_asl_random_cpp, '/opt/nec/ve/bin/nc++', 'random_normal_d',
                     (int64, void_p, float64, float64, uint64), int32)
        N, seed, D_M, D_S, val = self._prep()
        err = self.kern(N, veo.OnStack(seed, inout=veo.INTENT_IN), D_M, D_S, val.ve_adr,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(val, self._exec_random(seed, D_M, D_S, N),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_asl_random_f(self):
        self._helper(_test_asl_random_f, '/opt/nec/ve/bin/nfort', 'random_normal_d_',
                     (int64, void_p, float64, float64, uint64), int32)
        N, seed, D_M, D_S, val = self._prep()
        err = self.kern(N, veo.OnStack(seed, inout=veo.INTENT_IN), D_M, D_S, val.ve_adr,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(val, self._exec_random(seed, D_M, D_S, N),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

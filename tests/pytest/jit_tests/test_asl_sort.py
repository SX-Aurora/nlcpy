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
from nlcpy.ve_types import (int32, int64, uint64)

DIST_DIR = None
# DIST_DIR = 'jit_cache'

LOG_STREAM = None
# import sys
# LOG_STREAM = sys.stdout

_test_asl_sort_c = r'''
#include <asl.h>
int sort_ascending_s(int64_t nx, float *kyi, float *kyo) {
    asl_sort_t sort;
    /* Sorting Preparation (ascending order) */
    asl_sort_create_s(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
    asl_sort_preallocate(sort, nx);
    /* Sorting Execution */
    asl_sort_execute_s(sort, nx, kyi, ASL_NULL, kyo, ASL_NULL);
    /* Sorting Finalization */
    asl_sort_destroy(sort);
    return 0;
}
'''

_test_asl_sort_cpp = r'''
#include <asl.h>
void cppsort_ascending_s(int64_t nx, float *kyi, float *kyo) {
    asl_sort_t sort;
    /* Sorting Preparation (ascending order) */
    asl_sort_create_s(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
    asl_sort_preallocate(sort, nx);
    /* Sorting Execution */
    asl_sort_execute_s(sort, nx, kyi, ASL_NULL, kyo, ASL_NULL);
    /* Sorting Finalization */
    asl_sort_destroy(sort);
}

extern "C" {
int sort_ascending_s(int64_t nx, float *kyi, float *kyo) {
    cppsort_ascending_s(nx, kyi, kyo);
    return 0;
}}
'''

_test_asl_sort_f = r'''
function sort_ascending_s(nx, kyi, kyo)
    use asl_unified
    implicit none
    integer(kind=8), value :: nx
    integer(kind=8) :: sort
    real(4) :: kyi(nx), kyo(nx)
    integer(kind=4) :: sort_ascending_s
    ! Sorting Preparation
    call asl_sort_create_s(sort, &
       & ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO)
    call asl_sort_preallocate(sort, nx)
    ! Sorting Execution (forward)
    call asl_sort_execute_s(sort, nx, kyi, kyo=kyo)
    ! Sorting Finalization
    call asl_sort_destroy(sort)
    sort_ascending_s = 0
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
class TestAslSort(unittest.TestCase):

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
        NX = 30
        rng = nlcpy.random.default_rng(seed=0)
        kyi = rng.random(size=NX, dtype='f4')
        kyo = nlcpy.empty(NX, dtype='f4')
        return NX, kyi, kyo

    def _exec_sort(self, kyi):
        return nlcpy.sort(kyi)

    def test_asl_sort_c(self):
        self._helper(_test_asl_sort_c, '/opt/nec/ve/bin/ncc', 'sort_ascending_s',
                     (int64, uint64, uint64), int32)
        NX, kyi, kyo = self._prep()
        err = self.kern(NX, kyi.ve_adr, kyo.ve_adr,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(kyo, self._exec_sort(kyi),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_asl_sort_cpp(self):
        self._helper(_test_asl_sort_cpp, '/opt/nec/ve/bin/nc++', 'sort_ascending_s',
                     (int64, uint64, uint64), int32)
        NX, kyi, kyo = self._prep()
        err = self.kern(NX, kyi.ve_adr, kyo.ve_adr,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(kyo, self._exec_sort(kyi),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_asl_sort_f(self):
        self._helper(_test_asl_sort_f, '/opt/nec/ve/bin/nfort', 'sort_ascending_s_',
                     (int64, uint64, uint64), int32)
        NX, kyi, kyo = self._prep()
        err = self.kern(NX, kyi.ve_adr, kyo.ve_adr,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(kyo, self._exec_sort(kyi),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

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

_test_asl_fftw_c = r'''
#include <stdlib.h>
#include <aslfftw3.h>

int test_complex_1d(fftw_complex *zin, fftw_complex *zout_forward,
                    fftw_complex *zout_backward, const int64_t NX) {
    if (zin == NULL || zout_forward == NULL || zout_backward == NULL) {
        fprintf(stderr, "memory allocation error!!\n");
        return 1;
    }
    /* Plan Creation (out-of-place forward and backward FFT) */
    fftw_plan planf = fftw_plan_dft_1d(
        NX, zin, zout_forward, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan planb = fftw_plan_dft_1d(
        NX, zout_forward, zout_backward, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (planf == NULL || planb == NULL) {
        fprintf(stderr, "plan creation error!!\n");
        return 1;
    }
    /* FFT Execution (forward) */
    fftw_execute(planf);
    /* FFT Execution (backward) */
    fftw_execute(planb);
    /* Plan Destruction */
    fftw_destroy_plan(planf);
    fftw_destroy_plan(planb);

    return 0;
}
'''

_test_asl_fftw_cpp = r'''
#include <stdlib.h>
#include <aslfftw3.h>

int cpptest_complex_1d(fftw_complex *zin, fftw_complex *zout_forward,
                       fftw_complex *zout_backward, const int64_t NX) {
    if (zin == NULL || zout_forward == NULL || zout_backward == NULL) {
        fprintf(stderr, "memory allocation error!!\n");
        return 1;
    }
    /* Plan Creation (out-of-place forward and backward FFT) */
    fftw_plan planf = fftw_plan_dft_1d(
        NX, zin, zout_forward, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan planb = fftw_plan_dft_1d(
        NX, zout_forward, zout_backward, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (planf == NULL || planb == NULL) {
        fprintf(stderr, "plan creation error!!\n");
        return 1;
    }
    /* FFT Execution (forward) */
    fftw_execute(planf);
    /* FFT Execution (backward) */
    fftw_execute(planb);
    /* Plan Destruction */
    fftw_destroy_plan(planf);
    fftw_destroy_plan(planb);

    return 0;
}

extern "C" {
int test_complex_1d(fftw_complex *zin, fftw_complex *zout_forward,
                       fftw_complex *zout_backward, const int NX) {
    return cpptest_complex_1d(zin, zout_forward, zout_backward, NX);
}}
'''

_test_asl_fftw_f = r'''
function test_complex_1d(zin, zout_forward, zout_backward, NX)
    use, intrinsic :: iso_c_binding
    implicit none
    include 'aslfftw3.f03'
    integer, value :: NX
    complex(kind=8) :: zin(NX), zout_forward(NX), zout_backward(NX)
    integer(kind=4) :: test_complex_1d
    ! Variable Definition
    type(C_PTR) :: planf, planb
    ! Plan Creation (out-of-place forward and backward FFT)
    planf = fftw_plan_dft_1d(NX, zin, zout_forward, &
                            &FFTW_FORWARD, FFTW_ESTIMATE)
    planb = fftw_plan_dft_1d(NX, zout_forward, zout_backward, &
                            &FFTW_BACKWARD, FFTW_ESTIMATE)
    if ((.not. c_associated(planf)) .or. (.not. c_associated(planb))) then
       write(*,*) "plan creation error!!"
       test_complex_1d = 1
       stop
    end if
    ! FFT Execution (forward)
    call fftw_execute_dft(planf, zin, zout_forward)
    ! FFT Execution (backward)
    call fftw_execute_dft(planb, zout_forward, zout_backward)
    ! Plan Destruction
    call fftw_destroy_plan(planf)
    call fftw_destroy_plan(planb)
    test_complex_1d = 0
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
class TestAslFftw(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        nlcpy.jit.unload_library(self.lib)

    def _helper(self, code, compiler, name, args_type, ret_type, ext_cflags=()):
        self.lib = nlcpy.jit.CustomVELibrary(
            code=code,
            compiler=compiler,
            cflags=nlcpy.jit.get_default_cflags(openmp=self.openmp) + ext_cflags,
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
        NX = 10
        zin = nlcpy.empty(NX, dtype='c16')
        zout_forward = nlcpy.empty(NX, dtype='c16')
        zout_backward = nlcpy.empty(NX, dtype='c16')
        zin.real = nlcpy.arange(NX, dtype='f8')
        zin.imag = nlcpy.arange(NX, dtype='f8')
        return NX, zin, zout_forward, zout_backward

    def _exec_fft(self, zin, inverse=False):
        if inverse:
            return nlcpy.fft.ifft(zin) * zin.size
        else:
            return nlcpy.fft.fft(zin)

    def test_asl_fftw_c(self):
        self._helper(_test_asl_fftw_c, '/opt/nec/ve/bin/ncc', 'test_complex_1d',
                     (uint64, uint64, uint64, int64), int32)
        NX, zin, zout_forward, zout_backward = self._prep()
        err = self.kern(zin.ve_adr, zout_forward.ve_adr, zout_backward.ve_adr, NX,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(zout_forward,
                                self._exec_fft(zin, inverse=False),
                                err_msg='File-ID: {}'.format(self.lib.id),
                                rtol=1e-12)
        testing.assert_allclose(zout_backward,
                                self._exec_fft(zout_forward, inverse=True),
                                err_msg='File-ID: {}'.format(self.lib.id),
                                rtol=1e-12)
        if self.sync:
            assert err == 0

    def test_asl_fftw_cpp(self):
        self._helper(_test_asl_fftw_cpp, '/opt/nec/ve/bin/nc++', 'test_complex_1d',
                     (uint64, uint64, uint64, int64), int32)
        NX, zin, zout_forward, zout_backward = self._prep()
        err = self.kern(zin.ve_adr, zout_forward.ve_adr, zout_backward.ve_adr, NX,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(zout_forward,
                                self._exec_fft(zin, inverse=False),
                                err_msg='File-ID: {}'.format(self.lib.id),
                                rtol=1e-12)
        testing.assert_allclose(zout_backward,
                                self._exec_fft(zout_forward, inverse=True),
                                err_msg='File-ID: {}'.format(self.lib.id),
                                rtol=1e-12)
        if self.sync:
            assert err == 0

    def test_asl_fftw_f(self):
        self._helper(_test_asl_fftw_f, '/opt/nec/ve/bin/nfort', 'test_complex_1d_',
                     (uint64, uint64, uint64, int64), int32,
                     ext_cflags=('-fdefault-integer=8',))
        NX, zin, zout_forward, zout_backward = self._prep()
        err = self.kern(zin.ve_adr, zout_forward.ve_adr, zout_backward.ve_adr, NX,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(zout_forward,
                                self._exec_fft(zin, inverse=False),
                                err_msg='File-ID: {}'.format(self.lib.id),
                                rtol=1e-12)
        testing.assert_allclose(zout_backward,
                                self._exec_fft(zout_forward, inverse=True),
                                err_msg='File-ID: {}'.format(self.lib.id),
                                rtol=1e-12)
        if self.sync:
            assert err == 0

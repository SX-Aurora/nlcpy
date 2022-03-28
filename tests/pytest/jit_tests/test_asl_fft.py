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
from nlcpy.ve_types import (int32, int64, uint64)

DIST_DIR = None
# DIST_DIR = 'jit_cache'

LOG_STREAM = None
# import sys
# LOG_STREAM = sys.stdout


_test_asl_fft_c = r'''
#include <asl.h>
#include <complex.h>

int fft_complex_1d_multi_s(float complex *zin, float complex *zout_forward,
                           float complex *zout_backward, int64_t nx, int64_t m) {
    asl_fft_t fft;
    /* DFT Preparation */
    asl_fft_create_complex_1d_s(&fft, nx);
    asl_fft_set_spatial_stride_1d(fft, 1);
    asl_fft_set_frequency_stride_1d(fft, 1);
    asl_fft_set_multiplicity(fft, m);
    asl_fft_set_spatial_multiplicity_stride(fft, nx);
    asl_fft_set_frequency_multiplicity_stride(fft, nx);
    /* DFT Execution (forward) */
    asl_fft_execute_complex_forward_s(fft, zin, zout_forward);
    /* DFT Execution (backward) */
    asl_fft_execute_complex_backward_s(fft, zout_forward, zout_backward);
    /* DFT Finalization */
    asl_fft_destroy(fft);
    return 0;
}
'''

_test_asl_fft_cpp = r'''
#include <asl.h>
#include <complex>

int cppfft_complex_1d_multi_s(std::complex<float> *zin,
                              std::complex<float> *zout_forward,
                              std::complex<float> *zout_backward,
                              int64_t nx, int64_t m) {
    asl_fft_t fft;
    /* DFT Preparation */
    asl_fft_create_complex_1d_s(&fft, nx);
    asl_fft_set_spatial_stride_1d(fft, 1);
    asl_fft_set_frequency_stride_1d(fft, 1);
    asl_fft_set_multiplicity(fft, m);
    asl_fft_set_spatial_multiplicity_stride(fft, nx);
    asl_fft_set_frequency_multiplicity_stride(fft, nx);
    /* DFT Execution (forward) */
    asl_fft_execute_complex_forward_s(fft, zin, zout_forward);
    /* DFT Execution (backward) */
    asl_fft_execute_complex_backward_s(fft, zout_forward, zout_backward);
    /* DFT Finalization */
    asl_fft_destroy(fft);
    return 0;
}

extern "C" {
int fft_complex_1d_multi_s(std::complex<float> *zin, std::complex<float> *zout_forward,
                           std::complex<float> *zout_backward, int64_t nx, int64_t m) {
    return cppfft_complex_1d_multi_s(zin, zout_forward, zout_backward, nx, m);
}}
'''

_test_asl_fft_f = r'''
function fft_complex_1d_multi_s(zin, zout_forward, zout_backward, nx, m)
    use asl_unified
    implicit none
    integer(kind=8), value :: nx, m
    integer(kind=8) :: fft
    integer(kind=4) :: fft_complex_1d_multi_s
    complex(kind=4) :: zin(m, nx), zout_forward(m, nx), zout_backward(m, nx)
    ! DFT Preparation
    call asl_fft_create_complex_1d_s(fft, nx)
    call asl_fft_set_spatial_stride_1d(fft, 1)
    call asl_fft_set_frequency_stride_1d(fft, 1)
    call asl_fft_set_multiplicity(fft, m)
    call asl_fft_set_spatial_multiplicity_stride(fft, nx)
    call asl_fft_set_frequency_multiplicity_stride(fft, nx)
    ! DFT Execution (forward)
    call asl_fft_execute_complex_forward_s(fft, zin, zout_forward)
    ! DFT Execution (backward)
    call asl_fft_execute_complex_backward_s(fft, zout_forward, zout_backward)
    ! DFT Finalization
    call asl_fft_destroy(fft)
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
class TestAslFft(unittest.TestCase):

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
        NX = 10
        M = 3
        zin = nlcpy.empty((M, NX), dtype='c8')
        zout_forward = nlcpy.empty((M, NX), dtype='c8')
        zout_backward = nlcpy.empty((M, NX), dtype='c8')
        for im in range(M):
            for ix in range(NX):
                zin[im, ix] = (ix + im + 1) + 1j * ((ix + 1) * (im + 1))
        return zin, zout_forward, zout_backward, NX, M

    def _exec_fft(self, zin, inverse=False):
        if inverse:
            return nlcpy.fft.ifft(zin) * zin.shape[-1]
        else:
            return nlcpy.fft.fft(zin)

    def test_asl_fft_c(self):
        self._helper(_test_asl_fft_c, '/opt/nec/ve/bin/ncc', 'fft_complex_1d_multi_s',
                     (uint64, uint64, uint64, int64, int64), int32)
        zin, zout_forward, zout_backward, nx, m = self._prep()
        err = self.kern(zin.ve_adr, zout_forward.ve_adr, zout_backward.ve_adr,
                        nx, m, sync=self.sync, callback=self.callback)
        testing.assert_array_equal(zout_forward,
                                   self._exec_fft(zin, inverse=False),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        testing.assert_array_equal(zout_backward,
                                   self._exec_fft(zout_forward, inverse=True),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_asl_fft_cpp(self):
        self._helper(_test_asl_fft_cpp, '/opt/nec/ve/bin/nc++', 'fft_complex_1d_multi_s',
                     (uint64, uint64, uint64, int64, int64), int32)
        zin, zout_forward, zout_backward, nx, m = self._prep()
        err = self.kern(zin.ve_adr, zout_forward.ve_adr, zout_backward.ve_adr,
                        nx, m, sync=self.sync, callback=self.callback)
        testing.assert_array_equal(zout_forward,
                                   self._exec_fft(zin, inverse=False),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        testing.assert_array_equal(zout_backward,
                                   self._exec_fft(zout_forward, inverse=True),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_asl_fft_f(self):
        self._helper(_test_asl_fft_f, '/opt/nec/ve/bin/nfort', 'fft_complex_1d_multi_s_',
                     (uint64, uint64, uint64, int64, int64), int32)
        zin, zout_forward, zout_backward, nx, m = self._prep()
        err = self.kern(zin.ve_adr, zout_forward.ve_adr, zout_backward.ve_adr,
                        nx, m, sync=self.sync, callback=self.callback)
        testing.assert_array_equal(zout_forward,
                                   self._exec_fft(zin, inverse=False),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        testing.assert_array_equal(zout_backward,
                                   self._exec_fft(zout_forward, inverse=True),
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

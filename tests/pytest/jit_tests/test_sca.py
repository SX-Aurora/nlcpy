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
from nlcpy.ve_types import (void, int32, int64, uint64)

DIST_DIR = None
# DIST_DIR = 'jit_cache'

LOG_STREAM = None
# import sys
# LOG_STREAM = sys.stdout

_test_sca_c = r'''
#include <sca.h>
void create_image(sca_int_t nx, sca_int_t ny, float *img) {
    for (sca_int_t iy = 0; iy < ny; iy++) {
    for (sca_int_t ix = 0; ix < nx; ix++) {
        const float rx = (float)ix / (nx - 1) - 0.5f;
        const float ry = (float)iy / (ny - 1) - 0.5f;
        const float rr = rx * rx + ry * ry;
        img[ix + nx * iy] = ((rr < 0.5f * 0.5f && rr > 0.3f * 0.3f) ||
                             rr < 0.1f * 0.1f) ? 1.0f : 0.0f;
    }
    }
}
int sca_filtering_s(const float *flt, float *img, float *wrk,
                    sca_int_t NX, sca_int_t NY) {
    sca_stencil_t sten;
    sca_code_t code;
    sca_stencil_create_s(&sten);
    sca_stencil_append_elements_xy_s(sten, -2, 2, -2, 2, 1, NX + 4, NY + 4, 0,
                                     wrk + (2 + (NX + 4) * 2), flt);
    sca_stencil_set_output_array_s(sten, 1, NX, NY, 0, img);
    sca_code_create(&code, sten, NX, NY, 1, 1);
    sca_stencil_destroy(sten);
    sca_code_execute(code);
    sca_code_destroy(code);
    return 0;
}
'''

_test_sca_cpp = r'''
#include <sca.h>
extern "C" {
void create_image(sca_int_t nx, sca_int_t ny, float *img) {
    for (sca_int_t iy = 0; iy < ny; iy++) {
    for (sca_int_t ix = 0; ix < nx; ix++) {
        const float rx = (float)ix / (nx - 1) - 0.5f;
        const float ry = (float)iy / (ny - 1) - 0.5f;
        const float rr = rx * rx + ry * ry;
        img[ix + nx * iy] = ((rr < 0.5f * 0.5f && rr > 0.3f * 0.3f) ||
                             rr < 0.1f * 0.1f) ? 1.0f : 0.0f;
    }
    }
}}
int cppsca_filtering_s(const float *flt, float *img, float *wrk,
                       sca_int_t NX, sca_int_t NY) {
    sca_stencil_t sten;
    sca_code_t code;
    sca_stencil_create_s(&sten);
    sca_stencil_append_elements_xy_s(sten, -2, 2, -2, 2, 1, NX + 4, NY + 4, 0,
                                     wrk + (2 + (NX + 4) * 2), flt);
    sca_stencil_set_output_array_s(sten, 1, NX, NY, 0, img);
    sca_code_create(&code, sten, NX, NY, 1, 1);
    sca_stencil_destroy(sten);
    sca_code_execute(code);
    sca_code_destroy(code);
    return 0;
}

extern "C" {
int sca_filtering_s(const float *flt, float *img, float *wrk,
                       sca_int_t NX, sca_int_t NY) {
    return cppsca_filtering_s(flt, img, wrk, NX, NY);
}}
'''

_test_sca_f = r'''
subroutine create_image(nx, ny, img) bind(c)
    integer, value, intent(in) :: nx, ny
    real(4), intent(out) :: img(nx,ny)
    real(4) :: rx, ry, rr
    integer :: ix, iy
    do iy = 1, ny
    do ix = 1, nx
       rx = real(ix - 1, 4) / real(nx - 1, 4) - 0.5
       ry = real(iy - 1, 4) / real(ny - 1, 4) - 0.5
       rr = rx * rx + ry * ry
       if ((rr < 0.5 * 0.5 .and. rr > 0.3 * 0.3) .or. rr < 0.1 * 0.1) then
          img(ix,iy) = 1.0
       else
          img(ix,iy) = 0.0
       end if
    end do
    end do
end subroutine create_image

function sca_filtering_s(flt, img, wrk, NX, NY)
    use sca
    implicit none
    integer(kind=8), value :: NX, NY
    real(kind=4) :: flt(25), img(NX, NY), wrk(-1:NX+2, -1:NY+2)
    integer(kind=8) :: sten, code
    integer(kind=4) :: sca_filtering_s
    call sca_stencil_create_s(sten)
    call sca_stencil_append_elements_xy_s(sten, -2, 2, -2, 2, 1, NX + 4, NY + 4, 0, &
                                         &wrk(1, 1), flt)
    call sca_stencil_set_output_array_s(sten, 1, NX, NY, 0, img)
    call sca_code_create(code, sten, NX, NY, 1, 1)
    call sca_stencil_destroy(sten)
    call sca_code_execute(code)
    call sca_code_destroy(code)
    sca_filtering_s = 0
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
class TestSca(unittest.TestCase):

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
        NX = 32
        NY = 32
        flt = nlcpy.array([
            1.0 / 256,  4.0 / 256,  6.0 / 256,  4.0 / 256, 1.0 / 256,  # NOQA
            4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256,  # NOQA
            6.0 / 256, 24.0 / 256, 36.0 / 256, 24.0 / 256, 6.0 / 256,  # NOQA
            4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256,  # NOQA
            1.0 / 256,  4.0 / 256,  6.0 / 256,  4.0 / 256, 1.0 / 256   # NOQA
        ], dtype='f4')  # Gaussian Filter
        img = nlcpy.empty((NY, NX), dtype='f4')  # output
        wrk = nlcpy.empty((NY + 4, NX + 4), dtype='f4')  # input
        self.lib.get_function(
            'create_image',
            args_type=(int64, int64, uint64),
            ret_type=void
        )(NX, NY, img.ve_adr)
        wrk[2:-2, 2:-2] = img
        return NX, NY, flt, img, wrk

    def _exec_sca(self, wrk, flt):
        dwrk = nlcpy.sca.create_descriptor(wrk)
        desc = nlcpy.sca.empty_description()
        cnt = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                desc += dwrk[i, j] * flt[cnt]
                cnt += 1
        return nlcpy.sca.create_kernel(desc).execute()

    def test_sca_c(self):
        self._helper(_test_sca_c, '/opt/nec/ve/bin/ncc', 'sca_filtering_s',
                     (uint64, uint64, uint64, int64, int64), int32)
        NX, NY, flt, img, wrk = self._prep()
        err = self.kern(flt.ve_adr, img.ve_adr, wrk.ve_adr, NX, NY,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(img, self._exec_sca(wrk, flt)[2:-2, 2:-2],
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_sca_cpp(self):
        self._helper(_test_sca_cpp, '/opt/nec/ve/bin/nc++', 'sca_filtering_s',
                     (uint64, uint64, uint64, int64, int64), int32)
        NX, NY, flt, img, wrk = self._prep()
        err = self.kern(flt.ve_adr, img.ve_adr, wrk.ve_adr, NX, NY,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(img, self._exec_sca(wrk, flt)[2:-2, 2:-2],
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_sca_f(self):
        self._helper(_test_sca_f, '/opt/nec/ve/bin/nfort', 'sca_filtering_s_',
                     (uint64, uint64, uint64, int64, int64), int32)
        NX, NY, flt, img, wrk = self._prep()
        err = self.kern(flt.ve_adr, img.ve_adr, wrk.ve_adr, NX, NY,
                        sync=self.sync, callback=self.callback)
        testing.assert_array_equal(img, self._exec_sca(wrk, flt)[2:-2, 2:-2],
                                   err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

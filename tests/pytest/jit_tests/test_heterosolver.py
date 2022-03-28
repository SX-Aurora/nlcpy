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
from nlcpy import veo
from nlcpy import testing
from nlcpy.ve_types import (void_p, int64)

DIST_DIR = None
# DIST_DIR = 'jit_cache'

LOG_STREAM = None
# import sys
# LOG_STREAM = sys.stdout

_test_heterosolver_c = r'''
#include <stdlib.h>
#include <omp.h>
#include <heterosolver.h>
#define   MROW   7   /* The number of rows of matrix A */
#define   NCOL   7   /* The number of columns of matrix A */
#define   NRHS   1   /* The number of right-hand side vectors */
HS_int_t HS_csr_unsym_ind_0(double *x)
{
  /*             Matrix A                      X        B
     [ 1.1, 1.2,   0,   0,   0,   0,   0 ]   [1.0]   [  3.5]
     [   0, 2.2, 2.3,   0,   0,   0,   0 ]   [2.0]   [ 11.3]
     [   0,   0, 3.3, 3.4, 3.5,   0, 3.7 ]   [3.0]   [ 66.9]
     [ 4.1,   0,   0, 4.4,   0,   0,   0 ] * [4.0] = [ 21.7]
     [   0,   0, 5.3,   0, 5.5, 5.6,   0 ]   [5.0]   [ 77.0]
     [   0,   0,   0,   0, 6.5, 6.6, 6.7 ]   [6.0]   [119.0]
     [   0,   0, 7.3,   0,   0, 7.6, 7.7 ]   [7.0]   [121.4]
   */
   /* Values of the nonzero elements of A in row-major order */
   double aval[19] = {
                      1.1, 1.2, 2.2, 2.3, 3.3,
                      3.4, 3.5, 3.7, 4.1, 4.4,
                      5.3, 5.5, 5.6, 6.5, 6.6,
                      6.7, 7.3, 7.6, 7.7
                     };
   /* Column indices of nonzero elements */
   HS_int_t iaind[19] = {
                      0, 1, 1, 2, 2,
                      3, 4, 6, 0, 3,
                      2, 4, 5, 4, 5,
                      6, 2, 5, 6
                     };
   /* Starting points of the rows of the arrays aval and iaind */
   HS_int_t iaptr[MROW+1] = {0, 2, 4, 8, 10, 13, 16, 19};
   /* The right-hand side vector */
   double b[MROW] = {3.5, 11.3, 66.9, 21.7, 77.0, 119.0, 121.4};
   /* The target accuracy for the residual norm */
   double res = 1.0e-13;
   /* Local variables */
   int ierr;
   /* Handle Initialization */
   HS_handle_t hnd;
   ierr = HS_init_handle(&hnd, MROW, NCOL, HS_UNSYMMETRIC, HS_CSR);
   /* Specifying the number of OpenMP threads */
   omp_set_num_threads(8);
   /* Preprocessing Phase */
   ierr = HS_preprocess_rd(hnd, iaptr, iaind, aval);
   /* Numeric Factorization Phase */
   ierr = HS_factorize_rd(hnd, iaptr, iaind, aval);

   /* Solution Phase */
   ierr = HS_solve_rd(hnd, iaptr, iaind, aval, NRHS, b, x, &res);
   /* Handle Finalization */
   ierr = HS_finalize_handle(hnd);
   return ierr;
}
'''

_test_heterosolver_cpp = r'''
#include <stdlib.h>
#include <omp.h>
#include <heterosolver.h>
#define   MROW   7   /* The number of rows of matrix A */
#define   NCOL   7   /* The number of columns of matrix A */
#define   NRHS   1   /* The number of right-hand side vectors */
HS_int_t cppHS_csr_unsym_ind_0(double *x)
{
  /*             Matrix A                      X        B
     [ 1.1, 1.2,   0,   0,   0,   0,   0 ]   [1.0]   [  3.5]
     [   0, 2.2, 2.3,   0,   0,   0,   0 ]   [2.0]   [ 11.3]
     [   0,   0, 3.3, 3.4, 3.5,   0, 3.7 ]   [3.0]   [ 66.9]
     [ 4.1,   0,   0, 4.4,   0,   0,   0 ] * [4.0] = [ 21.7]
     [   0,   0, 5.3,   0, 5.5, 5.6,   0 ]   [5.0]   [ 77.0]
     [   0,   0,   0,   0, 6.5, 6.6, 6.7 ]   [6.0]   [119.0]
     [   0,   0, 7.3,   0,   0, 7.6, 7.7 ]   [7.0]   [121.4]
   */
   /* Values of the nonzero elements of A in row-major order */
   double aval[19] = {
                      1.1, 1.2, 2.2, 2.3, 3.3,
                      3.4, 3.5, 3.7, 4.1, 4.4,
                      5.3, 5.5, 5.6, 6.5, 6.6,
                      6.7, 7.3, 7.6, 7.7
                     };
   /* Column indices of nonzero elements */
   HS_int_t iaind[19] = {
                      0, 1, 1, 2, 2,
                      3, 4, 6, 0, 3,
                      2, 4, 5, 4, 5,
                      6, 2, 5, 6
                     };
   /* Starting points of the rows of the arrays aval and iaind */
   HS_int_t iaptr[MROW+1] = {0, 2, 4, 8, 10, 13, 16, 19};
   /* The right-hand side vector */
   double b[MROW] = {3.5, 11.3, 66.9, 21.7, 77.0, 119.0, 121.4};
   /* The target accuracy for the residual norm */
   double res = 1.0e-13;
   /* Local variables */
   int ierr;
   /* Handle Initialization */
   HS_handle_t hnd;
   ierr = HS_init_handle(&hnd, MROW, NCOL, HS_UNSYMMETRIC, HS_CSR);
   /* Specifying the number of OpenMP threads */
   omp_set_num_threads(8);
   /* Preprocessing Phase */
   ierr = HS_preprocess_rd(hnd, iaptr, iaind, aval);
   /* Numeric Factorization Phase */
   ierr = HS_factorize_rd(hnd, iaptr, iaind, aval);

   /* Solution Phase */
   ierr = HS_solve_rd(hnd, iaptr, iaind, aval, NRHS, b, x, &res);
   /* Handle Finalization */
   ierr = HS_finalize_handle(hnd);
   return ierr;
}

extern "C" {
HS_int_t HS_csr_unsym_ind_0(double *x) {
    return cppHS_csr_unsym_ind_0(x);
}}
'''

_test_heterosolver_f = r'''
function hs_csr_unsym_ind_0(x)
    use omp_lib
    use heterosolver
    implicit none
    integer :: hs_csr_unsym_ind_0
    !             Matrix A                      X        B
    ! [ 1.1, 1.2,   0,   0,   0,   0,   0 ]   [1.0]   [  3.5]
    ! [   0, 2.2, 2.3,   0,   0,   0,   0 ]   [2.0]   [ 11.3]
    ! [   0,   0, 3.3, 3.4, 3.5,   0, 3.7 ]   [3.0]   [ 66.9]
    ! [ 4.1,   0,   0, 4.4,   0,   0,   0 ] * [4.0] = [ 21.7]
    ! [   0,   0, 5.3,   0, 5.5, 5.6,   0 ]   [5.0]   [ 77.0]
    ! [   0,   0,   0,   0, 6.5, 6.6, 6.7 ]   [6.0]   [119.0]
    ! [   0,   0, 7.3,   0,   0, 7.6, 7.7 ]   [7.0]   [121.4]
    ! The number of rows of A
    integer, parameter ::  mrow=7
    ! The number of columns of A
    integer, parameter ::  ncol=7

    ! Values of the nonzero elements of A in row-major order
    real(8), dimension(19) :: aval = (/ &
       1.1d0, 1.2d0, 2.2d0, 2.3d0, 3.3d0, &
       3.4d0, 3.5d0, 3.7d0, 4.1d0, 4.4d0, &
       5.3d0, 5.5d0, 5.6d0, 6.5d0, 6.6d0, &
       6.7d0, 7.3d0, 7.6d0, 7.7d0 &
       /)
    ! Column indices of nonzero elements
    integer, dimension(19) :: iaind = (/ &
       0, 1, 1, 2, 2, &
       3, 4, 6, 0, 3, &
       2, 4, 5, 4, 5, &
       6, 2, 5, 6 &
       /)
    ! Starting points of the rows of the arrays aval and iaind
    integer, dimension(mrow+1) :: iaptr = (/ 0, 2, 4, 8, 10, 13, 16, 19 /)
    ! The number of right-hand side vectors
    integer :: nrhs = 1
    ! The right-hand side vector b
    real(8), dimension(mrow) :: b = (/ 3.5d0, 11.3d0, 66.9d0, 21.7d0, 77.0d0, &
                                   & 119.0d0, 121.4d0 /)

    ! The solution vector x
    real(8), dimension(ncol) :: x
    ! The target accuracy for the residual norm
    real(8) :: res = 1.0e-13
    ! Local variables
    integer    :: ierr, i, j, lx
    ! Handle Initialization
    integer(8) :: hnd
    call hs_init_handle(hnd, mrow, ncol, HS_UNSYMMETRIC, HS_CSR, ierr)
    ! Specifying the number of OpenMP threads
    call omp_set_num_threads(8)

    ! Specifying Zero-based indexing
    call hs_set_option(hnd, HS_INDEXING, HS_INDEXING_0, ierr)

    ! Preprocessing Phase
    call hs_preprocess_rd(hnd, iaptr, iaind, aval, ierr)

    ! Numeric Factorization Phase
    call hs_factorize_rd(hnd, iaptr, iaind, aval, ierr);

    ! Solution Phase
    call hs_solve_rd(hnd, iaptr, iaind, aval, nrhs, b, x, res, ierr)

    ! Handle Finalization
    call hs_finalize_handle(hnd,ierr)
    hs_csr_unsym_ind_0 = 0
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
class TestHeteroSolver(unittest.TestCase):

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
        x = numpy.empty(7, dtype='f8')
        return x

    def _make_ref(self):
        return [1., 2., 3., 4., 5., 6., 7.]

    def test_heterosolver_c(self):
        self._helper(_test_heterosolver_c, '/opt/nec/ve/bin/ncc',
                     'HS_csr_unsym_ind_0',
                     (void_p,), int64)
        x = self._prep()
        err = self.kern(veo.OnStack(x, inout=veo.INTENT_OUT),
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(x, self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_heterosolver_cpp(self):
        self._helper(_test_heterosolver_cpp, '/opt/nec/ve/bin/nc++',
                     'HS_csr_unsym_ind_0',
                     (void_p,), int64)
        x = self._prep()
        err = self.kern(veo.OnStack(x, inout=veo.INTENT_OUT),
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(x, self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_heterosolver_f(self):
        self._helper(_test_heterosolver_f, '/opt/nec/ve/bin/nfort',
                     'hs_csr_unsym_ind_0_',
                     (void_p,), int64,
                     ('-fdefault-integer=8',))
        x = self._prep()
        err = self.kern(veo.OnStack(x, inout=veo.INTENT_OUT),
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(x, self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

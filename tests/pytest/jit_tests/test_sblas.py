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
import numpy
from nlcpy import veo
from nlcpy import testing
from nlcpy.ve_types import (void_p, int64)

DIST_DIR = None
# DIST_DIR = 'jit_cache'

LOG_STREAM = None
# import sys
# LOG_STREAM = sys.stdout

_test_sblas_c = r'''
#include <stdlib.h>
#include <omp.h>
#include <sblas.h>
#define   MROW   5   /* The number of rows of matrix A */
#define   NCOL   7   /* The number of columns of matrix A */
#define   NNZ   13   /* The number of non-zero entries of matrix A */
sblas_int_t sblas_mv_csr_ind_0(double *y)
{
    /*             Matrix A                      X       Y
     [ 1.1  1.2    0    0    0    0    0 ]   [1.0]   [  3.5]
     [   0  2.2  2.3    0    0    0    0 ]   [2.0]   [ 11.3]
     [   0    0  3.3  3.4  3.5    0  3.7 ]   [3.0]   [ 66.9]
     [ 4.1    0    0  4.4    0    0    0 ] * [4.0] = [ 21.7]
     [   0    0  5.3    0  5.5  5.6    0 ]   [5.0]   [ 77.0]
                                             [6.0]
                                             [7.0]
    */

    /* Values of the non-zero entries of A in row-major order */
    double aval[NNZ] = {
                        1.1, 1.2, 2.2, 2.3, 3.3,
                        3.4, 3.5, 3.7, 4.1, 4.4,
                        5.3, 5.5, 5.6
                       };
    /* Column indices of non-zero entries */
    sblas_int_t iaind[NNZ] = {
                              0, 1, 1, 2, 2,
                              3, 4, 6, 0, 3,
                              2, 4, 5
                             };
    /* Starting points of the rows of the arrays aval and iaind */
    sblas_int_t iaptr[MROW+1] = {0, 2, 4, 8, 10, 13};
    /* The vector X */
    double x[NCOL] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    /* Error indicator */
    int ierr;
    /* Set the number of OpenMP threads */
    omp_set_num_threads(8);
    /* Creation of a handle from CSR format */
    sblas_handle_t a;
    ierr = sblas_create_matrix_handle_from_csr_rd(
        MROW, NCOL, iaptr, iaind, aval, SBLAS_INDEXING_0, SBLAS_GENERAL, &a);
    /* Analysis of the sparse matrix A */
    ierr = sblas_analyze_mv_rd(SBLAS_NON_TRANSPOSE, a);
    const double alpha = 1.0;
    const double beta = 0.0;
    /* Matrix-vector multiplication Y = A * X */
    ierr = sblas_execute_mv_rd(SBLAS_NON_TRANSPOSE, a, alpha, x, beta, y);
    /* Destruction of the handle */
    ierr = sblas_destroy_matrix_handle(a);
    return ierr;
}
'''

_test_sblas_cpp = r'''
#include <stdlib.h>
#include <omp.h>
#include <sblas.h>
#define   MROW   5   /* The number of rows of matrix A */
#define   NCOL   7   /* The number of columns of matrix A */
#define   NNZ   13   /* The number of non-zero entries of matrix A */
sblas_int_t cppsblas_mv_csr_ind_0(double *y)
{
    /*             Matrix A                      X       Y
     [ 1.1  1.2    0    0    0    0    0 ]   [1.0]   [  3.5]
     [   0  2.2  2.3    0    0    0    0 ]   [2.0]   [ 11.3]
     [   0    0  3.3  3.4  3.5    0  3.7 ]   [3.0]   [ 66.9]
     [ 4.1    0    0  4.4    0    0    0 ] * [4.0] = [ 21.7]
     [   0    0  5.3    0  5.5  5.6    0 ]   [5.0]   [ 77.0]
                                             [6.0]
                                             [7.0]
    */

    /* Values of the non-zero entries of A in row-major order */
    double aval[NNZ] = {
                        1.1, 1.2, 2.2, 2.3, 3.3,
                        3.4, 3.5, 3.7, 4.1, 4.4,
                        5.3, 5.5, 5.6
                       };
    /* Column indices of non-zero entries */
    sblas_int_t iaind[NNZ] = {
                              0, 1, 1, 2, 2,
                              3, 4, 6, 0, 3,
                              2, 4, 5
                             };
    /* Starting points of the rows of the arrays aval and iaind */
    sblas_int_t iaptr[MROW+1] = {0, 2, 4, 8, 10, 13};
    /* The vector X */
    double x[NCOL] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    /* Error indicator */
    int ierr;
    /* Set the number of OpenMP threads */
    omp_set_num_threads(8);
    /* Creation of a handle from CSR format */
    sblas_handle_t a;
    ierr = sblas_create_matrix_handle_from_csr_rd(
        MROW, NCOL, iaptr, iaind, aval, SBLAS_INDEXING_0, SBLAS_GENERAL, &a);
    /* Analysis of the sparse matrix A */
    ierr = sblas_analyze_mv_rd(SBLAS_NON_TRANSPOSE, a);
    const double alpha = 1.0;
    const double beta = 0.0;
    /* Matrix-vector multiplication Y = A * X */
    ierr = sblas_execute_mv_rd(SBLAS_NON_TRANSPOSE, a, alpha, x, beta, y);
    /* Destruction of the handle */
    ierr = sblas_destroy_matrix_handle(a);
    return ierr;
}

extern "C" {
sblas_int_t sblas_mv_csr_ind_0(double *y) {
    return cppsblas_mv_csr_ind_0(y);
}}
'''

_test_sblas_f = r'''
function sblas_mv_csr_ind_0(y)
  use omp_lib
  use sblas
  implicit none
  integer :: sblas_mv_csr_ind_0
  integer ,parameter :: mrow = 5   ! The number of rows of matrix A
  integer ,parameter :: ncol = 7   ! The number of columns of matrix A
  integer, parameter :: nnz = 13   ! The number of non-zero entries of matrix A
!                 Matrix A                      X       Y
!    [ 1.1  1.2    0    0    0    0    0 ]   [1.0]   [  3.5]
!    [   0  2.2  2.3    0    0    0    0 ]   [2.0]   [ 11.3]
!    [   0    0  3.3  3.4  3.5    0  3.7 ]   [3.0]   [ 66.9]
!    [ 4.1    0    0  4.4    0    0    0 ] * [4.0] = [ 21.7]
!    [   0    0  5.3    0  5.5  5.6    0 ]   [5.0]   [ 77.0]
!                                            [6.0]
!                                            [7.0]
!     Values of the nonzero entries of A in row-major order
  real(8) :: aval(nnz) =  (/1.1d0, 1.2d0, 2.2d0, 2.3d0, 3.3d0, &
                            3.4d0, 3.5d0, 3.7d0, 4.1d0, 4.4d0, &
                            5.3d0, 5.5d0, 5.6d0 /)
!     Column indices of nonzero entries
  integer :: iaind(nnz) = (/ 0, 1, 1, 2, 2, &
                             3, 4, 6, 0, 3, &
                             2, 4, 5 /)
! Starting points of the rows of the arrays aval and iaind
  integer :: iaptr(mrow+1) = (/ 0, 2, 4, 8, 10, 13 /)
! The vector X
  real(8) :: x(ncol) = (/ 1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0, 6.0d0, 7.0d0 /)
! The vector Y to be computed
  real(8) y(mrow)
! Handle
  integer(8) a
! Error indicator
  integer ierr
  real(8),parameter :: alpha = 1.0
  real(8),parameter :: beta = 0.0
! Set the number of OpenMP threads
  call omp_set_num_threads(8)
! Creation of a handle from CSR format
  call sblas_create_matrix_handle_from_csr_rd(mrow, ncol, iaptr, iaind, aval, &
                                              SBLAS_INDEXING_0, SBLAS_GENERAL, a, ierr)
! Analysis of the sparse matrix A
  call sblas_analyze_mv_rd(SBLAS_NON_TRANSPOSE, a, ierr)
! Matrix-vector multiplication Y = A * X
  call sblas_execute_mv_rd(SBLAS_NON_TRANSPOSE, a, alpha, x, beta, y, ierr)
! Destruction of the handle
  call sblas_destroy_matrix_handle(a,ierr)
  sblas_mv_csr_ind_0 = 0
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
class TestSblas(unittest.TestCase):

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
        y = numpy.empty(5, dtype='f8')
        return y

    def _make_ref(self):
        return [3.5, 11.3, 66.9, 21.7, 77.0]

    def test_sblas_c(self):
        self._helper(_test_sblas_c, '/opt/nec/ve/bin/ncc', 'sblas_mv_csr_ind_0',
                     (void_p,), int64)
        y = self._prep()
        err = self.kern(veo.OnStack(y, inout=veo.INTENT_OUT),
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_sblas_cpp(self):
        self._helper(_test_sblas_cpp, '/opt/nec/ve/bin/nc++', 'sblas_mv_csr_ind_0',
                     (void_p,), int64)
        y = self._prep()
        err = self.kern(veo.OnStack(y, inout=veo.INTENT_OUT),
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_sblas_f(self):
        self._helper(_test_sblas_f, '/opt/nec/ve/bin/nfort', 'sblas_mv_csr_ind_0_',
                     (void_p,), int64,
                     ('-fdefault-integer=8',))
        y = self._prep()
        err = self.kern(veo.OnStack(y, inout=veo.INTENT_OUT),
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, self._make_ref(), rtol=1e-12,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

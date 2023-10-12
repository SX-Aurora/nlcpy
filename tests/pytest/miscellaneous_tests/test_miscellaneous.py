#
# * The source code in this file is based on the soure code of CuPy.
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

import pytest
import unittest
import sys
import io
import os

import nlcpy


class Capture:

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._str = io.StringIO()  # redirect
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout  # restore

    def get_str(self):
        return self._str.getvalue().splitlines()


class TestShowConfig(unittest.TestCase):

    def test_show_config(self):

        with Capture() as c:
            nlcpy.show_config()
            _str = ''.join(c.get_str())
            assert 'OS' in _str
            assert 'Python Version' in _str
            assert 'NLC Library Path' in _str
            assert 'NLCPy Kernel Path' in _str
            assert 'NLCPy Version' in _str
            assert 'NumPy Version' in _str
            assert 'ncc Build Version' in _str
            assert 'VEO API Version' in _str
            assert 'VEO Version' in _str
            assert 'Assigned VE IDs' in _str
            assert 'VE Arch' in _str
            assert 'VE ncore' in _str
            assert 'VE Total Mem[KB]' in _str
            assert 'VE Used  Mem[KB]' in _str
            assert 'None' not in _str


class TestEnvironment(unittest.TestCase):

    def test_get_nlc_path(self):
        ld_path = os.environ.pop('VE_LD_LIBRARY_PATH', None)
        assert isinstance(nlcpy.__config__.get_nlc_lib_path(1), str)
        assert isinstance(nlcpy.__config__.get_nlc_lib_path(3), str)
        assert nlcpy.__config__.get_nlc_lib_path(2) is None
        if ld_path:
            os.environ['VE_LD_LIBRARY_PATH'] = ld_path

    def test_get_ncc_build_ver(self):
        build_ver_ve1 = nlcpy.__config__.get_ncc_build_ver(1)
        if build_ver_ve1:
            assert isinstance(build_ver_ve1, str)
        build_ver_ve3 = nlcpy.__config__.get_ncc_build_ver(3)
        if build_ver_ve3:
            assert isinstance(build_ver_ve3, str)
        with pytest.raises(RuntimeError):
            nlcpy.__config__.get_ncc_build_ver(2)

    def test_set_ld_preload(self):
        def _ld_preload_helper(arch):
            _ld_preload = os.environ.pop('VE_LD_PRELOAD', None)
            nlcpy._environment._set_ve_ld_preload(arch)
            ld_preload = os.environ['VE_LD_PRELOAD'].split(':')[0]
            assert 'libncc.so' in ld_preload
            if _ld_preload:
                os.environ['VE_LD_PRELOAD'] = _ld_preload
            else:
                os.environ.pop('VE_LD_PRELOAD', None)

        _ld_preload_helper(1)
        _ld_preload_helper(3)
        with pytest.raises(ValueError):
            nlcpy._environment._set_ve_ld_preload(2)

    def test_get_pool_size_invalid(self):
        _ps = os.environ.pop('VE_NLCPY_MEMPOOL_SIZE', None)
        os.environ['VE_NLCPY_MEMPOOL_SIZE'] = '1T'
        with pytest.raises(ValueError):
            nlcpy._environment._get_pool_size()
        if _ps:
            os.environ['VE_NLCPY_MEMPOOL_SIZE'] = _ps
        else:
            os.environ.pop('VE_NLCPY_MEMPOOL_SIZE', None)

    def test_is_mpi_not_decimal(self):
        _mpirank = os.environ.pop('MPIRANK', None)
        os.environ['MPIRANK'] = 'a'
        assert nlcpy._environment._is_mpi() is False
        if _mpirank:
            os.environ['MPIRANK'] = _mpirank
        else:
            os.environ.pop('MPIRANK', None)

    def test_get_mpi_local_size(self):
        _local_size = os.environ.pop('_MPI4PYVE_MPI_LOCAL_SIZE', None)
        _mpisize = os.environ.pop('MPISIZE', None)

        with pytest.raises(ValueError):
            nlcpy._environment._get_mpi_local_size()

        os.environ['MPISIZE'] = '2'
        assert nlcpy._environment._get_mpi_local_size() == 2

        if _local_size:
            os.environ['_MPI4PYVE_MPI_LOCAL_SIZE'] = _local_size
        if _mpisize:
            os.environ['MPISIZE'] = _mpisize
        else:
            os.environ.pop('MPISIZE', None)

    def test_get_nlc_home(self):
        _nlc_home = os.environ.pop('NLC_HOME', None)
        assert '/opt/nec/ve' in nlcpy._environment._get_nlc_home(1)
        assert '/opt/nec/ve3' in nlcpy._environment._get_nlc_home(3)
        with pytest.raises(ValueError):
            nlcpy._environment._get_nlc_home(2)
        if _nlc_home:
            os.environ['NLC_HOME'] = _nlc_home


class TestLibPath(unittest.TestCase):

    def test_lib_path(self):
        lib_ve1 = nlcpy._path.LibPath(1)
        lib_ve3 = nlcpy._path.LibPath(3)
        with pytest.raises(RuntimeError):
            nlcpy._path.LibPath(2)

        lib_common = 'libnlcpy_ve_kernel_common.so'
        lib_fast_math = 'libnlcpy_ve_kernel_fast_math.so'
        lib_no_fast_math = 'libnlcpy_ve_kernel_no_fast_math.so'
        lib_profiling = 'libnlcpy_profiling.so'

        assert lib_common in lib_ve1._common_kernel_path
        assert lib_common in lib_ve3._common_kernel_path

        assert lib_fast_math in lib_ve1._fast_math_kernel_path
        assert lib_fast_math in lib_ve3._fast_math_kernel_path

        assert lib_no_fast_math in lib_ve1._no_fast_math_kernel_path
        assert lib_no_fast_math in lib_ve3._no_fast_math_kernel_path

        assert lib_profiling in lib_ve1._profiling_kernel_path
        assert lib_profiling in lib_ve3._profiling_kernel_path

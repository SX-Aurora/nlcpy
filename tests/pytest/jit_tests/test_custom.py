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

import pytest
import unittest
import string
import tempfile
import subprocess
import os
import shutil
import io
import nlcpy
from nlcpy import testing
from nlcpy.ve_types import (uint64, void_p, void)

DIST_DIR = None
# DIST_DIR = 'jit_cache'  # comment in when you debug

LOG_STREAM = None
# import sys
# LOG_STREAM = sys.stdout


##########################
# source for using ve_adr
##########################

_test_ve_adr_c = r'''
#include <stdint.h>
uint64_t test_sum(uint64_t xadr, uint64_t yadr, uint64_t zadr, uint64_t n) {
    ${dtype} *px = (${dtype} *)xadr;
    ${dtype} *py = (${dtype} *)yadr;
    ${dtype} *pz = (${dtype} *)zadr;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        pz[i] = px[i] + py[i];
    }
    return 0;
}
'''

_test_ve_adr_cpp = r'''
#include <stdint.h>
#include <complex>
void cpptest_sum(uint64_t xadr, uint64_t yadr, uint64_t zadr, uint64_t n) {
    ${dtype} *px = (${dtype} *)xadr;
    ${dtype} *py = (${dtype} *)yadr;
    ${dtype} *pz = (${dtype} *)zadr;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        pz[i] = px[i] + py[i];
    }
}

extern "C"{
uint64_t test_sum(uint64_t xadr, uint64_t yadr, uint64_t zadr, uint64_t n) {
    cpptest_sum(xadr, yadr, zadr, n);
    return 0;
}
}
'''

_test_ve_adr_f = r'''
function test_sum(px, py, pz, n)
    implicit none
    integer(kind=8), value :: n
    integer(kind=8) :: test_sum, i
    ${dtype} :: px(n), py(n), pz(n)
#ifdef _OPENMP
!$$omp parallel do
#endif
    do i=1, n
        pz(i) = px(i) + py(i)
    end do
    test_sum = 0
end
'''


############################
# source for using ve_array
############################

_test_ve_array_c = r'''
#include <stdint.h>
#include <nlcpy.h>
#include <assert.h>

uint64_t test_sum(ve_array *x, ve_array *y, ve_array *z) {
    ${dtype} *px = (${dtype} *)x->ve_adr;
    ${dtype} *py = (${dtype} *)y->ve_adr;
    ${dtype} *pz = (${dtype} *)z->ve_adr;
    assert(x->size = y->size);
    assert(x->size = z->size);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < z->size; i++) {
        pz[i] = px[i] + py[i];
    }
    return 0;
}
'''

_test_ve_array_cpp = r'''
#include <stdint.h>
#include <nlcpy.h>
#include <assert.h>
#include <complex>

void cpptest_sum(ve_array *x, ve_array *y, ve_array *z) {
    ${dtype} *px = (${dtype} *)x->ve_adr;
    ${dtype} *py = (${dtype} *)y->ve_adr;
    ${dtype} *pz = (${dtype} *)z->ve_adr;
    assert(x->size = y->size);
    assert(x->size = z->size);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < z->size; i++) {
        pz[i] = px[i] + py[i];
    }
}

extern "C" {
uint64_t test_sum(ve_array *x, ve_array *y, ve_array *z) {
    cpptest_sum(x, y, z);
    return 0;
}
}
'''

####################################
# source for testing argument types
####################################

_test_arg_type_c = r'''
#include <stdint.h>
${dtype3} test_sum(${dtype1} x, ${dtype2} y) {
    return (${dtype3})(x + y);
}
'''

_test_arg_type_cpp = r'''
#include <stdint.h>
extern "C" {
${dtype3} test_sum(${dtype1} x, ${dtype2} y) {
    return (${dtype3})(x + y);
}}
'''

_test_arg_type_f = r'''
function test_sum(x, y)
    ${dtype1}, value :: x
    ${dtype2}, value :: y
    ${dtype2} :: test_sum
    test_sum = x + y
end
'''


####################################
# for dtype conversion
####################################

_ctypes = {
    'i4': 'int32_t',
    'i8': 'int64_t',
    'u4': 'uint32_t',
    'u8': 'uint64_t',
    'f4': 'float',
    'f8': 'double',
    'c8': 'float _Complex',
    'c16': 'double _Complex',
}

_cpptypes = {
    'i4': 'int32_t',
    'i8': 'int64_t',
    'u4': 'uint32_t',
    'u8': 'uint64_t',
    'f4': 'float',
    'f8': 'double',
    'c8': 'std::complex<float>',
    'c16': 'std::complex<double>',
}

_ftypes = {
    'i4': 'integer(kind=4)',
    'i8': 'integer(kind=8)',
    'u4': 'integer(kind=4)',
    'u8': 'integer(kind=8)',
    'f4': 'real(kind=4)',
    'f8': 'real(kind=8)',
    'c8': 'complex(kind=4)',
    'c16': 'complex(kind=8)',
}


def _callback(err):
    if err != 0:
        raise RuntimeError


def _prep(dtype):
    NX = 50
    NY = 60
    NZ = 40
    x1 = nlcpy.arange(NZ * NY * NX, dtype=dtype).reshape(NZ, NY, NX)
    x2 = nlcpy.ones((NZ, NY, NX), dtype=dtype)
    y = nlcpy.zeros((NZ, NY, NX), dtype=dtype)
    return x1, x2, y


def _helper1(cls, code, compiler, name, dtype, args_type=(void,), ret_type=void,
             ext_cflags=(), ext_ldflags=(), encode=False, ftrace=False):
    code = string.Template(code).substitute(dtype=dtype)
    if encode:
        code = code.encode('utf-8')
    cls.lib = nlcpy.jit.CustomVELibrary(
        code=code,
        compiler=compiler,
        cflags=nlcpy.jit.get_default_cflags(
            openmp=cls.openmp,
            opt_level=cls.opt_level
        ) + ext_cflags,
        ldflags=nlcpy.jit.get_default_ldflags(
            openmp=cls.openmp
        ) + ext_ldflags,
        dist_dir=DIST_DIR,
        log_stream=LOG_STREAM,
        ftrace=ftrace
    )
    cls.kern = cls.lib.get_function(
        name,
        args_type=args_type,
        ret_type=ret_type
    )


def _helper2(cls, raw, compiler, name, dtype, args_type, ret_type,
             ext_flags=(), encode=False):
    cmd = (compiler, '-fpic', '-shared', '-fopenmp') + ext_flags
    env = os.environ
    base_name = 'test'
    if 'ncc' in os.path.basename(compiler):
        suffix = '.c'
    elif 'nc++' in os.path.basename(compiler):
        suffix = '.cpp'
    elif 'nfort' in os.path.basename(compiler):
        suffix = '.f03'
    code = string.Template(raw).substitute(dtype=dtype)
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, base_name + suffix), "w") as f:
            f.write(code)
        cmd += (os.path.join(tmpdir, base_name + suffix),
                '-o', os.path.join(tmpdir, base_name + '.so'))
        subprocess.check_output(cmd,
                                env=env,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)
        if encode:
            cls.lib = nlcpy.jit.CustomVELibrary(
                path=os.path.join(tmpdir, base_name + '.so').encode('utf-8')
            )
        else:
            cls.lib = nlcpy.jit.CustomVELibrary(
                path=os.path.join(tmpdir, base_name + '.so')
            )
    cls.kern = cls.lib.get_function(
        name,
        args_type=args_type,
        ret_type=ret_type
    )


@testing.parameterize(*testing.product({
    'dtype': [
        'i4', 'i8', 'u4', 'u8',
        'f4', 'f8', 'c8', 'c16',
    ],
    'openmp': [True, False],
    'opt_level': [2, 3],
    'sync': [True, False],
    'callback': [_callback, None],
}))
class TestCustomLibraryFromStr(unittest.TestCase):

    def setUp(self):
        self.lib = None

    def tearDown(self):
        if self.lib:
            nlcpy.jit.unload_library(self.lib)

    def test_basic_c(self):
        _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                 _ctypes[self.dtype], (uint64, uint64, uint64, uint64), uint64)
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_basic_cpp(self):
        _helper1(self, _test_ve_adr_cpp, '/opt/nec/ve/bin/nc++', 'test_sum',
                 _cpptypes[self.dtype], (uint64, uint64, uint64, uint64), uint64)
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_basic_f(self):
        _helper1(self, _test_ve_adr_f, '/opt/nec/ve/bin/nfort', 'test_sum_',
                 _ftypes[self.dtype], (uint64, uint64, uint64, uint64), uint64,
                 ext_cflags=('-fpp',))
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_ve_array_c(self):
        _helper1(self, _test_ve_array_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                 _ctypes[self.dtype], (void_p, void_p, void_p), uint64)
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1, x2, y, sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_ve_array_cpp(self):
        _helper1(self, _test_ve_array_cpp, '/opt/nec/ve/bin/nc++', 'test_sum',
                 _cpptypes[self.dtype], (void_p, void_p, void_p), uint64)
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1, x2, y, sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0


@testing.parameterize(*testing.product({
    'dtype': [
        'i4', 'i8', 'u4', 'u8',
        'f4', 'f8', 'c8', 'c16',
    ],
    'sync': [True, False],
    'callback': [_callback, None],
}))
class TestCustomLibraryFromSO(unittest.TestCase):

    def setUp(self):
        self.lib = None

    def tearDown(self):
        if self.lib:
            nlcpy.jit.unload_library(self.lib)

    def test_from_so_c(self):
        _helper2(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                 _ctypes[self.dtype], (uint64, uint64, uint64, uint64), uint64)
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_from_so_cpp(self):
        _helper2(self, _test_ve_adr_cpp, '/opt/nec/ve/bin/nc++', 'test_sum',
                 _cpptypes[self.dtype], (uint64, uint64, uint64, uint64), uint64)
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_from_so_f(self):
        _helper2(self, _test_ve_adr_f, '/opt/nec/ve/bin/nfort', 'test_sum_',
                 _ftypes[self.dtype], (uint64, uint64, uint64, uint64), uint64,
                 ('-fpp',))
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0


@testing.parameterize(*testing.product({
    'dtype': [
        ['i4', 'i8', 'i8'],
        ['i8', 'u4', 'i8'],
        ['u4', 'u8', 'u8'],
        ['u8', 'f4', 'f4'],
        ['f4', 'f8', 'f8'],
        ['f8', 'i4', 'i4'],
    ],
}))
class TestCustomLibraryArgType(unittest.TestCase):

    def setUp(self):
        self.lib = None

    def tearDown(self):
        if self.lib:
            nlcpy.jit.unload_library(self.lib)

    def _helper(self, code, compiler, name, args_type, ret_type, ext_cflags=()):
        self.lib = nlcpy.jit.CustomVELibrary(
            code=code,
            compiler=compiler,
            cflags=nlcpy.jit.get_default_cflags() + ext_cflags,
            ldflags=nlcpy.jit.get_default_ldflags(),
            dist_dir=DIST_DIR,
            log_stream=LOG_STREAM,
        )
        self.kern = self.lib.get_function(
            name,
            args_type=args_type,
            ret_type=ret_type
        )

    def test_arg_type_c(self):
        code = string.Template(
            _test_arg_type_c
        ).substitute(
            dtype1=_ctypes[self.dtype[0]],
            dtype2=_ctypes[self.dtype[1]],
            dtype3=_ctypes[self.dtype[2]],
        )
        self._helper(code, '/opt/nec/ve/bin/ncc', 'test_sum',
                     (_ctypes[self.dtype[0]],
                      _ctypes[self.dtype[1]]),
                     _ctypes[self.dtype[2]])
        res = self.kern(1, 2, sync=True)
        testing.assert_array_equal(res, 3,
                                   err_msg='File-ID: {}'.format(self.lib.id))

    def test_arg_type_cpp(self):
        code = string.Template(
            _test_arg_type_cpp
        ).substitute(
            dtype1=_cpptypes[self.dtype[0]],
            dtype2=_cpptypes[self.dtype[1]],
            dtype3=_cpptypes[self.dtype[2]],
        )
        self._helper(code, '/opt/nec/ve/bin/nc++', 'test_sum',
                     (_ctypes[self.dtype[0]],
                      _ctypes[self.dtype[1]]),
                     _ctypes[self.dtype[2]])
        res = self.kern(1, 2, sync=True)
        testing.assert_array_equal(res, 3,
                                   err_msg='File-ID: {}'.format(self.lib.id))

    def test_arg_type_f(self):
        code = string.Template(
            _test_arg_type_f
        ).substitute(
            dtype1=_ftypes[self.dtype[0]],
            dtype2=_ftypes[self.dtype[1]],
            dtype3=_ftypes[self.dtype[2]],
        )
        self._helper(code, '/opt/nec/ve/bin/nfort', 'test_sum_',
                     (_ctypes[self.dtype[0]],
                      _ctypes[self.dtype[1]]),
                     _ctypes[self.dtype[2]])
        res = self.kern(1, 2, sync=True)
        testing.assert_array_equal(res, 3,
                                   err_msg='File-ID: {}'.format(self.lib.id))


class TestCodePathIsBytes(unittest.TestCase):

    def setUp(self):
        self.dtype = 'f8'
        self.openmp = True
        self.opt_level = 2
        self.sync = True
        self.callback = None
        self.lib = None

    def tearDown(self):
        if self.lib:
            nlcpy.jit.unload_library(self.lib)

    def test_code_is_bytes(self):
        _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                 _ctypes[self.dtype], (uint64, uint64, uint64, uint64), uint64,
                 encode=True)
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0

    def test_path_is_bytes(self):
        _helper2(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                 _ctypes[self.dtype], (uint64, uint64, uint64, uint64), uint64,
                 encode=True)
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0


class TestCustomLibraryFailure(unittest.TestCase):

    def setUp(self):
        self.dtype = 'f8'
        self.openmp = True
        self.opt_level = 2
        self.sync = True
        self.callback = None
        self.lib = None

    def tearDown(self):
        if self.lib:
            nlcpy.jit.unload_library(self.lib)

    def _helper(self, code, compiler, dtype, ext_cflags=(), ext_ldflags=()):
        self.lib = nlcpy.jit.CustomVELibrary(
            code=string.Template(
                code
            ).substitute(
                dtype=dtype
            ),
            compiler=compiler,
            cflags=nlcpy.jit.get_default_cflags() + ext_cflags,
            ldflags=nlcpy.jit.get_default_ldflags() + ext_ldflags,
            dist_dir=DIST_DIR,
            log_stream=LOG_STREAM,
        )

    def test_get_function_failure_c(self):
        with pytest.raises(RuntimeError) as ex:
            _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'dummy_name',
                     _ctypes[self.dtype], (uint64, uint64, uint64, uint64), uint64)
        assert ('veo_get_sym' in str(ex.value))

    def test_get_function_failure_cpp(self):
        with pytest.raises(RuntimeError) as ex:
            _helper1(self, _test_ve_adr_cpp, '/opt/nec/ve/bin/nc++', 'dummy_name',
                     _cpptypes[self.dtype], (uint64, uint64, uint64, uint64), uint64)
        assert ('veo_get_sym' in str(ex.value))

    def test_get_function_failure_f(self):
        with pytest.raises(RuntimeError) as ex:
            _helper1(self, _test_ve_adr_f, '/opt/nec/ve/bin/nfort', 'dummy_name',
                     _ftypes[self.dtype], (uint64, uint64, uint64, uint64), uint64)
        assert ('veo_get_sym' in str(ex.value))

    def test_get_function_name_not_str(self):
        with pytest.raises(TypeError) as ex:
            _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 0,
                     _ctypes[self.dtype], (uint64, uint64, uint64, uint64), uint64)
        assert ('func_name must be' in str(ex.value))

    # invalid cflags don't raise any errors, only raise warnings.

    def test_invalid_ldflags_c(self):
        with pytest.raises(nlcpy.jit.nccException) as ex:
            _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                     _ctypes[self.dtype], (uint64, uint64, uint64, uint64), uint64,
                     ext_cflags=('-dummyflag',), ext_ldflags=('-ldummy',))
        assert ('nld: cannot find' in str(ex.value))

    def test_invalid_ldflags_cpp(self):
        with pytest.raises(nlcpy.jit.ncppException) as ex:
            _helper1(self, _test_ve_adr_cpp, '/opt/nec/ve/bin/nc++', 'test_sum',
                     _cpptypes[self.dtype], (uint64, uint64, uint64, uint64), uint64,
                     ext_cflags=('-dummyflag',), ext_ldflags=('-ldummy',))
        assert ('nld: cannot find' in str(ex.value))

    def test_invalid_ldflags_f(self):
        with pytest.raises(nlcpy.jit.nfortException) as ex:
            _helper1(self, _test_ve_adr_f, '/opt/nec/ve/bin/nfort', 'test_sum_',
                     _ftypes[self.dtype], (uint64, uint64, uint64, uint64), uint64,
                     ext_cflags=('-dummyflag',), ext_ldflags=('-ldummy',))
        assert ('nld: cannot find' in str(ex.value))

    def test_c_ld_flags_not_tuple_list(self):
        code = r'''
            void hello(void) {
                printf("hello\n");
            }
'''
        with pytest.raises(TypeError) as ex:
            nlcpy.jit.CustomVELibrary(
                code=code,
                cflags=1
            )
        assert ('cflags must be given' in str(ex.value))

        with pytest.raises(TypeError) as ex:
            nlcpy.jit.CustomVELibrary(
                code=code,
                ldflags=1
            )
        assert ('ldflags must be given' in str(ex.value))

    # too many args_type
    def test_invalid_args_type1(self):
        _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                 _ctypes[self.dtype],
                 (uint64, uint64, uint64, uint64, uint64), uint64)
        x1, x2, y = _prep(self.dtype)
        with pytest.raises(ValueError) as ex:
            self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size)
        assert ('invalid number of arguments' in str(ex.value))

    # not enough args_type
    def test_invalid_args_type2(self):
        _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                 _ctypes[self.dtype],
                 (uint64, uint64, uint64), uint64)
        x1, x2, y = _prep(self.dtype)
        with pytest.raises(ValueError) as ex:
            self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size)
        assert ('invalid number of arguments' in str(ex.value))

    # invalid strings
    def test_invalid_args_type3(self):
        with pytest.raises(TypeError) as ex:
            _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                     _ctypes[self.dtype],
                     ('dummy1', 'dummy2', 'dummy3', 'dummy4'), uint64)
        assert ('Don\'t know how to convert' in str(ex.value))

    # invalid type
    def test_invalid_args_type4(self):
        with pytest.raises(TypeError) as ex:
            _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                     _ctypes[self.dtype],
                     0, uint64)
        assert ('args_type must be' in str(ex.value))

    # pass tuple
    def test_invalid_ret_type1(self):
        with pytest.raises(TypeError) as ex:
            _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                     _ctypes[self.dtype],
                     (uint64, uint64, uint64, uint64), (uint64,))
        assert ('ret_type must be given' in str(ex.value))

    # invalid strings
    def test_invalid_ret_type2(self):
        with pytest.raises(TypeError) as ex:
            _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                     _ctypes[self.dtype],
                     (uint64, uint64, uint64, uint64), 'dummy')
        assert ('Don\'t know how to convert' in str(ex.value))

    def test_invalid_code_c(self):
        code = r'''
            void test(void) {
                // missing semicolon
                double a
            }
'''
        with pytest.raises(nlcpy.jit.nccException):
            self._helper(
                code,
                '/opt/nec/ve/bin/ncc',
                _ctypes[self.dtype]
            )

    def test_invalid_code_cpp(self):
        code = r'''
            void test(void) {
                // missing semicolon
                double a
            }
'''
        with pytest.raises(nlcpy.jit.ncppException):
            self._helper(
                code,
                '/opt/nec/ve/bin/nc++',
                _cpptypes[self.dtype]
            )

    def test_invalid_code_f(self):
        code = r'''
            function test
                implicit none
                c = a + b
            end
'''
        with pytest.raises(nlcpy.jit.nfortException):
            self._helper(
                code,
                '/opt/nec/ve/bin/nfort',
                _ftypes[self.dtype]
            )

    def test_unknown_compiler(self):
        with pytest.raises(ValueError) as ex:
            self._helper(_test_ve_adr_c, 'unknown',
                         _ctypes[self.dtype])
        assert ('unknown compiler' in str(ex.value))

    def test_compiler_not_str(self):
        with pytest.raises(TypeError) as ex:
            self._helper(_test_ve_adr_c, 0,
                         _ctypes[self.dtype])
        assert ('compiler must be given' in str(ex.value))

    def test_invalid_ncc_command(self):
        with pytest.raises(FileNotFoundError):
            self._helper(_test_ve_adr_c, 'invalid_ncc',
                         _ctypes[self.dtype])

    def test_invalid_ncpp_command(self):
        with pytest.raises(FileNotFoundError):
            self._helper(_test_ve_adr_cpp, 'invalid_nc++',
                         _cpptypes[self.dtype])

    def test_invalid_nfort_command(self):
        with pytest.raises(FileNotFoundError):
            self._helper(_test_ve_adr_cpp, 'invalid_nfort',
                         _ftypes[self.dtype])

    def test_specify_code_and_path(self):
        with pytest.raises(TypeError) as ex:
            nlcpy.jit.CustomVELibrary(
                code=_test_ve_adr_c,
                path='invalid/path'
            )
        assert ('code and path' in str(ex.value))

    def test_invalid_code_type(self):
        with pytest.raises(TypeError):
            nlcpy.jit.CustomVELibrary(
                code=1
            )

    def test_invalid_path_type(self):
        with pytest.raises(TypeError):
            nlcpy.jit.CustomVELibrary(
                path=1
            )

    def test_unload_failure(self):
        with pytest.raises(TypeError) as ex:
            nlcpy.jit.unload_library(1)
        assert ('unrecognized input type' in str(ex.value))


class TestCustomLibraryDistDir(unittest.TestCase):

    def setUp(self):
        self.lib = None

    def tearDown(self):
        if self.lib:
            nlcpy.jit.unload_library(self.lib)

    def _helper(self, code, compiler, name, args_type=(void,), ret_type=void,
                ext_cflags=(), ext_ldflags=(), dist_dir=None):
        self.lib = nlcpy.jit.CustomVELibrary(
            code=code,
            compiler=compiler,
            cflags=nlcpy.jit.get_default_cflags() + ext_cflags,
            ldflags=nlcpy.jit.get_default_ldflags() + ext_ldflags,
            dist_dir=dist_dir,
            log_stream=LOG_STREAM,
        )
        self.kern = self.lib.get_function(
            name,
            args_type=args_type,
            ret_type=ret_type
        )

    def test_dist_dir(self):
        dist_dir = './jit_tmp_dir/'
        if os.path.exists(dist_dir):
            shutil.rmtree(dist_dir)
        code = r'''
            void hello(void) {
                printf("hello\n");
            }
'''
        self._helper(code, '/opt/nec/ve/bin/ncc', 'hello', dist_dir=dist_dir)
        self.kern()
        assert os.path.exists(dist_dir)
        assert os.path.exists(os.path.join(dist_dir, self.lib.id) + '.c')
        assert os.path.exists(os.path.join(dist_dir, self.lib.id) + '.o')
        assert os.path.exists(os.path.join(dist_dir, self.lib.id) + '.so')
        nlcpy.jit.unload_library(self.lib)
        self.lib = None
        shutil.rmtree(dist_dir)

    def test_dist_dir_permission_denied(self):
        dist_dir = './jit_tmp_dir/'
        if os.path.exists(dist_dir):
            shutil.rmtree(dist_dir)
        os.mkdir(dist_dir)
        os.chmod(dist_dir, 0o444)
        code = r'''
            void hello(void) {
                printf("hello\n");
            }
'''
        with pytest.raises(PermissionError):
            self._helper(code, '/opt/nec/ve/bin/ncc', 'hello', dist_dir=dist_dir)
        shutil.rmtree(dist_dir)


class TestCustomLibraryLogStream(unittest.TestCase):

    def setUp(self):
        self.lib = None

    def tearDown(self):
        if self.lib:
            nlcpy.jit.unload_library(self.lib)

    def _helper(self, code, compiler, name, args_type=(void,), ret_type=void,
                ext_cflags=(), ext_ldflags=(), log_stream=None):
        self.lib = nlcpy.jit.CustomVELibrary(
            code=code,
            compiler=compiler,
            cflags=nlcpy.jit.get_default_cflags() + ext_cflags,
            ldflags=nlcpy.jit.get_default_ldflags() + ext_ldflags,
            dist_dir=DIST_DIR,
            log_stream=log_stream,
        )

    def test_log_stream(self):
        file_name = './tmp.log'
        if os.path.exists(file_name):
            os.remove(file_name)
        code = r'''
            void hello(void) {
                printf("hello\n");
            }
'''
        with open(file_name, 'w') as fs:
            self._helper(code, '/opt/nec/ve/bin/ncc', 'hello', log_stream=fs)
        assert os.path.isfile(file_name)
        assert os.path.getsize(file_name) > 0
        os.remove(file_name)

    def test_log_stream_not_writable(self):
        file_name = './tmp.log'
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, 'w'):
            pass
        code = r'''
            void hello(void) {
                printf("hello\n");
            }
'''
        with pytest.raises(io.UnsupportedOperation):
            with open(file_name, 'r') as fs:
                self._helper(code, '/opt/nec/ve/bin/ncc', 'hello', log_stream=fs)
        os.remove(file_name)

    def test_log_stream_with_compile_failure(self):
        file_name = './tmp.log'
        if os.path.exists(file_name):
            os.remove(file_name)
        code = r'''
            void test(void) {
                // missing semicolon
                double a
            }
'''
        with pytest.raises(nlcpy.jit.nccException):
            with open(file_name, 'w') as fs:
                self._helper(code, '/opt/nec/ve/bin/ncc', 'hello', log_stream=fs)
        assert os.path.isfile(file_name)
        assert os.path.getsize(file_name) > 0
        os.remove(file_name)


class TestCustomLibraryFtrace(unittest.TestCase):

    def setUp(self):
        self.dtype = 'f8'
        self.openmp = True
        self.opt_level = 2
        self.sync = True
        self.callback = _callback
        self.lib = None

    def tearDown(self):
        pass

    def test_ftrace_c(self):
        file_name = './ftrace.out'
        if os.path.exists(file_name):
            os.remove(file_name)
        _helper1(self, _test_ve_adr_c, '/opt/nec/ve/bin/ncc', 'test_sum',
                 _ctypes[self.dtype], (uint64, uint64, uint64, uint64), uint64,
                 ftrace=True)
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0
        nlcpy.jit.unload_library(self.lib)
        assert os.path.exists(file_name)
        assert os.path.getsize(file_name) > 0
        os.remove(file_name)

    def test_ftrace_cpp(self):
        file_name = './ftrace.out'
        if os.path.exists(file_name):
            os.remove(file_name)
        _helper1(self, _test_ve_adr_cpp, '/opt/nec/ve/bin/nc++', 'test_sum',
                 _cpptypes[self.dtype], (uint64, uint64, uint64, uint64), uint64,
                 ftrace=True)
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0
        nlcpy.jit.unload_library(self.lib)
        assert os.path.exists(file_name)
        assert os.path.getsize(file_name) > 0
        os.remove(file_name)

    def test_ftrace_f(self):
        file_name = './ftrace.out'
        if os.path.exists(file_name):
            os.remove(file_name)
        _helper1(self, _test_ve_adr_f, '/opt/nec/ve/bin/nfort', 'test_sum_',
                 _ftypes[self.dtype], (uint64, uint64, uint64, uint64), uint64,
                 ext_cflags=('-fpp',), ftrace=True)
        x1, x2, y = _prep(self.dtype)
        err = self.kern(x1.ve_adr, x2.ve_adr, y.ve_adr, y.size,
                        sync=self.sync, callback=self.callback)
        testing.assert_allclose(y, x1 + x2,
                                err_msg='File-ID: {}'.format(self.lib.id))
        if self.sync:
            assert err == 0
        nlcpy.jit.unload_library(self.lib)
        assert os.path.exists(file_name)
        assert os.path.getsize(file_name) > 0
        os.remove(file_name)


class TestCustomKernelFailure(unittest.TestCase):

    def test_invalid_type(self):
        with pytest.raises(TypeError) as ex:
            nlcpy.jit.CustomVEKernel(0)
        assert ('func must be given' in str(ex.value))

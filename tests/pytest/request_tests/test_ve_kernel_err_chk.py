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
import pytest
import string

import nlcpy
from nlcpy.ve_types import (int32, uint64)

_err_case = {
    'ERROR_OK': 0,
    'ERROR_NDIM': 1,
    'ERROR_DTYPE': 2,
    'ERROR_MEMORY': 3,
    'ERROR_FUNCNUM': 4,
    'ERROR_FUNCTYPE': 5,
    'ERROR_INDEX': 6,
    'ERROR_ASL': 7,
    'ERROR_SCA': 8,
    'ERROR_OTHER': 9
}

############################
# source for VE kernel
############################

_test_ve_kernel_ret_err_code = string.Template(r'''
#include <stdint.h>
#include <nlcpy.h>
#include <assert.h>

uint64_t test_err_code(int x) {
    switch(x) {
    case ${ERROR_OK}:
        return NLCPY_ERROR_OK;
    case ${ERROR_NDIM}:
        return NLCPY_ERROR_NDIM;
    case ${ERROR_DTYPE}:
        return NLCPY_ERROR_DTYPE;
    case ${ERROR_MEMORY}:
        return NLCPY_ERROR_MEMORY;
    case ${ERROR_FUNCNUM}:
        return NLCPY_ERROR_FUNCNUM;
    case ${ERROR_FUNCTYPE}:
        return NLCPY_ERROR_FUNCTYPE;
    case ${ERROR_INDEX}:
        return NLCPY_ERROR_INDEX;
    case ${ERROR_ASL}:
        return NLCPY_ERROR_ASL;
    case ${ERROR_SCA}:
        return NLCPY_ERROR_SCA;
    default:
        return ~ (uint64_t)(NLCPY_ERROR_OK |
                            NLCPY_ERROR_NDIM |
                            NLCPY_ERROR_DTYPE |
                            NLCPY_ERROR_MEMORY |
                            NLCPY_ERROR_FUNCNUM |
                            NLCPY_ERROR_FUNCTYPE |
                            NLCPY_ERROR_INDEX |
                            NLCPY_ERROR_ASL |
                            NLCPY_ERROR_SCA );
    }
}
''').substitute(**_err_case)


class TestVEKernelErrCode(unittest.TestCase):

    def setUp(self):
        self._lib = None

    def tearDown(self):
        if self._lib is not None:
            nlcpy.jit.unload_library(self._lib)

    def test_jit_c_ve_adr(self):
        code = _test_ve_kernel_ret_err_code
        ve_lib = nlcpy.jit.CustomVELibrary(
            code=code,
            compiler='ncc'
        )
        self._lib = ve_lib
        ve_kern = ve_lib.get_function(
            'test_err_code',
            args_type=(int32,),
            ret_type=uint64
        )

        # NLCPY_ERROR_OK
        _ = ve_kern(_err_case['ERROR_OK'], sync=True,
                    callback=nlcpy.request.ve_kernel.check_error)

        # NLCPY_ERROR_NDIM
        with pytest.raises(RuntimeError) as ex:
            _ = ve_kern(_err_case['ERROR_NDIM'], sync=True,
                        callback=nlcpy.request.ve_kernel.check_error)
        assert ('invalid ndim' in str(ex.value))

        # NLCPY_ERROR_DTYPE
        with pytest.raises(RuntimeError) as ex:
            _ = ve_kern(_err_case['ERROR_DTYPE'], sync=True,
                        callback=nlcpy.request.ve_kernel.check_error)
        assert ('invalid dtype' in str(ex.value))

        # NLCPY_ERROR_MEMORY
        with pytest.raises(RuntimeError) as ex:
            _ = ve_kern(_err_case['ERROR_MEMORY'], sync=True,
                        callback=nlcpy.request.ve_kernel.check_error)
        assert ('invalid ve_adr' in str(ex.value))

        # NLCPY_ERROR_FUNCNUM
        with pytest.raises(RuntimeError) as ex:
            _ = ve_kern(_err_case['ERROR_FUNCNUM'], sync=True,
                        callback=nlcpy.request.ve_kernel.check_error)
        assert ('invalid function number' in str(ex.value))

        # NLCPY_ERROR_FUNCTYPE
        with pytest.raises(RuntimeError) as ex:
            _ = ve_kern(_err_case['ERROR_FUNCTYPE'], sync=True,
                        callback=nlcpy.request.ve_kernel.check_error)
        assert ('invalid function type' in str(ex.value))

        # NLCPY_ERROR_INDEX
        with pytest.raises(RuntimeError) as ex:
            _ = ve_kern(_err_case['ERROR_INDEX'], sync=True,
                        callback=nlcpy.request.ve_kernel.check_error)
        assert ('invalid index' in str(ex.value))

        # NLCPY_ERROR_ASL
        with pytest.raises(RuntimeError) as ex:
            _ = ve_kern(_err_case['ERROR_ASL'], sync=True,
                        callback=nlcpy.request.ve_kernel.check_error)
        assert ('ASL error' in str(ex.value))

        # NLCPY_ERROR_SCA
        with pytest.raises(RuntimeError) as ex:
            _ = ve_kern(_err_case['ERROR_SCA'], sync=True,
                        callback=nlcpy.request.ve_kernel.check_error)
        assert ('SCA error' in str(ex.value))

        # NLCPY_ERROR_OTHER
        with pytest.raises(RuntimeError) as ex:
            _ = ve_kern(_err_case['ERROR_OTHER'], sync=True,
                        callback=nlcpy.request.ve_kernel.check_error)
        assert ('unknown error' in str(ex.value))

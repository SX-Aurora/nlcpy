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

import os
import nlcpy
from nlcpy import _path

include "basic_kernel_list.pxi"
include "math_kernel_list.pxi"
include "indexing_kernel_list.pxi"
include "creation_kernel_list.pxi"
include "manipulation_kernel_list.pxi"
include "cblas_kernel_list.pxi"
include "searching_kernel_list.pxi"
include "linalg_kernel_list.pxi"
include "random_kernel_list.pxi"
include "fft_kernel_list.pxi"
include "reduceat_kernel_list.pxi"
include "sca_kernel_list.pxi"

_here = _path._here


def _register_ve_kernel(p):
    fast_math = os.environ.get('VE_NLCPY_FAST_MATH', 'no')
    if fast_math in ('yes', 'YES'):
        lib = p.load_library(
            (_here + "/lib/libnlcpy_ve_kernel_fast_math.so").encode('utf-8'))
    else:
        lib = p.load_library(
            (_here + "/lib/libnlcpy_ve_kernel_no_fast_math.so").encode('utf-8'))
    if lib is None:
        raise RuntimeError("cannot detect ve kernel")
    all_kernel_list = {
        **_basic_kernel_list,
        **_indexing_kernel_list,
        **_creation_kernel_list,
        **_manipulation_kernel_list,
        **_math_kernel_list,
        **_cblas_kernel_list,
        **_searching_kernel_list,
        **_linalg_kernel_list,
        **_random_kernel_list,
        **_fft_kernel_list,
        **_reduceat_kernel_list,
        **_sca_kernel_list,
    }
    for k, v in all_kernel_list.items():
        try:
            f = lib.find_function(k.encode('utf-8'))
        except RuntimeError:
            continue
        if f is not None:
            fargs = v["args"]
            f.args_type(*fargs)
            f.ret_type(v["ret"])

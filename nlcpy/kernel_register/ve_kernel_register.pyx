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

import nlcpy
from nlcpy import _path
from nlcpy import _environment

include "basic_kernel_list.pxi"
include "asluni_kernel_list.pxi"
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
include "profiling_kernel_list.pxi"


def _register_ve_kernel(p):
    lib = None
    lib_prof = None
    fast_math = _environment._is_fast_math()
    if fast_math:
        lib = p.load_library(_path._fast_math_kernel_path.encode('utf-8'))
    else:
        lib = p.load_library(_path._no_fast_math_kernel_path.encode('utf-8'))
    lib_prof = p.load_library(_path._profiling_kernel_path.encode('utf-8'))
    if lib is None or lib_prof is None:
        raise RuntimeError("failed to load ve kernel")
    all_kernel_list = {
        **_basic_kernel_list,
        **_asluni_kernel_list,
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

    profiling_kernel_list = {**_profiling_kernel_list}
    for k, v in profiling_kernel_list.items():
        try:
            f = lib_prof.find_function(k.encode('utf-8'))
        except RuntimeError:
            continue
        if f is not None:
            fargs = v["args"]
            f.args_type(*fargs)
            f.ret_type(v["ret"])

    return lib, lib_prof

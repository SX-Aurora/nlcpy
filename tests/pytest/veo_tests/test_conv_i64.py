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
import string

import nlcpy
from nlcpy import testing


_test_c = r'''
#include <stdint.h>
${dt_to} test_conv_i64(${dt_from} x) {
    return (${dt_to})x;
}
'''


def _helper(src, dt_from, dt_to, func_name):
    code = string.Template(src).substitute(dt_from=dt_from, dt_to=dt_to)
    ve_lib = nlcpy.jit.CustomVELibrary(
        code=code,
        compiler='ncc'
    )
    return ve_lib.get_function(
        func_name,
        args_type=(dt_from,),
        ret_type=dt_to
    )


@testing.parameterize(*testing.product({
    'dt': [
        'char',
        'short',
        'int',
        'int32_t',
        'long',
        'int64_t',
        'unsigned char',
        'unsigned short',
        'unsigned int',
        'uint32_t',
        'unsigned long',
        'uint64_t',
        'float',
        'double',
    ],
}))
class TestConvI64(unittest.TestCase):

    def test_conv_from_to(self):
        _helper(_test_c, self.dt, self.dt, 'test_conv_i64')(12, sync=True) == 12

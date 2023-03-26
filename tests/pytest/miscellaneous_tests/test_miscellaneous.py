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

import unittest
import sys
import io

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

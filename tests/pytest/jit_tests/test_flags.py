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


class TestCflags(unittest.TestCase):

    def test_openmp(self):
        got = nlcpy.jit.get_default_cflags(openmp=True)
        exp = (
            '-c',
            '-fpic',
            '-O2',
            '-I',
            nlcpy.get_include(),
            '-fopenmp'
        )
        assert got == exp

        got = nlcpy.jit.get_default_cflags(openmp=False)
        exp = (
            '-c',
            '-fpic',
            '-O2',
            '-I',
            nlcpy.get_include(),
        )
        assert got == exp

    def test_opt_level_normal(self):
        got = nlcpy.jit.get_default_cflags(opt_level=0)
        exp = (
            '-c',
            '-fpic',
            '-O0',
            '-I',
            nlcpy.get_include(),
            '-fopenmp'
        )
        assert got == exp

        got = nlcpy.jit.get_default_cflags(opt_level=4)
        exp = (
            '-c',
            '-fpic',
            '-O4',
            '-I',
            nlcpy.get_include(),
            '-fopenmp'
        )
        assert got == exp

    def test_opt_level_failure(self):
        with self.assertRaises(ValueError):
            nlcpy.jit.get_default_cflags(opt_level=-1)
        with self.assertRaises(ValueError):
            nlcpy.jit.get_default_cflags(opt_level=-10)
        with self.assertRaises(ValueError):
            nlcpy.jit.get_default_cflags(opt_level=5)
        with self.assertRaises(ValueError):
            nlcpy.jit.get_default_cflags(opt_level=100)
        with self.assertRaises(TypeError):
            nlcpy.jit.get_default_cflags(opt_level='1')

    def test_debug(self):
        got = nlcpy.jit.get_default_cflags(debug=True)
        exp = (
            '-c',
            '-fpic',
            '-O2',
            '-I',
            nlcpy.get_include(),
            '-fopenmp',
            '-g'
        )
        assert got == exp

        got = nlcpy.jit.get_default_cflags(debug=False)
        exp = (
            '-c',
            '-fpic',
            '-O2',
            '-I',
            nlcpy.get_include(),
            '-fopenmp',
        )
        assert got == exp

    def test_combination(self):
        got = nlcpy.jit.get_default_cflags(openmp=True, opt_level=3, debug=True)
        exp = (
            '-c',
            '-fpic',
            '-O3',
            '-I',
            nlcpy.get_include(),
            '-fopenmp',
            '-g'
        )
        assert got == exp

        got = nlcpy.jit.get_default_cflags(openmp=False, opt_level=1, debug=True)
        exp = (
            '-c',
            '-fpic',
            '-O1',
            '-I',
            nlcpy.get_include(),
            '-g'
        )
        assert got == exp


class TestLdflags(unittest.TestCase):

    def test_openmp(self):
        got = nlcpy.jit.get_default_ldflags(openmp=True)
        exp = (
            '-fpic',
            '-shared',
            '-fopenmp'
        )
        assert got == exp

        got = nlcpy.jit.get_default_ldflags(openmp=False)
        exp = (
            '-fpic',
            '-shared',
        )
        assert got == exp

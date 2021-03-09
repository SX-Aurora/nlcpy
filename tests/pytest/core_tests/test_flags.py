#
# * The source code in this file is based on the soure code of CuPy.
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
# # CuPy License #
#
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

import unittest

from nlcpy.core import flags
from nlcpy import testing


class TestFlags(unittest.TestCase):

    def setUp(self):
        self.flags = flags.Flags(1, 2, 3)

    def test_c_contiguous(self):
        self.assertEqual(1, self.flags['C_CONTIGUOUS'])

    def test_f_contiguous(self):
        self.assertEqual(2, self.flags['F_CONTIGUOUS'])

    def test_owndata(self):
        self.assertEqual(3, self.flags['OWNDATA'])

    def test_key_error(self):
        with self.assertRaises(KeyError):
            self.flags['unknown key']

    def test_repr(self):
        self.assertEqual('''  C_CONTIGUOUS : 1
  F_CONTIGUOUS : 2
  OWNDATA : 3''', repr(self.flags))


@testing.parameterize(
    *testing.product({
        'order': ['C', 'F', 'non-contiguous'],
        'shape': [(8, ), (4, 8)],
    })
)
class TestContiguityFlags(unittest.TestCase):

    def setUp(self):
        self.flags = None

    def init_flags(self, xp):
        if self.order == 'non-contiguous':
            a = xp.empty(self.shape)[::2]
        else:
            a = xp.empty(self.shape, order=self.order)
        self.flags = a.flags

    @testing.numpy_nlcpy_equal()
    def test_fnc(self, xp):
        self.init_flags(xp)
        return self.flags.fnc

    @testing.numpy_nlcpy_equal()
    def test_forc(self, xp):
        self.init_flags(xp)
        return self.flags.forc

    @testing.numpy_nlcpy_equal()
    def test_f_contiguous(self, xp):
        self.init_flags(xp)
        return self.flags.f_contiguous

    @testing.numpy_nlcpy_equal()
    def test_c_contiguous(self, xp):
        self.init_flags(xp)
        return self.flags.c_contiguous

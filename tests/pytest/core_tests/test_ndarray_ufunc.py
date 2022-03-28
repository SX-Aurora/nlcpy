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

import numpy as np

import nlcpy
from nlcpy import testing


class TestArrayUfunc(unittest.TestCase):

    @testing.with_requires('numpy>=1.13')
    @testing.for_all_dtypes(no_bool=True)
    def test_unary_op(self, dtype):
        a = nlcpy.array(np.array([0, 1, 2]), dtype=dtype)
        outa = nlcpy.sin(a)
        # nlcpy operation produced a nlcpy array
        self.assertTrue(isinstance(outa, nlcpy.ndarray))
        b = a.get()
        outb = np.sin(b)
        # numpy operation produced a numpy array
        self.assertTrue(isinstance(outa, nlcpy.ndarray))
        self.assertTrue(np.allclose(outa.get(), outb))

    @testing.with_requires('numpy>=1.13')
    @testing.for_all_dtypes(no_bool=True)
    def test_unary_op_out(self, dtype):
        a = nlcpy.array(np.array([0, 1, 2]), dtype=dtype)
        b = a.get()
        outb = np.sin(b)
        # pre-make output with same type as input
        outa = nlcpy.array(np.array([0, 1, 2]), dtype=outb.dtype)
        nlcpy.sin(a, out=outa)
        self.assertTrue(np.allclose(outa.get(), outb))

    @testing.with_requires('numpy>=1.13')
    @testing.for_all_dtypes()
    def test_binary_op(self, dtype):
        a1 = nlcpy.array(np.array([0, 1, 2]), dtype=dtype)
        a2 = nlcpy.array(np.array([0, 1, 2]), dtype=dtype)
        outa = a1 + a2
        # nlcpy operation produced a nlcpy array
        self.assertTrue(isinstance(outa, nlcpy.ndarray))
        b1 = a1.get()
        b2 = a2.get()
        outb = np.add(b1, b2)
        # numpy operation produced a numpy array
        self.assertTrue(isinstance(outb, np.ndarray))
        self.assertTrue(np.allclose(outa.get(), outb))

    @testing.with_requires('numpy>=1.13')
    @testing.for_all_dtypes()
    def test_binary_op_out(self, dtype):
        a1 = nlcpy.array(np.array([0, 1, 2]), dtype=dtype)
        a2 = nlcpy.array(np.array([0, 1, 2]), dtype=dtype)
        outa = nlcpy.array(np.array([0, 1, 2]), dtype=dtype)
        nlcpy.add(a1, a2, out=outa)
        b1 = a1.get()
        b2 = a2.get()
        outb = np.add(b1, b2)
        self.assertTrue(np.allclose(outa.get(), outb))

    @testing.numpy_nlcpy_array_equal()
    def test_indexing(self, xp):
        a = nlcpy.testing.shaped_arange((3, 1), xp)[:, :, None]
        b = nlcpy.testing.shaped_arange((3, 2), xp)[:, None, :]
        return a * b

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_nlcpy_array_equal()
    def test_shares_memory(self, xp):
        a = nlcpy.testing.shaped_arange((1000, 1000), xp, 'int64')
        b = xp.transpose(a)
        a += b
        return a

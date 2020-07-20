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

import nlcpy

from nlcpy import testing


class TestJoin(unittest.TestCase):

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate1(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=2)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_large_2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        d = testing.shaped_arange((2, 3, 5), xp, dtype)
        e = testing.shaped_arange((2, 3, 2), xp, dtype)
        return xp.concatenate((a, b, c, d, e) * 2, axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_large_3(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 1), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 1), xp, dtype)
        return xp.concatenate((a, b) * 10, axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_large_4(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 4), xp, dtype)
        return xp.concatenate((a, b) * 10, axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_large_5(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 4), xp, 'i')
        return xp.concatenate((a, b) * 10, axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_f_contiguous(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((2, 3, 2), xp, dtype).T
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_large_f_contiguous(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((2, 3, 2), xp, dtype).T
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        d = testing.shaped_arange((2, 3, 2), xp, dtype).T
        e = testing.shaped_arange((2, 3, 2), xp, dtype)
        return xp.concatenate((a, b, c, d, e) * 2, axis=-1)

    @testing.numpy_nlcpy_array_equal()
    def test_concatenate_many_multi_dptye(self, xp):
        a = testing.shaped_arange((2, 1), xp, 'i')
        b = testing.shaped_arange((2, 1), xp, 'f')
        return xp.concatenate((a, b) * 1024, axis=1)

    def test_concatenate_wrong_ndim(self):
        a = nlcpy.empty((2, 3))
        b = nlcpy.empty((2,))
        with self.assertRaises(ValueError):
            nlcpy.concatenate((a, b))

    def test_concatenate_wrong_shape(self):
        a = nlcpy.empty((2, 3, 4))
        b = nlcpy.empty((3, 3, 4))
        c = nlcpy.empty((4, 4, 4))
        with self.assertRaises(ValueError):
            nlcpy.concatenate((a, b, c))

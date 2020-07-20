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


class TestTranspose(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis1(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 1], [1, 2])

    @testing.numpy_nlcpy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis2(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, 1, -1)

    @testing.numpy_nlcpy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis3(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 2], [1, 0])

    @testing.numpy_nlcpy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis4(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [2, 0], [1, 0])

    @testing.numpy_nlcpy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis5(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [2, 0], [0, 1])

    @testing.numpy_nlcpy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis6(self, xp):
        a = testing.shaped_arange((2, 3, 4, 5, 6), xp)
        return xp.moveaxis(a, [0, 2, 1], [3, 4, 0])

    # dim is too large
    @testing.numpy_nlcpy_raises(ignore_msg=True)
    @testing.with_requires('numpy>=1.13')
    def test_moveaxis_invalid1_1(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 1], [1, 3])

    def test_moveaxis_invalid1_2(self):
        a = testing.shaped_arange((2, 3, 4), nlcpy)
        with self.assertRaises(nlcpy.AxisError):
            return nlcpy.moveaxis(a, [0, 1], [1, 3])

    # dim is too small
    @testing.numpy_nlcpy_raises(ignore_msg=True)
    @testing.with_requires('numpy>=1.13')
    def test_moveaxis_invalid2_1(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, -4], [1, 2])

    def test_moveaxis_invalid2_2(self):
        a = testing.shaped_arange((2, 3, 4), nlcpy)
        with self.assertRaises(nlcpy.AxisError):
            return nlcpy.moveaxis(a, [0, -4], [1, 2])

    # len(source) != len(destination)
    @testing.numpy_nlcpy_raises()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis_invalid3(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 1, 2], [1, 2])

    # len(source) != len(destination)
    @testing.numpy_nlcpy_raises()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis_invalid4(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 1], [1, 2, 0])

    # Use the same axis twice
    def test_moveaxis_invalid5_1(self):
        a = testing.shaped_arange((2, 3, 4), nlcpy)
        with self.assertRaises(nlcpy.AxisError):
            return nlcpy.moveaxis(a, [1, -1], [1, 3])

    def test_moveaxis_invalid5_2(self):
        a = testing.shaped_arange((2, 3, 4), nlcpy)
        with self.assertRaises(nlcpy.AxisError):
            return nlcpy.moveaxis(a, [0, 1], [-1, 2])

    def test_moveaxis_invalid5_3(self):
        a = testing.shaped_arange((2, 3, 4), nlcpy)
        with self.assertRaises(nlcpy.AxisError):
            return nlcpy.moveaxis(a, [0, 1], [1, 1])

    @testing.numpy_nlcpy_array_equal()
    def test_rollaxis(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.rollaxis(a, 2)

    def test_rollaxis_failure(self):
        a = testing.shaped_arange((2, 3, 4))
        with self.assertRaises(ValueError):
            nlcpy.rollaxis(a, 3)

#    @testing.numpy_nlcpy_array_equal()
#    def test_swapaxes(self, xp):
#        a = testing.shaped_arange((2, 3, 4), xp)
#        return xp.swapaxes(a, 2, 0)
#
#    def test_swapaxes_failure(self):
#        a = testing.shaped_arange((2, 3, 4))
#        with self.assertRaises(ValueError):
#            nlcpy.swapaxes(a, 3, 0)
#
    @testing.numpy_nlcpy_array_equal()
    def test_transpose(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.transpose(-1, 0, 1)

    @testing.numpy_nlcpy_array_equal()
    def test_transpose_empty(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.transpose()

    @testing.numpy_nlcpy_array_equal()
    def test_transpose_none(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.transpose(None)

    @testing.numpy_nlcpy_array_equal()
    def test_external_transpose(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.transpose(a, (-1, 0, 1))

    @testing.numpy_nlcpy_array_equal()
    def test_external_transpose_all(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.transpose(a)

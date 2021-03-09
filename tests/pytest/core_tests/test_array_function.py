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

import numpy

import nlcpy
from nlcpy import testing


class TestArrayFunction(unittest.TestCase):

    @testing.with_requires('numpy>=1.17.0')
    def test_array_function(self):
        a = numpy.random.randn(100, 100)
        a_cpu = numpy.asarray(a)
        a_ve = nlcpy.asarray(a)

        # The numpy call for both CPU and ve arrays is intentional to test the
        # __array_function__ protocol
        qr_cpu = numpy.linalg.qr(a_cpu)
        qr_ve = numpy.linalg.qr(a_ve)

        if isinstance(qr_cpu, tuple):
            for b_cpu, b_ve in zip(qr_cpu, qr_ve):
                self.assertEqual(b_cpu.dtype, b_ve.dtype)
                nlcpy.testing.assert_allclose(b_cpu, b_ve, atol=1e-4)
        else:
            self.assertEqual(qr_cpu.dtype, qr_ve.dtype)
            nlcpy.testing.assert_allclose(qr_cpu, qr_ve, atol=1e-4)

    @testing.with_requires('numpy>=1.17.0')
    @testing.numpy_nlcpy_equal()
    def test_array_function_can_cast(self, xp):
        return numpy.can_cast(xp.arange(2), 'f4')

    @testing.with_requires('numpy>=1.17.0')
    @testing.numpy_nlcpy_equal()
    def test_array_function_common_type(self, xp):
        return numpy.common_type(xp.arange(2, dtype='f8'),
                                 xp.arange(2, dtype='f4'))

    @testing.with_requires('numpy>=1.17.0')
    @testing.numpy_nlcpy_equal()
    def test_array_function_result_type(self, xp):
        return numpy.result_type(3, xp.arange(2, dtype='f8'))

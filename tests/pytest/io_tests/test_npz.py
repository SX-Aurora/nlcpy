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
import io
import pickle

import numpy
from nlcpy import testing


class TestNpz(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_save_load(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype=dtype)

        with io.BytesIO() as f:
            xp.save(f, a)
            val = f.getvalue()

        with io.BytesIO(val) as f:
            b = xp.load(f)

        return b

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_savez(self, xp, dtype):
        a1 = testing.shaped_arange((2, 3, 4), xp, dtype=dtype)
        a2 = testing.shaped_arange((3, 4, 5), xp, dtype=dtype)

        with io.BytesIO() as f:
            xp.savez(f, a1, a2)
            val = f.getvalue()

        with io.BytesIO(val) as f:
            d = xp.load(f)
            b1 = d['arr_0']
            b2 = d['arr_1']

        return b1, b2

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_savez_compressed(self, xp, dtype):
        a1 = testing.shaped_arange((2, 3, 4), xp, dtype=dtype)
        a2 = testing.shaped_arange((3, 4, 5), xp, dtype=dtype)

        with io.BytesIO() as f:
            xp.savez_compressed(f, a1, a2)
            val = f.getvalue()

        with io.BytesIO(val) as f:
            d = xp.load(f)
            b1 = d['arr_0']
            b2 = d['arr_1']

        return b1, b2

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_dump(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype=dtype)

        with io.BytesIO() as f:
            a.dump(f)
            val = f.getvalue()

        with io.BytesIO(val) as f:
            b = xp.load(f, allow_pickle=True)

        return b


@testing.parameterize(
    {'data': testing.shaped_arange((2, 3, 4), numpy)},
    {'data': set((1, (0, 1), 1.5))},
    {'data': (1, 2.5, 4)},
    {'data': [1, 2.5, 4]},
    {'data': "string"},
)
class TestNpzPickle(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_load_pickle(self, xp):

        with io.BytesIO() as f:
            pickle.dump(self.data, f)
            val = f.getvalue()

        with io.BytesIO(val) as f:
            b = xp.load(f, allow_pickle=True)

        return b


class TestNpzFailure(unittest.TestCase):

    @testing.numpy_nlcpy_raises()
    def test_load_pickle_failure(self, xp):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype='float32')

        with io.BytesIO() as f:
            a.dump(f)
            val = f.getvalue()

        with io.BytesIO(val) as f:
            xp.load(f, allow_pickle=False)

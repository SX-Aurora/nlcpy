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

import numpy
import nlcpy
from nlcpy import testing


def is_same_contiguity(na, va):
    if na.flags.c_contiguous != va.flags.c_contiguous or \
       na.flags.f_contiguous != va.flags.f_contiguous or \
       na._data.flags.c_contiguous != va._data.flags.c_contiguous or \
       na._data.flags.f_contiguous != va._data.flags.f_contiguous or \
       na._mask.flags.c_contiguous != va._mask.flags.c_contiguous or \
       na._mask.flags.f_contiguous != va._mask.flags.f_contiguous or \
       na._fill_value.flags.c_contiguous != va._fill_value.flags.c_contiguous or \
       na._fill_value.flags.f_contiguous != va._fill_value.flags.f_contiguous:
        return False
    else:
        return True


class TestArrayContiguity(unittest.TestCase):

    def test_is_contiguous(self):
        data = testing.shaped_arange((2, 3, 4), numpy)
        mask = testing.shaped_arange((2, 3, 4), numpy)
        fill_value = testing.shaped_arange((2, 3, 4), numpy)
        na = nlcpy.ma.array(data, mask=mask, fill_value=fill_value)

        data = testing.shaped_arange((2, 3, 4), nlcpy)
        mask = testing.shaped_arange((2, 3, 4), nlcpy)
        fill_value = testing.shaped_arange((2, 3, 4), nlcpy)
        va = nlcpy.ma.array(data, mask=mask, fill_value=fill_value)

        assert (is_same_contiguity(na, va))

        nb = na.transpose(2, 0, 1)
        vb = va.transpose(2, 0, 1)
        assert (is_same_contiguity(nb, vb))

        nc = na[::-1]
        vc = va[::-1]
        assert (is_same_contiguity(nc, vc))

        nd = na[:, :, ::2]
        vd = va[:, :, ::2]
        assert (is_same_contiguity(nd, vd))

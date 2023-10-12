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
import pytest
import numpy
import nlcpy
from nlcpy.core import dtype


class TestDtype(unittest.TestCase):

    def test_check_dtype_is_valid(self):
        assert dtype._check_dtype_is_valid(nlcpy.dtype('bool')) is True
        assert dtype._check_dtype_is_valid(nlcpy.dtype('i4')) is True
        assert dtype._check_dtype_is_valid(nlcpy.dtype('i8')) is True
        assert dtype._check_dtype_is_valid(nlcpy.dtype('u4')) is True
        assert dtype._check_dtype_is_valid(nlcpy.dtype('u8')) is True
        assert dtype._check_dtype_is_valid(nlcpy.dtype('f4')) is True
        assert dtype._check_dtype_is_valid(nlcpy.dtype('f8')) is True
        assert dtype._check_dtype_is_valid(nlcpy.dtype('c8')) is True
        assert dtype._check_dtype_is_valid(nlcpy.dtype('c16')) is True

        assert dtype._check_dtype_is_valid(nlcpy.dtype('i2')) is False
        assert dtype._check_dtype_is_valid(nlcpy.dtype('f2')) is False
        assert dtype._check_dtype_is_valid(nlcpy.dtype('object')) is False
        assert dtype._check_dtype_is_valid(nlcpy.dtype('i8')) is True

    def test_dtype_char_convert(self):
        assert nlcpy.ndarray(1, dtype='bool').itemsize == 4
        assert nlcpy.ndarray(1, dtype='q').dtype.char == 'l'
        assert nlcpy.ndarray(1, dtype='Q').dtype.char == 'L'
        assert nlcpy.ndarray(1, dtype='i4').dtype.char == 'i'

    def test_get_dtype_number(self):
        for s in ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                  'uint32', 'uint64', 'float16', 'float32', 'float64',
                  'complex64', 'complex128']:
            dt = numpy.dtype(s)
            assert dtype.get_dtype_number(dt) == dt.num

        with pytest.raises(ValueError):
            dtype.get_dtype_number(numpy.dtype('object'))

    def test_promote_dtype_to_supported(self):
        assert dtype.promote_dtype_to_supported(numpy.dtype('i1')) is numpy.dtype('i4')
        assert dtype.promote_dtype_to_supported(numpy.dtype('i2')) is numpy.dtype('i4')
        assert dtype.promote_dtype_to_supported(numpy.dtype('u1')) is numpy.dtype('u4')
        assert dtype.promote_dtype_to_supported(numpy.dtype('u2')) is numpy.dtype('u4')
        assert dtype.promote_dtype_to_supported(numpy.dtype('f2')) is numpy.dtype('f4')
        assert dtype.promote_dtype_to_supported(numpy.dtype('i4')) is numpy.dtype('i4')
        assert dtype.promote_dtype_to_supported(numpy.dtype('u4')) is numpy.dtype('u4')
        assert dtype.promote_dtype_to_supported(numpy.dtype('f4')) is numpy.dtype('f4')

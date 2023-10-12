#
# * The source code in this file is developed independently by NEC Corporation.
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
import numpy
import nlcpy
from nlcpy.fft import _fft
from nlcpy import testing


class TestFftCoverage(unittest.TestCase):

    def test_get_complex_real_ndarray(self):
        def _helper(xp, get_func, dt_i, dt_o):
            ret = get_func(xp.empty(10, dtype=dt_i))
            assert ret.dtype.name == numpy.dtype(dt_o).name
            assert isinstance(ret, nlcpy.ndarray)

        get_func_complex = _fft._get_complex_ndarray
        _helper(numpy, get_func_complex, 'c8', 'c8')
        _helper(numpy, get_func_complex, 'c16', 'c16')
        _helper(numpy, get_func_complex, 'f4', 'c8')
        _helper(numpy, get_func_complex, 'f8', complex)

        get_func_complex_sub = _fft._get_complex_ndarray_sub
        _helper(numpy, get_func_complex_sub, 'c8', 'c8')
        _helper(numpy, get_func_complex_sub, 'c16', 'c16')
        _helper(numpy, get_func_complex_sub, 'f4', complex)
        _helper(numpy, get_func_complex_sub, 'f8', complex)
        _helper(numpy, get_func_complex_sub, 'i8', complex)

        get_func_real = _fft._get_real_ndarray
        _helper(numpy, get_func_real, 'f4', 'f4')
        _helper(numpy, get_func_real, 'i4', float)

    def test_get_real_ndarray_failure(self):
        get_func_real = _fft._get_real_ndarray
        with self.assertRaises(TypeError):
            get_func_real(nlcpy.empty(10, dtype='c8'))
        with self.assertRaises(TypeError):
            get_func_real(nlcpy.empty(10, dtype='c16'))
        with self.assertRaises(TypeError):
            get_func_real(numpy.empty(10, dtype='c8'))
        with self.assertRaises(TypeError):
            get_func_real(numpy.empty(10, dtype='c16'))
        with self.assertRaises(ValueError):
            get_func_real([0 + 0j, 1 + 1j])

    def test_is_int_cast_for_iterable_ret_false(self):
        assert _fft._is_int_cast_for_iterable((1 + 2j, 4j)) is False

    def test_fft_1d_axis_tuple_failure(self):
        with self.assertRaises(ValueError):
            nlcpy.fft.fft([1, 2, 3], axis=(0,))
        with self.assertRaises(TypeError):
            nlcpy.fft.fft([1, 2, 3], axis=('a',))

    def test_fft_nd_axes_failure(self):
        with self.assertRaises(TypeError):
            nlcpy.fft.fftn([[1, 2], [3, 4]], axes=(0 + 0j,))
        with self.assertRaises(ValueError):
            nlcpy.fft.fftn([[1, 2], [3, 4]], s=(1 + 2j,))
        with self.assertRaises(TypeError):
            nlcpy.fft.fftn([[1, 2], [3, 4]], s=1)
        with self.assertRaises(TypeError):
            nlcpy.fft.fftn([[1, 2], [3, 4]], axes=1)

    def test_fft_nd_axes_s_len_not_equal(self):
        with self.assertRaises(ValueError):
            nlcpy.fft.fftn([[1, 2], [3, 4]], axes=(1, 0), s=(0,))

    def test_fftn_axes_len_0(self):
        x = nlcpy.random.rand(2, 3, 4)
        ret = nlcpy.fft.fftn(x, axes=())
        testing.assert_array_equal(x, ret)

    def test_ifftn_axes_len_0(self):
        x = nlcpy.random.rand(2, 3, 4)
        ret = nlcpy.fft.ifftn(x, axes=())
        testing.assert_array_equal(x, ret)

    def test_rfftn_axes_len_0(self):
        x = nlcpy.random.rand(2, 3, 4)
        ret = nlcpy.fft.rfftn(x, axes=())
        testing.assert_array_equal(x, ret)

    def test_irfftn_axes_len_0(self):
        x = nlcpy.random.rand(2, 3, 4)
        ret = nlcpy.fft.irfftn(x, axes=())
        testing.assert_array_equal(x, ret)

    def test_irfft_invalid_dat_points(self):
        with self.assertRaises(ValueError):
            nlcpy.fft.irfft(nlcpy.empty((2, 1, 3)), axis=1)

    def test_fftfreq_not_integer(self):
        with self.assertRaises(ValueError):
            nlcpy.fft.fftfreq(1.2)

    def test_rfftfreq_not_integer(self):
        with self.assertRaises(ValueError):
            nlcpy.fft.rfftfreq(1.2)

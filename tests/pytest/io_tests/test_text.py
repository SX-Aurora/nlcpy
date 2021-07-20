#
# * The source code in this file is based on the soure code of NumPy.
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
# # NumPy License #
#
#     Copyright (c) 2005-2020, NumPy Developers.
#     All rights reserved.
#
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of the NumPy Developers nor the names of any contributors may be
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
import tempfile
from io import BytesIO, StringIO

from nlcpy import testing


class TestSaveTxt(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_array_float(self, xp):
        a = xp.array([[1, 2], [3, 4]], float)
        fmt = "%.18e"
        c = BytesIO()
        xp.savetxt(c, a, fmt=fmt)
        c.seek(0)
        return c.readlines()

    @testing.numpy_nlcpy_array_equal()
    def test_array_int(self, xp):
        a = xp.array([[1, 2], [3, 4]], int)
        c = BytesIO()
        xp.savetxt(c, a, fmt='%d')
        c.seek(0)
        return c.readlines()

    @testing.numpy_nlcpy_array_equal()
    def test_1D(self, xp):
        a = xp.array([1, 2, 3, 4], int)
        c = BytesIO()
        xp.savetxt(c, a, fmt='%d')
        c.seek(0)
        return c.readlines()

    @testing.numpy_nlcpy_raises()
    def test_0D(self, xp):
        c = BytesIO()
        xp.savetxt(c, xp.array(1))

    @testing.numpy_nlcpy_raises()
    def test_3D(self, xp):
        c = BytesIO()
        xp.savetxt(c, xp.array([[[1], [2]]]))

    @testing.numpy_nlcpy_array_equal()
    def test_delimiter(self, xp):
        a = xp.array([[1., 2.], [3., 4.]])
        c = BytesIO()
        xp.savetxt(c, a, delimiter=',', fmt='%d')
        c.seek(0)
        return c.readlines()

    @testing.numpy_nlcpy_array_equal()
    def test_sequence_of_format(self, xp):
        a = xp.array([(1, 2), (3, 4)])
        c = BytesIO()
        xp.savetxt(c, a, fmt=['%02d', '%3.1f'])
        c.seek(0)
        return c.readlines()

    @testing.numpy_nlcpy_array_equal()
    def test_multiformat_string(self, xp):
        a = xp.array([(1, 2), (3, 4)])
        c = BytesIO()
        xp.savetxt(c, a, fmt='%02d : %3.1f')
        c.seek(0)
        return c.readlines()

    @testing.numpy_nlcpy_array_equal()
    def test_fmt_specify_delimiter(self, xp):
        a = xp.array([(1, 2), (3, 4)])
        c = BytesIO()
        xp.savetxt(c, a, fmt='%02d : %3.1f', delimiter=',')
        c.seek(0)
        return c.readlines()

    @testing.numpy_nlcpy_raises()
    def test_invalid_fmt_specify(self, xp):
        a = xp.array([(1, 2), (3, 4)])
        c = BytesIO()
        xp.savetxt(c, a, fmt=99)

    @testing.numpy_nlcpy_array_equal()
    def test_header(self, xp):
        c = BytesIO()
        a = xp.array([(1, 2), (3, 4)], dtype=int)
        test_header_footer = 'Test header / footer'

        # Test the header keyword argument
        xp.savetxt(c, a, fmt='%1d', header=test_header_footer)
        c.seek(0)
        return c.read()

    @testing.numpy_nlcpy_array_equal()
    def test_footer(self, xp):
        c = BytesIO()
        a = xp.array([(1, 2), (3, 4)], dtype=int)
        test_header_footer = 'Test header / footer'
        xp.savetxt(c, a, fmt='%1d', footer=test_header_footer)
        c.seek(0)
        return c.read()

    @testing.numpy_nlcpy_array_equal()
    def test_header_with_comments(self, xp):
        c = BytesIO()
        a = xp.array([(1, 2), (3, 4)], dtype=int)
        test_header_footer = 'Test header / footer'
        commentstr = '% '
        xp.savetxt(c, a, fmt='%1d',
                   header=test_header_footer, comments=commentstr)
        c.seek(0)
        return c.read()

    @testing.numpy_nlcpy_array_equal()
    def test_footer_with_comments(self, xp):
        c = BytesIO()
        a = xp.array([(1, 2), (3, 4)], dtype=int)
        test_header_footer = 'Test header / footer'
        commentstr = '% '
        xp.savetxt(c, a, fmt='%1d',
                   footer=test_header_footer, comments=commentstr)
        c.seek(0)
        return c.read()

    @testing.numpy_nlcpy_array_equal()
    def test_file_roundtrip(self, xp):
        a = xp.array([(1, 2), (3, 4)])
        with tempfile.TemporaryDirectory() as path:
            f = path + 'tempfile'
            xp.savetxt(f, a)
            return xp.loadtxt(f)

    @testing.numpy_nlcpy_array_equal()
    def test_complex_arrays_one_fmt(self, xp):
        a = xp.full([2, 2], xp.pi + 1.0j * xp.e)
        c = BytesIO()
        xp.savetxt(c, a, fmt=' %+.3e')
        c.seek(0)
        return c.readlines()

    @testing.numpy_nlcpy_array_equal()
    def test_complex_arrays_one_fmt_for_real_and_imag(self, xp):
        a = xp.full([2, 2], xp.pi + 1.0j * xp.e)
        c = BytesIO()
        xp.savetxt(c, a, fmt='  %+.3e' * 2 * 2)
        c.seek(0)
        return c.readlines()

    @testing.numpy_nlcpy_array_equal()
    def test_complex_arrays_one_fmt_for_complex(self, xp):
        a = xp.full([2, 2], xp.pi + 1.0j * xp.e)
        c = BytesIO()
        xp.savetxt(c, a, fmt=['(%.3e%+.3ej)'] * 2)
        c.seek(0)
        return c.readlines()

    @testing.numpy_nlcpy_array_equal()
    def test_complex_negative_exponent(self, xp):
        a = xp.full([2, 2], xp.pi + 1.0j * xp.e)
        c = BytesIO()
        xp.savetxt(c, a, fmt='%.3e')
        c.seek(0)
        return c.readlines()

    @testing.numpy_nlcpy_array_equal()
    def test_custom_writer(self, xp):
        class CustomWriter(list):
            def write(self, text):
                self.extend(text.split(b'\n'))
        w = CustomWriter()
        a = xp.array([(1, 2), (3, 4)])
        xp.savetxt(w, a)
        return xp.loadtxt(w)


@testing.parameterize(*(
    testing.product({
        'fmt': [u"%f", b"%f"],
        'iotype': [StringIO, BytesIO]
    })))
class TestUnicodeAndBytesFmt(unittest.TestCase):

    @testing.numpy_nlcpy_array_equal()
    def test_unicode_and_bytes_fmt(self, xp):
        a = xp.array([1.])
        s = self.iotype()
        xp.savetxt(s, a, fmt=self.fmt)
        s.seek(0)
        return s.read()

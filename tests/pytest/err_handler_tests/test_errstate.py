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

import pytest
import unittest
import sys
import io

import nlcpy
from nlcpy import testing


class Capture:

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._str = io.StringIO()  # redirect
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout  # restore

    def get_str(self):
        return self._str.getvalue().splitlines()


class TestErrState(unittest.TestCase):

    def setUp(self):
        self._prev = nlcpy.geterr()

    def tearDown(self):
        nlcpy.seterr(**self._prev)

    def test_seterr_all(self):
        x = nlcpy.array(1e20, dtype='f4')
        y = nlcpy.array(1e-30, dtype='f4')

        nlcpy.seterr(all='ignore')
        state = nlcpy.geterr()
        for s in state.values():
            assert s == 'ignore'
        nlcpy.divide(1, 0).get()  # zero divide
        nlcpy.multiply(x, x).get()  # overflow
        nlcpy.divide(y, 1e20).get()  # underflow
        nlcpy.sqrt(-1).get()  # invalid

        nlcpy.seterr(all='warn')
        state = nlcpy.geterr()
        for s in state.values():
            assert s == 'warn'
        with testing.assert_warns(RuntimeWarning):
            nlcpy.divide(1, 0).get()  # zero divide
        with testing.assert_warns(RuntimeWarning):
            nlcpy.multiply(x, x).get()  # overflow
        with testing.assert_warns(RuntimeWarning):
            nlcpy.divide(y, 1e20).get()  # underflow
        with testing.assert_warns(RuntimeWarning):
            nlcpy.sqrt(-1).get()  # invalid

        nlcpy.seterr(all='raise')
        state = nlcpy.geterr()
        for s in state.values():
            assert s == 'raise'
        with pytest.raises(FloatingPointError):
            nlcpy.divide(1, 0).get()  # zero divide
        with pytest.raises(FloatingPointError):
            nlcpy.multiply(x, x).get()  # overflow
        with pytest.raises(FloatingPointError):
            nlcpy.divide(y, 1e20).get()  # underflow
        with pytest.raises(FloatingPointError):
            nlcpy.sqrt(-1).get()  # invalid

        nlcpy.seterr(all='print')
        state = nlcpy.geterr()
        for s in state.values():
            assert s == 'print'
        with Capture() as c:
            nlcpy.divide(1, 0).get()  # zero divide
            assert 'divide by zero' in c.get_str()[-1]
            nlcpy.multiply(x, x).get()  # overflow
            assert 'overflow' in c.get_str()[-1]
            nlcpy.divide(y, 1e20).get()  # underflow
            assert 'underflow' in c.get_str()[-1]
            nlcpy.sqrt(-1).get()  # invalid
            assert 'invalid' in c.get_str()[-1]

    def test_seterr_divide(self):
        nlcpy.seterr(divide='ignore')
        assert nlcpy.geterr()['divide'] == 'ignore'
        nlcpy.divide(1, 0).get()

        nlcpy.seterr(divide='warn')
        assert nlcpy.geterr()['divide'] == 'warn'
        with testing.assert_warns(RuntimeWarning):
            nlcpy.divide(1, 0).get()

        nlcpy.seterr(divide='raise')
        assert nlcpy.geterr()['divide'] == 'raise'
        with pytest.raises(FloatingPointError):
            nlcpy.divide(1, 0).get()

        nlcpy.seterr(divide='print')
        assert nlcpy.geterr()['divide'] == 'print'
        with Capture() as c:
            nlcpy.divide(1, 0).get()
            assert 'RuntimeWarning' in c.get_str()[0]

    def test_seterr_over(self):
        x = nlcpy.array(1e20, dtype='f4')

        nlcpy.seterr(over='ignore')
        assert nlcpy.geterr()['over'] == 'ignore'
        nlcpy.multiply(x, x).get()

        nlcpy.seterr(over='warn')
        assert nlcpy.geterr()['over'] == 'warn'
        with testing.assert_warns(RuntimeWarning):
            nlcpy.multiply(x, x).get()

        nlcpy.seterr(over='raise')
        assert nlcpy.geterr()['over'] == 'raise'
        with pytest.raises(FloatingPointError):
            nlcpy.multiply(x, x).get()

        nlcpy.seterr(over='print')
        assert nlcpy.geterr()['over'] == 'print'
        with Capture() as c:
            nlcpy.multiply(x, x).get()
            assert 'RuntimeWarning' in c.get_str()[0]

    def test_seterr_under(self):
        x = nlcpy.array(1e-30, dtype='f4')

        nlcpy.seterr(under='ignore')
        assert nlcpy.geterr()['under'] == 'ignore'
        nlcpy.divide(x, 1e20).get()

        nlcpy.seterr(under='warn')
        assert nlcpy.geterr()['under'] == 'warn'
        with testing.assert_warns(RuntimeWarning):
            nlcpy.divide(x, 1e20).get()

        nlcpy.seterr(under='raise')
        assert nlcpy.geterr()['under'] == 'raise'
        with pytest.raises(FloatingPointError):
            nlcpy.divide(x, 1e20).get()

        nlcpy.seterr(under='print')
        assert nlcpy.geterr()['under'] == 'print'
        with Capture() as c:
            nlcpy.divide(x, 1e20).get()
            assert 'RuntimeWarning' in c.get_str()[0]

    def test_seterr_invalid(self):
        nlcpy.seterr(invalid='ignore')
        assert nlcpy.geterr()['invalid'] == 'ignore'
        nlcpy.sqrt(-1).get()

        nlcpy.seterr(invalid='warn')
        assert nlcpy.geterr()['invalid'] == 'warn'
        with testing.assert_warns(RuntimeWarning):
            nlcpy.sqrt(-1).get()

        nlcpy.seterr(invalid='raise')
        assert nlcpy.geterr()['invalid'] == 'raise'
        with pytest.raises(FloatingPointError):
            nlcpy.sqrt(-1).get()

        nlcpy.seterr(invalid='print')
        assert nlcpy.geterr()['invalid'] == 'print'
        with Capture() as c:
            nlcpy.sqrt(-1).get()
            assert 'RuntimeWarning' in c.get_str()[0]

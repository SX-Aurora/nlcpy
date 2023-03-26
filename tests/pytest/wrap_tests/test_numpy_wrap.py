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
import io
import warnings
import platform
import pytest

import numpy

import nlcpy
from nlcpy import testing


def _is_skip():
    v = list(map(lambda x: int(x), platform.python_version().split('.')))
    return True if v[1] <= 6 else False


class TestNumpyWrap(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_numpy_wrap(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([1, 2, xp.nan, 4, 5])
        out = xp.empty([5])
        ret = xp.nancumsum(a, out=out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_numpy_wrap_out_in_kwargs(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([1, 2, xp.nan, 4, 5])
        out = xp.empty([5])
        ret = xp.nancumsum(a, out=out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_numpy_wrap_out_in_args(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([1, 2, xp.nan, 4, 5])
        out = xp.empty([5])
        ret = xp.nancumsum(a, 0, numpy.float64, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_allclose()
    def test_apply_along_axis(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(12).reshape(3, 4)
        ret = xp.apply_along_axis(xp.sum, 1, a)
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_allclose()
    def test_piecewise(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.linspace(-2.5, 2.5, 6)
        condlist = (a < 0, a >= 0)
        funclist = [lambda a: -a, lambda a: a]
        ret = xp.piecewise(a, condlist=condlist, funclist=funclist)
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_allclose()
    def test_piecewise_numpy_func(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.linspace(-2.5, 2.5, 6)
        condlist = (a < 0, a >= 0)
        funclist = [numpy.positive, numpy.negative]
        out = numpy.zeros([3])
        ret = xp.piecewise(a, condlist, funclist, out=out)
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_allclose()
    def test_piecewise_nlcpy_func(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.linspace(-2.5, 2.5, 6)
        condlist = (a < 0, a >= 0)
        funclist = [nlcpy.positive, nlcpy.negative]
        out = nlcpy.zeros([3])
        ret = xp.piecewise(a, condlist, funclist, out=out)
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_allclose()
    def test_fromfunction(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        ret = xp.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_put_1(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(5)
        xp.put(a, [0, 2], v=(-44, -55))
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_put_2(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(5)
        xp.put(a=a, ind=[0, 2], v=(-44, -55))
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_put(self, xp):
        a = xp.arange(5)
        a.put([0, 2], [-44, -55])
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_put_along_axis_1(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(6).reshape(2, 3)
        ind = xp.array([[1], [0]])
        xp.put_along_axis(a, ind, 99, axis=1)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_put_along_axis_2(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(6).reshape(2, 3)
        ind = xp.array([[1], [0]])
        xp.put_along_axis(arr=a, indices=ind, values=99, axis=1)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_putmask(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(6).reshape(2, 3)
        xp.putmask(a, a > 2, a**2)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_place_1(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(6).reshape(2, 3)
        xp.place(a, a > 2, [44, 55])
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_place_2(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(6).reshape(2, 3)
        xp.place(arr=a, mask=a > 2, vals=[44, 55])
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_nan_to_num_1(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([1, 2, xp.inf, xp.nan, 5])
        ret = xp.nan_to_num(a)
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_nan_to_num_2(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([1, 2, xp.inf, xp.nan, 5])
        xp.nan_to_num(a, False)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_nan_to_num_3(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([1, 2, xp.inf, xp.nan, 5])
        xp.nan_to_num(x=a, copy=False)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_nan_to_num_4(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([1, 2, xp.inf, xp.nan, 5])
        xp.nan_to_num(x=a, copy=0)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_itemset(self, xp):
        a = xp.arange(3)
        a.itemset(1, 10)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_byteswap_1(self, xp):
        a = xp.array([1, 256, 8755])
        a.byteswap(True)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_byteswap_2(self, xp):
        a = xp.array([1, 256, 8755])
        a.byteswap(inplace=True)
        return a

    def test_lookfor(self):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        sio = io.StringIO()
        nlcpy.lookfor("svd", output=sio)
        res = sio.getvalue()
        assert (res.count("nlcpy") > 0)

    def test_info(self):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        sio = io.StringIO()
        nlcpy.info("arange", output=sio)
        res = sio.getvalue()
        assert (res.count("nlcpy") > 0)

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_partition(self, xp):
        a = xp.array([3, 4, 2, 1])
        a.partition(3)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_setfield(self, xp):
        a = xp.eye(3)
        a.setfield(3, numpy.int32)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_choose_with_out_1(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        choices = [-10, 10]
        out = xp.zeros([3, 3])
        ret = xp.choose(a, choices, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_choose_with_out_2(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        choices = [-10, 10]
        out = xp.zeros([3, 3])
        ret = xp.choose(a, choices, out=out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_compress_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(6).reshape(3, 2)
        out = xp.zeros([1, 2], dtype='l')
        ret = xp.compress([0, 1], a, 0, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_compress_with_out(self, xp):
        a = xp.arange(6).reshape(3, 2)
        out = xp.zeros([1, 2], dtype='l')
        ret = a.compress([0, 1], 0, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_cumprod_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(6).reshape(2, 3)
        out = xp.zeros([2, 3])
        ret = xp.cumprod(a, 0, 'd', out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_cumprod_with_out(self, xp):
        a = xp.arange(6).reshape(2, 3)
        out = xp.zeros([2, 3])
        ret = a.cumprod(0, 'd', out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_trace_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(8).reshape(2, 2, 2)
        out = xp.zeros(2)
        ret = xp.trace(a, 0, 1, 0, 'd', out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_trace_with_out(self, xp):
        a = xp.arange(8).reshape(2, 2, 2)
        out = xp.zeros(2)
        ret = a.trace(0, 1, 0, 'd', out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_around_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(3)
        out = xp.zeros(3)
        ret = xp.around(a, 0, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_round_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(3)
        out = xp.zeros(3)
        ret = xp.round_(a, 0, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_round_with_out(self, xp):
        a = xp.arange(3)
        out = xp.zeros(3)
        ret = a.round(0, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_fix_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        out = xp.array(0, dtype='d')
        ret = xp.fix(3.14, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_nanprod_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([[1, 2], [3, xp.nan]])
        out = xp.array(0)
        ret = xp.nanprod(a, None, 'd', out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_nansum_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([[1, 2], [3, xp.nan]])
        out = xp.array(0)
        ret = xp.nansum(a, None, 'd', out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_nancumprod_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([[1, 2], [3, xp.nan]])
        out = xp.zeros(4)
        ret = xp.nancumprod(a, None, 'd', out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_nancumsum_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([[1, 2], [3, xp.nan]])
        out = xp.zeros(4)
        ret = xp.nancumsum(a, None, 'd', out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_lcm_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = [xp.array(10), xp.array(12)]
        b = [xp.array(12), xp.array(20)]
        out = xp.zeros(2)
        ret = xp.lcm(a, b, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_gcd_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        out = xp.array(0)
        ret = xp.gcd(12, 20, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_isnat_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        out = xp.array(0)
        ret = xp.isnat(numpy.datetime64("NaT"), out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_isneginf_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        out = xp.array(0)
        ret = xp.isneginf(xp.NINF, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_isposinf_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        out = xp.array(0)
        ret = xp.isposinf(xp.inf, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_float_power_with_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(6)
        out = xp.zeros(6)
        ret = xp.float_power(a, 2, out)
        assert (isinstance(ret, xp.ndarray))
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_modf_with_out_1(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        out1 = xp.zeros(2)
        out2 = xp.zeros(2)
        ret = xp.modf([0, 3.5], out1, out2)
        assert (isinstance(ret[0], xp.ndarray))
        assert (isinstance(ret[1], xp.ndarray))
        assert (ret[0] is out1)
        assert (ret[1] is out2)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_modf_with_list_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        out = (xp.zeros(2), xp.zeros(2))
        ret = xp.modf([0, 3.5], out=out)
        assert (isinstance(ret[0], xp.ndarray))
        assert (isinstance(ret[1], xp.ndarray))
        assert (ret[0] is out[0])
        assert (ret[1] is out[1])
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_frexp_with_out_1(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        out1 = xp.zeros(2)
        out2 = xp.zeros(2)
        ret = xp.frexp([1, 2], out1, out2)
        assert (isinstance(ret[0], xp.ndarray))
        assert (isinstance(ret[1], xp.ndarray))
        assert (ret[0] is out1)
        assert (ret[1] is out2)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_frexp_with_list_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        out = (xp.zeros(2), xp.zeros(2))
        ret = xp.frexp([1, 2], out=out)
        assert (isinstance(ret[0], xp.ndarray))
        assert (isinstance(ret[1], xp.ndarray))
        assert (ret[0] is out[0])
        assert (ret[1] is out[1])
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_divmol_with_out_1(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(5)
        out1 = xp.zeros(5)
        out2 = xp.zeros(5)
        ret = xp.divmod(a, 3, out1, out2)
        assert (isinstance(ret[0], xp.ndarray))
        assert (isinstance(ret[1], xp.ndarray))
        assert (ret[0] is out1)
        assert (ret[1] is out2)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_divmol_with_list_out(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(5)
        out = (xp.zeros(5), xp.zeros(5))
        ret = xp.divmod(a, 3, out=out)
        assert (isinstance(ret[0], xp.ndarray))
        assert (isinstance(ret[1], xp.ndarray))
        assert (ret[0] is out[0])
        assert (ret[1] is out[1])
        return ret

    @testing.numpy_nlcpy_allclose()
    def test_det(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([[1, 2], [3, 4]])
        ret = xp.linalg.det(a)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_sort_complex(self, xp):
        a = xp.arange(5, dtype='F')
        ret = xp.sort(a)
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_sort_complex(self, xp):
        a = xp.arange(5, dtype='F')
        a.sort()
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_sort_kind(self, xp):
        a = xp.arange(5)
        ret = xp.sort(a, kind='quicksort')
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_sort_kind(self, xp):
        a = xp.arange(5)
        a.sort(kind='quicksort')
        return a

    def test_sort_order(self):
        a = nlcpy.arange(5)
        with self.assertRaises(ValueError):
            nlcpy.sort(a, order='a')

    def test_ndarray_sort_order(self):
        a = nlcpy.arange(5)
        with self.assertRaises(ValueError):
            a.sort(order='a')

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_astype_casting(self, xp):
        a = xp.array([1 + 2j, 3 + 4j])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            return a.astype('i', casting='unsafe')

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_astype_inplace(self, xp):
        a = xp.array([1 + 2j, 3 + 4j])
        ret = a.astype('D', casting='unsafe', copy=False)
        assert (ret is a)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_take_clip(self, xp):
        a = xp.arange(5)
        out = xp.empty(2, dtype='l')
        ret = xp.take(a, (4, 10), mode='clip', out=out)
        assert (out is ret)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_take_clip_2(self, xp):
        a = xp.arange(5)
        out = xp.empty(2, dtype='l')
        ret = xp.take(a, (4, 10), None, out, 'clip')
        assert (out is ret)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_take_clip(self, xp):
        a = xp.arange(5)
        out = xp.empty(2, dtype='l')
        ret = a.take((4, 10), mode='clip', out=out)
        assert (out is ret)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_take_clip_2(self, xp):
        a = xp.arange(5)
        out = xp.empty(2, dtype='l')
        ret = a.take((4, 10), None, out, 'clip')
        assert (out is ret)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_flatten_order(self, xp):
        a = xp.arange(4).reshape(2, 2)
        ret = a.flatten(order='F')
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_dot_bool(self, xp):
        a = xp.array([[True, False]])
        b = xp.array([[True], [False]])
        ret = xp.dot(a, b)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_dot_with_out(self, xp):
        a = xp.arange(4).reshape(2, 2)
        out = xp.empty([2, 2], dtype='l')
        ret = xp.dot(a, a, out=out)
        assert (ret is out)
        return out

    @testing.numpy_nlcpy_array_equal()
    def test_dot_with_out_2(self, xp):
        a = xp.arange(4).reshape(2, 2)
        out = xp.empty([2, 2], dtype='l')
        ret = xp.dot(a, a, out)
        assert (ret is out)
        return out

    @testing.numpy_nlcpy_array_equal()
    def test_matmul_bool(self, xp):
        a = xp.array([[True, False]])
        b = xp.array([[True], [False]])
        ret = xp.matmul(a, b)
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_matmul_with_out(self, xp):
        a = xp.arange(4).reshape(2, 2)
        out = xp.empty([2, 2], dtype='l')
        ret = xp.matmul(a, a, out=out)
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_matmul_with_out_2(self, xp):
        a = xp.arange(4).reshape(2, 2)
        out = xp.empty([2, 2], dtype='l')
        ret = xp.matmul(a, a, out)
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_matmul_3d(self, xp):
        a = xp.arange(8).reshape(2, 2, 2)
        ret = xp.matmul(a, a)
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_inner_2d(self, xp):
        a = xp.arange(4).reshape(2, 2)
        ret = xp.inner(a, a)
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_unique_complex(self, xp):
        a = xp.array([1 + 2j, 3 + 4j, 1 + 2j, 3 + 5j])
        ret = xp.unique(a)
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_mean_multi_axis(self, xp):
        a = xp.arange(16).reshape(2, 2, 2, 2)
        ret = xp.mean(a, axis=(1, 2))
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_ndarray_mean_multi_axis(self, xp):
        a = xp.arange(16).reshape(2, 2, 2, 2)
        ret = a.mean(axis=(1, 2))
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_at_unary(self, xp):
        a = xp.arange(4)
        xp.negative.at(a, [0, 1])
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_at_binary(self, xp):
        a = xp.arange(4)
        b = xp.arange(2)
        xp.add.at(a, [0, 1], b)
        return a

    @testing.numpy_nlcpy_array_equal()
    def test_reduce_where_broadcast(self, xp):
        a = xp.array([[1, 2], [3, 4]])
        w = xp.array([[True], [False]])
        ret = xp.add.reduce(a, where=w)
        assert (isinstance(ret, xp.ndarray) or numpy.isscalar(ret))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_reduce_where_broadcast_with_out(self, xp):
        a = xp.array([[1, 2], [3, 4]])
        w = xp.array([[True], [False]])
        out = xp.zeros(2, dtype='l')
        ret = xp.add.reduce(a, where=w, out=out)
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_reduce_where_broadcast_with_out_2(self, xp):
        a = xp.array([[1, 2], [3, 4]])
        w = xp.array([[True], [False]])
        out = xp.array(0, dtype='l')
        ret = xp.add.reduce(a, None, None, out, where=w)
        assert (ret is out)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_outer_unsafe(self, xp):
        a = xp.array([1, 2])
        b = xp.array([1 + 2j, 3 + 4j])
        ret = xp.subtract.outer(a, b, casting='unsafe')
        assert (isinstance(ret, xp.ndarray))
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_cond(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])
        ret = xp.linalg.cond(a, 1)
        return ret

    @testing.numpy_nlcpy_array_equal()
    def test_multi_dot(self, xp):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = xp.arange(12).reshape(3, 4)
        b = xp.arange(8).reshape(4, 2)
        c = xp.arange(6).reshape(2, 3)
        out = xp.zeros([3, 3], dtype='l')
        ret = xp.linalg.multi_dot([a, b, c], out=out)
        assert (ret is out)
        return ret


@testing.parameterize(*testing.product({
    'attr': [
        'asmatrix',
        'byte_bounds',
        'get_array_wrap',
        'getbufsize',
        'geterrcall',
        'mafromtxt',
        'maximum_sctype',
        'memmap',
        'min_scalar_type',
        'mintypecode',
        'nditer',
        'nested_iters',
        'set_numeric_ops',
        'setbufsize',
        'seterrcall',
        'shares_memory',
        'test',
        'trim_zeros',
        'vectorize',
        'who',
    ]}
))
class TestNotSupported(unittest.TestCase):
    def test_notsupported(self):
        with self.assertRaises(AttributeError):
            getattr(nlcpy, self.attr)


class TestNotSupportedArray(unittest.TestCase):
    def test_ndarray_flat(self):
        with self.assertRaises(AttributeError):
            nlcpy.arange(3).flat

    def test_maskedarray(self):
        with self.assertRaises(AttributeError):
            a = nlcpy.ma.array(1)
            a.byteswap()


@testing.parameterize(*testing.product({
    'arg': [
        ('beta', 2, 4, 10),
        ('chisquare', 4.5, 10),
        ('choice', 5, 3),
        ('dirichlet', [0.7, 1.2], 2),
        ('f', 1, 48, 1000),
        ('hypergeometric', 100, 2, 10, 1000),
        ('laplace', 0.0, 1.0, 10),
        ('logseries', 0.6, 100),
        ('multinomial', 20, [1 / 6.] * 6, 19),
        ('multivariate_normal', [0, 0], [[1, 0], [0, 100]]),
        ('negative_binomial', 1, 0.1, 1000),
        ('noncentral_chisquare', 3, 20),
        ('noncentral_f', 3, 20, 3.0),
        ('pareto', 3),
        ('power', 5),
        ('rayleigh', 3),
        ('triangular', -3, 0, 8),
        ('standard_t', 10, 10),
        ('vonmises', 0, 4, 10),
        ('wald', 3, 2, 10),
        ('zipf', 4.0, 100),
    ]}
))
class TestRandom(unittest.TestCase):
    # checks followings
    # - numpy_wrap function does not change the seed of NumPy
    # - the random number values are depend on only the seed of NLCPy
    # - the random number values change every called
    def test_random(self):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        np_f = getattr(numpy.random, self.arg[0])
        vp_f = getattr(nlcpy.random, self.arg[0])
        numpy.random.seed(0)
        na1 = np_f(*self.arg[1:])
        numpy.random.seed(0)
        nlcpy.random.seed(1)
        va1 = vp_f(*self.arg[1:])
        na2 = np_f(*self.arg[1:])
        nlcpy.random.seed(1)
        va2 = vp_f(*self.arg[1:])
        state = nlcpy.random.get_state()
        va3 = vp_f(*self.arg[1:])
        nlcpy.random.set_state(state)
        va4 = vp_f(*self.arg[1:])
        assert numpy.all(na1 == na2)
        assert nlcpy.all(va1 == va2)
        assert nlcpy.any(va2 != va3)
        assert nlcpy.all(va3 == va4)


class TestRandom2(unittest.TestCase):

    def test_random_pos_feedback(self):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        vs = nlcpy.random.get_state()
        nlcpy.random.beta(1, 1, 1)
        v1 = nlcpy.random.rand()

        nlcpy.random.set_state(vs)
        nlcpy.random.beta(1, 1, 10)
        v2 = nlcpy.random.rand()
        assert (v1 != v2)


@testing.parameterize(*testing.product({
    'attr': [
        'PCG64',
        'Philox',
        'SFC64',
    ]}
))
class TestNotSupportedRandom(unittest.TestCase):
    def test_notsupported(self):
        with self.assertRaises(AttributeError):
            getattr(nlcpy.random, self.attr)

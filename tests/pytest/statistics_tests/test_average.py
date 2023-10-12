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
import warnings
import platform
import pytest
import numpy
import nlcpy
from nlcpy import testing


def _is_skip():
    v = list(map(lambda x: int(x), platform.python_version().split('.')))
    return True if v[1] <= 6 else False


_complex_dtypes = (numpy.complex64, numpy.complex128)
_float_dtypes = (numpy.float64, numpy.float32)

_base_param = (
    {'shape': 10, 'axis': None},
    {'shape': (10,), 'axis': None},
    {'shape': (10,), 'axis': 0},
    {'shape': (10,), 'axis': -1},
    {'shape': (3, 4), 'axis': None},
    {'shape': (3, 4), 'axis': 0},
    {'shape': (3, 4), 'axis': 1},
    {'shape': (3, 4), 'axis': -1},
    {'shape': (3, 4), 'axis': -2},
    {'shape': (3, 5, 4), 'axis': None},
    {'shape': (3, 5, 4), 'axis': 0},
    {'shape': (3, 5, 4), 'axis': 1},
    {'shape': (3, 5, 4), 'axis': 2},
    {'shape': (3, 5, 4), 'axis': -1},
    {'shape': (3, 5, 4), 'axis': -2},
    {'shape': (3, 5, 4), 'axis': -3},
    {'shape': (3, 5, 6, 4), 'axis': None},
    {'shape': (3, 5, 6, 4), 'axis': 0},
    {'shape': (3, 5, 6, 4), 'axis': 1},
    {'shape': (3, 5, 6, 4), 'axis': 2},
    {'shape': (3, 5, 6, 4), 'axis': 3},
    {'shape': (3, 5, 6, 4), 'axis': -1},
    {'shape': (3, 5, 6, 4), 'axis': -2},
    {'shape': (3, 5, 6, 4), 'axis': -3},
    {'shape': (3, 5, 6, 4), 'axis': -4},
)


###########################################
# average
###########################################

@testing.parameterize(*_base_param)
class TestAverage(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_average(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.average(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_average_returned(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.average(a, axis=self.axis, returned=True)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_average_weights(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        w = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.average(a, weights=w, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_average_weights_returned(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        w = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.average(a, weights=w, axis=self.axis, returned=True)


@testing.parameterize(
    {'a_shape': (3, 2), 'w_shape': (2,), 'axis': 1},
    {'a_shape': (3, 2), 'w_shape': (3,), 'axis': 0},
    {'a_shape': (3, 2, 5), 'w_shape': (5,), 'axis': 2},
    {'a_shape': (3, 4, 5), 'w_shape': (4,), 'axis': 1},
    {'a_shape': (3, 4, 5), 'w_shape': (3,), 'axis': 0},
    {'a_shape': (3, 4, 5), 'w_shape': (4,), 'axis': -2},
)
class TestAverageWeightsShapeDiffer(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_average_weights_shape_differ(self, xp, dtype):
        a = testing.shaped_arange(self.a_shape, xp, dtype=dtype)
        w = testing.shaped_arange(self.w_shape, xp, dtype=dtype)
        return xp.average(a, weights=w, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_average_weights_shape_differ_returned(self, xp, dtype):
        a = testing.shaped_arange(self.a_shape, xp, dtype=dtype)
        w = testing.shaped_arange(self.w_shape, xp, dtype=dtype)
        return xp.average(a, weights=w, axis=self.axis, returned=True)


class TestAverageArrayNone(unittest.TestCase):

    def test_average_array_none(self):
        assert nlcpy.average(None) is None


class TestAverageFailure(unittest.TestCase):

    def test_average_weights_0(self):
        a = nlcpy.ones(10)
        w = nlcpy.zeros(10)
        with self.assertRaises(ZeroDivisionError):
            nlcpy.average(a, weights=w)

    def test_average_axis_none(self):
        a = nlcpy.ones((3, 2))
        w = nlcpy.ones((3,))
        with self.assertRaises(TypeError):
            nlcpy.average(a, weights=w, axis=None)

    def test_average_weights_not_1d(self):
        a = nlcpy.ones((3, 2, 3))
        w = nlcpy.ones((3, 2))
        with self.assertRaises(TypeError):
            nlcpy.average(a, weights=w, axis=0)

    def test_average_weights_shape_mismatch(self):
        a = nlcpy.ones((3, 2))
        w = nlcpy.ones((3))
        with self.assertRaises(ValueError):
            nlcpy.average(a, weights=w, axis=1)


###########################################
# mean
###########################################

@testing.parameterize(*_base_param)
class TestMean(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_mean(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.mean(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_mean_dtype(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.mean(a, axis=self.axis, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_mean_keepdims(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.mean(a, axis=self.axis, keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_mean_out(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        o_shape = []
        if self.axis is not None:
            if self.axis < 0:
                self.axis = a.ndim + self.axis
            if a.ndim > 1:
                for i, s in enumerate(list(self.shape)):
                    if i != self.axis:
                        o_shape.append(s)
        o = xp.empty(o_shape, dtype=dtype)
        o2 = xp.mean(a, out=o, axis=self.axis)
        assert id(o) == id(o2)
        return o


class TestMeanArrayNone(unittest.TestCase):

    def test_mean_array_none(self):
        assert nlcpy.mean(None) is None


class TestMeanFailure(unittest.TestCase):

    def test_mean_out_shape_mismatch(self):
        a = nlcpy.ones((3, 2))
        o = nlcpy.ones((3))
        with self.assertRaises(ValueError):
            nlcpy.mean(a, out=o)

    def test_mean_out_not_array(self):
        a = nlcpy.ones(10)
        o = numpy.empty(())
        with self.assertRaises(TypeError):
            nlcpy.mean(a, out=o)


###########################################
# median
###########################################

@testing.parameterize(*_base_param)
class TestMedian(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_median(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.median(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_median_overwrite(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.median(a, axis=self.axis, overwrite_input=True)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_median_keepdims(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.median(a, axis=self.axis, keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_median_out(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        o_shape = []
        if self.axis is not None:
            if self.axis < 0:
                self.axis = a.ndim + self.axis
            if a.ndim > 1:
                for i, s in enumerate(list(self.shape)):
                    if i != self.axis:
                        o_shape.append(s)
        o = xp.empty(o_shape, dtype=dtype)
        o2 = xp.median(a, out=o, axis=self.axis)
        assert id(o) == id(o2)
        return o


class TestMedianArrayNone(unittest.TestCase):

    def test_median_array_none(self):
        assert nlcpy.median(None) is None


class TestMedianFailure(unittest.TestCase):

    def test_median_out_shape_mismatch(self):
        a = nlcpy.ones((3, 2))
        o = nlcpy.ones((3))
        with self.assertRaises(ValueError):
            nlcpy.median(a, out=o)

    def test_median_out_not_array(self):
        a = nlcpy.ones(10)
        o = numpy.empty(())
        with self.assertRaises(TypeError):
            nlcpy.median(a, out=o)


###########################################
# std
###########################################

@testing.parameterize(*_base_param)
class TestStd(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_std(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.std(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_std_ddof(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ret = xp.std(a, axis=self.axis, ddof=1)
            if xp is nlcpy:
                nlcpy.request.flush()
        return ret

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_std_dtype(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.std(a, axis=self.axis, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_std_keepdims(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.std(a, axis=self.axis, keepdims=True)

    @testing.for_all_dtypes(name='dtype_i')
    @testing.for_float_dtypes(name='dtype_o')
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_std_out(self, xp, dtype_i, dtype_o):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype_i)
        o_shape = []
        if self.axis is not None:
            if self.axis < 0:
                self.axis = a.ndim + self.axis
            if a.ndim > 1:
                for i, s in enumerate(list(self.shape)):
                    if i != self.axis:
                        o_shape.append(s)
        o = xp.empty(o_shape, dtype=dtype_o)
        o2 = xp.std(a, out=o, axis=self.axis)
        assert id(o) == id(o2)
        return o


class TestStdArrayNone(unittest.TestCase):

    def test_std_array_none(self):
        assert nlcpy.std(None) is None


class TestStdFailure(unittest.TestCase):

    def test_std_out_shape_mismatch(self):
        a = nlcpy.ones((3, 2))
        o = nlcpy.ones((3))
        with self.assertRaises(ValueError):
            nlcpy.std(a, out=o)

    def test_std_out_not_array(self):
        a = nlcpy.ones(10)
        o = numpy.empty(())
        with self.assertRaises(TypeError):
            nlcpy.std(a, out=o)

    def test_std_zero_size_array(self):
        with self.assertRaises(ValueError):
            nlcpy.std([])

    def test_std_ddof_not_int(self):
        with self.assertRaises(ValueError):
            nlcpy.std([1, 2], ddof=1.2)


###########################################
# var
###########################################

@testing.parameterize(*_base_param)
class TestVar(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_var(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.var(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_var_ddof(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ret = xp.var(a, axis=self.axis, ddof=1)
            if xp is nlcpy:
                nlcpy.request.flush()
        return ret

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_var_dtype(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.var(a, axis=self.axis, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_var_keepdims(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.var(a, axis=self.axis, keepdims=True)

    @testing.for_all_dtypes(name='dtype_i')
    @testing.for_float_dtypes(name='dtype_o')
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_var_out(self, xp, dtype_i, dtype_o):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype_i)
        o_shape = []
        if self.axis is not None:
            if self.axis < 0:
                self.axis = a.ndim + self.axis
            if a.ndim > 1:
                for i, s in enumerate(list(self.shape)):
                    if i != self.axis:
                        o_shape.append(s)
        o = xp.empty(o_shape, dtype=dtype_o)
        o2 = xp.var(a, out=o, axis=self.axis)
        assert id(o) == id(o2)
        return o


class TestVarArrayNone(unittest.TestCase):

    def test_var_array_none(self):
        assert nlcpy.var(None) is None


class TestVarFailure(unittest.TestCase):

    def test_var_out_shape_mismatch(self):
        a = nlcpy.ones((3, 2))
        o = nlcpy.ones((3))
        with self.assertRaises(ValueError):
            nlcpy.var(a, out=o)

    def test_var_out_not_array(self):
        a = nlcpy.ones(10)
        o = numpy.empty(())
        with self.assertRaises(TypeError):
            nlcpy.var(a, out=o)

    def test_var_zero_size_array(self):
        with self.assertRaises(ValueError):
            nlcpy.var([])

    def test_var_ddof_not_int(self):
        with self.assertRaises(ValueError):
            nlcpy.var([1, 2], ddof=1.2)


###########################################
# nanmean
###########################################

@testing.parameterize(*_base_param)
class TestNanmean(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_nanmean(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.nanmean(a, axis=self.axis)

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_nanmean_with_nan(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        if a.size > 1:
            a[a.ndim * (1,)] = xp.nan
        return xp.nanmean(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_nanmean_dtype(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.nanmean(a, axis=self.axis, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_nanmean_keepdims(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.nanmean(a, axis=self.axis, keepdims=True)

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_nanmean_keepdims_with_nan(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        if a.size > 1:
            a[a.ndim * (1,)] = xp.nan
        return xp.nanmean(a, axis=self.axis, keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_nanmean_out(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        o_shape = []
        if self.axis is not None:
            if self.axis < 0:
                self.axis = a.ndim + self.axis
            if a.ndim > 1:
                for i, s in enumerate(list(self.shape)):
                    if i != self.axis:
                        o_shape.append(s)
        o = xp.empty(o_shape, dtype=dtype)
        o2 = xp.nanmean(a, out=o, axis=self.axis)
        assert id(o) == id(o2)
        return o


class TestNanmeanArrayNone(unittest.TestCase):

    def test_nanmean_array_none(self):
        assert nlcpy.nanmean(None) is None


class TestNanmeanFailure(unittest.TestCase):

    def test_nanmean_out_shape_mismatch(self):
        a = nlcpy.ones((3, 2))
        o = nlcpy.ones((3))
        with self.assertRaises(ValueError):
            nlcpy.nanmean(a, out=o)

    def test_nanmean_out_not_array(self):
        a = nlcpy.ones(10)
        o = numpy.empty(())
        with self.assertRaises(TypeError):
            nlcpy.nanmean(a, out=o)

    def test_nanmean_zero_size_array(self):
        with self.assertRaises(ValueError):
            nlcpy.nanmean([])


###########################################
# nanmedian
###########################################

@testing.parameterize(*_base_param)
class TestNanmedian(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_nanmedian(self, xp, dtype):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.nanmedian(a, axis=self.axis)

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_nanmedian_with_nan(self, xp, dtype):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        if a.size > 1:
            a[a.ndim * (1,)] = xp.nan
        return xp.nanmedian(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_nanmedian_keepdims(self, xp, dtype):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.nanmedian(a, axis=self.axis, keepdims=True)

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_nanmedian_keepdims_with_nan(self, xp, dtype):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        if a.size > 1:
            a[a.ndim * (1,)] = xp.nan
        return xp.nanmedian(a, axis=self.axis, keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_nanmedian_out(self, xp, dtype):
        if _is_skip():
            pytest.skip('Python3.6 is not testable')
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        o_shape = []
        if self.axis is not None:
            if self.axis < 0:
                self.axis = a.ndim + self.axis
            if a.ndim > 1:
                for i, s in enumerate(list(self.shape)):
                    if i != self.axis:
                        o_shape.append(s)
        o = xp.empty(o_shape, dtype=dtype)
        o2 = xp.nanmedian(a, out=o, axis=self.axis)
        assert id(o) == id(o2)
        return o


###########################################
# nanstd
###########################################

@testing.parameterize(*_base_param)
class TestNanstd(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_nanstd(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.nanstd(a, axis=self.axis)

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_nanstd_with_nan(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        if a.size > 1:
            a[a.ndim * (1,)] = xp.nan
        return xp.nanstd(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_nanstd_ddof(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ret = xp.nanstd(a, axis=self.axis, ddof=1)
            if xp is nlcpy:
                nlcpy.request.flush()
        return ret

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_nanstd_dtype(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.nanstd(a, axis=self.axis, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_nanstd_keepdims(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.nanstd(a, axis=self.axis, keepdims=True)

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_nanstd_keepdims_with_nan(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        if a.size > 1:
            a[a.ndim * (1,)] = xp.nan
        return xp.nanstd(a, axis=self.axis, keepdims=True)

    @testing.for_all_dtypes(name='dtype_i')
    @testing.for_float_dtypes(name='dtype_o')
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_nanstd_out(self, xp, dtype_i, dtype_o):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype_i)
        o_shape = []
        if self.axis is not None:
            if self.axis < 0:
                self.axis = a.ndim + self.axis
            if a.ndim > 1:
                for i, s in enumerate(list(self.shape)):
                    if i != self.axis:
                        o_shape.append(s)
        o = xp.empty(o_shape, dtype=dtype_o)
        o2 = xp.nanstd(a, out=o, axis=self.axis)
        assert id(o) == id(o2)
        return o

    @testing.for_float_dtypes(name='dtype_i')
    @testing.for_float_dtypes(name='dtype_o')
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_nanstd_out_with_nan(self, xp, dtype_i, dtype_o):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype_i)
        if a.size > 1:
            a[a.ndim * (1,)] = xp.nan
        o_shape = []
        if self.axis is not None:
            if self.axis < 0:
                self.axis = a.ndim + self.axis
            if a.ndim > 1:
                for i, s in enumerate(list(self.shape)):
                    if i != self.axis:
                        o_shape.append(s)
        o = xp.empty(o_shape, dtype=dtype_o)
        o2 = xp.nanstd(a, out=o, axis=self.axis)
        assert id(o) == id(o2)
        return o


class TestNanstdArrayNone(unittest.TestCase):

    def test_nanstd_array_none(self):
        assert nlcpy.nanstd(None) is None


class TestNanstdFailure(unittest.TestCase):

    def test_nanstd_out_shape_mismatch(self):
        a = nlcpy.ones((3, 2))
        o = nlcpy.ones((3))
        with self.assertRaises(ValueError):
            nlcpy.nanstd(a, out=o)

    def test_nanstd_out_not_array(self):
        a = nlcpy.ones(10)
        o = numpy.empty(())
        with self.assertRaises(TypeError):
            nlcpy.nanstd(a, out=o)

    def test_nanstd_zero_size_array(self):
        with self.assertRaises(ValueError):
            nlcpy.nanstd([])

    def test_nanstd_ddof_not_int(self):
        with self.assertRaises(ValueError):
            nlcpy.nanstd([1, 2], ddof=1.2)


###########################################
# nanvar
###########################################

@testing.parameterize(*_base_param)
class TestNanvar(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_nanvar(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.nanvar(a, axis=self.axis)

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_nanvar_with_nan(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        if a.size > 1:
            a[a.ndim * (1,)] = xp.nan
        return xp.nanvar(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_nanvar_ddof(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ret = xp.nanvar(a, axis=self.axis, ddof=1)
            if xp is nlcpy:
                nlcpy.request.flush()
        return ret

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_nanvar_dtype(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.nanvar(a, axis=self.axis, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_nanvar_keepdims(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        return xp.nanvar(a, axis=self.axis, keepdims=True)

    @testing.for_float_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_nanvar_keepdims_with_nan(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        if a.size > 1:
            a[a.ndim * (1,)] = xp.nan
        return xp.nanvar(a, axis=self.axis, keepdims=True)

    @testing.for_all_dtypes(name='dtype_i')
    @testing.for_float_dtypes(name='dtype_o')
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_nanvar_out(self, xp, dtype_i, dtype_o):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype_i)
        o_shape = []
        if self.axis is not None:
            if self.axis < 0:
                self.axis = a.ndim + self.axis
            if a.ndim > 1:
                for i, s in enumerate(list(self.shape)):
                    if i != self.axis:
                        o_shape.append(s)
        o = xp.empty(o_shape, dtype=dtype_o)
        o2 = xp.nanvar(a, out=o, axis=self.axis)
        assert id(o) == id(o2)
        return o

    @testing.for_dtypes(_complex_dtypes + _float_dtypes, name='dtype_i')
    @testing.for_float_dtypes(name='dtype_o')
    @testing.numpy_nlcpy_allclose(rtol=1e-3)
    def test_nanvar_out_with_nan(self, xp, dtype_i, dtype_o):
        a = testing.shaped_arange(self.shape, xp, dtype=dtype_i)
        if a.size > 1:
            a[a.ndim * (1,)] = xp.nan
        o_shape = []
        if self.axis is not None:
            if self.axis < 0:
                self.axis = a.ndim + self.axis
            if a.ndim > 1:
                for i, s in enumerate(list(self.shape)):
                    if i != self.axis:
                        o_shape.append(s)
        o = xp.empty(o_shape, dtype=dtype_o)
        o2 = xp.nanvar(a, out=o, axis=self.axis)
        assert id(o) == id(o2)
        return o


class TestNanvarArrayNone(unittest.TestCase):

    def test_nanvar_array_none(self):
        assert nlcpy.nanvar(None) is None


class TestNanvarFailure(unittest.TestCase):

    def test_nanvar_out_shape_mismatch(self):
        a = nlcpy.ones((3, 2))
        o = nlcpy.ones((3))
        with self.assertRaises(ValueError):
            nlcpy.nanvar(a, out=o)

    def test_nanvar_out_not_array(self):
        a = nlcpy.ones(10)
        o = numpy.empty(())
        with self.assertRaises(TypeError):
            nlcpy.nanvar(a, out=o)

    def test_nanvar_out_not_array_with_nan(self):
        a = nlcpy.ones(10)
        o = numpy.empty(3, dtype='object')
        a[::2] = nlcpy.nan
        with self.assertRaises(TypeError):
            nlcpy.nanvar(a, out=o)

    def test_nanvar_zero_size_array(self):
        with self.assertRaises(ValueError):
            nlcpy.nanvar([])

    def test_nanvar_ddof_not_int(self):
        with self.assertRaises(ValueError):
            nlcpy.nanvar([1, 2], ddof=1.2)

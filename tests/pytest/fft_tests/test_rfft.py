import functools # NOQA

import unittest
import pytest

import numpy as np

import nlcpy
from nlcpy import testing
from nlcpy.testing import (  # NOQA
    assert_array_equal, assert_allclose)


@testing.parameterize(*testing.product({
    'n': [None, 0, 5, 10, 15],
    'shape': [(10,), (5, 10), (3, 4, 5)],
    'norm': [None, 'ortho', ''],
}))
@testing.with_requires('numpy>=1.10.0')
class TestRfft(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-7,
                                  accept_error=(ValueError, TypeError),
                                  contiguous_check=False)
    def test_rfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        tmp = a.copy()
        out = xp.fft.rfft(a, n=self.n, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        elif xp == np and dtype is not np.float32:
            out = out.astype(np.complex128)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-7,
                                  accept_error=(ValueError, TypeError),
                                  contiguous_check=False)
    # NumPy 1.17.0 and 1.17.1 raises ZeroDivisonError due to a bug
    @testing.with_requires('numpy!=1.17.0')
    @testing.with_requires('numpy!=1.17.1')
    def test_irfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        tmp = a.copy()
        out = xp.fft.irfft(a, n=self.n, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out


@testing.parameterize(*testing.product({
    # 'n': [None, -1, 0, 0j, 1+2j, [1,2,3], [], [-1], (1,2,3), (), (-1,), ],
    'n': [None, -1, 0, ],
    'axis': [-1, 0j, 1 + 2j, [1, 2, 3], [], [-1], (1, 2, 3), (), (1, 2, -1), ],
}))
@testing.with_requires('numpy>=1.10.0')
class TestRfft1DRaise(unittest.TestCase):
    shape = (5, 5)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_raises()
    @testing.with_requires('numpy<1.20')
    def test_rfft(self, xp, dtype):
        if self.n is None and self.axis == -1:
            raise Exception("ignore case")

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfft(a, n=self.n, axis=self.axis)

        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        elif xp == np and dtype is not np.float32:
            out = out.astype(np.complex128)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_raises()
    # NumPy 1.17.0 and 1.17.1 raises ZeroDivisonError due to a bug
    @testing.with_requires('numpy!=1.17.0')
    @testing.with_requires('numpy!=1.17.1')
    @testing.with_requires('numpy<1.20')
    def test_irfft(self, xp, dtype):
        if self.n is None and self.axis == -1:
            raise Exception("ignore case")

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.rfft.irfft(a, n=self.n, axis=self.axis)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out


class TestRfft1DInvalidParam(object):

    @pytest.mark.parametrize('a', (1, 1 + 2j,
                             ["aaa"], [],
                             ("aaa",), (),))
    def _test_rfft_param_array(self, a):
        with pytest.raises(TypeError):
            nlcpy.fft.rfft(a)

    @pytest.mark.parametrize('a', ([1 + 2j], [[1 + 2j], [2 + 3j]]))
    def _test_rfft_complex_array(self, a):
        with pytest.raises(TypeError):
            nlcpy.fft.rfft(a)

    @pytest.mark.parametrize('a', (1, 1 + 2j,
                             ["aaa"], [],
                             ("aaa",), (),))
    def test_irfft_param_array(self, a):
        with pytest.raises(ValueError):
            nlcpy.fft.irfft(a)

    @pytest.mark.parametrize('param', (([1, 2, 3], 1), ([[1, 2, 3], [4, 5, 6]], -3), ))
    def test_rfft_param_axis(self, param):
        with pytest.raises(ValueError):
            nlcpy.fft.rfft(param[0], axis=param[1])

    @pytest.mark.parametrize('param', (([1, 2, 3], 1), ([[1, 2, 3], [4, 5, 6]], -3), ))
    def test_irfft_param_axis(self, param):
        with pytest.raises(ValueError):
            nlcpy.fft.irfft(param[0], axis=param[1])

    @pytest.mark.parametrize('n', (1 + 2j, [0, 1], [], [-1], (0, 1), (), (1, 2, -1), ))
    def test_rfft_param_n_list(self, n):
        with pytest.raises(TypeError):
            nlcpy.fft.rfft([[1, 2, 3], [4, 5, 6]], n=n)

    @pytest.mark.parametrize('n', ([0, 1], [], [-1], (0, 1), (), (1, 2, -1), ))
    def test_irfft_param_n_list(self, n):
        with pytest.raises(TypeError):
            nlcpy.fft.irfft([[1, 2, 3], [4, 5, 6]], n=n)


@testing.parameterize(*testing.product({
    'shape': [(5, 10), (10, 5, 10), (10, 5, 7, 9), (13, 11, 9, 7, 5)],
    'n': [None, 1, 5, 15],
    'data_order': ['F', 'C'],
    'axis': [0, 1, 2, 3, 4, 5],
    'norm': [None, 'ortho']
}))
@testing.with_requires('numpy>=1.10.0')
class TestFftOrder(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-5,
                                  accept_error=(ValueError, TypeError),
                                  contiguous_check=False)
    def test_rfft_1(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=self.data_order)
        if xp == np and self.axis >= len(self.shape) and self.norm == 'ortho':
            raise ValueError
        if xp == np and self.axis > a.ndim - 1:
            raise ValueError
        tmp = a.copy()
        out = xp.fft.rfft(a, axis=self.axis, n=self.n, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        elif xp == np and dtype is not np.float32:
            out = out.astype(np.complex128)

        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-5,
                                  accept_error=(ValueError, TypeError),
                                  contiguous_check=False)
    def test_rfft_2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=self.data_order)
        if xp == np and self.axis >= len(self.shape) and self.norm == 'ortho':
            raise ValueError
        if xp == np and self.axis > a.ndim - 1:
            raise ValueError
        tmp = a.copy()
        out = xp.fft.rfft(a, norm=self.norm, n=self.n, axis=self.axis)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        elif xp == np and dtype is not np.float32:
            out = out.astype(np.complex128)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-5,
                                  accept_error=(ValueError, TypeError),
                                  contiguous_check=False)
    def test_irfft_1(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=self.data_order)
        if xp == np and self.axis > a.ndim - 1:
            raise ValueError
        tmp = a.copy()
        out = xp.fft.irfft(a, axis=self.axis, n=self.n, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-5,
                                  accept_error=(ValueError, TypeError),
                                  contiguous_check=False)
    def test_irfft_2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=self.data_order)
        if xp == np and self.axis > a.ndim - 1:
            raise ValueError
        tmp = a.copy()
        out = xp.fft.irfft(a, norm=self.norm, n=self.n, axis=self.axis)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out

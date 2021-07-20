import functools  # NOQA
import unittest
import pytest

import numpy as np  # NOQA

import nlcpy
from nlcpy import testing


@testing.parameterize(*testing.product({
    'n': [None, 0, 5, 10, 15],
    'shape': [(10,), (5, 10), (3, 4, 5)],
    'norm': [None, 'ortho', ''],
}))
@testing.with_requires('numpy>=1.10.0')
class TestFft(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7,
                                  accept_error=ValueError,
                                  contiguous_check=False)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7,
                                  accept_error=ValueError,
                                  contiguous_check=False)
    # NumPy 1.17.0 and 1.17.1 raises ZeroDivisonError due to a bug
    @testing.with_requires('numpy!=1.17.0')
    @testing.with_requires('numpy!=1.17.1')
    def test_ifft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out


@testing.parameterize(*testing.product({
    'n': [None, -1, 0, ],
    'axis': [-1, 0j, 1 + 2j, [1, 2, 3], [], [1, 2, -1], (1, 2, 3), (), (1, 2, -1)],
}))
@testing.with_requires('numpy>=1.10.0')
class TestFft1DRaise(unittest.TestCase):
    shape = (5, 5)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_raises()
    def test_fft(self, xp, dtype):
        if self.n is None and self.axis == -1:
            raise Exception("ignore case")

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft(a, n=self.n, axis=self.axis)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_raises()
    # NumPy 1.17.0 and 1.17.1 raises ZeroDivisonError due to a bug
    @testing.with_requires('numpy!=1.17.0')
    @testing.with_requires('numpy!=1.17.1')
    def test_ifft(self, xp, dtype):
        if self.n is None and self.axis == -1:
            raise Exception("ignore case")

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifft(a, n=self.n, axis=self.axis)
        return out


class TestFft1DInvalidParam(object):

    @pytest.mark.parametrize('a', (1, 1 + 2j,
                             ["aaa"], [], ("aaa",), (),))
    def test_fft_param_array(self, a):
        with pytest.raises(ValueError):
            # ,match='Dimension n should be a positive integer not larger '
            #       'than the shape of the array along the chosen axis'):
            nlcpy.fft.fft(a)

    @pytest.mark.parametrize('a', (1, 1 + 2j,
                             ["aaa"], [], ("aaa",), (),))
    def test_ifft_param_array(self, a):
        with pytest.raises(ValueError):
            # ,match='First argument must be a complex or real sequence of single '
            #       'or double precision'):
            nlcpy.fft.ifft(a)

    @pytest.mark.parametrize('param', (([1, 2, 3], 1), ([[1, 2, 3], [4, 5, 6]], -3), ))
    def test_fft_param_axis(self, param):
        with pytest.raises(ValueError):
            # ,match='Invalid axis (-3) specified.'):
            nlcpy.fft.fft(param[0], axis=param[1])

    @pytest.mark.parametrize('param', (([1, 2, 3], 1), ([[1, 2, 3], [4, 5, 6]], -3), ))
    def test_ifft_param_axis(self, param):
        with pytest.raises(ValueError):
            # ,match='Invalid axis (-3) specified.'):
            nlcpy.fft.ifft(param[0], axis=param[1])

    @pytest.mark.parametrize('n', (1 + 2j, [0, 1], [], [-1], (0, 1), (), (-1, ), ))
    def test_fft_param_n_list(self, n):
        with pytest.raises(TypeError):
            # ,match='an integer is required (got type tuple)'):
            nlcpy.fft.fft([[1, 2, 3], [4, 5, 6]], n=n)

    @pytest.mark.parametrize('n', ([0, 1], [], [-1], (0, 1), (), (-1, ), ))
    def test_ifft_param_n_list(self, n):
        with pytest.raises(TypeError):
            # ,match='an integer is required (got type tuple)'):
            nlcpy.fft.ifft([[1, 2, 3], [4, 5, 6]], n=n)


@testing.parameterize(*testing.product({
    'shape': [(5, 10), (10, 5, 10), (10, 5, 7, 9), (13, 11, 9, 7, 5)],
    'n': [None, 1, 5, 15],
    'data_order': ['F', 'C'],
    'axis': [0, 1, 2, 3, 4, 5],
    'norm': [None, 'ortho']
}))
@testing.with_requires('numpy>=1.10.0')
class TestFftOrder(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_fft_1(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=self.data_order)
        if xp == np and self.axis > a.ndim - 1:
            raise ValueError
        out = xp.fft.fft(a, axis=self.axis, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_fft_2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=self.data_order)
        if xp == np and self.axis > a.ndim - 1:
            raise ValueError
        out = xp.fft.fft(a, norm=self.norm, n=self.n, axis=self.axis)

        if xp == np and dtype in [np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_ifft_1(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=self.data_order)
        if xp == np and self.axis > a.ndim - 1:
            raise ValueError
        out = xp.fft.ifft(a, axis=self.axis, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_ifft_2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=self.data_order)
        if xp == np and self.axis > a.ndim - 1:
            raise ValueError
        out = xp.fft.ifft(a, norm=self.norm, n=self.n, axis=self.axis)

        if xp == np and dtype in [np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

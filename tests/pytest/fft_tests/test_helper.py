import functools    # NOQA
import unittest
import pytest       # NOQA

import numpy as np  # NOQA

import nlcpy        # NOQA
from nlcpy import testing


@testing.parameterize(
    {'n': 1, 'd': 1},
    {'n': 10, 'd': 0.5},
    {'n': 100, 'd': 2},
)
@testing.with_requires('numpy>=1.10.0')
class TestFftfreq(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fftfreq(self, xp, dtype):
        out = xp.fft.fftfreq(self.n, self.d)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfftfreq(self, xp, dtype):
        out = xp.fft.rfftfreq(self.n, self.d)

        return out


@testing.parameterize(
    {'shape': (5,), 'axes': None},
    {'shape': (5,), 'axes': 0},
    {'shape': (10,), 'axes': None},
    {'shape': (10,), 'axes': 0},
    {'shape': (10, 10), 'axes': None},
    {'shape': (10, 10), 'axes': 0},
    {'shape': (10, 10), 'axes': 1},
    {'shape': (10, 10), 'axes': (0, 1)},
    {'shape': (10, 10), 'axes': (0, 0)},
    {'shape': (10, 10, 10), 'axes': None},
    {'shape': (10, 10, 10), 'axes': 0},
    {'shape': (10, 10, 10), 'axes': 1},
    {'shape': (10, 10, 10), 'axes': 2},
    {'shape': (10, 10, 10), 'axes': (0, 1)},
    {'shape': (10, 10, 10), 'axes': (0, 2)},
    {'shape': (10, 10, 10), 'axes': (1, 2)},
    {'shape': (10, 10, 10), 'axes': (0, 1, 2)},
    {'shape': (10, 10, 10), 'axes': (1, 1, 2)},
    {'shape': (10, 10, 10, 10), 'axes': None},
    {'shape': (10, 10, 10, 10), 'axes': 0},
    {'shape': (10, 10, 10, 10), 'axes': 1},
    {'shape': (10, 10, 10, 10), 'axes': 2},
    {'shape': (10, 10, 10, 10), 'axes': 3},
    {'shape': (10, 10, 10, 10), 'axes': (0, 1)},
    {'shape': (10, 10, 10, 10), 'axes': (0, 2)},
    {'shape': (10, 10, 10, 10), 'axes': (0, 3)},
    {'shape': (10, 10, 10, 10), 'axes': (1, 2)},
    {'shape': (10, 10, 10, 10), 'axes': (1, 3)},
    {'shape': (10, 10, 10, 10), 'axes': (2, 3)},
    {'shape': (10, 10, 10, 10), 'axes': (0, 1, 2)},
    {'shape': (10, 10, 10, 10), 'axes': (0, 1, 3)},
    {'shape': (10, 10, 10, 10), 'axes': (0, 2, 3)},
    {'shape': (10, 10, 10, 10), 'axes': (1, 2, 3)},
    {'shape': (10, 10, 10, 10), 'axes': (1, 3, 3)},
    {'shape': (10, 10, 10, 10), 'axes': (0, 1, 2, 3)},
)
@testing.with_requires('numpy>=1.10.0')
class TestFftshift(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fftshift(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fftshift(x, self.axes)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifftshift(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifftshift(x, self.axes)

        return out

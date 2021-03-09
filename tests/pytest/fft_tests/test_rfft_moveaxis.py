import functools  # NOQA
import unittest

import numpy as np  # NOQA

import nlcpy  # NOQA
from nlcpy import testing


@testing.parameterize(*testing.product({
    'shape': [(2, 3, 4), (2, 3, 4, 5), (6, 5, 4, 3, 2)],
    'move_1': [(0, 1), (1, 2), (-2, -1), (-2, -3)],
    'move_2': [(0, 1), (1, 2), (-2, -1), (-2, -3)],
}))
@testing.with_requires('numpy>=1.10.0')
class TestRfftMoveAxis(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7,
                                  contiguous_check=False)
    def test_rfft_move(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        ma = xp.moveaxis(a, self.move_1, self.move_2)
        out = xp.fft.rfft(ma)
        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        elif xp == np and dtype is not np.float32:
            out = out.astype(np.complex128)

        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7,
                                  contiguous_check=False)
    def test_rfft_move_slice(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        ma = xp.moveaxis(a, self.move_1, self.move_2)
        out = xp.fft.rfft(ma[1:, 1:])
        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        elif xp == np and dtype is not np.float32:
            out = out.astype(np.complex128)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7,
                                  contiguous_check=False)
    def test_irfft_move(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        ma = xp.moveaxis(a, self.move_1, self.move_2)
        out = xp.fft.irfft(ma)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7,
                                  contiguous_check=False)
    def test_irfft_move_slice(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        ma = xp.moveaxis(a, self.move_1, self.move_2)
        out = xp.fft.irfft(ma[1:, 1:])

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out

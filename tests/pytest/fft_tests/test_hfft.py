import functools  # NOQA
import unittest
import pytest     # NOQA

import numpy as np

import nlcpy      # NOQA
from nlcpy import testing


@testing.parameterize(
    *testing.product(
        {
            'n': [None, 5, 10, 15],
            'shape': [(10,), (10, 10)],
            'axis': [-1, 0],
            'norm': [None, 'ortho'],
        }
    )
)
@testing.with_requires('numpy>=1.10.0')
class TestHfft(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-7, contiguous_check=False)
    def test_hfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.hfft(a, n=self.n, axis=self.axis, norm=self.norm)

#       if xp == np and dtype in [np.float16, np.float32, np.complex64]:
#           out = out.astype(np.float32)
        if out.dtype in [np.float16, np.float32]:
            out = out.astype(np.float64)

        if out.dtype in [np.complex64]:
            out = out.astype(np.complex128)

        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-7, contiguous_check=False)
    def test_ihfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ihfft(a, n=self.n, axis=self.axis, norm=self.norm)

#        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
#            out = out.astype(np.complex64)
        if out.dtype in [np.float16, np.float32]:
            out = out.astype(np.float64)

        if out.dtype in [np.complex64]:
            out = out.astype(np.complex128)

        return out

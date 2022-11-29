import unittest
import numpy as np
import nlcpy as vp
from nlcpy import testing

nan_dtypes = (
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
)

shapes = (
    (10,),
)


@testing.parameterize(*(
    testing.product({
        'shape': shapes,
    })
))
class TestCorr(unittest.TestCase):
    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_array_equal()
    def test_case_01(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        v = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)
        v = xp.asarray(v)
        return xp.correlate(a, v)


def test_me_1():
    a = np.array([1, 2, 3])
    v = np.array([0, 1, 0.5])
    testing.assert_allclose(
        np.correlate(a, v, 'same'),
        vp.correlate(a, v, 'same')
    )


def test_me_2():
    a = np.array([1, 2, 3])
    v = np.array([0, 1, 0.5])
    testing.assert_allclose(
        np.correlate(a, v, 'full'),
        vp.correlate(a, v, 'full')
    )


def test_me_3():
    a = np.array([1 + 1j, 2, 3 - 1j])
    v = np.array([0, 1, 0.5j])
    testing.assert_allclose(
        np.correlate(a, v, 'full'),
        vp.correlate(a, v, 'full')
    )

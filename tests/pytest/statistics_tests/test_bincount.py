import unittest
import numpy
from nlcpy import testing

nan_dtypes = (
    numpy.float32,
    numpy.float64,
    numpy.complex64,
    numpy.complex128,
)

shapes = (
    (4,),
)


@testing.parameterize(*(
    testing.product({
        'shape': shapes,
    })
))
class TestBincount(unittest.TestCase):
    @testing.for_dtypes(['i', 'q'])
    @testing.numpy_nlcpy_array_equal()
    def test_case_01(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)
        return xp.bincount(a)

    @testing.for_dtypes(['i', 'q'])
    @testing.numpy_nlcpy_array_equal()
    def test_case_02(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)
        size = xp.bincount(a).size
        maxx = xp.amax(a) + 1
        return size == maxx

    @testing.numpy_nlcpy_array_equal()
    def test_case_03(self, xp):
        x = xp.array([0, 1, 1, 2, 2, 2])
        w = xp.array([0.3, 0.5, 0.2, 0.7, 1., -0.6])
        return xp.bincount(x, weights=w)

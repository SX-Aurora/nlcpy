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
    (3, 4),
    (2, 3, 4),
)


@testing.parameterize(*(
    testing.product({
        'shape': shapes,
    })
))
class TestPtp(unittest.TestCase):
    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_array_equal()
    def test_case_00(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)
        return xp.ptp(a)

    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_array_equal()
    def test_case_01(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)
        return xp.ptp(a, axis=0)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_2(self, xp):
        x = xp.array([[4, 9, 2, 10], [6, 9, 7, 12]])
        return xp.ptp(x, axis=1)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_3(self, xp):
        x = xp.array([[4, 9, 2, 10], [6, 9, 7, 12]])
        return xp.ptp(x, axis=0)

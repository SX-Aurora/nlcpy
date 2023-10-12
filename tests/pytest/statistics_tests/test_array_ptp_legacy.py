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
    # ndarray methods
    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_array_equal()
    def test_case_mem_01(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)
        return a.ptp()

    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_array_equal()
    def test_case_mem_02(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)
        return a.ptp(axis=0)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_mem_1(self, xp):
        x = xp.array([[4, 9, 2, 10], [6, 9, 7, 12]])
        return x.ptp(axis=1)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_mem_2(self, xp):
        x = xp.array([[4, 9, 2, 10], [6, 9, 7, 12]])
        return x.ptp(axis=0)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_mem_3(self, xp):
        x = xp.array([[1, 127], [0, 127], [-1, 127], [-2, 127]], dtype=xp.int32)
        return x.ptp(axis=1)

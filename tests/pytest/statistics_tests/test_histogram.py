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
class TestHistogram(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_case_01(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)

        bins = xp.asarray([0, 1, 2, 3])

        return xp.histogram(a, bins)

    @testing.numpy_nlcpy_array_equal()
    def test_me_1(self, xp):
        a = xp.array([1, 2, 1])
        bins = xp.array([0, 1, 2, 3])
        return xp.histogram(a, bins)

    @testing.numpy_nlcpy_array_equal()
    def test_me_2(self, xp):
        a = xp.arange(4)
        bins = xp.arange(5)
        return xp.histogram(a, bins, density=True)

    @testing.numpy_nlcpy_array_equal()
    def test_me_3(self, xp):
        a = xp.array([[1, 2, 1], [1, 0, 1]])
        bins = xp.array([0, 1, 2, 3])
        return xp.histogram(a, bins)

    @testing.numpy_nlcpy_array_equal()
    def test_me_4(self, xp):
        a = xp.arange(5)
        hist, bin_edges = xp.histogram(a, density=True)
        return hist

    @testing.numpy_nlcpy_array_equal()
    def test_me_5(self, xp):
        a = xp.arange(5)
        hist, bin_edges = xp.histogram(a, density=True)
        return xp.sum(hist)

    @testing.numpy_nlcpy_array_equal()
    def test_me_6(self, xp):
        a = xp.arange(5)
        hist, bin_edges = xp.histogram(a, density=True)
        return xp.sum(hist * xp.diff(bin_edges))

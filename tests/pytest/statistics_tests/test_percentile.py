import unittest
import numpy
import nlcpy
from nlcpy import testing
import nlcpy as vp

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
class TestPercentile(unittest.TestCase):
    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_array_equal()
    def test_case_00(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)
        return xp.percentile(a, 50)

    @testing.for_dtypes(['i', 'q', 'f', 'd', 'F', 'D'])
    @testing.numpy_nlcpy_array_equal()
    def test_case_01(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)
        return xp.percentile(a, 50, axis=0)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_1(self, xp):
        a = xp.array([[10, 7, 4], [3, 2, 1]])
        return xp.percentile(a, 50)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_2(self, xp):
        a = xp.array([[10, 7, 4], [3, 2, 1]])
        return xp.percentile(a, 50, axis=0)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_3(self, xp):
        a = xp.array([[10, 7, 4], [3, 2, 1]])
        return xp.percentile(a, 50, axis=1)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_4(self, xp):
        a = xp.array([[10, 7, 4], [3, 2, 1]])
        return xp.percentile(a, 50, axis=1, keepdims=True)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_5(self, xp):
        a = xp.array([[10, 7, 4], [3, 2, 1]])
        m = xp.percentile(a, 50, axis=0)
        out = xp.zeros_like(m)
        return xp.percentile(a, 50, axis=0, out=out)


def testinge_case_6():
    a = vp.array([[10, 7, 4], [3, 2, 1]])
    b = a.copy()
    vp.percentile(b, 50, axis=1, overwrite_input=True)
    return vp.all(a == b)

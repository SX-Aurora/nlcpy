import unittest
import warnings
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
class TestQuantile(unittest.TestCase):
    @testing.for_dtypes(['f', 'F'])
    @testing.numpy_nlcpy_array_equal()
    def test_case_00(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)
        a[1] = xp.nan
        return xp.nanquantile(a, 0.5)

    @testing.for_dtypes(['f', 'F'])
    @testing.numpy_nlcpy_array_equal()
    def test_case_01(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a)
        a[1] = xp.nan
        return xp.quantile(a, 0.5, axis=0)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_1(self, xp):
        a = xp.array([[10, 7, 4], [3, 2, 1]], dtype=xp.float64)
        a[1] = xp.nan
        return xp.nanquantile(a, 0.5)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_2(self, xp):
        a = xp.array([[10, 7, 4], [3, 2, 1]], dtype=xp.float64)
        a[1] = xp.nan
        return xp.nanquantile(a, 0.5, axis=0)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_3(self, xp):
        a = xp.array([[10, 7, 4], [3, 2, 1]], dtype=xp.float64)
        a[1] = xp.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return xp.nanquantile(a, 0.5, axis=1)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_4(self, xp):
        a = xp.array([[10, 7, 4], [3, 2, 1]], dtype=xp.float64)
        a[1] = xp.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return xp.nanquantile(a, 0.5, axis=1, keepdims=True)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_5(self, xp):
        a = xp.array([[10, 7, 4], [3, 2, 1]], dtype=xp.float64)
        a[1] = xp.nan
        m = xp.nanquantile(a, 0.5, axis=0)
        out = xp.zeros_like(m)
        return xp.nanquantile(a, 0.5, axis=0, out=out)

    @testing.numpy_nlcpy_array_equal()
    def test_me_case_6(self, xp):
        a = xp.array([[10., 7., 4.], [3., 2., 1.]])
        a[1] = xp.nan
        b = a.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return xp.nanquantile(b, 0.5, axis=1, overwrite_input=True)

import unittest
import numpy
import nlcpy
from nlcpy import testing
import nlcpy as ny

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
class TestNanmedian(unittest.TestCase):
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_case_01(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        if dtype in nan_dtypes:
            a[1] = xp.nan
        return xp.nanmedian(a)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_case_02(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        if dtype in nan_dtypes:
            a[1] = xp.nan
        return xp.nanmedian(a, axis=0)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_case_03(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        if dtype in nan_dtypes:
            a[1] = xp.nan
        if a.ndim > 1:
            return xp.nanmedian(a, axis=1)
        else:
            return xp.nanmedian(a)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_case_04(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        if dtype in nan_dtypes:
            a[1] = xp.nan
        if a.ndim > 2:
            return xp.nanmedian(a, axis=2)
        else:
            return xp.nanmedian(a)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_nanmedian_with_out(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        out = xp.empty(self.shape, dtype=dtype, order=order)
        if dtype in nan_dtypes:
            a[1] = xp.nan
        xp.nanmedian(a, out=out)
        return out


def test_me_case_1():
    a = nlcpy.array([[10, 7, 4], [3, 2, 1]])
    b = a.copy()
    ny.median(b, axis=1, overwrite_input=True)
    assert not ny.all(a == b)


def test_me_case_2():
    a = nlcpy.array([[10, 7, 4], [3, 2, 1]])
    b = a.copy()
    ny.median(b, axis=None, overwrite_input=True)
    assert not ny.all(a == b)

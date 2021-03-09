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
)


@testing.parameterize(*(
    testing.product({
        'shape': shapes,
    })
))
class TestMean(unittest.TestCase):
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_case_01(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        return a.mean()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_case_02(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        return a.mean(axis=0)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_case_03(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        if a.ndim > 1:
            return a.mean(axis=1)
        else:
            return a.mean()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_case_04(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        if a.ndim > 2:
            return a.mean(axis=2)
        else:
            return a.mean()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-6)
    def test_mean_with_out(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        ans = xp.mean(a)
        out = xp.empty(ans.shape, dtype=dtype, order=order)
        a.mean(out=out)
        return out

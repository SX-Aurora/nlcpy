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
class TestVar(unittest.TestCase):
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_case_01(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        return a.var()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_case_02(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        return a.var(axis=0)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_case_03(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        if a.ndim > 1:
            return a.var(axis=1)
        else:
            return a.var()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_case_04(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        if a.ndim > 2:
            return a.var(axis=2)
        else:
            return a.var()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose(rtol=1e-4)
    def test_mean_with_out(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        ans = xp.var(a)
        out = xp.empty(ans.shape, dtype=dtype, order=order)
        a.var(out=out)
        return out

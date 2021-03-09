import unittest
from nlcpy import testing


class TestDegitize(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_case_01(self, xp):
        x = xp.array([0.2, 6.4, 3.0, 1.6])
        bins = xp.array([0.0, 1.0, 2.5, 4.0, 10.0])
        return xp.digitize(x, bins)

    @testing.numpy_nlcpy_array_equal()
    def test_case_02(self, xp):
        x = xp.array([1.2, 10.0, 12.4, 15.5, 20.])
        bins = xp.array([0, 5, 10, 15, 20])
        return xp.digitize(x, bins, right=True)

    @testing.numpy_nlcpy_array_equal()
    def test_case_03(self, xp):
        x = xp.array([1.2, 10.0, 12.4, 15.5, 20.])
        bins = xp.array([0, 5, 10, 15, 20])
        return xp.digitize(x, bins, right=False)

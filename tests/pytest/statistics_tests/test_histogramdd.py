import unittest
from nlcpy import testing


class TestHistogramddd(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_me_1(self, xp):
        r = testing.shaped_random((3, 3), xp)
        H, edges = xp.histogramdd(r, bins=(5, 8, 4))
        return H

    @testing.numpy_nlcpy_array_equal()
    def test_me_2(self, xp):
        r = testing.shaped_random((3, 3), xp)
        H, edges = xp.histogramdd(r, bins=(5, 8, 4))
        return edges

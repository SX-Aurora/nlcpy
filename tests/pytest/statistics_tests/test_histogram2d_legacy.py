import unittest
from nlcpy import testing


class TestHistogram2d(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_me_1(self, xp):
        xedges = [0, 1, 3, 5]
        yedges = [0, 2, 3, 4, 6]
        x = testing.shaped_random((100,), xp)
        y = testing.shaped_random((100,), xp)
        H, xedges, yedges = xp.histogram2d(x, y, bins=(xedges, yedges))
        return H

    @testing.numpy_nlcpy_array_equal()
    def test_me_2(self, xp):
        xedges = [0, 1, 3, 5]
        yedges = [0, 2, 3, 4, 6]
        x = testing.shaped_random((100,), xp)
        y = testing.shaped_random((100,), xp)
        H, xedges, yedges = xp.histogram2d(x, y, bins=(xedges, yedges))
        return xedges

    @testing.numpy_nlcpy_array_equal()
    def test_me_3(self, xp):
        xedges = [0, 1, 3, 5]
        yedges = [0, 2, 3, 4, 6]
        x = testing.shaped_random((100,), xp)
        y = testing.shaped_random((100,), xp)
        H, xedges, yedges = xp.histogram2d(x, y, bins=(xedges, yedges))
        return yedges

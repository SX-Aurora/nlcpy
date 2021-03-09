import unittest
from nlcpy import testing


class TestHistogramBinEdges(unittest.TestCase):
    @testing.numpy_nlcpy_array_equal()
    def test_case_01(self, xp):
        arr = xp.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
        return xp.histogram_bin_edges(arr, bins='auto', range=(0, 1))

    @testing.numpy_nlcpy_array_equal()
    def test_case_02(self, xp):
        arr = xp.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
        return xp.histogram_bin_edges(arr, bins=2)

    @testing.numpy_nlcpy_array_equal()
    def test_case_03(self, xp):
        arr = xp.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
        return xp.histogram_bin_edges(arr, [1, 2])

    @testing.numpy_nlcpy_array_equal()
    def test_case_04(self, xp):
        arr = xp.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
        return xp.histogram_bin_edges(arr, bins='auto')

    @testing.numpy_nlcpy_array_equal()
    def test_case_05(self, xp):
        arr = xp.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
        group_id = xp.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
        shared_bins = xp.histogram_bin_edges(arr, bins='auto')
        hist_0, _ = xp.histogram(arr[group_id == 0], bins=shared_bins)
        hist_1, _ = xp.histogram(arr[group_id == 1], bins=shared_bins)
        return hist_0

    @testing.numpy_nlcpy_array_equal()
    def test_case_06(self, xp):
        arr = xp.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
        group_id = xp.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
        shared_bins = xp.histogram_bin_edges(arr, bins='auto')
        hist_0, _ = xp.histogram(arr[group_id == 0], bins=shared_bins)
        hist_1, _ = xp.histogram(arr[group_id == 1], bins=shared_bins)
        return hist_1

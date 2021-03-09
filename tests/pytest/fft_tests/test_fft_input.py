import functools # NOQA
import unittest
import pytest # NOQA

import numpy
import numpy as np # NOQA

import nlcpy
from nlcpy import testing


@testing.parameterize(*testing.product({
    'a': [
        [1, 2, 3, 4, 5],
        (1, 2, 3),
        range(10),
        bytearray(b'abc'),
        memoryview(b'abc'),
        numpy.asarray([1, 2]),
        nlcpy.asarray([1, 2]),
        [True, False],
        [1, 2, 3],
        [2.3, 4.5],
        [3. + 0.1j, 4. + 0.2j],
        [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        [(1, 2, 3), (1, 2, 3)],
        [range(10), range(10)],
        [bytearray(b'abc'), bytearray(b'abc')],
        [memoryview(b'abc'), memoryview(b'abc')],
        [numpy.asarray([1, 2]), numpy.asarray([1, 2])],
        [nlcpy.asarray([1, 2]), nlcpy.asarray([1, 2])],
        (True, False),
        (1, 2, 3),
        (2.3, 4.5),
        (3. + 0.1j, 4. + 0.2j),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ((1, 2, 3), (1, 2, 3)),
        (range(10), range(10)),
        (bytearray(b'abc'), bytearray(b'abc')),
        (memoryview(b'abc'), memoryview(b'abc')),
        (numpy.asarray([1, 2]), numpy.asarray([1, 2])),
        (nlcpy.asarray([1, 2]), nlcpy.asarray([1, 2]))
    ],
}))
@testing.with_requires('numpy>=1.10.0')
class TestFftInput(unittest.TestCase):

    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7,
                                  accept_error=ValueError,
                                  contiguous_check=False)
    def test_fft(self, xp):
        out = xp.fft.fft(self.a)

        return out


@testing.parameterize(*testing.product({
    'a': [
        [1, 2, 3, 4, 5],
        (1, 2, 3),
        range(10),
        bytearray(b'abc'),
        memoryview(b'abc'),
        numpy.asarray([1, 2]),
        nlcpy.asarray([1, 2]),
        [True, False],
        [1, 2, 3],
        [2.3, 4.5],
        # [3. + 0.1j, 4. + 0.2j], # numpy.fft.rfft use this. return TypeError.
        [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        [(1, 2, 3), (1, 2, 3)],
        [range(10), range(10)],
        [bytearray(b'abc'), bytearray(b'abc')],
        [memoryview(b'abc'), memoryview(b'abc')],
        [numpy.asarray([1, 2]), numpy.asarray([1, 2])],
        [nlcpy.asarray([1, 2]), nlcpy.asarray([1, 2])],
        (True, False),
        (1, 2, 3),
        (2.3, 4.5),
        # (3. + 0.1j, 4. + 0.2j), #  numpy.fft.rfft use this. return TypeError.
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ((1, 2, 3), (1, 2, 3)),
        (range(10), range(10)),
        (bytearray(b'abc'), bytearray(b'abc')),
        (memoryview(b'abc'), memoryview(b'abc')),
        (numpy.asarray([1, 2]), numpy.asarray([1, 2])),
        (nlcpy.asarray([1, 2]), nlcpy.asarray([1, 2]))
    ],
}))
@testing.with_requires('numpy>=1.10.0')
class TestRfftInput(unittest.TestCase):
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7,
                                  accept_error=ValueError,
                                  contiguous_check=False)
    def _test_rfft(self, xp):
        out = xp.fft.rfft(self.a)

        return out

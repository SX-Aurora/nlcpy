import nlcpy
from nlcpy import random
from nlcpy import request


def _warmup():
    _iteration = 20
    _size1 = 10
    _size2 = 10

    """
    2-D warm-up
    """
    for i in range(_iteration):
        # (10, 10)
        nlcpy.add(random.rand(_size1, _size2), random.rand(_size1, _size2))
    request.flush()

    _size2 *= 10
    for i in range(_iteration):
        # (10, 100)
        nlcpy.add(random.rand(_size1, _size2), random.rand(_size1, _size2))
    request.flush()

    _size1 *= 10
    for i in range(_iteration):
        # (100, 100)
        nlcpy.add(random.rand(_size1, _size2), random.rand(_size1, _size2))
    request.flush()

    _size2 *= 10
    for i in range(_iteration):
        # (100, 1000)
        nlcpy.add(random.rand(_size1, _size2), random.rand(_size1, _size2))
    request.flush()

    _size1 *= 10
    for i in range(_iteration):
        # (1000, 1000)
        nlcpy.add(random.rand(_size1, _size2), random.rand(_size1, _size2))
    request.flush()

    _iteration = 1

    _size2 *= 10
    for i in range(_iteration):
        # (10000, 1000)
        nlcpy.add(random.rand(_size1, _size2), random.rand(_size1, _size2))
    request.flush()

    _size1 *= 10
    for i in range(_iteration):
        # (10000, 10000)
        nlcpy.add(random.rand(_size1, _size2), random.rand(_size1, _size2))
    request.flush()

    """
    1-D warm-up
    """
    _iteration = 5
    _size1 = 100
    for i in range(_iteration):
        # (100)
        nlcpy.add(random.rand(_size1), random.rand(_size1))
    request.flush()

    _size1 *= 100
    for i in range(_iteration):
        # (10000)
        nlcpy.add(random.rand(_size1), random.rand(_size1))
    request.flush()

    _size1 *= 100
    for i in range(_iteration):
        # (1000000)
        nlcpy.add(random.rand(_size1), random.rand(_size1))
    request.flush()

    _size1 *= 100
    for i in range(_iteration):
        # (100000000)
        nlcpy.add(random.rand(_size1), random.rand(_size1))
    request.flush()

import nlcpy
from nlcpy import random
from nlcpy import request


def _warmup():
    _iteration = 5
    _size1 = 10
    _size2 = 10

    """
    2-D warm-up
    """
    is_even = True
    while _size1 <= 1000 or _size2 <= 1000:
        x = random.rand(_size1, _size2)
        y = random.rand(_size1, _size2)
        out = nlcpy.zeros_like(x)
        for i in range(_iteration):
            nlcpy.add(x, y, out=out)
        if is_even:
            _size2 *= 10
        else:
            _size1 *= 10
        is_even = not(is_even)

    """
    1-D warm-up
    """
    _iteration = 5
    _size1 = 100
    while _size1 <= 100000000:
        for i in range(_iteration):
            x = random.rand(_size1)
            y = random.rand(_size1)
            out = nlcpy.zeros_like(x)
            nlcpy.add(x, y, out=out)
        request.flush()
        _size1 *= 100

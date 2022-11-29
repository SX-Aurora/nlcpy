import numpy # NOQA
from nlcpy.wrapper.numpy_wrap import _make_wrap_func # NOQA

from nlcpy.random import sample
from nlcpy.random import permutations # NOQA
from nlcpy.random import distributions # NOQA
from nlcpy.random import generator # NOQA
from nlcpy.random import _generator # NOQA

from nlcpy.random.sample import rand # NOQA
from nlcpy.random.sample import randn # NOQA
from nlcpy.random.sample import randint # NOQA
from nlcpy.random.sample import random_integers # NOQA
from nlcpy.random.sample import random_sample # NOQA
from nlcpy.random.sample import random # NOQA
from nlcpy.random.sample import ranf # NOQA
from nlcpy.random.sample import sample # NOQA
from nlcpy.random.sample import bytes # NOQA

from nlcpy.random.permutations import shuffle # NOQA
from nlcpy.random.permutations import permutation # NOQA

from nlcpy.random.distributions import binomial # NOQA
from nlcpy.random.distributions import exponential # NOQA
from nlcpy.random.distributions import gamma # NOQA
from nlcpy.random.distributions import geometric # NOQA
from nlcpy.random.distributions import gumbel # NOQA
from nlcpy.random.distributions import logistic # NOQA
from nlcpy.random.distributions import lognormal # NOQA
from nlcpy.random.distributions import normal # NOQA
from nlcpy.random.distributions import poisson # NOQA
from nlcpy.random.distributions import standard_cauchy # NOQA
from nlcpy.random.distributions import standard_exponential # NOQA
from nlcpy.random.distributions import standard_gamma # NOQA
from nlcpy.random.distributions import standard_normal # NOQA
from nlcpy.random.distributions import uniform # NOQA
from nlcpy.random.distributions import weibull # NOQA

from nlcpy.random.generator import get_state # NOQA
from nlcpy.random.generator import set_state # NOQA
from nlcpy.random.generator import seed # NOQA
from nlcpy.random.generator import RandomState # NOQA

from nlcpy.random._generator import Generator # NOQA
from nlcpy.random._generator import BitGenerator # NOQA
from nlcpy.random._generator import MT19937 # NOQA
from nlcpy.random._generator import SeedSequence # NOQA
from nlcpy.random._generator import default_rng # NOQA


_numpy_state = (0, 0.0)


def __getattr__(attr):
    if attr in (
            'PCG64',
            'Philox',
            'SFC64'
    ):
        raise AttributeError(
            "module 'nlcpy.random' has no attribute '{}'.".format(attr))

    try:
        f = getattr(numpy.random, attr)
    except AttributeError as _err:
        raise AttributeError(
            "module 'nlcpy.random' has no attribute '{}'."
            .format(attr)) from _err
    if not callable(f):
        raise AttributeError(
            "module 'nlcpy.random' has no attribute '{}'.".format(attr))
    return _make_wrap_func(f)

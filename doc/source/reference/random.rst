.. module:: nlcpy.random

Random Sampling
===============

.. contents:: :local:

NLCPy random number routines produce pseudo random numbers and create sample from different statistical distributions.

Generator
---------

The Generator provides access to a wide variety of probability distributions, and serves as a replacement for RandomState.

An easy example of Generator is below:

.. code-block:: python

    from nlcpy.random import Generator, MT19937
    rng = Generator(MT19937(12345))
    rng.standard_normal()

And, an easy example of using default_rng is below:

.. code-block:: python

    import nlcpy as vp
    rng = vp.random.default_rng()
    rng.standard_normal()


Available Functions and Methods
-------------------------------

The following tables show that nlcpy.random.Generator class methods to generate random numbers.

Construct Generator
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.random.default_rng

Simple Random Data
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.random.Generator.bytes
    nlcpy.random.Generator.integers
    nlcpy.random.Generator.random

Permutations
^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.random.Generator.permutation
    nlcpy.random.Generator.shuffle

Distributions
^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.random.Generator.binomial
    nlcpy.random.Generator.exponential
    nlcpy.random.Generator.gamma
    nlcpy.random.Generator.geometric
    nlcpy.random.Generator.gumbel
    nlcpy.random.Generator.logistic
    nlcpy.random.Generator.lognormal
    nlcpy.random.Generator.normal
    nlcpy.random.Generator.poisson
    nlcpy.random.Generator.standard_cauchy
    nlcpy.random.Generator.standard_exponential
    nlcpy.random.Generator.standard_gamma
    nlcpy.random.Generator.standard_normal
    nlcpy.random.Generator.uniform
    nlcpy.random.Generator.weibull


RandomState
-----------

The RandomState provides access to legacy generators.
An easy example of RandomState is below:

.. code-block:: python

    # Uses the nlcpy.random.RandomState
    from nlcpy import random
    random.standard_normal()
    # or
    rst = random.RandomState()
    rst.standard_normal()

Available Functions and Methods
-------------------------------

Seeding and State
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.random.RandomState.get_state
    nlcpy.random.RandomState.seed
    nlcpy.random.RandomState.set_state

Simple Random Data
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.random.RandomState.bytes
    nlcpy.random.RandomState.rand
    nlcpy.random.RandomState.randint
    nlcpy.random.RandomState.randn
    nlcpy.random.RandomState.random
    nlcpy.random.RandomState.random_integers
    nlcpy.random.RandomState.random_sample
    nlcpy.random.RandomState.ranf
    nlcpy.random.RandomState.sample
    nlcpy.random.RandomState.tomaxint

Permutations
^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.random.RandomState.permutation
    nlcpy.random.RandomState.shuffle

Distributions
^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.random.RandomState.binomial
    nlcpy.random.RandomState.exponential
    nlcpy.random.RandomState.gamma
    nlcpy.random.RandomState.geometric
    nlcpy.random.RandomState.gumbel
    nlcpy.random.RandomState.logistic
    nlcpy.random.RandomState.lognormal
    nlcpy.random.RandomState.normal
    nlcpy.random.RandomState.poisson
    nlcpy.random.RandomState.standard_cauchy
    nlcpy.random.RandomState.standard_exponential
    nlcpy.random.RandomState.standard_gamma
    nlcpy.random.RandomState.standard_normal
    nlcpy.random.RandomState.uniform
    nlcpy.random.RandomState.weibull

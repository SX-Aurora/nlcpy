#
# * The source code in this file is based on the soure code of CuPy.
#
# # NLCPy License #
#
#     Copyright (c) 2020 NEC Corporation
#     All rights reserved.
#
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither NEC Corporation nor the names of its contributors may be
#       used to endorse or promote products derived from this software
#       without specific prior written permission.
#
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#     ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#     WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#     ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#     (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#     SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# # CuPy License #
#
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
from __future__ import absolute_import
import atexit
import functools
import numpy
import os
import random
import types
import unittest

import nlcpy


_old_python_random_state = None
_old_numpy_random_state = None
_old_nlcpy_random_states = None


def do_setup(deterministic=True):
    global _old_python_random_state
    global _old_numpy_random_state
    global _old_nlcpy_random_states
    _old_python_random_state = random.getstate()
    _old_numpy_random_state = numpy.random.get_state()
    _old_nlcpy_random_states = nlcpy.random.generator._random_states
    nlcpy.random.reset_states()
    # Check that _random_state has been recreated in
    # nlcpy.random.reset_states(). Otherwise the contents of
    # _old_nlcpy_random_states would be overwritten.
    assert (nlcpy.random.generator._random_states is not
            _old_nlcpy_random_states)

    if not deterministic:
        random.seed()
        numpy.random.seed()
        nlcpy.random.seed()
    else:
        random.seed(99)
        numpy.random.seed(100)
        nlcpy.random.seed(101)


def do_teardown():
    global _old_python_random_state
    global _old_numpy_random_state
    global _old_nlcpy_random_states
    random.setstate(_old_python_random_state)
    numpy.random.set_state(_old_numpy_random_state)
    nlcpy.random.generator._random_states = _old_nlcpy_random_states
    _old_python_random_state = None
    _old_numpy_random_state = None
    _old_nlcpy_random_states = None


# In some tests (which utilize condition.repeat or condition.retry),
# setUp/tearDown is nested. _setup_random() and _teardown_random() do their
# work only in the outermost setUp/tearDown pair.
_nest_count = 0


@atexit.register
def _check_teardown():
    assert _nest_count == 0, ('_setup_random() and _teardown_random() '
                              'must be called in pairs.')


def _setup_random():
    """Sets up the deterministic random states of ``numpy`` and ``nlcpy``.

    """
    global _nest_count
    if _nest_count == 0:
        nondeterministic = bool(int(os.environ.get(
            'nlcpy_TEST_RANDOM_NONDETERMINISTIC', '0')))
        do_setup(not nondeterministic)
    _nest_count += 1


def _teardown_random():
    """Tears down the deterministic random states set up by ``_setup_random``.

    """
    global _nest_count
    assert _nest_count > 0, '_setup_random has not been called'
    _nest_count -= 1
    if _nest_count == 0:
        do_teardown()


def generate_seed():
    assert _nest_count > 0, 'random is not set up'
    return numpy.random.randint(0x7fffffff)


def fix_random():
    """Decorator that fixes random numbers in a test.

    This decorator can be applied to either a test case class or a test method.
    It should not be applied within ``condition.retry`` or
    ``condition.repeat``.
    """

    # TODO: Prevent this decorator from being applied within
    #    condition.repeat or condition.retry decorators. That would repeat
    #    tests with the same random seeds. It's okay to apply this outside
    #    these decorators.

    def decorator(impl):
        if (isinstance(impl, types.FunctionType) and
                impl.__name__.startswith('test_')):
            # Applied to test method
            @functools.wraps(impl)
            def test_func(self, *args, **kw):
                _setup_random()
                try:
                    impl(self, *args, **kw)
                finally:
                    _teardown_random()
            return test_func
        elif isinstance(impl, type) and issubclass(impl, unittest.TestCase):
            # Applied to test case class
            klass = impl

            def wrap_setUp(f):
                def func(self):
                    _setup_random()
                    f(self)
                return func

            def wrap_tearDown(f):
                def func(self):
                    try:
                        f(self)
                    finally:
                        _teardown_random()
                return func

            klass.setUp = wrap_setUp(klass.setUp)
            klass.tearDown = wrap_tearDown(klass.tearDown)
            return klass
        else:
            raise ValueError('Can\'t apply fix_random to {}'.format(impl))

    return decorator

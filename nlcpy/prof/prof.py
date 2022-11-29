#
# * The source code in this file is developed independently by NEC Corporation.
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

import time
import functools
import contextlib
import warnings
import numpy
import nlcpy
from nlcpy import veo

# profiling status
NOT_PROFILING = 0
UNDER_PROFILING = 1
END_PROFILING = 2


class Profiling():
    def __init__(self):
        self.n_alloc_mem = 0
        self.t_alloc_mem = 0
        self.n_free_mem = 0
        self.t_free_mem = 0
        self.n_write_mem = 0
        self.t_write_mem = 0
        self.n_read_mem = 0
        self.t_read_mem = 0
        self.n_wait_result = 0
        self.t_wait_result = 0
        self.vh_runtime = 0
        self.total_runtime = 0
        self.status = NOT_PROFILING

    def start(self):
        if self.status == UNDER_PROFILING:
            raise Exception('under profiling')
        self.clear()
        self.status = UNDER_PROFILING
        self.total_runtime = time.time()

    def stop(self):
        if self.status != UNDER_PROFILING:
            raise Exception('not started profiling')
        self.total_runtime = time.time() - self.total_runtime
        self.vh_runtime = self.total_runtime - (
            self.t_alloc_mem + self.t_free_mem + self.t_write_mem
            + self.t_read_mem + self.t_wait_result)
        self.status = END_PROFILING

    def get_stats(self):
        if _prof.status != END_PROFILING:
            raise Exception('not finished profiling')
        _stats = {
            'veo_alloc_mem': {
                'elapsed_time': 0,
                'number_of_call': 0,
            },
            'veo_free_mem': {
                'elapsed_time': 0,
                'number_of_call': 0,
            },
            'veo_write_mem': {
                'elapsed_time': 0,
                'number_of_call': 0,
            },
            'veo_read_mem': {
                'elapsed_time': 0,
                'number_of_call': 0,
            },
            'veo_wait_result': {
                'elapsed_time': 0,
                'number_of_call': 0,
            },
            'vh_runtime': {
                'elapsed_time': 0,
            },
            'total_runtime': {
                'elapsed_time': 0,
            },
        }
        # alloc mem
        _stats['veo_alloc_mem']['elapsed_time'] = _prof.t_alloc_mem
        _stats['veo_alloc_mem']['number_of_call'] = _prof.n_alloc_mem
        # free mem
        _stats['veo_free_mem']['elapsed_time'] = _prof.t_free_mem
        _stats['veo_free_mem']['number_of_call'] = _prof.n_free_mem
        # write mem
        _stats['veo_write_mem']['elapsed_time'] = _prof.t_write_mem
        _stats['veo_write_mem']['number_of_call'] = _prof.n_write_mem
        # read mem
        _stats['veo_read_mem']['elapsed_time'] = _prof.t_read_mem
        _stats['veo_read_mem']['number_of_call'] = _prof.n_read_mem
        # VE runtime
        _stats['veo_wait_result']['elapsed_time'] = _prof.t_wait_result
        _stats['veo_wait_result']['number_of_call'] = _prof.n_wait_result
        # VH runtime
        _stats['vh_runtime']['elapsed_time'] = _prof.vh_runtime
        # total runtime
        _stats['total_runtime']['elapsed_time'] = _prof.total_runtime
        return _stats

    def clear(self):
        self.n_alloc_mem = 0
        self.t_alloc_mem = 0
        self.n_free_mem = 0
        self.t_free_mem = 0
        self.n_write_mem = 0
        self.t_write_mem = 0
        self.n_read_mem = 0
        self.t_read_mem = 0
        self.n_wait_result = 0
        self.t_wait_result = 0
        self.total_s = 0
        self.total_e = 0
        self.in_analyze = 0
        self.status = NOT_PROFILING


_prof = Profiling()


def start_profiling():
    """Starts profiling.

    Profiling the code block between :func:`nlcpy.prof.start_profiling` and
    :func:`nlcpy.prof.stop_profiling`.

    Notes
    -----
    .. deprecated:: 2.0.0

    See Also
    --------
    nlcpy.prof.print_run_stats : Prints NLCPy run stats.
    nlcpy.prof.get_run_stats : Gets dict of NLCPy run stats.
    """
    warnings.warn('This routine is deprecated since version 2.0.0. '
                  'Please use nlcpy.prof.ftrace_region().',
                  UserWarning)
    _prof.start()


def stop_profiling():
    """Stops profiling.

    Profiling the code block between :func:`nlcpy.prof.start_profiling` and
    :func:`nlcpy.prof.stop_profiling`.

    Notes
    -----
    .. deprecated:: 2.0.0

    See Also
    --------
    nlcpy.prof.print_run_stats : Prints NLCPy run stats.
    nlcpy.prof.get_run_stats : Gets dict of NLCPy run stats.
    """
    warnings.warn('This routine is deprecated since version 2.0.0. '
                  'Please use nlcpy.prof.ftrace_region().',
                  UserWarning)
    _prof.stop()


def profile_alloc_mem(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        if _prof.status != UNDER_PROFILING:
            return func(*args, **kwargs)
        pre_wait_result = _prof.t_wait_result
        pre_write_mem = _prof.t_write_mem
        pre_free_mem = _prof.t_free_mem
        s = time.time()
        res = func(*args, **kwargs)
        e = time.time()
        _prof.n_alloc_mem += 1
        _prof.t_alloc_mem += (e - s)
        if pre_wait_result != _prof.t_wait_result:
            _prof.t_alloc_mem -= (_prof.t_wait_result - pre_wait_result)
        if pre_write_mem != _prof.t_write_mem:
            _prof.t_alloc_mem -= (_prof.t_write_mem - pre_write_mem)
        if pre_free_mem != _prof.t_free_mem:
            _prof.t_free_mem -= (_prof.t_free_mem - pre_free_mem)
        return res
    return wrap_func


def profile_free_mem(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        if _prof.status != UNDER_PROFILING:
            return func(*args, **kwargs)
        s = time.time()
        res = func(*args, **kwargs)
        e = time.time()
        _prof.n_free_mem += 1
        _prof.t_free_mem += (e - s)
        return res
    return wrap_func


def profile_write_mem(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        if _prof.status != UNDER_PROFILING:
            return func(*args, **kwargs)
        s = time.time()
        res = func(*args, **kwargs)
        e = time.time()
        _prof.n_write_mem += 1
        _prof.t_write_mem += (e - s)
        return res
    return wrap_func


def profile_read_mem(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        if _prof.status != UNDER_PROFILING:
            return func(*args, **kwargs)
        s = time.time()
        res = func(*args, **kwargs)
        e = time.time()
        _prof.n_read_mem += 1
        _prof.t_read_mem += (e - s)
        return res
    return wrap_func


def profile_wait_result(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        if _prof.status != UNDER_PROFILING:
            return func(*args, **kwargs)
        s = time.time()
        res = func(*args, **kwargs)
        e = time.time()
        _prof.n_wait_result += 1
        _prof.t_wait_result += (e - s)
        return res
    return wrap_func


def _print_impl(msg, val, is_exp):
    if is_exp:
        print("{} {:.3e} [sec]".format(msg, val))
    else:
        print("{} {} times".format(msg, val))


def print_run_stats():
    """Prints NLCPy run stats.

    Notes
    -----
    .. deprecated:: 2.0.0

    Examples
    --------
    Sample Program::

        # sample.py
        import nlcpy as vp
        vp.prof.start_profiling()
        for i in range(10):
            vp.random.rand(10000)
        vp.prof.stop_profiling()
        vp.prof.print_run_stats()

    Execution::

        $ python sample.py

        ----------- NLCPy Run Stats ------------
        alloc memory on VE:
          total: 1.097e-05 [sec]
          veo_alloc_mem was called 10 times
        free memory on VE:
          total: 9.298e-06 [sec]
          veo_free_mem was called 10 times
        write memory on VE:
          total: 0.000e+00 [sec]
          veo_write_mem was called 0 times
        read memory from VE:
          total: 0.000e+00 [sec]
          veo_read_mem was called 0 times
        VE runtime(include offload overhead):
          total: 3.016e-04 [sec]
          veo_wait_result was called 10 times
        other VH runtime:
          total: 2.632e-04 [sec]
        total runtime:
          total: 5.851e-04 [sec]
        ----------------------------------------

    """
    warnings.warn('This routine is deprecated since version 2.0.0. '
                  'Please use nlcpy.prof.ftrace_region().',
                  UserWarning)
    if _prof.status != END_PROFILING:
        raise Exception('profiling not finished')
    print("\n----------- NLCPy Run Stats ------------")
    # alloc mem
    print("alloc memory on VE:")
    _print_impl("  total:", _prof.t_alloc_mem, True)
    _print_impl("  veo_alloc_mem was called", _prof.n_alloc_mem, False)
    # free mem
    print("free memory on VE:")
    _print_impl("  total:", _prof.t_free_mem, True)
    _print_impl("  veo_free_mem was called", _prof.n_free_mem, False)
    # write mem
    print("write memory on VE:")
    _print_impl("  total:", _prof.t_write_mem, True)
    _print_impl("  veo_write_mem was called", _prof.n_write_mem, False)
    # read mem
    print("read memory from VE:")
    _print_impl("  total:", _prof.t_read_mem, True)
    _print_impl("  veo_read_mem was called", _prof.n_read_mem, False)
    # VE runtime
    print("VE runtime(include offload overhead):")
    _print_impl("  total:", _prof.t_wait_result, True)
    _print_impl("  veo_wait_result was called", _prof.n_wait_result, False)
    # VH runtime
    print("other VH runtime:")
    _print_impl("  total:", _prof.vh_runtime, True)
    # total runtime
    print("total runtime:")
    _print_impl("  total:", _prof.total_runtime, True)
    print("----------------------------------------\n")


def get_run_stats():
    """Gets dict of NLCPy run stats.

    Notes
    -----
    .. deprecated:: 2.0.0

    Returns
    -------
    out : dict

    Examples
    --------
    Sample Program::

        # sample.py
        import nlcpy as vp
        from pprint import pprint
        vp.prof.start_profiling()
        for i in range(10):
            vp.random.rand(10000)
        vp.prof.stop_profiling()
        stats = vp.prof.get_run_stats()
        pprint(stats)

    Execution::

        $ python sample.py
        {'total_runtime': {'elapsed_time': 0.004348278045654297},
        'veo_alloc_mem': {'elapsed_time': 2.574920654296875e-05, 'number_of_call': 10},
        'veo_free_mem': {'elapsed_time': 4.100799560546875e-05, 'number_of_call': 10},
        'veo_read_mem': {'elapsed_time': 0, 'number_of_call': 0},
        'veo_wait_result': {'elapsed_time': 0.0034487247467041016,
                            'number_of_call': 10},
        'veo_write_mem': {'elapsed_time': 0, 'number_of_call': 0},
        'vh_runtime': {'elapsed_time': 0.0008327960968017578}}
    """
    warnings.warn('This routine is deprecated since version 2.0.0. '
                  'Please use nlcpy.prof.ftrace_region().',
                  UserWarning)
    return _prof.get_stats()


def ftrace_region_begin(message):
    """Begins an ftrace region.

    A file ftrace.out is generated after running your program that invokes
    this routine.
    The ftrace.out includes performance information of your program.

    Notes
    -----
    It is necessary to specify an identical string *message* to
    :func:`ftrace_region_begin` and :func:`ftrace_region_end`.

    Parameters
    ----------
    message : str
        Any string can be specified to distinguish a user-specified region.

    See Also
    --------
    ftrace_region: Enables an ftrace region.
    ftrace_region_end : Ends an ftrace region.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.random.rand(10000, 10000)
    >>> vp.prof.ftrace_region_begin('dgemm')
    >>> # something you want to profile
    >>> _ = x @ x
    >>> vp.prof.ftrace_region_end('dgemm')
    """

    nlcpy.request.flush()
    venode = nlcpy.venode.VE()
    if type(message) is not bytes:
        message = message.encode('utf-8')
    buff = numpy.frombuffer(message, dtype=numpy.uint8)
    req = venode.lib_prof.func[b"nlcpy_profiling_region_begin"](
        venode.ctx, veo.OnStack(buff))
    req.wait_result()


def ftrace_region_end(message):
    """Ends an ftrace region.

    A file ftrace.out is generated after running your program that invokes
    this routine.
    The ftrace.out includes performance information of your program.

    Notes
    -----
    It is necessary to specify an identical string *message* to
    :func:`ftrace_region_begin` and :func:`ftrace_region_end`.

    Parameters
    ----------
    message : str
        Any string can be specified to distinguish a user-specified region.

    See Also
    --------
    ftrace_region : Enables an ftrace region.
    ftrace_region_begin : Begins an ftrace region.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.random.rand(10000, 10000)
    >>> vp.prof.ftrace_region_begin('dgemm')
    >>> # something you want to profile
    >>> _ = x @ x
    >>> vp.prof.ftrace_region_end('dgemm')
    """

    nlcpy.request.flush()
    venode = nlcpy.venode.VE()
    if type(message) is not bytes:
        message = message.encode('utf-8')
    buff = numpy.frombuffer(message, dtype=numpy.uint8)
    req = venode.lib_prof.func[b"nlcpy_profiling_region_end"](
        venode.ctx, veo.OnStack(buff))
    req.wait_result()


@contextlib.contextmanager
def ftrace_region(message):
    """Enables profiling with an ftrace region during \'with\' statement.

    A file ftrace.out is generated after running your program that invokes
    this routine.
    The ftrace.out includes performance information of your program.

    Parameters
    ----------
    message : str
        Any string can be specified to distinguish a user-specified region.

    See Also
    --------
    ftrace_region_begin : Begins ftrace region.
    ftrace_region_end : Ends ftrace region.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.random.rand(10000, 10000)
    >>> with vp.prof.ftrace_region('dgemm'):
    ...     # something you want to profile
    ...     _ = x @ x
    """

    ftrace_region_begin(message)
    try:
        yield
    finally:
        ftrace_region_end(message)

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

import pytest


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--test", default='standard')
        parser.addoption(
            '--fast_math', action='store_true')
        parser.addoption(
            '--import_err', action='store_true')
        parser.addoption(
            '--ctxt_close', action='store_true')
        parser.addoption(
            '--proc_destroy', action='store_true')
        parser.addoption(
            '--ftrace_gen', action='store_true')
        parser.addoption(
            '--ftrace_chk', action='store_true')
    except ValueError:
        pass


def pytest_runtest_setup(item):
    option = item.config.getoption("--test")
    fast_math = item.config.getoption("--fast_math")
    import_err = item.config.getoption("--import_err")
    ctxt_close = item.config.getoption("--ctxt_close")
    proc_destroy = item.config.getoption("--proc_destroy")
    ftrace_gen = item.config.getoption("--ftrace_gen")
    ftrace_chk = item.config.getoption("--ftrace_chk")
    if option == 'standard':
        if 'full' in item.keywords:
            pytest.skip(
                "need --test=full option to run this test")
    elif option == 'full':
        pass  # run all tests

    if fast_math:
        if 'no_fast_math' in item.keywords:
            pytest.skip(
                "do not specify --test=fast_math option to run this test")
    else:
        if 'fast_math' in item.keywords:
            pytest.skip(
                "need --test=fast_math option to run this test")

    if not import_err:
        if 'import_err' in item.keywords:
            pytest.skip(
                "need --import_err option to run this test")

    if not ctxt_close:
        if 'ctxt_close' in item.keywords:
            pytest.skip(
                "need --ctxt_close option to run this test")

    if not proc_destroy:
        if 'proc_destroy' in item.keywords:
            pytest.skip(
                "need --proc_destroy option to run this test")

    if not ftrace_gen:
        if 'ftrace_gen' in item.keywords:
            pytest.skip(
                "need --ftrace_gen option to run this test")

    if not ftrace_chk:
        if 'ftrace_chk' in item.keywords:
            pytest.skip(
                "need --ftrace_chk option to run this test")

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

from __future__ import division, absolute_import, print_function
import random

from numpy.testing import (
    assert_equal,
    assert_array_equal,
)

import pytest
import numpy as np
import nlcpy as ny

typedata = {
    '1': ('ca1', [[1, 2], [4, 5]], 'int64', None, None, None),
    '2': ('ca1', 10, 'int64', None, None, None),
    '3': ('ca1', [10, 2], 'int64', None, None, None),
    '4': ('ca1', [[[8, 4, 1], [7, 2, 12]], [[6, 11, 3],
                                            [5, 9, 10]]], 'int64', None, None, None),
    '5': ('ca2', [[[8, 4, 1], [7, 2, 12]], [[6, 11, 3],
                                            [5, 9, 10]]], 'int64', 0, None, None),
    '6': ('ca2', [[[8, 4, 1], [7, 2, 12]], [[6, 11, 3],
                                            [5, 9, 10]]], 'int64', 1, None, None),
    '7': ('ca2', [[[8, 4, 1], [7, 2, 12]], [[6, 11, 3],
                                            [5, 9, 10]]], 'int64', 2, None, None),
    '8': ('ca3', (2, 3, 4), 'int64', None, None, None),
    '9': ('ca3', (2, 3, 4), 'int32', None, None, None),
    '10': ('ca3', (2, 3, 4), 'uint64', None, None, None),
    '11': ('ca3', (2, 3, 4), 'uint32', None, None, None),
    '12': ('ca3', (2, 3, 4), 'float32', None, None, None),
    '13': ('ca4', (2, 3, 4), 'float64', None, None, None),
    '14': ('ca4', (2, 3, 4), 'float64', None, None, None),
    '15': ('ca5', (2, 3, 4), 'float64', None, None, False),
    '16': ('ca5', (2, 3, 4), 'float64', None, None, True),

}

fil = "tmp.npy"


def ca1(arg):
    d = arg[0]

    na = np.array(d)
    src = np.median(na)
    np.save(fil, src)

    ok = np.load(fil)
    dst = ny.median(na)

    return dst, ok


def ca2(arg):
    d = arg[0]
    a = arg[2]

    na = np.array(d)
    src = np.median(na, axis=a)
    np.save(fil, src)

    ok = np.load(fil)
    dst = ny.median(na, axis=a)

    return dst, ok


def ca3(arg):
    d = arg[0]
    t = arg[1]
    a = arg[2]

    ss = 1
    for x in d:
        ss *= x

    if t == 'int32' or t == 'int64' or t == 'uint32' or t == 'uint64':
        seed = [random.randint(1, ss * ss) for i in range(ss)]
    elif t == 'float64' or t == 'float32':
        seed = [random.random() for i in range(ss)]
    else:
        pass

    na = np.array(seed)
    src = np.median(na, axis=a)
    np.save(fil, src)

    ok = np.load(fil)
    dst = ny.median(na, axis=a)

    return dst, ok


def ca4(arg):
    d = arg[0]
    t = arg[1]
    a = arg[2]

    ss = 1
    for x in d:
        ss *= x

    if t == 'int32' or t == 'int64' or t == 'uint32' or t == 'uint64':
        seed = [random.randint(1, ss * ss) for i in range(ss)]
    elif t == 'float64' or t == 'float32':
        seed = [random.random() for i in range(ss)]
    else:
        pass

    na = ny.array(seed)
    ans = ny.median(na, axis=a)
    out = ny.zeros_like(ans)
    dst = ny.median(na, axis=a, out=out)

    return dst, out


def ca5(arg):
    d = arg[0]
    t = arg[1]
    a = arg[2]
    k = arg[4]

    ss = 1
    for x in d:
        ss *= x

    if t == 'int32' or t == 'int64' or t == 'uint32' or t == 'uint64':
        seed = [random.randint(1, ss * ss) for i in range(ss)]
    elif t == 'float64' or t == 'float32':
        seed = [random.random() for i in range(ss)]
    else:
        pass

    na = np.array(seed)
    ans = np.median(na, axis=a, keepdims=k)
    np.save(fil, ans)

    ok = np.load(fil)
    dst = ny.median(na, axis=a, keepdims=k)

    return ok, dst


@pytest.mark.parametrize('k,v', typedata.items())
def test_run2(k, v):
    l_v = tuple(v)
    ans1, ans2 = eval(l_v[0])(l_v[1:])

    print("ans1={} ans2={}".format(ans1, ans2))
    assert_equal(ans1, ans2)


def test_me_case_1():
    a = ny.array([[10, 7, 4], [3, 2, 1]])
    b = a.copy()
    ny.median(b, overwrite_input=False)
    print("a={} b={}".format(a, b))
    assert ny.all(a == b)


def test_me_case_2():
    ny_a = ny.array([[10, ny.nan, 4], [3, 2, 1]])
    np_a = np.array([[10, np.nan, 4], [3, 2, 1]])
    ans_ny = ny.median(ny_a)
    ans_np = np.median(np_a)

    print("ans1={} ans2={}".format(ans_ny, ans_np))
    assert_array_equal(ans_np, ans_ny.get())


def test_me_case_3():
    ny_a = ny.array([[10, ny.nan, 4], [3, 2, 1]])
    np_a = np.array([[10, np.nan, 4], [3, 2, 1]])
    ans_ny = ny.median(ny_a, axis=1)
    ans_np = np.median(np_a, axis=1)

    print("ans1={} ans2={}".format(ans_ny, ans_np))
    assert_array_equal(ans_np, ans_ny.get())

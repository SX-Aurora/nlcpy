#
# * The source code in this file is developed independently by NEC Corporation.
#
# # NLCPy License #
#
#     Copyright (c) 2020-2021 NEC Corporation
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

from numpy.testing import assert_array_almost_equal

import numpy as np
import nlcpy as ny


def test_me_case_1():
    np_a = np.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])
    ny_a = ny.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])

    ans_np = np.cov(np_a)
    ans_ny = ny.cov(ny_a)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def test_me_case_2():
    np_a = np.array([-2.1, -1, 4.3])
    ny_a = ny.array([-2.1, -1, 4.3])

    ans_np = np.cov(np_a)
    ans_ny = ny.cov(ny_a)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def test_me_case_3():
    np_a = np.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])
    ny_a = ny.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])
    np_y = np.array([2, 1, 1, 8, 9, 4, 3, 5, 7])
    ny_y = ny.array([2, 1, 1, 8, 9, 4, 3, 5, 7])

    ans_np = np.cov(np_a, np_y)
    ans_ny = ny.cov(ny_a, ny_y)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def test_me_case_4():
    np_a = np.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])
    ny_a = ny.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])

    ans_np = np.cov(np_a.T, rowvar=False)
    ans_ny = ny.cov(ny_a.T, rowvar=False)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def test_me_case_5():
    np_a = np.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])
    ny_a = ny.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])

    ans_np = np.cov(np_a.T, rowvar=True)
    ans_ny = ny.cov(ny_a.T, rowvar=True)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def test_me_case_6():
    np_a = np.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])
    ny_a = ny.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])

    ans_np = np.cov(np_a, bias=False)
    ans_ny = ny.cov(ny_a, bias=False)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def test_me_case_7():
    np_a = np.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])
    ny_a = ny.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])

    ans_np = np.cov(np_a, bias=True)
    ans_ny = ny.cov(ny_a, bias=True)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def test_me_case_8():
    np_a = np.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])
    ny_a = ny.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])

    ans_np = np.cov(np_a, ddof=None)
    ans_ny = ny.cov(ny_a, ddof=None)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def test_me_case_9():
    np_a = np.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])
    ny_a = ny.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])

    ans_np = np.cov(np_a, ddof=0)
    ans_ny = ny.cov(ny_a, ddof=0)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def test_me_case_10():
    np_a = np.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])
    ny_a = ny.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])

    ans_np = np.cov(np_a, ddof=1)
    ans_ny = ny.cov(ny_a, ddof=1)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def test_me_case_11():
    np_a = np.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])
    ny_a = ny.array([[1, 2, 1, 9, 10, 3, 2, 6, 7],
                     [2, 1, 8, 3, 7, 5, 10, 7, 2]])

    ans_np = np.cov(np_a, ddof=2)
    ans_ny = ny.cov(ny_a, ddof=2)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def testt_me_case_12():
    np_a = np.array([[10, 5, 2, 4, 9, 3, 2], [10, 2, 8, 3, 7, 4, 1]])
    np_y = np.array([1, 2, 2, 1, 1, 1, 1])
    ny_a = ny.array([[10, 5, 2, 4, 9, 3, 2], [10, 2, 8, 3, 7, 4, 1]])
    ny_y = ny.array([1, 2, 2, 1, 1, 1, 1])

    ans_np = np.cov(np_a, fweights=np_y)
    ans_ny = ny.cov(ny_a, fweights=ny_y)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def testt_me_case_13():
    np_a = np.array([[10, 5, 2, 4, 9, 3, 2], [10, 2, 8, 3, 7, 4, 1]])
    ny_a = ny.array([[10, 5, 2, 4, 9, 3, 2], [10, 2, 8, 3, 7, 4, 1]])

    ans_np = np.cov(np_a, aweights=None)
    ans_ny = ny.cov(ny_a, aweights=None)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())


def testt_me_case_14():
    np_a = np.array([[10, 5, 2, 4, 9, 3, 2], [10, 2, 8, 3, 7, 4, 1]])
    ny_a = ny.array([[10, 5, 2, 4, 9, 3, 2], [10, 2, 8, 3, 7, 4, 1]])

    np_w = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    ny_w = ny.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1])

    ans_np = np.cov(np_a, aweights=np_w)
    ans_ny = ny.cov(ny_a, aweights=ny_w)

    print("numpy={} nlcpy={}".format(ans_np, ans_ny))
    assert_array_almost_equal(ans_np, ans_ny.get())

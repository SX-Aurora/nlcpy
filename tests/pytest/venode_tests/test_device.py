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

import threading
import unittest

import pytest

import numpy
import nlcpy
from nlcpy.core import internal
from nlcpy import venode
from nlcpy import testing


nve = nlcpy.venode.get_num_available_venodes()


@testing.multi_ve(2)
class TestVENodeComparison(unittest.TestCase):

    def check_eq(self, result, obj1, obj2):
        if result:
            assert obj1 == obj2
            assert obj2 == obj1
            assert not (obj1 != obj2)
            assert not (obj2 != obj1)
        else:
            assert obj1 != obj2
            assert obj2 != obj1
            assert not (obj1 == obj2)
            assert not (obj2 == obj1)

    def test_equality(self):
        self.check_eq(True, venode.VE(0), venode.VE(0))
        self.check_eq(True, venode.VE(1), venode.VE(1))
        self.check_eq(False, venode.VE(0), venode.VE(1))
        self.check_eq(False, venode.VE(0), 0)
        self.check_eq(False, venode.VE(0), None)
        self.check_eq(False, venode.VE(0), object())

    def test_lt_ve(self):
        assert venode.VE(0) < venode.VE(1)
        assert not (venode.VE(0) < venode.VE(0))
        assert not (venode.VE(1) < venode.VE(0))

    def test_le_ve(self):
        assert venode.VE(0) <= venode.VE(1)
        assert venode.VE(0) <= venode.VE(0)
        assert not (venode.VE(1) <= venode.VE(0))

    def test_gt_ve(self):
        assert not (venode.VE(0) > venode.VE(0))
        assert not (venode.VE(0) > venode.VE(0))
        assert venode.VE(1) > venode.VE(0)

    def test_ge_ve(self):
        assert not (venode.VE(0) >= venode.VE(1))
        assert venode.VE(0) >= venode.VE(0)
        assert venode.VE(1) >= venode.VE(0)

    def check_comparison_other_type(self, obj1, obj2):
        with pytest.raises(TypeError):
            obj1 < obj2
        with pytest.raises(TypeError):
            obj1 <= obj2
        with pytest.raises(TypeError):
            obj1 > obj2
        with pytest.raises(TypeError):
            obj1 >= obj2
        with pytest.raises(TypeError):
            obj2 < obj1
        with pytest.raises(TypeError):
            obj2 <= obj1
        with pytest.raises(TypeError):
            obj2 > obj1
        with pytest.raises(TypeError):
            obj2 >= obj1

    def test_comparison_other_type(self):
        self.check_comparison_other_type(venode.VE(0), 0)
        self.check_comparison_other_type(venode.VE(0), 1)
        self.check_comparison_other_type(venode.VE(1), 0)
        self.check_comparison_other_type(venode.VE(1), None)
        self.check_comparison_other_type(venode.VE(1), object())


@testing.multi_ve(2)
class TestVESwitch(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)

    def tearDown(self):
        self._prev_ve.apply()

    def test_use(self):
        ve0 = venode.VE(0)
        ve1 = venode.VE(1)

        assert ve1.use() is ve1
        assert 1 == venode.VE().id
        assert ve0.use() is ve0
        assert 0 == venode.VE().id

    def test_apply(self):
        ve0 = venode.VE(0)
        ve1 = venode.VE(1)

        assert ve1.apply() is ve1
        assert 1 == venode.VE().id
        assert ve0.apply() is ve0
        assert 0 == venode.VE().id

    def test_context(self):
        ve0 = venode.VE(0)
        ve1 = venode.VE(1)

        with ve0:
            assert 0 == venode.VE().id
            with ve1:
                assert 1 == venode.VE().id
            assert 0 == venode.VE().id
            with ve0:
                assert 0 == venode.VE().id
                with ve1:
                    assert 1 == venode.VE().id
                assert 0 == venode.VE().id
            assert 0 == venode.VE().id
        assert 0 == venode.VE().id

        with ve1:
            assert 1 == venode.VE().id
            with ve0:
                assert 0 == venode.VE().id
            assert 1 == venode.VE().id
            with ve1:
                assert 1 == venode.VE().id
                with ve0:
                    assert 0 == venode.VE().id
                assert 1 == venode.VE().id
            assert 1 == venode.VE().id
        assert 0 == venode.VE().id

    def test_context_and_use(self):
        ve0 = venode.VE(0)
        ve1 = venode.VE(1)

        ve1.use()
        assert 1 == venode.VE().id
        with ve0:
            assert 0 == venode.VE().id
            ve1.use()
            assert 1 == venode.VE().id
            with ve1:
                assert 1 == venode.VE().id
            assert 0 == venode.VE().id
        assert 0 == venode.VE().id

    def test_context_and_apply(self):
        ve0 = venode.VE(0)
        ve1 = venode.VE(1)

        ve1.apply()
        assert 1 == venode.VE().id
        with ve0:
            assert 0 == venode.VE().id
            ve1.apply()
            assert 1 == venode.VE().id
            with ve1:
                assert 1 == venode.VE().id
            assert 1 == venode.VE().id
        assert 1 == venode.VE().id

    def test_thread_safe(self):
        ve0 = venode.VE(0)
        ve1 = venode.VE(1)

        t0_setup = threading.Event()
        t1_setup = threading.Event()
        t0_first_exit = threading.Event()

        t0_ves = []
        t1_ves = []

        def t0_seq():
            t0_ves.append(venode.VE().id)  # 0
            with ve0:
                t0_ves.append(venode.VE().id)  # 0
                with ve1:
                    t0_ves.append(venode.VE().id)  # 1
                    t0_setup.set()
                    t1_setup.wait()
                    t0_ves.append(venode.VE().id)  # 1
                t0_first_exit.set()
                t0_ves.append(venode.VE().id)  # 0
            t0_ves.append(venode.VE().id)  # 0

        def t1_seq():
            t1_ves.append(venode.VE().id)  # 0
            t0_setup.wait()
            t1_ves.append(venode.VE().id)  # 0
            with ve1:
                t1_ves.append(venode.VE().id)  # 1
                with ve0:
                    t1_ves.append(venode.VE().id)  # 0
                    t1_setup.set()
                    t0_first_exit.wait()
                    t1_ves.append(venode.VE().id)  # 0
                t1_ves.append(venode.VE().id)  # 1
            t1_ves.append(venode.VE().id)  # 0

        t1 = threading.Thread(target=t1_seq)
        t1.start()
        t0_seq()
        t1.join()
        assert t0_ves == [0, 0, 1, 1, 0, 0]
        assert t1_ves == [0, 0, 1, 0, 0, 1, 0]

    def test_invalid(self):
        with pytest.raises(ValueError):
            venode.VE(100)
        with venode.VE(0):
            pass
        with venode.VE(1):
            pass


@testing.parameterize(*(
    testing.product({
        'shape': [
            [100], [10, 20], [3, 10, 7], [2, 3, 4, 5],
            [7, 6, 5, 2, 3, 4], [1, 1, 1], [1, 100, 1],
        ],
    })
))
@testing.multi_ve(2)
class TestArrayDistributeMultiVE(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)

    def tearDown(self):
        self._prev_ve.apply()

    @testing.for_all_dtypes()
    def test_ones(self, dtype):
        ve0 = venode.VE(0)
        ve1 = venode.VE(1)

        with ve0:
            x_ve0 = nlcpy.ones(self.shape, dtype)

        with ve1:
            x_ve1 = nlcpy.ones(self.shape, dtype)

        assert x_ve0.venode == ve0
        assert x_ve1.venode == ve1
        with ve0:
            assert nlcpy.all(x_ve0 == venode.transfer_array(x_ve1, ve0))
        with ve1:
            assert nlcpy.all(x_ve1 == venode.transfer_array(x_ve0, ve1))

    @testing.for_all_dtypes()
    def test_random(self, dtype):
        ve0 = venode.VE(0)
        ve1 = venode.VE(1)

        with ve0:
            nlcpy.random.seed(123)
            x_ve0 = nlcpy.random.rand(*self.shape).astype(dtype)

        with ve1:
            nlcpy.random.seed(123)
            x_ve1 = nlcpy.random.rand(*self.shape).astype(dtype)

        assert x_ve0.venode == ve0
        assert x_ve1.venode == ve1
        with ve0:
            assert nlcpy.all(x_ve0 == venode.transfer_array(x_ve1, ve0))
        with ve1:
            assert nlcpy.all(x_ve1 == venode.transfer_array(x_ve0, ve1))

    @testing.for_all_dtypes()
    def test_random_nested(self, dtype):
        ve0 = venode.VE(0)
        ve1 = venode.VE(1)

        with ve0:
            nlcpy.random.seed(123)
            x_ve0 = nlcpy.random.rand(*self.shape).astype(dtype)

            with ve1:
                nlcpy.random.seed(123)
                x_ve1 = nlcpy.random.rand(*self.shape).astype(dtype)

        assert x_ve0.venode == ve0
        assert x_ve1.venode == ve1
        with ve0:
            assert nlcpy.all(x_ve0 == venode.transfer_array(x_ve1, ve0))
        with ve1:
            assert nlcpy.all(x_ve1 == venode.transfer_array(x_ve0, ve1))

    @testing.for_all_dtypes()
    def test_binary_op(self, dtype):
        ve0 = venode.VE(0)
        ve1 = venode.VE(1)

        with ve0:
            x_ve0 = testing.shaped_arange(self.shape, nlcpy, dtype=dtype)

        with ve1:
            x_ve1 = testing.shaped_arange(self.shape, nlcpy, dtype=dtype)

        with ve0:
            y_ve0 = x_ve0 + venode.transfer_array(x_ve1, ve0)
            assert y_ve0.venode == ve0
            testing.assert_array_equal(y_ve0, x_ve0 + x_ve0)

        with ve1:
            y_ve1 = venode.transfer_array(x_ve0, ve1) + x_ve1
            assert y_ve1.venode == ve1
            testing.assert_array_equal(y_ve1, x_ve1 + x_ve1)


@testing.parameterize(*(
    testing.product({
        'shape': [
            [100], [10, 20], [3, 10, 7], [2, 3, 4, 5],
            [7, 6, 5, 2, 3, 4], [1], [1, 100, 1],
            [1024, 3, 1024], [2, 124, 3, 1024],
        ],
    })
))
@testing.multi_ve(nve)
class TestTransferArray(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_transfer_array1(self, dtype):
        ves = [venode.VE(n) for n in range(nve)]
        x_src = []
        x_dst = []
        for ve in ves:
            with ve:
                x_src.append(testing.shaped_arange(self.shape, nlcpy, dtype=dtype))
                x_dst.append(nlcpy.empty(self.shape, dtype=dtype))
        for i, ve in enumerate(ves):
            for j, _ve in enumerate(ves):
                prev_ve = venode.VE()
                y = venode.transfer_array(x_src[i], target_ve=_ve)
                testing.assert_array_equal(x_src[i], y)
                y = venode.transfer_array(x_src[i], target_ve=_ve, dst=x_dst[j])
                assert id(x_dst[j]) == id(y)
                testing.assert_array_equal(x_src[i], x_dst[j])
                assert prev_ve == venode.VE()

    @testing.for_all_dtypes()
    def test_transfer_array2(self, dtype):
        ves = [venode.VE(n) for n in range(nve)]
        x_src = []
        x_dst = []
        for ve in ves:
            with ve:
                x_src.append(testing.shaped_arange(
                    internal.prod(self.shape), nlcpy, dtype=dtype))
                x_dst.append(nlcpy.empty(self.shape, dtype=dtype))
        for i, ve in enumerate(ves):
            for j, _ve in enumerate(ves):
                prev_ve = venode.VE()
                y = venode.transfer_array(x_src[i], target_ve=_ve)
                testing.assert_array_equal(x_src[i], y)
                y = venode.transfer_array(x_src[i], target_ve=_ve, dst=x_dst[j])
                assert id(x_dst[j]) == id(y)
                testing.assert_array_equal(x_src[i], x_dst[j])
                assert prev_ve == venode.VE()

    @testing.for_all_dtypes()
    def test_transfer_array_broadcast(self, dtype):
        ves = [venode.VE(n) for n in range(nve)]
        x_src = nlcpy.array(1, dtype=dtype)
        x_dst = []
        for ve in ves:
            with ve:
                x_dst.append(nlcpy.empty(self.shape, dtype=dtype))
        for i, ve in enumerate(ves):
            for j, _ve in enumerate(ves):
                prev_ve = venode.VE()
                y = venode.transfer_array(x_src, target_ve=_ve, dst=x_dst[j])
                assert id(x_dst[j]) == id(y)
                testing.assert_array_equal(x_src, x_dst[j])
                assert prev_ve == venode.VE()

    @testing.for_all_dtypes()
    def test_transfer_array_samenode(self, dtype):
        ves = [venode.VE(n) for n in range(nve)]
        x_src = []
        x_dst = []
        for ve in ves:
            with ve:
                x_src.append(testing.shaped_arange(self.shape, nlcpy, dtype=dtype))
                x_dst.append(nlcpy.empty(self.shape, dtype=dtype))
        for i, ve in enumerate(ves):
            prev_ve = venode.VE()
            y = venode.transfer_array(x_src[i], target_ve=ve)
            assert x_src[i].venode == ve
            testing.assert_array_equal(x_src[i], y)
            y = venode.transfer_array(x_src[i], target_ve=ve, dst=x_dst[i])
            assert id(x_dst[i]) == id(y)
            assert x_src[i].venode == x_dst[i].venode == ve
            testing.assert_array_equal(x_src[i], x_dst[i])
            assert prev_ve == venode.VE()


@testing.multi_ve(2)
class TestTransferArrayErr(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_transfer_array_size_mismatch(self, dtype):
        with venode.VE(0):
            x = nlcpy.empty(10, dtype=dtype)
        with venode.VE(1):
            y = nlcpy.empty(5, dtype=dtype)
        with pytest.raises(ValueError):
            venode.transfer_array(x, y.venode, y)
        with pytest.raises(ValueError):
            venode.transfer_array(y, x.venode, x)

    def test_transfer_array_venode_mismatch1(self):
        with venode.VE(0):
            x = nlcpy.empty(10)
        with venode.VE(1):
            y = nlcpy.empty(10)
        with pytest.raises(ValueError):
            venode.transfer_array(x, x.venode, y)

    def test_transfer_array_venode_mismatch2(self):
        with venode.VE(0):
            x = nlcpy.empty(10)
            y = nlcpy.empty(10)
        with pytest.raises(ValueError):
            venode.transfer_array(x, venode.VE(1), y)


@testing.multi_ve(2)
class TestTransferArrayWithUse(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)

    def tearDown(self):
        self._prev_ve.apply()

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_transfer_array_no_dst_with_use(self, dtype, order):
        with venode.VE(0):
            src = testing.shaped_arange((2, 3, 4), nlcpy, dtype, order)
        venode.VE(1).use()
        dst = venode.transfer_array(src, venode.VE(1))
        assert venode.VE() == venode.VE(1)
        testing.assert_array_equal(src, dst)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_transfer_array_broadcast_with_use(self, dtype, order):
        with venode.VE(0):
            src = nlcpy.ones(4, dtype, order)
        venode.VE(1).use()
        dst = nlcpy.empty((2, 3, 4), dtype, order)
        venode.transfer_array(src, venode.VE(1), dst)
        assert venode.VE() == venode.VE(1)
        testing.assert_array_equal(dst, numpy.ones((2, 3, 4), dtype, order))


class TestConnect(unittest.TestCase):

    def test_connect(self):
        for veid in range(venode.get_num_available_venodes()):
            venode.VE(veid).connect()


class TestDisconnect(unittest.TestCase):

    def test_disconnect(self):
        for veid in range(venode.get_num_available_venodes()):
            with pytest.raises(NotImplementedError):
                venode.VE(veid).disconnect()

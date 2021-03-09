#
# * The source code in this file is based on the soure code of CuPy.
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

import numpy
import math
import sys
import unittest

import nlcpy
from nlcpy import testing


class TestRanges(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_arange(self, xp, dtype):
        return xp.arange(10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_arange2(self, xp, dtype):
        return xp.arange(5, 10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_arange3(self, xp, dtype):
        return xp.arange(1, 11, 2, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_arange4(self, xp, dtype):
        return xp.arange(20, 2, -3, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_arange5(self, xp, dtype):
        return xp.arange(0, 100, None, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_arange6(self, xp, dtype):
        return xp.arange(0, 2, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_arange7(self, xp, dtype):
        return xp.arange(10, 11, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_equal()
    def test_arange8(self, xp, dtype):
        return xp.arange(10, 8, -1, dtype=dtype)

    @testing.numpy_nlcpy_raises()
    def test_arange9(self, xp):
        return xp.arange(10, dtype=xp.bool_)

    @testing.numpy_nlcpy_array_equal()
    def test_arange_no_dtype_int(self, xp):
        return xp.arange(1, 11, 2)

    @testing.numpy_nlcpy_array_equal()
    def test_arange_no_dtype_float(self, xp):
        return xp.arange(1.0, 11.0, 2.0)

    @testing.numpy_nlcpy_array_equal()
    def test_arange_negative_size(self, xp):
        return xp.arange(3, 1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace(self, xp, dtype):
        return xp.linspace(0, 10, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace2(self, xp, dtype):
        return xp.linspace(10, 0, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_zero_num(self, xp, dtype):
        return xp.linspace(0, 10, 0, dtype=dtype)

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_zero_num_no_endopoint_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 0, dtype=dtype, endpoint=False,
                              retstep=True)
        self.assertTrue(math.isnan(step))
        return x

    @testing.with_requires('numpy>=1.10', 'numpy<1.18')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_one_num_no_endopoint_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 1, dtype=dtype, endpoint=False,
                              retstep=True)
        self.assertTrue(math.isnan(step))
        return x

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_one_num(self, xp, dtype):
        return xp.linspace(0, 2, 1, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_no_endpoint(self, xp, dtype):
        return xp.linspace(0, 10, 5, dtype=dtype, endpoint=False)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_nlcpy_array_equal()
    def test_linspace_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 5, dtype=dtype, retstep=True)
        self.assertEqual(step, 2.5)
        return x

    @testing.numpy_nlcpy_allclose()
    def test_linspace_no_dtype_int(self, xp):
        return xp.linspace(0, 10)

    @testing.numpy_nlcpy_allclose()
    def test_linspace_no_dtype_float(self, xp):
        return xp.linspace(0.0, 10.0)

    @testing.numpy_nlcpy_allclose()
    def test_linspace_float_args_with_int_dtype(self, xp):
        return xp.linspace(0.1, 9.1, 11, dtype=int)

    @testing.for_dtypes('?ilILdD', name='dtype_s')
    @testing.for_dtypes('?ilILdD', name='dtype_e')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_linspace_array(self, xp, dtype_s, dtype_e):
        start = testing.shaped_random([5], xp, dtype=dtype_s)
        stop = testing.shaped_random([4, 5], xp, dtype=dtype_e)
        return xp.linspace(start, stop, axis=0)

    @testing.for_dtypes('?ilILdD', name='dtype_s')
    @testing.for_dtypes('fF', name='dtype_e')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_linspace_array2(self, xp, dtype_s, dtype_e):
        start = testing.shaped_random([4, 5], xp, dtype=dtype_s)
        stop = testing.shaped_random([3, 4, 5], xp, dtype=dtype_e)
        return xp.linspace(start, stop, axis=1)

    @testing.for_dtypes('fF', name='dtype_s')
    @testing.for_dtypes('?ilILdD', name='dtype_e')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_linspace_array3(self, xp, dtype_s, dtype_e):
        start = testing.shaped_random([4, 5], xp, dtype=dtype_s)
        stop = testing.shaped_random([3, 4, 5], xp, dtype=dtype_e)
        return xp.linspace(start, stop, axis=2)

    @testing.for_dtypes('fF', name='dtype_s')
    @testing.for_dtypes('fF', name='dtype_e')
    @testing.numpy_nlcpy_allclose(atol=1e-5, rtol=1e-5)
    def test_linspace_array4(self, xp, dtype_s, dtype_e):
        start = testing.shaped_random([4, 5], xp, dtype=dtype_s)
        stop = testing.shaped_random([3, 4, 5], xp, dtype=dtype_e)
        return xp.linspace(start, stop, axis=xp.array(1))

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_nlcpy_raises()
    def test_linspace_neg_num(self, xp):
        return xp.linspace(0, 10, -1)

    @testing.numpy_nlcpy_allclose()
    def test_linspace_float_overflow(self, xp):
        return xp.linspace(0., sys.float_info.max / 5, 10, dtype=float)

    # @testing.with_requires('numpy>=1.10')
    # @testing.numpy_nlcpy_array_equal()
    # def test_linspace_float_underflow(self, xp):
    #     # find minimum subnormal number
    #     x = sys.float_info.min
    #     while x / 2 > 0:
    #         x /= 2
    #     return xp.linspace(0., x, 10, dtype=float)


@testing.parameterize(*(
    testing.product({
        'size': [1, 3, 5],
        'num': [1, 2, 3, 5, 7],
    })
))
class TestMeshgrid(unittest.TestCase):
    @testing.numpy_nlcpy_array_list_equal()
    def test_meshgrid(self, xp):
        arrays = [testing.shaped_random([self.size], xp) for _ in range(self.num)]
        return xp.meshgrid(*arrays)


class TestMeshgrid2(unittest.TestCase):
    @testing.numpy_nlcpy_array_list_equal()
    def test_meshgrid_empty(self, xp):
        return xp.meshgrid()

    @testing.numpy_nlcpy_array_list_equal()
    def test_meshgrid_empty_list(self, xp):
        return xp.meshgrid([])

    @testing.numpy_nlcpy_array_list_equal()
    def test_meshgrid_empty_tuple(self, xp):
        return xp.meshgrid(())

    @testing.numpy_nlcpy_array_list_equal()
    def test_meshgrid_scalar(self, xp):
        return xp.meshgrid(1)

    @testing.numpy_nlcpy_array_list_equal()
    def test_meshgrid_scalars(self, xp):
        return xp.meshgrid(True, 1.0, 2)


@testing.parameterize(*(
    testing.product({
        'indexing': ['xy', 'ij'],
        'sparse': [True, False],
        'copy': [True, False],
    })
))
class TestMeshgridKwargs(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_array_list_equal()
    def test_meshgrid_kwargs(self, xp, dtype):
        arrays = [testing.shaped_random([10], xp, dtype=dtype) for _ in range(3)]
        return xp.meshgrid(
            *arrays, indexing=self.indexing, sparse=self.sparse, copy=self.copy)

    def test_meshgrid_owndata(self):
        arrays = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        nx = numpy.meshgrid(
            *arrays, indexing=self.indexing, sparse=self.sparse, copy=self.copy)

        vx = nlcpy.meshgrid(
            *arrays, indexing=self.indexing, sparse=self.sparse, copy=self.copy)

        for _nx, _vx in zip(nx, vx):
            assert _nx.flags.owndata == _vx.flags.owndata


class TestMeshgridFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_meshgrid_incompatible_indexing(self, xp):
        return xp.meshgrid(1, indexing='foo')

    @testing.numpy_nlcpy_raises()
    def test_meshgrid_incompatible_kwargs(self, xp):
        return xp.meshgrid(1, foo='var')


@testing.parameterize(*(
    testing.product({
        'start': [-3, 0, 2, 2.5, 1 + 4j],
        'stop': [-2, 0, 3, 3.5, 2 - 2j],
        'num': [0, 1, 50],
        'base': [1, 5.5, 10.0, 3 + 2j],
        'endpoint': [True, False]
    })
))
class TestLogspace(unittest.TestCase):
    @testing.numpy_nlcpy_allclose()
    def test_logspace(self, xp):
        args = dict()
        args["start"] = self.start
        args["stop"] = self.stop
        args["num"] = self.num
        args["endpoint"] = self.endpoint
        args["base"] = self.base
        args["dtype"] = None
        args["axis"] = 0
        return xp.logspace(**args)


class TestLogspace2(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_nlcpy_allclose()
    def test_logspace_dtype(self, xp, dtype):
        return xp.logspace(2, 5, 50, True, 5.5, dtype, 0)

    @testing.numpy_nlcpy_allclose()
    def test_logspace_base_is_zero(self, xp):
        return xp.logspace(2, 5, 50, True, 0)

    @testing.with_requires('numpy<1.18')
    @testing.numpy_nlcpy_allclose()
    def test_logspace_float_num(self, xp):
        return xp.logspace(1, 10, 10.5)

    @testing.numpy_nlcpy_array_equal()
    def test_logspace_cast_to_bool(self, xp):
        return xp.logspace(1.4 + 1.2j, 2.3 - 3.5j, dtype=bool)

    @testing.for_dtypes('ilIL', name='dtype')
    @testing.numpy_nlcpy_allclose(atol=1, rtol=1e-12)
    def test_logspace_cast_to_int(self, xp, dtype):
        return xp.logspace(1.4 + 1.2j, 2.3 - 3.5j, dtype=dtype)

    @testing.for_dtypes('fF', name='dtype')
    @testing.numpy_nlcpy_allclose(atol=1e-5, rtol=1e-5)
    def test_logspace_cast_to_single(self, xp, dtype):
        return xp.logspace(1.4 + 1.2j, 2.3 - 3.5j, dtype=dtype)

    @testing.for_dtypes('dD', name='dtype')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_logspace_cast_to_double(self, xp, dtype):
        return xp.logspace(1.4 + 1.2j, 2.3 - 3.5j, dtype=dtype)

    @testing.for_dtypes('?ilILdD', name='dtype_s')
    @testing.for_dtypes('?ilILdD', name='dtype_e')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_logspace_array(self, xp, dtype_s, dtype_e):
        start = testing.shaped_random([5], xp, dtype=dtype_s)
        stop = testing.shaped_random([4, 5], xp, dtype=dtype_e)
        return xp.logspace(start, stop)

    @testing.for_dtypes('?ilILdD', name='dtype_s')
    @testing.for_dtypes('fF', name='dtype_e')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_logspace_array2(self, xp, dtype_s, dtype_e):
        start = testing.shaped_random([4, 5], xp, dtype=dtype_s)
        stop = testing.shaped_random([3, 4, 5], xp, dtype=dtype_e)
        return xp.logspace(start, stop)

    @testing.for_dtypes('fF', name='dtype_s')
    @testing.for_dtypes('?ilILdD', name='dtype_e')
    @testing.numpy_nlcpy_allclose(atol=1e-12, rtol=1e-12)
    def test_logspace_array3(self, xp, dtype_s, dtype_e):
        start = testing.shaped_random([4, 5], xp, dtype=dtype_s)
        stop = testing.shaped_random([3, 4, 5], xp, dtype=dtype_e)
        return xp.logspace(start, stop)

    @testing.for_dtypes('fF', name='dtype_s')
    @testing.for_dtypes('fF', name='dtype_e')
    @testing.numpy_nlcpy_allclose(atol=1e-5, rtol=1e-5)
    def test_logspace_array4(self, xp, dtype_s, dtype_e):
        start = testing.shaped_random([4, 5], xp, dtype=dtype_s)
        stop = testing.shaped_random([3, 4, 5], xp, dtype=dtype_e)
        return xp.logspace(start, stop)


class TestLogspaceFailure(unittest.TestCase):
    @testing.numpy_nlcpy_raises()
    def test_logspace_negative_num(self, xp):
        return xp.logspace(0, 2, -1)

    @testing.numpy_nlcpy_raises()
    def test_logspace_complex_num(self, xp):
        return xp.logspace(0, 2, 1 + 2j)

    @testing.numpy_nlcpy_raises()
    def test_logspace_shape_mismatch(self, xp):
        return xp.logspace(xp.zeros([2, 3]), xp.ones([3, 3]))

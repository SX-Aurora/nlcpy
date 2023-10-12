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

import unittest
import pytest
import pickle
import io
import nlcpy
import numpy
from nlcpy import testing
from nlcpy.core import internal
from nlcpy.core import manipulation


class MyVEArray:

    def __init__(self, x):
        self.shape = x.shape
        self.typestr = x.dtype.str
        self.strides = x.strides
        self.data = (x.veo_hmem, False)

    @property
    def __ve_array_interface__(self):
        return {
            'shape': self.shape,
            'typestr': self.typestr,
            'version': 1,
            'strides': self.strides,
            'data': self.data
        }


class MyVEArrayWithMask(MyVEArray):

    @property
    def __ve_array_interface__(self):
        return {
            'shape': self.shape,
            'typestr': self.typestr,
            'version': 1,
            'strides': self.strides,
            'data': self.data,
            'mask': True
        }


class MyVEArraySizeNegative(MyVEArray):

    @property
    def __ve_array_interface__(self):
        return {
            'shape': [-1 * s for s in self.shape],
            'typestr': self.typestr,
            'version': 1,
            'strides': self.strides,
            'data': self.data,
        }


class TestNdarray(unittest.TestCase):

    def test_dtype_invalid(self):
        with pytest.raises(TypeError):
            nlcpy.ndarray(10, dtype='i2')

    def test_order_invalid(self):
        with pytest.raises(ValueError):
            nlcpy.ndarray(10, order='A')

    def test_repr(self):
        x = nlcpy.empty(1)
        assert isinstance(x.__repr__(), str)

    def test_ve_array_interface(self):
        x = nlcpy.empty(10)
        vai = x.__ve_array_interface__
        assert vai['data'][0] == x.veo_hmem
        y = x[::2]
        vai = y.__ve_array_interface__
        assert vai['data'][0] == y.veo_hmem
        assert vai['strides'] == y.strides
        z = nlcpy.array([])
        vai = z.__ve_array_interface__
        assert vai['data'][0] == 0

    def test_richcomp_none(self):
        x = nlcpy.empty(3)
        assert (x > None) is None
        assert (None > x) is None

    def test_complex(self):
        ref = 1 + 1j
        x = nlcpy.array(ref)
        assert complex(x) == ref

    def test_reduce(self):
        exp = nlcpy.arange(10)
        with io.BytesIO() as f:
            pickle.dump(exp, f)
            f.seek(0)
            act = pickle.load(f)
        testing.assert_array_equal(exp, act)

    def test_pow_with_modulo(self):
        x = nlcpy.arange(10)
        y = pow(x, 2, 3)
        z = (x ** 2) % 3
        testing.assert_array_equal(y, z)

    def test_imatmul(self):
        x = nlcpy.empty((3, 3), dtype='f8')
        y = x.copy()
        with pytest.raises(TypeError):
            x @= y

    def test_astype_subok(self):
        x = nlcpy.empty(5, dtype='f4')
        with pytest.raises(NotImplementedError):
            x.astype('f8', subok=True)

    def test_all_keepdims_novalue(self):
        x = nlcpy.array([True, True, True])
        z = x.all(keepdims=nlcpy._NoValue)
        assert z

    def test_any_keepdims_novalue(self):
        x = nlcpy.array([False, True, False])
        z = x.any(keepdims=nlcpy._NoValue)
        assert z

    def test_set_sh_and_std_length_mismatch(self):
        x = nlcpy.empty((3, 4))
        with pytest.raises(ValueError):
            x._set_shape_and_strides((3, 4), (8,), False, False)

    def test_ndarray_from_vai(self):
        x = nlcpy.random.rand(10)
        myarr = MyVEArray(x)
        y = nlcpy.array(myarr)
        testing.assert_array_equal(x, y)

    def test_ndarray_from_vai_with_mask(self):
        x = nlcpy.random.rand(10)
        myarr = MyVEArrayWithMask(x)
        with pytest.raises(NotImplementedError):
            nlcpy.array(myarr)

    def test_ndarray_from_vai_size_negative(self):
        x = nlcpy.random.rand(10)
        myarr = MyVEArraySizeNegative(x)
        with pytest.raises(ValueError):
            nlcpy.array(myarr)

    def test_array_subok(self):
        with pytest.raises(NotImplementedError):
            nlcpy.core.core.array(0, subok=True)

    def test_array_ndmin_above_maxndim(self):
        maxndim = nlcpy.core.get_nlcpy_maxndim()
        with pytest.raises(ValueError):
            nlcpy.array(0, ndmin=maxndim + 1)

    def test_array_ndmin(self):
        x = nlcpy.array([0, 1, 2])
        y = nlcpy.array([0, 1, 2])
        z = nlcpy.array([x, y], ndmin=10)
        assert z.ndim == 10

    def test_array_from_None(self):
        with pytest.raises(NotImplementedError):
            nlcpy.array(None)

    def test_may_share_memory_max_work(self):
        with pytest.raises(NotImplementedError):
            nlcpy.may_share_memory(
                nlcpy.array([1, 2]), nlcpy.array([3, 4]), max_work=1)

    def test_may_share_memory_ret_false(self):
        assert nlcpy.may_share_memory(
            nlcpy.array([1, 2]), nlcpy.array([3, 4])) is False


class TestInternal(unittest.TestCase):

    def test_get_size_invalid(self):
        with pytest.raises(ValueError):
            internal.get_size(1.1)

    def test_infer_unknown_dimension_cnt_above_1(self):
        with pytest.raises(ValueError):
            internal.infer_unknown_dimension((-1, -2, 3), 27)

    def test_infer_unknown_dimension_reshape_failed(self):
        with pytest.raises(ValueError):
            internal.infer_unknown_dimension((-1, 3), 5)

    def test_normalize_order_not_understood(self):
        with pytest.raises(TypeError):
            nlcpy.ndarray(1, order='X')

    def test_complete_slice_list_multiple_ellipsis(self):
        with pytest.raises(ValueError):
            internal.complete_slice_list([Ellipsis, Ellipsis], 2)


class TestManipulation(unittest.TestCase):

    def test_shape_setter_incompatible(self):
        x = nlcpy.zeros((3, 3))
        with pytest.raises(AttributeError):
            x.shape = 3

    def test_ndarray_ravel_order_k(self):
        with pytest.raises(NotImplementedError):
            nlcpy.zeros((3, 3)).ravel(order='K')

    def test_resize_not_contiguous(self):
        x = nlcpy.empty((3, 3))[..., ::2]
        with pytest.raises(ValueError):
            manipulation._resize(x, x.shape, 1)

    def test_resize_newsize_negative(self):
        x = nlcpy.empty((3, 3))
        with pytest.raises(ValueError):
            manipulation._resize(x, (-1, 3), 1)

    def test_resize_refcheck_base_not_none(self):
        x = nlcpy.empty((3, 3)).view()
        with pytest.raises(ValueError):
            manipulation._resize(x, (3, 3, 3), 1)

    def test_resize_refcheck_refcnt_above_3(self):
        x0 = nlcpy.empty((3, 3))
        _ = x0  # refcnt: 4
        with pytest.raises(ValueError):
            # manipulation._resize(x0, (3, 3, 3), 1)
            x0.resize(3, 3, 3)

    def test_resize_newshape_0d(self):
        x = nlcpy.empty((3, 3))
        manipulation._resize(x, [], 0)
        assert x.shape == ()

    def test_transpose_axes_mismatch(self):
        with pytest.raises(ValueError):
            nlcpy.empty((3, 3)).transpose(1)

    def test_transpose_axis_out_of_bounds(self):
        with pytest.raises(nlcpy.AxisError):
            nlcpy.empty((3, 3)).transpose((2, 3))

    def test_transpose_axis_repeated(self):
        with pytest.raises(ValueError):
            nlcpy.empty((3, 3)).transpose((0, 0))

    def test_rollaxis_axis_start_negative(self):
        x = nlcpy.empty((3, 4, 5, 6))
        z = nlcpy.rollaxis(x, -2, -3)
        assert z.shape == (3, 5, 4, 6)

    def test_concatenate_op_len_0(self):
        ret = nlcpy.empty((2, 3))
        with pytest.raises(ValueError):
            manipulation._ndarray_concatenate([], 0, ret)

    def test_concatenate_axis_none_ret_not_1d_array(self):
        op = [nlcpy.empty(3) for _ in range(2)]
        ret = nlcpy.empty((2, 3))
        with pytest.raises(ValueError):
            manipulation._ndarray_concatenate(op, None, ret)

    def test_concatenate_axis_none_size_mismatch(self):
        op = [nlcpy.empty(3) for _ in range(2)]
        ret = nlcpy.empty(7)
        with pytest.raises(ValueError):
            manipulation._ndarray_concatenate(op, None, ret)


class TestScalar(unittest.TestCase):

    def test_conver_scalar_numpy_ndarray(self):
        x = numpy.array(1)
        y = nlcpy.core.scalar.convert_scalar(x)
        assert isinstance(y, nlcpy.ndarray)

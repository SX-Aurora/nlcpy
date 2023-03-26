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

import numpy
import unittest
import pytest

import nlcpy
from nlcpy import testing
from nlcpy.testing.types import float_types


TOL_SINGLE = 1e-6
TOL_DOUBLE = 1e-12

types = [
    'xyza',
    'xywa',
    'xzwa',
    'yzwa',
]

shapes = [
    (22, 21, 20),
    (20, 21, 22, 23),
]

stencil_scales_full = [0, 1, 2, 3, 4]
stencil_scales_small = [0, 1]


# --------------------------------------------------
# Individual routines for each stencil shapes
# --------------------------------------------------

def create_description_xyza(dxin, N, factor=None, coef=None, prefix=True):
    assert dxin.arr.ndim >= 3
    desc = nlcpy.sca.empty_description()
    loc1 = [(i, 0, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i, 0) for i in range(-N, N + 1)]
    loc3 = [(0, 0, i) for i in range(-N, N + 1)]
    for i, _l in enumerate(set(loc1 + loc2 + loc3)):
        if factor is None and coef is None:
            desc += dxin[..., _l[0], _l[1], _l[2]]
        elif factor is not None and coef is None:
            assert len(factor) == get_n_stencil_elem(N)
            if prefix:
                desc += factor[i] * dxin[..., _l[0], _l[1], _l[2]]
            else:
                desc += dxin[..., _l[0], _l[1], _l[2]] * factor[i]
        elif factor is None and coef is not None:
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * dxin[..., _l[0], _l[1], _l[2]]
            else:
                desc += dxin[..., _l[0], _l[1], _l[2]] * coef[i]
        else:
            assert len(factor) == get_n_stencil_elem(N)
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * (factor[i] * dxin[..., _l[0], _l[1], _l[2]])
            else:
                desc += (dxin[..., _l[0], _l[1], _l[2]] * factor[i]) * coef[i]
    return desc


def create_description_xywa(dxin, N, factor=None, coef=None, prefix=True):
    assert dxin.arr.ndim >= 4
    desc = nlcpy.sca.empty_description()
    loc1 = [(i, 0, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i, 0) for i in range(-N, N + 1)]
    loc3 = [(0, 0, i) for i in range(-N, N + 1)]
    for i, _l in enumerate(set(loc1 + loc2 + loc3)):
        if factor is None and coef is None:
            desc += dxin[_l[0], :, _l[1], _l[2]]
        elif factor is not None and coef is None:
            assert len(factor) == get_n_stencil_elem(N)
            if prefix:
                desc += factor[i] * dxin[_l[0], :, _l[1], _l[2]]
            else:
                desc += dxin[_l[0], :, _l[1], _l[2]] * factor[i]
        elif factor is None and coef is not None:
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * dxin[_l[0], :, _l[1], _l[2]]
            else:
                desc += dxin[_l[0], :, _l[1], _l[2]] * coef[i]
        else:
            assert len(factor) == get_n_stencil_elem(N)
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * (factor[i] * dxin[_l[0], :, _l[1], _l[2]])
            else:
                desc += (dxin[_l[0], :, _l[1], _l[2]] * factor[i]) * coef[i]
    return desc


def create_description_xzwa(dxin, N, factor=None, coef=None, prefix=True):
    assert dxin.arr.ndim >= 4
    desc = nlcpy.sca.empty_description()
    loc1 = [(i, 0, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i, 0) for i in range(-N, N + 1)]
    loc3 = [(0, 0, i) for i in range(-N, N + 1)]
    for i, _l in enumerate(set(loc1 + loc2 + loc3)):
        if factor is None and coef is None:
            desc += dxin[_l[0], _l[1], :, _l[2]]
        elif factor is not None and coef is None:
            assert len(factor) == get_n_stencil_elem(N)
            if prefix:
                desc += factor[i] * dxin[_l[0], _l[1], :, _l[2]]
            else:
                desc += dxin[_l[0], _l[1], :, _l[2]] * factor[i]
        elif factor is None and coef is not None:
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * dxin[_l[0], _l[1], :, _l[2]]
            else:
                desc += dxin[_l[0], _l[1], :, _l[2]] * coef[i]
        else:
            assert len(factor) == get_n_stencil_elem(N)
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * (factor[i] * dxin[_l[0], _l[1], :, _l[2]])
            else:
                desc += (dxin[_l[0], _l[1], :, _l[2]] * factor[i]) * coef[i]
    return desc


def create_description_yzwa(dxin, N, factor=None, coef=None, prefix=True):
    assert dxin.arr.ndim >= 4
    desc = nlcpy.sca.empty_description()
    loc1 = [(i, 0, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i, 0) for i in range(-N, N + 1)]
    loc3 = [(0, 0, i) for i in range(-N, N + 1)]
    for i, _l in enumerate(set(loc1 + loc2 + loc3)):
        if factor is None and coef is None:
            desc += dxin[_l[0], _l[1], _l[2], :]
        elif factor is not None and coef is None:
            assert len(factor) == get_n_stencil_elem(N)
            if prefix:
                desc += factor[i] * dxin[_l[0], _l[1], _l[2], :]
            else:
                desc += dxin[_l[0], _l[1], _l[2], :] * factor[i]
        elif factor is None and coef is not None:
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * dxin[_l[0], _l[1], _l[2], :]
            else:
                desc += dxin[_l[0], _l[1], _l[2], :] * coef[i]
        else:
            assert len(factor) == get_n_stencil_elem(N)
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * (factor[i] * dxin[_l[0], _l[1], _l[2], :])
            else:
                desc += (dxin[_l[0], _l[1], _l[2], :] * factor[i]) * coef[i]
    return desc


def compute_with_sca(type, xin, N, is_out=True, factor=None, coef=None, prefix=True,
                     optimize=False, change_coef=False):
    if optimize:
        xin = nlcpy.sca.convert_optimized_array(xin, xin.dtype)
    if is_out:
        xout = nlcpy.zeros_like(xin)
        if optimize:
            xout = nlcpy.sca.convert_optimized_array(xout, xout.dtype)
        dxin, dxout = nlcpy.sca.create_descriptor((xin, xout))
    else:
        dxin = nlcpy.sca.create_descriptor((xin))
        xout = None
    _coef = nlcpy.zeros_like(coef) if change_coef else coef
    if type == 'xyza':
        if xin.ndim < 3:
            return True, True
        desc = create_description_xyza(
            dxin, N, factor=factor, coef=_coef, prefix=prefix)
    elif type == 'xywa':
        if xin.ndim < 4:
            return True, True
        desc = create_description_xywa(
            dxin, N, factor=factor, coef=_coef, prefix=prefix)
    elif type == 'xzwa':
        if xin.ndim < 4:
            return True, True
        desc = create_description_xzwa(
            dxin, N, factor=factor, coef=_coef, prefix=prefix)
    elif type == 'yzwa':
        if xin.ndim < 4:
            return True, True
        desc = create_description_yzwa(
            dxin, N, factor=factor, coef=_coef, prefix=prefix)
    else:
        raise TypeError
    if is_out:
        kern = nlcpy.sca.create_kernel(desc, dxout[...])
    else:
        kern = nlcpy.sca.create_kernel(desc)
    if change_coef:
        _coef[...] = coef
    res = kern.execute()
    return res, xout


def naive_xyza(xin, xout, N, factor=None, coef=None):
    assert xin.ndim >= 3
    loc1 = [(i, 0, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i, 0) for i in range(-N, N + 1)]
    loc3 = [(0, 0, i) for i in range(-N, N + 1)]

    if N == 0:
        xout_v = xout[...]
    else:
        xout_v = xout[..., N:-N, N:-N, N:-N]
    for i, _l in enumerate(set(loc1 + loc2 + loc3)):
        if factor is None and coef is None:
            xout_v += xin[..., N + _l[0]:xin.shape[-3] - N + _l[0],
                          N + _l[1]:xin.shape[-2] - N + _l[1],
                          N + _l[2]:xin.shape[-1] - N + _l[2]]
        elif factor is not None and coef is None:
            xout_v += factor[i] * xin[..., N + _l[0]:xin.shape[-3] - N + _l[0],
                                      N + _l[1]:xin.shape[-2] - N + _l[1],
                                      N + _l[2]:xin.shape[-1] - N + _l[2]]
        elif factor is None and coef is not None:
            xout_v += coef[i] * xin[..., N + _l[0]:xin.shape[-3] - N + _l[0],
                                    N + _l[1]:xin.shape[-2] - N + _l[1],
                                    N + _l[2]:xin.shape[-1] - N + _l[2]]
        else:
            xout_v += factor[i] * coef[i] * \
                xin[..., N + _l[0]:xin.shape[-3] - N + _l[0],
                    N + _l[1]:xin.shape[-2] - N + _l[1],
                    N + _l[2]:xin.shape[-1] - N + _l[2]]


def naive_xywa(xin, xout, N, factor=None, coef=None):
    assert xin.ndim >= 4
    loc1 = [(i, 0, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i, 0) for i in range(-N, N + 1)]
    loc3 = [(0, 0, i) for i in range(-N, N + 1)]

    if N == 0:
        xout_v = xout[...]
    else:
        xout_v = xout[N:-N, :, N:-N, N:-N]
    for i, _l in enumerate(set(loc1 + loc2 + loc3)):
        if factor is None and coef is None:
            xout_v += xin[N + _l[0]:xin.shape[-4] - N + _l[0], :,
                          N + _l[1]:xin.shape[-2] - N + _l[1],
                          N + _l[2]:xin.shape[-1] - N + _l[2]]
        elif factor is not None and coef is None:
            xout_v += factor[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0], :,
                                      N + _l[1]:xin.shape[-2] - N + _l[1],
                                      N + _l[2]:xin.shape[-1] - N + _l[2]]
        elif factor is None and coef is not None:
            xout_v += coef[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0], :,
                                    N + _l[1]:xin.shape[-2] - N + _l[1],
                                    N + _l[2]:xin.shape[-1] - N + _l[2]]
        else:
            xout_v += factor[i] * coef[i] * \
                xin[N + _l[0]:xin.shape[-4] - N + _l[0], :,
                    N + _l[1]:xin.shape[-2] - N + _l[1],
                    N + _l[2]:xin.shape[-1] - N + _l[2]]


def naive_xzwa(xin, xout, N, factor=None, coef=None):
    assert xin.ndim >= 4
    loc1 = [(i, 0, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i, 0) for i in range(-N, N + 1)]
    loc3 = [(0, 0, i) for i in range(-N, N + 1)]

    if N == 0:
        xout_v = xout[...]
    else:
        xout_v = xout[N:-N, N:-N, :, N:-N]
    for i, _l in enumerate(set(loc1 + loc2 + loc3)):
        if factor is None and coef is None:
            xout_v += xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                          N + _l[1]:xin.shape[-3] - N + _l[1], :,
                          N + _l[2]:xin.shape[-1] - N + _l[2]]
        elif factor is not None and coef is None:
            xout_v += factor[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                                      N + _l[1]:xin.shape[-3] - N + _l[1], :,
                                      N + _l[2]:xin.shape[-1] - N + _l[2]]
        elif factor is None and coef is not None:
            xout_v += coef[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                                    N + _l[1]:xin.shape[-3] - N + _l[1], :,
                                    N + _l[2]:xin.shape[-1] - N + _l[2]]
        else:
            xout_v += factor[i] * coef[i] * \
                xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                    N + _l[1]:xin.shape[-3] - N + _l[1], :,
                    N + _l[2]:xin.shape[-1] - N + _l[2]]


def naive_yzwa(xin, xout, N, factor=None, coef=None):
    assert xin.ndim >= 4
    loc1 = [(i, 0, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i, 0) for i in range(-N, N + 1)]
    loc3 = [(0, 0, i) for i in range(-N, N + 1)]

    if N == 0:
        xout_v = xout[...]
    else:
        xout_v = xout[N:-N, N:-N, N:-N, :]
    for i, _l in enumerate(set(loc1 + loc2 + loc3)):
        if factor is None and coef is None:
            xout_v += xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                          N + _l[1]:xin.shape[-3] - N + _l[1],
                          N + _l[2]:xin.shape[-2] - N + _l[2], :]
        elif factor is not None and coef is None:
            xout_v += factor[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                                      N + _l[1]:xin.shape[-3] - N + _l[1],
                                      N + _l[2]:xin.shape[-2] - N + _l[2], :]
        elif factor is None and coef is not None:
            xout_v += coef[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                                    N + _l[1]:xin.shape[-3] - N + _l[1],
                                    N + _l[2]:xin.shape[-2] - N + _l[2], :]
        else:
            xout_v += factor[i] * coef[i] * \
                xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                    N + _l[1]:xin.shape[-3] - N + _l[1],
                    N + _l[2]:xin.shape[-2] - N + _l[2], :]


def compute_with_naive(type, xin, N, factor=None, coef=None):
    xout = nlcpy.zeros_like(xin)
    if type == 'xyza':
        if xin.ndim < 3:
            return True
        naive_xyza(xin, xout, N, factor=factor, coef=coef)
    elif type == 'xywa':
        if xin.ndim < 4:
            return True
        naive_xywa(xin, xout, N, factor=factor, coef=coef)
    elif type == 'xzwa':
        if xin.ndim < 4:
            return True
        naive_xzwa(xin, xout, N, factor=factor, coef=coef)
    elif type == 'yzwa':
        if xin.ndim < 4:
            return True
        naive_yzwa(xin, xout, N, factor=factor, coef=coef)
    else:
        raise TypeError
    return xout


def get_n_stencil_elem(N):
    return 6 * N + 1


def get_axis_numbers_from_strtype(t):
    if t == 'xyza':
        return [-1, -2, -3]
    elif t == 'xywa':
        return [-1, -2, -4]
    elif t == 'xzwa':
        return [-1, -3, -4]
    elif t == 'yzwa':
        return [-2, -3, -4]
    else:
        raise TypeError


# --------------------------------------------------
# Testing routines for full tests
# --------------------------------------------------

@testing.parameterize(*(
    testing.product({
        'type': types,
        'shape': shapes,
        'dtype': float_types,
        'stencil_scale': stencil_scales_full,
        'is_out': [True, False],
        'optimize': [True, False],
    })
))
@pytest.mark.full
class Test3dAxialFull(unittest.TestCase):

    def test_3d_axial(self):
        nlcpy.random.seed(0)
        xin = testing.shaped_random(self.shape, nlcpy).astype(self.dtype)
        rtol = TOL_SINGLE if self.dtype == numpy.float32 else TOL_DOUBLE
        sca_res, sca_out = compute_with_sca(
            self.type,
            xin,
            self.stencil_scale,
            is_out=self.is_out,
            optimize=self.optimize,
        )
        naive_res = compute_with_naive(self.type, xin, self.stencil_scale)
        if self.is_out:
            assert id(sca_res) == id(sca_out)
        testing.assert_allclose(sca_res, naive_res, rtol=rtol)


@testing.parameterize(*(
    testing.product({
        'type': types,
        'shape': shapes,
        'dtype': float_types,
        'stencil_scale': stencil_scales_full,
        'is_out': [True, False],
        'prefix': [True, False],
        'optimize': [True, False],
    })
))
@pytest.mark.full
class Test3dAxialFactorFull(unittest.TestCase):

    def test_3d_axial_with_factor(self):
        xin = testing.shaped_arange(self.shape, nlcpy).astype(self.dtype) * 0.1
        n_elem = get_n_stencil_elem(self.stencil_scale)
        factor = (nlcpy.arange(n_elem) * 0.1).tolist()
        rtol = TOL_SINGLE if self.dtype == numpy.float32 else TOL_DOUBLE
        sca_res, sca_out = compute_with_sca(
            self.type,
            xin,
            self.stencil_scale,
            factor=factor,
            is_out=self.is_out,
            prefix=self.prefix,
            optimize=self.optimize
        )
        naive_res = compute_with_naive(self.type, xin, self.stencil_scale, factor=factor)
        if self.is_out:
            assert id(sca_res) == id(sca_out)
        testing.assert_allclose(sca_res, naive_res, rtol=rtol)


@testing.parameterize(*(
    testing.product({
        'type': types,
        'shape': shapes,
        'dtype': float_types,
        'stencil_scale': stencil_scales_full,
        'is_out': [True, False],
        'prefix': [True, False],
        'optimize': [True, False],
        'change_coef': [True, False],
        'coef_array': [True, False],
    })
))
@pytest.mark.full
class Test3dAxialCoefFull(unittest.TestCase):

    def test_3d_axial_with_coef(self):
        xin = testing.shaped_arange(self.shape, nlcpy).astype(self.dtype) * 0.1
        n_elem = get_n_stencil_elem(self.stencil_scale)
        if self.coef_array:
            coef_shape = list(xin.shape)
            for an in get_axis_numbers_from_strtype(self.type):
                if abs(an) <= xin.ndim:
                    coef_shape[an] -= 2 * self.stencil_scale
            coef = testing.shaped_arange(
                [n_elem, ] + coef_shape, nlcpy, dtype=self.dtype
            ) * 0.01
        else:
            coef = nlcpy.arange(n_elem) * 0.1
            coef = coef.astype(dtype=self.dtype)
        rtol = TOL_SINGLE if self.dtype == numpy.float32 else TOL_DOUBLE
        sca_res, sca_out = compute_with_sca(
            self.type,
            xin,
            self.stencil_scale,
            coef=coef,
            is_out=self.is_out,
            prefix=self.prefix,
            optimize=self.optimize,
            change_coef=self.change_coef
        )
        naive_res = compute_with_naive(self.type, xin, self.stencil_scale, coef=coef)
        if self.is_out:
            assert id(sca_res) == id(sca_out)
        testing.assert_allclose(sca_res, naive_res, rtol=rtol)


@testing.parameterize(*(
    testing.product({
        'type': types,
        'shape': shapes,
        'dtype': float_types,
        'stencil_scale': stencil_scales_full,
        'is_out': [True, False],
        'prefix': [True, False],
        'optimize': [True, False],
        'change_coef': [True, False],
        'coef_array': [True, False],
    })
))
@pytest.mark.full
class Test3dAxialFactorCoefFull(unittest.TestCase):

    def test_3d_axial_with_factor_and_coef(self):
        xin = testing.shaped_arange(self.shape, nlcpy).astype(self.dtype) * 0.1
        n_elem = get_n_stencil_elem(self.stencil_scale)
        nlcpy.random.seed(0)
        # to avoid loss of digits, create from arenge
        factor = ((nlcpy.arange(n_elem) - 0.5) * 10).tolist()
        if self.coef_array:
            coef_shape = list(xin.shape)
            for an in get_axis_numbers_from_strtype(self.type):
                if abs(an) <= xin.ndim:
                    coef_shape[an] -= 2 * self.stencil_scale
            coef = testing.shaped_arange(
                [n_elem, ] + coef_shape, nlcpy, dtype=self.dtype
            ) * 0.01
        else:
            coef = (nlcpy.arange(n_elem) - 0.5) * 10
            coef = coef.astype(dtype=self.dtype)
        rtol = TOL_SINGLE if self.dtype == numpy.float32 else TOL_DOUBLE
        sca_res, sca_out = compute_with_sca(
            self.type,
            xin,
            self.stencil_scale,
            factor=factor,
            coef=coef,
            is_out=self.is_out,
            prefix=self.prefix,
            optimize=self.optimize,
            change_coef=self.change_coef
        )
        naive_res = compute_with_naive(
            self.type,
            xin,
            self.stencil_scale,
            factor=factor,
            coef=coef
        )

        if self.is_out:
            assert id(sca_res) == id(sca_out)
        testing.assert_allclose(sca_res, naive_res, rtol=rtol)


# --------------------------------------------------
# Testing routines for small tests
# --------------------------------------------------

@testing.parameterize(*(
    testing.product({
        'type': types,
        'shape': shapes,
        'dtype': float_types,
        'stencil_scale': stencil_scales_small,
        'is_out': [True, False],
        'optimize': [True, False],
    })
))
@pytest.mark.small
class Test3dAxialSmall(unittest.TestCase):

    def test_3d_axial(self):
        nlcpy.random.seed(0)
        xin = testing.shaped_random(self.shape, nlcpy).astype(self.dtype)
        rtol = TOL_SINGLE if self.dtype == numpy.float32 else TOL_DOUBLE
        sca_res, sca_out = compute_with_sca(
            self.type,
            xin,
            self.stencil_scale,
            is_out=self.is_out,
            optimize=self.optimize,
        )
        naive_res = compute_with_naive(self.type, xin, self.stencil_scale)
        if self.is_out:
            assert id(sca_res) == id(sca_out)
        testing.assert_allclose(sca_res, naive_res, rtol=rtol)


@testing.parameterize(*(
    testing.product({
        'type': types,
        'shape': shapes,
        'dtype': float_types,
        'stencil_scale': stencil_scales_small,
        'is_out': [True, False],
        'prefix': [True, False],
        'optimize': [True, False],
    })
))
@pytest.mark.small
class Test3dAxialFactorSmall(unittest.TestCase):

    def test_3d_axial_with_factor(self):
        xin = testing.shaped_arange(self.shape, nlcpy).astype(self.dtype) * 0.1
        n_elem = get_n_stencil_elem(self.stencil_scale)
        factor = (nlcpy.arange(n_elem) * 0.1).tolist()
        rtol = TOL_SINGLE if self.dtype == numpy.float32 else TOL_DOUBLE
        sca_res, sca_out = compute_with_sca(
            self.type,
            xin,
            self.stencil_scale,
            factor=factor,
            is_out=self.is_out,
            prefix=self.prefix,
            optimize=self.optimize
        )
        naive_res = compute_with_naive(self.type, xin, self.stencil_scale, factor=factor)
        if self.is_out:
            assert id(sca_res) == id(sca_out)
        testing.assert_allclose(sca_res, naive_res, rtol=rtol)


@testing.parameterize(*(
    testing.product({
        'type': types,
        'shape': shapes,
        'dtype': float_types,
        'stencil_scale': stencil_scales_small,
        'is_out': [True, False],
        'prefix': [True, False],
        'optimize': [True, False],
        'change_coef': [True, False],
        'coef_array': [True, False],
    })
))
@pytest.mark.small
class Test3dAxialCoefSmall(unittest.TestCase):

    def test_3d_axial_with_coef(self):
        xin = testing.shaped_arange(self.shape, nlcpy).astype(self.dtype) * 0.1
        n_elem = get_n_stencil_elem(self.stencil_scale)
        if self.coef_array:
            coef_shape = list(xin.shape)
            for an in get_axis_numbers_from_strtype(self.type):
                if abs(an) <= xin.ndim:
                    coef_shape[an] -= 2 * self.stencil_scale
            coef = testing.shaped_arange(
                [n_elem, ] + coef_shape, nlcpy, dtype=self.dtype
            ) * 0.01
        else:
            coef = nlcpy.arange(n_elem) * 0.1
            coef = coef.astype(dtype=self.dtype)
        rtol = TOL_SINGLE if self.dtype == numpy.float32 else TOL_DOUBLE
        sca_res, sca_out = compute_with_sca(
            self.type,
            xin,
            self.stencil_scale,
            coef=coef,
            is_out=self.is_out,
            prefix=self.prefix,
            optimize=self.optimize,
            change_coef=self.change_coef
        )
        naive_res = compute_with_naive(self.type, xin, self.stencil_scale, coef=coef)
        if self.is_out:
            assert id(sca_res) == id(sca_out)
        testing.assert_allclose(sca_res, naive_res, rtol=rtol)


@testing.parameterize(*(
    testing.product({
        'type': types,
        'shape': shapes,
        'dtype': float_types,
        'stencil_scale': stencil_scales_small,
        'is_out': [True, False],
        'prefix': [True, False],
        'optimize': [True, False],
        'change_coef': [True, False],
        'coef_array': [True, False],
    })
))
@pytest.mark.small
class Test3dAxialFactorCoefSmall(unittest.TestCase):

    def test_3d_axial_with_factor_and_coef(self):
        xin = testing.shaped_arange(self.shape, nlcpy).astype(self.dtype) * 0.1
        n_elem = get_n_stencil_elem(self.stencil_scale)
        nlcpy.random.seed(0)
        # to avoid loss of digits, create from arenge
        factor = ((nlcpy.arange(n_elem) - 0.5) * 10).tolist()
        if self.coef_array:
            coef_shape = list(xin.shape)
            for an in get_axis_numbers_from_strtype(self.type):
                if abs(an) <= xin.ndim:
                    coef_shape[an] -= 2 * self.stencil_scale
            coef = testing.shaped_arange(
                [n_elem, ] + coef_shape, nlcpy, dtype=self.dtype
            ) * 0.01
        else:
            coef = (nlcpy.arange(n_elem) - 0.5) * 10
            coef = coef.astype(dtype=self.dtype)
        rtol = TOL_SINGLE if self.dtype == numpy.float32 else TOL_DOUBLE
        sca_res, sca_out = compute_with_sca(
            self.type,
            xin,
            self.stencil_scale,
            factor=factor,
            coef=coef,
            is_out=self.is_out,
            prefix=self.prefix,
            optimize=self.optimize,
            change_coef=self.change_coef
        )
        naive_res = compute_with_naive(
            self.type,
            xin,
            self.stencil_scale,
            factor=factor,
            coef=coef
        )

        if self.is_out:
            assert id(sca_res) == id(sca_out)
        testing.assert_allclose(sca_res, naive_res, rtol=rtol)

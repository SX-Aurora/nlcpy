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


f64_type = [numpy.float64, ]
float_types = [numpy.float32, numpy.float64]
complex_types = [numpy.complex64, numpy.complex128]
signed_int_types = [numpy.int32, numpy.int64]
unsigned_int_types = [numpy.uint32, numpy.uint64]
int_types = signed_int_types + unsigned_int_types
no_bool_types = float_types + int_types + complex_types
no_bool_no_uint_types = float_types + signed_int_types + complex_types
all_types = [numpy.bool] + float_types + int_types + complex_types
negative_types = (
    [numpy.bool] + float_types + signed_int_types + complex_types)
negative_no_complex_types = [numpy.bool] + float_types + signed_int_types
no_complex_types = [numpy.bool] + float_types + int_types
no_bool_no_complex_types = float_types + int_types


TOL_SINGLE = 1e-6
TOL_DOUBLE = 1e-12

types = [
    'xya',
    'xza',
    'xwa',
    'yza',
    'ywa',
    'zwa',
]

shapes = [
    (21, 24),
    (22, 21, 20),
    (20, 21, 22, 23),
]

stencil_scales_full = [0, 1, 2, 3, 4]
stencil_scales_small = [0, 1]


# --------------------------------------------------
# Individual routines for each stencil shapes
# --------------------------------------------------

def create_description_xya(dxin, N, factor=None, coef=None, prefix=True):
    assert dxin.arr.ndim >= 2
    desc = nlcpy.sca.empty_description()
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            desc += dxin[..., _l[0], _l[1]]
        elif factor is not None and coef is None:
            assert len(factor) == get_n_stencil_elem(N)
            if prefix:
                desc += factor[i] * dxin[..., _l[0], _l[1]]
            else:
                desc += dxin[..., _l[0], _l[1]] * factor[i]
        elif factor is None and coef is not None:
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * dxin[..., _l[0], _l[1]]
            else:
                desc += dxin[..., _l[0], _l[1]] * coef[i]
        else:
            assert len(factor) == get_n_stencil_elem(N)
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * (factor[i] * dxin[..., _l[0], _l[1]])
            else:
                desc += (dxin[..., _l[0], _l[1]] * factor[i]) * coef[i]
    return desc


def create_description_xza(dxin, N, factor=None, coef=None, prefix=True):
    assert dxin.arr.ndim >= 3
    desc = nlcpy.sca.empty_description()
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            desc += dxin[..., _l[0], :, _l[1]]
        elif factor is not None and coef is None:
            assert len(factor) == get_n_stencil_elem(N)
            if prefix:
                desc += factor[i] * dxin[..., _l[0], :, _l[1]]
            else:
                desc += dxin[..., _l[0], :, _l[1]] * factor[i]
        elif factor is None and coef is not None:
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * dxin[..., _l[0], :, _l[1]]
            else:
                desc += dxin[..., _l[0], :, _l[1]] * coef[i]
        else:
            assert len(factor) == get_n_stencil_elem(N)
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * (factor[i] * dxin[..., _l[0], :, _l[1]])
            else:
                desc += (dxin[..., _l[0], :, _l[1]] * factor[i]) * coef[i]
    return desc


def create_description_xwa(dxin, N, factor=None, coef=None, prefix=True):
    assert dxin.arr.ndim >= 4
    desc = nlcpy.sca.empty_description()
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            desc += dxin[_l[0], ..., _l[1]]
        elif factor is not None and coef is None:
            assert len(factor) == get_n_stencil_elem(N)
            if prefix:
                desc += factor[i] * dxin[_l[0], ..., _l[1]]
            else:
                desc += dxin[_l[0], ..., _l[1]] * factor[i]
        elif factor is None and coef is not None:
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * dxin[_l[0], ..., _l[1]]
            else:
                desc += dxin[_l[0], ..., _l[1]] * coef[i]
        else:
            assert len(factor) == get_n_stencil_elem(N)
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * (factor[i] * dxin[_l[0], ..., _l[1]])
            else:
                desc += (dxin[_l[0], ..., _l[1]] * factor[i]) * coef[i]
    return desc


def create_description_yza(dxin, N, factor=None, coef=None, prefix=True):
    assert dxin.arr.ndim >= 3
    desc = nlcpy.sca.empty_description()
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            desc += dxin[..., _l[0], _l[1], :]
        elif factor is not None and coef is None:
            assert len(factor) == get_n_stencil_elem(N)
            if prefix:
                desc += factor[i] * dxin[..., _l[0], _l[1], :]
            else:
                desc += dxin[..., _l[0], _l[1], :] * factor[i]
        elif factor is None and coef is not None:
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * dxin[..., _l[0], _l[1], :]
            else:
                desc += dxin[..., _l[0], _l[1], :] * coef[i]
        else:
            assert len(factor) == get_n_stencil_elem(N)
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * (factor[i] * dxin[..., _l[0], _l[1], :])
            else:
                desc += (dxin[..., _l[0], _l[1], :] * factor[i]) * coef[i]
    return desc


def create_description_ywa(dxin, N, factor=None, coef=None, prefix=True):
    assert dxin.arr.ndim >= 4
    desc = nlcpy.sca.empty_description()
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            desc += dxin[_l[0], :, _l[1], :]
        elif factor is not None and coef is None:
            assert len(factor) == get_n_stencil_elem(N)
            if prefix:
                desc += factor[i] * dxin[_l[0], :, _l[1], :]
            else:
                desc += dxin[_l[0], :, _l[1], :] * factor[i]
        elif factor is None and coef is not None:
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * dxin[_l[0], :, _l[1], :]
            else:
                desc += dxin[_l[0], :, _l[1], :] * coef[i]
        else:
            assert len(factor) == get_n_stencil_elem(N)
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * (factor[i] * dxin[_l[0], :, _l[1], :])
            else:
                desc += (dxin[_l[0], :, _l[1], :] * factor[i]) * coef[i]
    return desc


def create_description_zwa(dxin, N, factor=None, coef=None, prefix=True):
    assert dxin.arr.ndim >= 4
    desc = nlcpy.sca.empty_description()
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            desc += dxin[_l[0], _l[1], ...]
        elif factor is not None and coef is None:
            assert len(factor) == get_n_stencil_elem(N)
            if prefix:
                desc += factor[i] * dxin[_l[0], _l[1], ...]
            else:
                desc += dxin[_l[0], _l[1], ...] * factor[i]
        elif factor is None and coef is not None:
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * dxin[_l[0], _l[1], ...]
            else:
                desc += dxin[_l[0], _l[1], ...] * coef[i]
        else:
            assert len(factor) == get_n_stencil_elem(N)
            # assert coef.size == get_n_stencil_elem(N)
            if prefix:
                desc += coef[i] * (factor[i] * dxin[_l[0], _l[1], ...])
            else:
                desc += (dxin[_l[0], _l[1], ...] * factor[i]) * coef[i]
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
    if type == 'xya':
        if xin.ndim < 2:
            return True, True
        desc = create_description_xya(
            dxin, N, factor=factor, coef=_coef, prefix=prefix)
    elif type == 'xza':
        if xin.ndim < 3:
            return True, True
        desc = create_description_xza(
            dxin, N, factor=factor, coef=_coef, prefix=prefix)
    elif type == 'xwa':
        if xin.ndim < 4:
            return True, True
        desc = create_description_xwa(
            dxin, N, factor=factor, coef=_coef, prefix=prefix)
    elif type == 'yza':
        if xin.ndim < 3:
            return True, True
        desc = create_description_yza(
            dxin, N, factor=factor, coef=_coef, prefix=prefix)
    elif type == 'ywa':
        if xin.ndim < 4:
            return True, True
        desc = create_description_ywa(
            dxin, N, factor=factor, coef=_coef, prefix=prefix)
    elif type == 'zwa':
        if xin.ndim < 4:
            return True, True
        desc = create_description_zwa(
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


def naive_xya(xin, xout, N, factor=None, coef=None):
    assert xin.ndim >= 2
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]

    if N == 0:
        xout_v = xout[...]
    else:
        xout_v = xout[..., N:-N, N:-N]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            xout_v += xin[..., N + _l[0]:xin.shape[-2] - N + _l[0],
                          N + _l[1]:xin.shape[-1] - N + _l[1]]
        elif factor is not None and coef is None:
            xout_v += factor[i] * xin[..., N + _l[0]:xin.shape[-2] - N + _l[0],
                                      N + _l[1]:xin.shape[-1] - N + _l[1]]
        elif factor is None and coef is not None:
            xout_v += coef[i] * xin[..., N + _l[0]:xin.shape[-2] - N + _l[0],
                                    N + _l[1]:xin.shape[-1] - N + _l[1]]
        else:
            xout_v += factor[i] * coef[i] * \
                xin[..., N + _l[0]:xin.shape[-2] - N + _l[0],
                    N + _l[1]:xin.shape[-1] - N + _l[1]]


def naive_xza(xin, xout, N, factor=None, coef=None):
    assert xin.ndim >= 3
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]

    if N == 0:
        xout_v = xout[...]
    else:
        xout_v = xout[..., N:-N, :, N:-N]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            xout_v += xin[..., N + _l[0]:xin.shape[-3] - N + _l[0], :,
                          N + _l[1]:xin.shape[-1] - N + _l[1]]
        elif factor is not None and coef is None:
            xout_v += factor[i] * xin[..., N + _l[0]:xin.shape[-3] - N + _l[0], :,
                                      N + _l[1]:xin.shape[-1] - N + _l[1]]
        elif factor is None and coef is not None:
            xout_v += coef[i] * xin[..., N + _l[0]:xin.shape[-3] - N + _l[0], :,
                                    N + _l[1]:xin.shape[-1] - N + _l[1]]
        else:
            xout_v += factor[i] * coef[i] * \
                xin[..., N + _l[0]:xin.shape[-3] - N + _l[0], :,
                    N + _l[1]:xin.shape[-1] - N + _l[1]]


def naive_xwa(xin, xout, N, factor=None, coef=None):
    assert xin.ndim >= 4
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]

    if N == 0:
        xout_v = xout[...]
    else:
        xout_v = xout[N:-N, ..., N:-N]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            xout_v += xin[N + _l[0]:xin.shape[-4] - N + _l[0], ...,
                          N + _l[1]:xin.shape[-1] - N + _l[1]]
        elif factor is not None and coef is None:
            xout_v += factor[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0], ...,
                                      N + _l[1]:xin.shape[-1] - N + _l[1]]
        elif factor is None and coef is not None:
            xout_v += coef[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0], ...,
                                    N + _l[1]:xin.shape[-1] - N + _l[1]]
        else:
            xout_v += factor[i] * coef[i] * \
                xin[N + _l[0]:xin.shape[-4] - N + _l[0], ...,
                    N + _l[1]:xin.shape[-1] - N + _l[1]]


def naive_yza(xin, xout, N, factor=None, coef=None):
    assert xin.ndim >= 3
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]

    if N == 0:
        xout_v = xout[...]
    else:
        xout_v = xout[..., N:-N, N:-N, :]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            xout_v += xin[..., N + _l[0]:xin.shape[-3] - N + _l[0],
                          N + _l[1]:xin.shape[-2] - N + _l[1], :]
        elif factor is not None and coef is None:
            xout_v += factor[i] * xin[..., N + _l[0]:xin.shape[-3] - N + _l[0],
                                      N + _l[1]:xin.shape[-2] - N + _l[1], :]
        elif factor is None and coef is not None:
            xout_v += coef[i] * xin[..., N + _l[0]:xin.shape[-3] - N + _l[0],
                                    N + _l[1]:xin.shape[-2] - N + _l[1], :]
        else:
            xout_v += factor[i] * coef[i] * \
                xin[..., N + _l[0]:xin.shape[-3] - N + _l[0],
                    N + _l[1]:xin.shape[-2] - N + _l[1], :]


def naive_ywa(xin, xout, N, factor=None, coef=None):
    assert xin.ndim >= 4
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]

    if N == 0:
        xout_v = xout[...]
    else:
        xout_v = xout[N:-N, :, N:-N, :]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            xout_v += xin[N + _l[0]:xin.shape[-4] - N + _l[0], :,
                          N + _l[1]:xin.shape[-2] - N + _l[1], :]
        elif factor is not None and coef is None:
            xout_v += factor[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0], :,
                                      N + _l[1]:xin.shape[-2] - N + _l[1], :]
        elif factor is None and coef is not None:
            xout_v += coef[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0], :,
                                    N + _l[1]:xin.shape[-2] - N + _l[1], :]
        else:
            xout_v += factor[i] * coef[i] * \
                xin[N + _l[0]:xin.shape[-4] - N + _l[0], :,
                    N + _l[1]:xin.shape[-2] - N + _l[1], :]


def naive_zwa(xin, xout, N, factor=None, coef=None):
    assert xin.ndim >= 4
    loc1 = [(i, 0) for i in range(-N, N + 1)]
    loc2 = [(0, i) for i in range(-N, N + 1)]

    if N == 0:
        xout_v = xout[...]
    else:
        xout_v = xout[N:-N, N:-N, ...]
    for i, _l in enumerate(set(loc1 + loc2)):
        if factor is None and coef is None:
            xout_v += xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                          N + _l[1]:xin.shape[-3] - N + _l[1], ...]
        elif factor is not None and coef is None:
            xout_v += factor[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                                      N + _l[1]:xin.shape[-3] - N + _l[1], ...]
        elif factor is None and coef is not None:
            xout_v += coef[i] * xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                                    N + _l[1]:xin.shape[-3] - N + _l[1], ...]
        else:
            xout_v += factor[i] * coef[i] * \
                xin[N + _l[0]:xin.shape[-4] - N + _l[0],
                    N + _l[1]:xin.shape[-3] - N + _l[1], ...]


def compute_with_naive(type, xin, N, factor=None, coef=None):
    xout = nlcpy.zeros_like(xin)
    if type == 'xya':
        if xin.ndim < 2:
            return True
        naive_xya(xin, xout, N, factor=factor, coef=coef)
    elif type == 'xza':
        if xin.ndim < 3:
            return True
        naive_xza(xin, xout, N, factor=factor, coef=coef)
    elif type == 'xwa':
        if xin.ndim < 4:
            return True
        naive_xwa(xin, xout, N, factor=factor, coef=coef)
    elif type == 'yza':
        if xin.ndim < 3:
            return True
        naive_yza(xin, xout, N, factor=factor, coef=coef)
    elif type == 'ywa':
        if xin.ndim < 4:
            return True
        naive_ywa(xin, xout, N, factor=factor, coef=coef)
    elif type == 'zwa':
        if xin.ndim < 4:
            return True
        naive_zwa(xin, xout, N, factor=factor, coef=coef)
    else:
        raise TypeError
    return xout


def get_n_stencil_elem(N):
    return 4 * N + 1


def get_axis_numbers_from_strtype(t):
    if t == 'xya':
        return [-1, -2]
    elif t == 'xza':
        return [-1, -3]
    elif t == 'xwa':
        return [-1, -4]
    elif t == 'yza':
        return [-2, -3]
    elif t == 'ywa':
        return [-2, -4]
    elif t == 'zwa':
        return [-3, -4]
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
class Test2dAxialFull(unittest.TestCase):

    def test_2d_axial(self):
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
class Test2dAxialFactorFull(unittest.TestCase):

    def test_2d_axial_with_factor(self):
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
class Test2dAxialCoefFull(unittest.TestCase):

    def test_2d_axial_with_coef(self):
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
class Test2dAxialFactorCoefFull(unittest.TestCase):

    def test_2d_axial_with_factor_and_coef(self):
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
class Test2dAxialSmall(unittest.TestCase):

    def test_2d_axial(self):
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
class Test2dAxialFactorSmall(unittest.TestCase):

    def test_2d_axial_with_factor(self):
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
class Test2dAxialCoefSmall(unittest.TestCase):

    def test_2d_axial_with_coef(self):
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
class Test2dAxialFactorCoefSmall(unittest.TestCase):

    def test_2d_axial_with_factor_and_coef(self):
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

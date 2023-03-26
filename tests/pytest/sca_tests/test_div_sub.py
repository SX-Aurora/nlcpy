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

import nlcpy
from nlcpy import testing
from nlcpy.testing.types import float_types


TOL_SINGLE = 1e-6
TOL_DOUBLE = 1e-12

types = [
    'xa',
]

shapes = [
    (21, ),
    (21, 24),
    (22, 21, 20),
    (20, 21, 22, 23),
]

stencil_scales = [0, 1, 2, 3, 4]


# --------------------------------------------------
# Individual routines for each stencil shapes
# --------------------------------------------------

def create_description_xa(dxin, N, factor=None, coef=None, prefix=True,
                          div=False, sub=False):
    assert not (prefix is True and div is True)
    desc = nlcpy.sca.empty_description()
    for i in range(-N, N + 1):
        if factor is not None and coef is not None:
            assert len(factor) == get_n_stencil_elem(N)
            assert coef.size == get_n_stencil_elem(N)
            if prefix:
                if sub:
                    desc -= coef[i + N] * (factor[i + N] * dxin[..., i])
                else:
                    desc += coef[i + N] * (factor[i + N] * dxin[..., i])
            else:
                if sub and div:
                    desc -= (dxin[..., i] / factor[i + N]) / coef[i + N]
                elif sub and not div:
                    desc -= (dxin[..., i] * factor[i + N]) * coef[i + N]
                elif not sub and div:
                    desc += (dxin[..., i] / factor[i + N]) / coef[i + N]
                else:
                    desc += (dxin[..., i] * factor[i + N]) * coef[i + N]
        else:
            raise RuntimeError
    return desc


def compute_with_sca(type, xin, N, is_out=True, factor=None, coef=None, prefix=True,
                     optimize=False, change_coef=False, div=False, sub=False):
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
    _coef = nlcpy.ones_like(coef) if change_coef else coef
    if type == 'xa':
        desc = create_description_xa(
            dxin, N, factor=factor, coef=_coef, prefix=prefix, div=div, sub=sub)
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


def naive_xa(xin, xout, N, factor=None, coef=None, div=False, sub=False):
    loc = [i for i in range(-N, N + 1)]

    if N == 0:
        xout_v = xout[...]
    else:
        xout_v = xout[..., N:-N]
    for i, _l in enumerate(loc):
        if factor is not None and coef is not None:
            if sub and div:
                xout_v -= xin[..., N + _l:xin.shape[-1] - N + _l] / factor[i] / coef[i]
            elif sub and not div:
                xout_v -= xin[..., N + _l:xin.shape[-1] - N + _l] * factor[i] * coef[i]
            elif not sub and div:
                xout_v += xin[..., N + _l:xin.shape[-1] - N + _l] / factor[i] / coef[i]
            else:
                xout_v += xin[..., N + _l:xin.shape[-1] - N + _l] * factor[i] * coef[i]
        else:
            raise RuntimeError


def compute_with_naive(type, xin, N, factor=None, coef=None, div=False, sub=False):
    xout = nlcpy.zeros_like(xin)
    if type == 'xa':
        naive_xa(xin, xout, N, factor=factor, coef=coef, div=div, sub=sub)
    else:
        raise TypeError
    return xout


def get_n_stencil_elem(N):
    return 2 * N + 1


# --------------------------------------------------
# Testing routines
# --------------------------------------------------

@testing.parameterize(*(
    testing.product({
        'type': types,
        'shape': shapes,
        'dtype': float_types,
        'stencil_scale': stencil_scales,
        'is_out': [True, False],
        'prefix': [True, False],
        'optimize': [True, False],
        'change_coef': [True, False],
        'div': [True, False],
        'sub': [True, False],
    })
))
class Test1dAxialFactorCoefDivSub(unittest.TestCase):

    def test_1d_axial_with_factor_and_coef_div_sub(self):
        if self.prefix and self.div:
            return True
        if self.change_coef and self.div:
            return True
        xin = testing.shaped_arange(self.shape, nlcpy).astype(self.dtype) * 0.1
        n_elem = get_n_stencil_elem(self.stencil_scale)
        nlcpy.random.seed(0)
        factor = ((nlcpy.random.rand(n_elem) - 0.5) * 10).tolist()
        coef = (nlcpy.random.rand(n_elem) - 0.5) * 10
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
            change_coef=self.change_coef,
            div=self.div,
            sub=self.sub
        )
        naive_res = compute_with_naive(
            self.type,
            xin,
            self.stencil_scale,
            factor=factor,
            coef=coef,
            div=self.div,
            sub=self.sub
        )

        if self.is_out:
            assert id(sca_res) == id(sca_out)
        testing.assert_allclose(sca_res, naive_res, rtol=rtol)

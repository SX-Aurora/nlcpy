from __future__ import division, absolute_import, print_function

import numpy
import nlcpy as np
import pytest
from nlcpy.random import random
from nlcpy.testing import (  # NOQA
    assert_array_equal, assert_allclose)
from numpy.testing import assert_raises
# import threading
# import sys
# if sys.version_info[0] >= 3:
#    import queue
# else:
#    import Queue as queue


def fft1(x):
    L = len(x)
    phase = -2j * np.pi * (np.arange(L) / float(L))
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(x * np.exp(phase), axis=1)


class TestFFTShift(object):

    def test_fft_n(self):
        assert_raises(ValueError, np.fft.fft, [1, 2, 3], 0)


class TestFFT1D(object):

    # TODO
    def test_identity(self):
        maxlen = 512
        x = random(maxlen) + 1j * random(maxlen)
        # xr = random(maxlen)  # local variable 'xr' is assigned to but never used
        for i in range(1, maxlen):
            assert_allclose(np.fft.ifft(np.fft.fft(x[0:i])), x[0:i],
                            atol=1e-12)
            # assert_allclose(np.fft.irfft(np.fft.rfft(xr[0:i]),i),
            #                xr[0:i], atol=1e-12)

    def test_fft(self):
        x = random(30) + 1j * random(30)
        assert_allclose(fft1(x), np.fft.fft(x), atol=1e-6)
        assert_allclose(fft1(x) / np.sqrt(30),
                        np.fft.fft(x, norm="ortho"), atol=1e-6)

    @pytest.mark.parametrize('norm', (None, 'ortho'))
    def test_ifft(self, norm):
        x = random(30) + 1j * random(30)
        assert_allclose(
            x, np.fft.ifft(np.fft.fft(x, norm=norm), norm=norm),
            atol=1e-6)
        # Ensure we get the correct error message
        with pytest.raises(ValueError):
            # ,match='Invalid number of FFT data points'):
            np.fft.ifft([], norm=norm)

    def test_fft2(self):
        x = random((30, 20)) + 1j * random((30, 20))
        assert_allclose(np.fft.fft(np.fft.fft(x, axis=1), axis=0),
                        np.fft.fft2(x), atol=1e-6)
        assert_allclose(np.fft.fft2(x) / np.sqrt(30 * 20),
                        np.fft.fft2(x, norm="ortho"), atol=1e-6)

    def test_ifft2(self):
        x = random((30, 20)) + 1j * random((30, 20))
        assert_allclose(np.fft.ifft(np.fft.ifft(x, axis=1), axis=0),
                        np.fft.ifft2(x), atol=1e-6)
        assert_allclose(np.fft.ifft2(x) * np.sqrt(30 * 20),
                        np.fft.ifft2(x, norm="ortho"), atol=1e-6)

    def test_fftn(self):
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        assert_allclose(
            np.fft.fft(np.fft.fft(np.fft.fft(x, axis=2), axis=1), axis=0),
            np.fft.fftn(x), atol=1e-6)
        assert_allclose(np.fft.fftn(x) / np.sqrt(30 * 20 * 10),
                        np.fft.fftn(x, norm="ortho"), atol=1e-6)

    def test_ifftn(self):
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        assert_allclose(
            np.fft.ifft(np.fft.ifft(np.fft.ifft(x, axis=2), axis=1), axis=0),
            np.fft.ifftn(x), atol=1e-6)
        assert_allclose(np.fft.ifftn(x) * np.sqrt(30 * 20 * 10),
                        np.fft.ifftn(x, norm="ortho"), atol=1e-6)

    def test_rfft(self):
        x = random(30)
        for n in [x.size, 2 * x.size]:
            for norm in [None, 'ortho']:
                assert_allclose(
                    np.fft.fft(x, n=n, norm=norm)[:(n // 2 + 1)],
                    np.fft.rfft(x, n=n, norm=norm), atol=1e-6)
            assert_allclose(
                np.fft.rfft(x, n=n) / np.sqrt(n),
                np.fft.rfft(x, n=n, norm="ortho"), atol=1e-6)

    def test_irfft(self):
        x = random(30)
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x)), atol=1e-6)
        assert_allclose(
            x, np.fft.irfft(np.fft.rfft(x, norm="ortho"), norm="ortho"), atol=1e-6)

    def test_rfft2(self):
        x = random((30, 20))
        assert_allclose(np.fft.fft2(x)[:, :11], np.fft.rfft2(x), atol=1e-6)
        assert_allclose(np.fft.rfft2(x) / np.sqrt(30 * 20),
                        np.fft.rfft2(x, norm="ortho"), atol=1e-6)

    def test_irfft2(self):
        x = random((30, 20))
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x)), atol=1e-6)
        assert_allclose(
            x, np.fft.irfft2(np.fft.rfft2(x, norm="ortho"), norm="ortho"), atol=1e-6)

    def test_rfftn(self):
        x = random((30, 20, 10))
        assert_allclose(np.fft.fftn(x)[:, :, :6], np.fft.rfftn(x), atol=1e-6)
        assert_allclose(np.fft.rfftn(x) / np.sqrt(30 * 20 * 10),
                        np.fft.rfftn(x, norm="ortho"), atol=1e-6)

    def test_irfftn(self):
        x = random((30, 20, 10))
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x)), atol=1e-6)
        assert_allclose(
            x, np.fft.irfftn(np.fft.rfftn(x, norm="ortho"), norm="ortho"), atol=1e-6)

    def test_hfft(self):
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        # x = np.concatenate((x_herm, x[::-1].conj()))
        x = np.concatenate((x_herm, np.conj(x[::-1])))
        assert_allclose(np.fft.fft(x), np.fft.hfft(x_herm), atol=1e-6)
        assert_allclose(np.fft.hfft(x_herm) / np.sqrt(30),
                        np.fft.hfft(x_herm, norm="ortho"), atol=1e-6)

    def test_ihttf(self):
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        # x = np.concatenate((x_herm, x[::-1].conj()))
        x = np.concatenate((x_herm, np.conj(x[::-1])))
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm)), atol=1e-6)
        assert_allclose(
            x_herm, np.fft.ihfft(np.fft.hfft(x_herm, norm="ortho"),
                                 norm="ortho"), atol=1e-6)

    @pytest.mark.parametrize("op", [np.fft.fftn, np.fft.ifftn,
                                    np.fft.rfftn, np.fft.irfftn])
    def _test_axes(self, op):
        x = random((30, 20, 10))
        axes = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        for a in axes:
            op_tr = op(np.transpose(x, a))
            tr_op = np.transpose(op(x, axes=a), a)
            assert_allclose(op_tr, tr_op, atol=1e-6)

    # TODO
    def test_all_1d_norm_preserving(self):
        # verify that round-trip transforms are norm-preserving
        x = random(30)
        # x_norm = np.linalg.norm(x)
        x_norm = numpy.linalg.norm(x)
        n = x.size * 2
        func_pairs = [(np.fft.fft, np.fft.ifft),
                      # (np.fft.rfft, np.fft.irfft),
                      # hfft: order so the first function takes x.size samples
                      #       (necessary for comparison to x_norm above)
                      # (np.fft.ihfft, np.fft.hfft),
                      ]
        for forw, back in func_pairs:
            for n in [x.size, 2 * x.size]:
                for norm in [None, 'ortho']:
                    tmp = forw(x, n=n, norm=norm)
                    tmp = back(tmp, n=n, norm=norm)
                    assert_allclose(x_norm,
                                    # np.linalg.norm(tmp),
                                    numpy.linalg.norm(tmp),
                                    atol=1e-6)

    # TODO
    @pytest.mark.parametrize("dtype", [np.single,   # np.half,
                                       np.double])  # numpy.longdouble
    def test_dtypes(self, dtype):
        # make sure that all input precisions are accepted and internally
        # converted to 64bit
        # x = random(30).astype(dtype)
        x = random(30).astype(dtype)
        assert_allclose(np.fft.ifft(np.fft.fft(x)), x, atol=1e-6)
        # assert_allclose(np.fft.irfft(np.fft.rfft(x)), x, atol=1e-6)


# TODO
@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize(
    "order",
    ["F", "C"]  # 'non-contiguous'
)
@pytest.mark.parametrize(
    "fft",
    [np.fft.fft,   # np.fft.fft2, np.fft.fftn,
     np.fft.ifft]  # ,np.fft.ifft2, np.fft.ifftn
)
def test_fft_with_order(dtype, order, fft):
    # Check that FFT/IFFT produces identical results for C, Fortran and
    # non contiguous arrays
    rng = np.random.RandomState(42)
    # X = rng.rand(8, 7, 13).astype(dtype, copy=False)
    X = rng.rand((8, 7, 13)).astype(dtype, copy=False)
    # See discussion in pull/14178
    _tol = 8.0 * np.sqrt(np.log2(X.size)) * np.finfo(X.dtype).eps
    if order == 'F':
        # Y = np.asfortranarray(X)
        Y = np.asarray(X, order='F')
    else:
        # Make a non contiguous array
        # #Y = X[::-1]
        # #X = np.ascontiguousarray(X[::-1])
        Y = X[:-1]
        X = np.asarray(X[:-1], order='C')

    if fft.__name__.endswith('fft'):
        for axis in range(3):
            X_res = fft(X, axis=axis)
            Y_res = fft(Y, axis=axis)
            assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
    elif fft.__name__.endswith(('fft2', 'fftn')):
        axes = [(0, 1), (1, 2), (0, 2)]
        if fft.__name__.endswith('fftn'):
            axes.extend([(0,), (1,), (2,), None])
        for ax in axes:
            X_res = fft(X, axes=ax)
            Y_res = fft(Y, axes=ax)
            assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
    else:
        raise ValueError()

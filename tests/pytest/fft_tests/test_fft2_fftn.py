import functools
import unittest
import pytest      # NOQA

import numpy as np
from numpy.testing import assert_allclose
import nlcpy       # NOQA
from nlcpy import testing
from nlcpy.testing.types import int_types

global enable_nd_planning
enable_nd_planning = True


def nd_planning_states(states=[True, False], name='enable_nd'):
    """Decorator for parameterized tests with and wihout nd planning

    Tests are repeated with enable_nd_planning set to True and False

    Args:
         states(list of bool): The boolean cases to test.
         name(str): Argument name to which specified dtypes are passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixture. Then, it runs the fixtures in parallel
    by passing the each element of ``dtypes`` to the named
    argument.
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            # get original global planning state
            global enable_nd_planning
            planning_state = enable_nd_planning
            try:
                for nd_planning in states:
                    try:
                        # enable or disable nd planning
                        enable_nd_planning = nd_planning

                        kw[name] = nd_planning
                        impl(self, *args, **kw)
                    except Exception:
                        print(name, 'is', nd_planning)
                        raise
            finally:
                # restore original global planning state
                enable_nd_planning = planning_state

        return test_func
    return decorator


def _numpy_fftn_correct_dtype(xp, a):
    if xp == np and a.dtype in int_types + [np.bool_]:
        a = xp.asarray(a, dtype=np.float64)
    return a


def _size_last_transform_axis(shape, s, axes):
    if s is not None:
        if s[-1] is not None:
            return s[-1]
    elif axes is not None:
        return shape[axes[-1]]
    return shape[-1]


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.with_requires('numpy>=1.10.0')
class TestFft2(unittest.TestCase):

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_fft2(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.fft2(a, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_ifft2(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.ifft2(a, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out


class TestFft2DInvalidParam(object):
    @pytest.mark.parametrize('a', (1, 1 + 2j,
                             ["aaa"], [],
                             ("aaa",), (),
                             ))
    def test_fft2_param_array(self, a):
        with pytest.raises(ValueError):
            nlcpy.fft.fft2(a)

    @pytest.mark.parametrize('a', (
                             [[1, 2], [3, "4"]],
                             ((1, 2), (3, "4")),
                             ([1, 2], [3, "4"]), [(1, 2), (3, "4")],
                             [[1, 2], (3, "4")], ((1, 2), [3, "4"]),))
    def test_fft2_param_array_U21(self, a):
        if np.__version__ < np.lib.NumpyVersion('1.19.1'):
            with pytest.raises(ValueError):
                nlcpy.fft.fft2(a)
        else:
            assert_allclose(nlcpy.fft.fft2(a), np.fft.fft2(a))

    @pytest.mark.parametrize('a', (1, 1 + 2j,
                             ["aaa"], [],
                             ("aaa",), (),
                             ))
    def test_ifft2_param_array(self, a):
        with pytest.raises(ValueError):
            nlcpy.fft.ifft2(a)

    @pytest.mark.parametrize('a', (
                             [[1, 2], [3, "4"]],
                             ((1, 2), (3, "4")),
                             ([1, 2], [3, "4"]), [(1, 2), (3, "4")],
                             [[1, 2], (3, "4")], ((1, 2), [3, "4"]),))
    def test_ifft2_param_array_U21(self, a):
        if np.__version__ < np.lib.NumpyVersion('1.19.1'):
            with pytest.raises(ValueError):
                nlcpy.fft.ifft2(a)
        else:
            assert_allclose(nlcpy.fft.ifft2(a), np.fft.ifft2(a))

    @pytest.mark.parametrize('param', (
                             ([[1, 2, 3], [4, 5, 6]], (-1, -3)),
                             ([[1, 2, 3], [4, 5, 6]], (0, 2)),
                             ([[1, 2, 3], [4, 5, 6]], (0, 0, 0, 0, 5)),
                             ([[1, 2, 3], [4, 5, 6]], (5, 0, 0, 0, 0)),
                             ))
    def test_fft2_param_axes(self, param):
        with pytest.raises(ValueError):
            nlcpy.fft.fft2(param[0], axes=param[1])

    @pytest.mark.parametrize('param', (
                             ([[1, 2, 3], [4, 5, 6]], (-1, -3)),
                             ([[1, 2, 3], [4, 5, 6]], (0, 2)),
                             ([[1, 2, 3], [4, 5, 6]], (0, 0, 0, 0, 5)),
                             ([[1, 2, 3], [4, 5, 6]], (5, 0, 0, 0, 0)),
                             ))
    def test_ifft2_param_axes(self, param):
        with pytest.raises(ValueError):
            nlcpy.fft.ifft2(param[0], axes=param[1])

    @pytest.mark.parametrize('s', (1, 1 + 2j, ""))
    def test_fft2_param_s_TypeError(self, s):
        with pytest.raises(TypeError):
            nlcpy.fft.fft2([[1, 2, 3], [4, 5, 6]], s=s)

    @pytest.mark.parametrize('s', ([0, 1], [], [-1], [""], (0, 1), (), (-1, ), ("",)))
    def test_fft2_param_s_ValueError(self, s):
        with pytest.raises(ValueError):
            nlcpy.fft.fft2([[1, 2, 3], [4, 5, 6]], s=s)

    @pytest.mark.parametrize('s', (1, 1 + 2j, ""))
    def test_ifft2_param_s_TypeError(self, s):
        with pytest.raises(TypeError):
            nlcpy.fft.ifft2([[1, 2, 3], [4, 5, 6]], s=s)

    @pytest.mark.parametrize('s', ([0, 1], [], [-1], [""], (0, 1), (), (-1, ), ("",)))
    def test_ifft2_param_s_ValueError(self, s):
        with pytest.raises(ValueError):
            nlcpy.fft.ifft2([[1, 2, 3], [4, 5, 6]], s=s)

    @pytest.mark.parametrize('norm', (None, 'ortho'))
    @pytest.mark.parametrize('param', (
                             ((2, 3), (0, 1, 2)),
                             ((2,), (0, 1, 2)),
                             ((2,), (0, 1))
                             ))
    def test_fft2_invalid_axes_s(self, param, norm):
        a = nlcpy.arange(24).reshape(2, 3, 4)
        with pytest.raises(ValueError):
            nlcpy.fft.fft2(a, s=param[0], axes=param[1], norm=norm)

    @pytest.mark.parametrize('norm', (None, 'ortho'))
    @pytest.mark.parametrize('param', (
                             ((2, 3), (0, 1, 2)),
                             ((2,), (0, 1, 2)),
                             ((2,), (0, 1))
                             ))
    def test_ifft2_invalid_axes_s(self, param, norm):
        a = nlcpy.arange(24).reshape(2, 3, 4)
        with pytest.raises(ValueError):
            nlcpy.fft.ifft2(a, s=param[0], axes=param[1], norm=norm)


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 10, 4), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (4, 1, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (1, 10, 4), 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (4, 1, 10), 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -1, -2), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -3, -2), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -1, -2), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -3, -2), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (0, 1, 2, 3), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (3, 2, 1, 0), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (4, 3, 2, 1), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (1, 2, 3, 4), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (0, 2, 1, 3), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (4, 2, 3, 1), 'norm': None},
)
@testing.with_requires('numpy>=1.10.0')
class TestFftn(unittest.TestCase):

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_fftn(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_ifftn(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 100), 's': None, 'axes': None, 'norm': None},
    {'shape': (100, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 1, 100), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 100, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (100, 1, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 1, 1, 100), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 1, 100, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 100, 1, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (100, 1, 1, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 100), 's': (1, 100), 'axes': None, 'norm': None},
    {'shape': (100, 1), 's': (100, 1), 'axes': None, 'norm': None},
    {'shape': (1, 1, 100), 's': (1, 1, 100), 'axes': None, 'norm': None},
    {'shape': (1, 100, 1), 's': (1, 100, 1), 'axes': None, 'norm': None},
    {'shape': (100, 1, 1), 's': (100, 1, 1), 'axes': None, 'norm': None},
    {'shape': (1, 1, 1, 100), 's': (1, 1, 1, 100), 'axes': None, 'norm': None},
    {'shape': (1, 1, 100, 1), 's': (1, 1, 100, 1), 'axes': None, 'norm': None},
    {'shape': (1, 100, 1, 1), 's': (1, 100, 1, 1), 'axes': None, 'norm': None},
    {'shape': (100, 1, 1, 1), 's': (100, 1, 1, 1), 'axes': None, 'norm': None},
    {'shape': (20, 30, 40), 's': (1, 1, 100), 'axes': None, 'norm': None},
    {'shape': (20, 30, 40), 's': (1, 100, 1), 'axes': None, 'norm': None},
    {'shape': (20, 30, 40), 's': (100, 1, 1), 'axes': None, 'norm': None},
)
@testing.with_requires('numpy>=1.10.0')
class TestRfft2(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_rfft2(self, xp, dtype, order):
        # the scaling of old Numpy is incorrect
        if np.__version__ < np.lib.NumpyVersion('1.13.0'):
            if self.s is not None:
                return xp.empty(0)

        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.rfft2(a, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_irfft2(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.irfft2(a, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.float32:
            out = out.astype(np.float64)
        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -1, -2), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -3, -2), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -1, -2), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -3, -2), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': None},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (0, 1, 2, 3), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (3, 2, 1, 0), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (4, 3, 2, 1), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (1, 2, 3, 4), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (0, 2, 1, 3), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (4, 2, 3, 1), 'norm': None},
    {'shape': (1, 100), 's': None, 'axes': None, 'norm': None},
    {'shape': (100, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 1, 100), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 100, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (100, 1, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 1, 1, 100), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 1, 100, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 100, 1, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (100, 1, 1, 1), 's': None, 'axes': None, 'norm': None},
    {'shape': (1, 100), 's': (1, 100), 'axes': None, 'norm': None},
    {'shape': (100, 1), 's': (100, 1), 'axes': None, 'norm': None},
    {'shape': (1, 1, 100), 's': (1, 1, 100), 'axes': None, 'norm': None},
    {'shape': (1, 100, 1), 's': (1, 100, 1), 'axes': None, 'norm': None},
    {'shape': (100, 1, 1), 's': (100, 1, 1), 'axes': None, 'norm': None},
    {'shape': (1, 1, 1, 100), 's': (1, 1, 1, 100), 'axes': None, 'norm': None},
    {'shape': (1, 1, 100, 1), 's': (1, 1, 100, 1), 'axes': None, 'norm': None},
    {'shape': (1, 100, 1, 1), 's': (1, 100, 1, 1), 'axes': None, 'norm': None},
    {'shape': (100, 1, 1, 1), 's': (100, 1, 1, 1), 'axes': None, 'norm': None},
    {'shape': (20, 30, 40), 's': (1, 1, 100), 'axes': None, 'norm': None},
    {'shape': (20, 30, 40), 's': (1, 100, 1), 'axes': None, 'norm': None},
    {'shape': (20, 30, 40), 's': (100, 1, 1), 'axes': None, 'norm': None},
)
@testing.with_requires('numpy>=1.10.0')
class TestRfftn(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_rfftn(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        return out

    @testing.for_dtypes('DF')
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-3, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_irfftn(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out


@testing.parameterize(
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (4, 10, 1), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (4, 1, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (4, 10, 1), 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (4, 1, 10), 'axes': None, 'norm': 'ortho'},
)
class TestRfftn_sub(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_rfftn(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        return out

    @testing.with_requires('numpy>=1.18.0')
    @testing.for_dtypes('DF')
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_irfftn(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out


@testing.parameterize(*testing.product({
    'shape': [(3, 3), (3, 3, 3), (3, 3, 3, 3)],
    's': [None] + [(i, j) for i in [2, 3, 4] for j in [2, 3, 4]],
    'axes': [(i, j) for i in (-2, -1) for j in (-2, -1)],
    'norm': [None, 'ortho']
}))
@testing.with_requires('numpy>=1.10.0')
class TestFft2DAxes_s(unittest.TestCase):

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-5, accept_error=ValueError,
                                  contiguous_check=False)
    def test_fft_2d_axes_s(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.fftn(a, axes=self.axes, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-6, accept_error=ValueError,
                                  contiguous_check=False)
    def test_ifft_2d_axes_s(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.ifftn(a, axes=self.axes, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_rfft_2d_axes_s(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        return out

    @testing.with_requires('numpy>=1.18.0')
    @testing.for_dtypes('DF')
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                  contiguous_check=False)
    def test_irfft_2d_axes_s(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out


@testing.parameterize(*testing.product({
    'shape': [(3, 3, 3), ],
    's': [None, ] +
         [(i, j, k) for i in [2, 3, 4] for j in [2, 3, 4] for k in [2, 3, 4]],
    'axes': [(i, j, k) for i in (0, 1, 2) for j in (0, 1, 2) for k in (0, 1, 2)],
    'norm': [None, 'ortho']
}))
@testing.with_requires('numpy>=1.10.0')
class TestFft3DAxes_s(unittest.TestCase):

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-5, accept_error=ValueError,
                                  contiguous_check=False)
    def test_fft_3d_axes_s(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.fftn(a, axes=self.axes, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-5, accept_error=ValueError,
                                  contiguous_check=False)
    def test_ifft_3d_axes_s(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.ifftn(a, axes=self.axes, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-5, accept_error=ValueError,
                                  contiguous_check=False)
    def test_rfft_3d_axes_s(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        return out

    @testing.with_requires('numpy>=1.18.0')
    @testing.for_dtypes('DF')
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-5, accept_error=ValueError,
                                  contiguous_check=False)
    def test_irfft_3d_axes_s(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out


@testing.parameterize(*testing.product({
    'shape': [(10, 10, 10, 10), ],
    's': [None, ] +
         [(i, j, k) for i in [5, 10, 15] for j in [5, 10, 15] for k in [5, 10, 15]],
    'axes': [(i, j, k) for i in (0, 1, 2, 3)
             for j in (0, 1, 2, 3)
             for k in (0, 1, 2, 3)],
    'norm': [None, 'ortho']
}))
@testing.with_requires('numpy>=1.10.0')
class TestFft3DAxes_s_4d(unittest.TestCase):

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-3, accept_error=ValueError,
                                  contiguous_check=False)
    def test_fft_3d_axes_s_4d(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.fftn(a, axes=self.axes, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-5, accept_error=ValueError,
                                  contiguous_check=False)
    def test_ifft_3d_axes_s_4d(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.ifftn(a, axes=self.axes, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-3, accept_error=ValueError,
                                  contiguous_check=False)
    def test_rfft_3d_axes_s_4d(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        return out

    @testing.with_requires('numpy>=1.18.0')
    @testing.for_dtypes('DF')
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-4, accept_error=ValueError,
                                  contiguous_check=False)
    def test_irfft_3d_axes_s_4d(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out


@testing.parameterize(*testing.product({
    'shape': [(10, 10, 10, 10), ],
    's': [None, ],
    'axes': [(i, j, k, h) for i in (0, 1, 2, 3)
             for j in (0, 1, 2, 3)
             for k in (0, 1, 2, 3)
             for h in (0, 1, 2, 3)],
    'norm': [None, 'ortho']
}))
@testing.with_requires('numpy>=1.10.0')
class TestFft4DAxes(unittest.TestCase):

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-3, accept_error=ValueError,
                                  contiguous_check=False)
    def test_fft_4d_axes(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.fftn(a, axes=self.axes, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-5, accept_error=ValueError,
                                  contiguous_check=False)
    def test_ifft_4d_axes(self, xp, dtype, order, enable_nd):
        global enable_nd_planning
        assert enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        a = _numpy_fftn_correct_dtype(xp, a)
        tmp = a.copy()
        out = xp.fft.ifftn(a, axes=self.axes, s=self.s, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-3, accept_error=ValueError,
                                  contiguous_check=False)
    def test_rfft_4d_axes(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.float32:
            out = out.astype(np.complex64)
        return out

    @testing.with_requires('numpy>=1.18.0')
    @testing.for_dtypes('DF')
    @testing.for_orders("CF")
    @testing.numpy_nlcpy_allclose(rtol=1e-4, atol=1e-4, accept_error=ValueError,
                                  contiguous_check=False)
    def test_irfft_4d_axes(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        a = xp.asarray(a, order=order)
        tmp = a.copy()
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        assert_allclose(a, tmp)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.float32)
        elif xp == np and dtype is not np.complex64:
            out = out.astype(np.float64)
        return out

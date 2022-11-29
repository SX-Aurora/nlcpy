#
# * The source code in this file is based on the soure code of NumPy.
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
# # NumPy License #
#
#     Copyright (c) 2005-2020, NumPy Developers.
#     All rights reserved.
#
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of the NumPy Developers nor the names of any contributors may be
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
# distutils: language = c++

import warnings

import nlcpy
import numpy

from nlcpy import ndarray
from nlcpy.core import core
from nlcpy.core import internal
from nlcpy.core import dtype as _dtype
from nlcpy.ufuncs import operations as ufunc_op
from nlcpy.request import request
from numpy.ma import masked

MaskType = numpy.bool_
nomask = MaskType(0)

ufunc_domain = {}
ufunc_fills = {}

default_filler = {
    'b': True,
    'c': 1.e20 + 0.0j,
    'f': 1.e20,
    'i': 999999,
    'u': 999999,
}


class MAError(Exception):
    pass


class MaskError(MAError):
    pass


def _arraymethod(funcname, onmask=True):
    """ Returns a class method wrapper around a basic array method.

    Creates a class method which returns a masked array, where the new
    ``_data`` array is the output of the corresponding basic method called
    on the original ``_data``.
    If `onmask` is True, the new mask is the output of the method called
    on the initial mask. Otherwise, the new mask is just a reference
    to the initial mask.

    Parameters
    ----------
    funcname : str
        Name of the function to apply on data.
    onmask : bool
        Whether the mask must be processed also (True) or left
        alone (False). Default is True. Make available as `_onmask`
        attribute.

    Returns
    -------
    method : instancemethod
        Class method wrapper of the specified basic array method.
    """
    def wrapped_method(self, *args, **params):
        result = getattr(self._data, funcname)(*args, **params)
        result = result.view(type(self))
        result._update_from(self)
        mask = self._mask
        if not onmask:
            result.__setmask__(mask)
        elif mask is not nomask:
            # __setmask__ makes a copy, which we don't want
            result._mask = getattr(mask, funcname)(*args, **params)
        return result
    methdoc = getattr(ndarray, funcname, None) or getattr(nlcpy, funcname, None)
    if methdoc is not None:
        wrapped_method.__doc__ = methdoc.__doc__
    wrapped_method.__name__ = funcname
    return wrapped_method


class MaskedArray(ndarray):
    """ An array class with possibly masked values.

    Masked values of True exclude the corresponding element from any
    computation.

    Parameters
    ----------
    data : array_like
        Input data.
    mask : sequence, optional
        Mask. Must be convertible to an array of booleans with the same
        shape as `data`. True indicates a masked (i.e. invalid) data.
    dtype : dtype, optional
        Data type of the output.
        If `dtype` is None, the type of the data argument (``data.dtype``)
        is used. If `dtype` is not None and different from ``data.dtype``,
        a copy is performed.
    copy : bool, optional
        Whether to copy the input data (True), or to use a reference instead.
        Default is False.
    subok : bool, optional
        Whether to return a subclass of `MaskedArray` if possible (True) or a
        plain `MaskedArray`.
        subok=True is Not Implemented.
    ndmin : int, optional
        Minimum number of dimensions. Default is 0.
    fill_value : scalar, optional
        Value used to fill in the masked values when necessary.
        If None, a default based on the data-type is used.
    keep_mask : bool, optional
        Whether to combine `mask` with the mask of the input data, if any
        (True), or to use only `mask` for the output (False). Default is True.
    hard_mask : bool, optional
        Whether to use a hard mask or not. With a hard mask, masked values
        cannot be unmasked. Default is False.
    shrink : bool, optional
        Whether to force compression of an empty mask. Default is True.
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C', then the array
        will be in C-contiguous order (last-index varies the fastest).
        If order is 'F', then the returned array will be in
        Fortran-contiguous order (first-index varies the fastest).
        If order is 'A' (default), then the returned array may be
        in any order (either C-, Fortran-contiguous, or even discontiguous),
        unless a copy is required, in which case it will be C-contiguous.

    Examples
    --------
    The ``mask`` can be initialized with an array of boolean values
    with the same shape as ``data``.

    >>> import nlcpy as vp
    >>> import numpy as np
    >>> data = vp.arange(6).reshape((2, 3))
    >>> vp.ma.MaskedArray(data, mask=[[False, True, False],
    ...                               [False, False, True]])
    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)

    Alternatively, the ``mask`` can be initialized to homogeneous boolean
    array with the same shape as ``data`` by passing in a scalar
    boolean value:

    >>> vp.ma.MaskedArray(data, mask=False)
    masked_array(
      data=[[0, 1, 2],
            [3, 4, 5]],
      mask=[[False, False, False],
            [False, False, False]],
      fill_value=999999)
    >>> vp.ma.MaskedArray(data, mask=True)
    masked_array(
      data=[[--, --, --],
            [--, --, --]],
      mask=[[ True,  True,  True],
            [ True,  True,  True]],
      fill_value=999999,
      dtype=int64)

    .. note::
        The recommended practice for initializing ``mask`` with a scalar
        boolean value is to use ``True``/``False`` rather than
        ``np.True_``/``np.False_``. The reason is :attr:`nomask`
        is represented internally as ``np.False_``.

        >>> np.False_ is vp.ma.nomask
        True
    """
    __array_priority__ = 15

    def __new__(self, data=None, mask=nomask, dtype=None, copy=False,
                subok=None, ndmin=0, fill_value=None, keep_mask=True,
                hard_mask=None, shrink=True, order=None):
        if subok is not None:
            raise NotImplementedError('subok is not implemented yet')

        if isinstance(data, MaskedArray):
            _data = data._data
        else:
            _data = data
        if not isinstance(_data, ndarray) or copy or \
           _data.dtype != dtype or _data.ndim < ndmin or \
           (order == 'C' and not _data.flags.c_contiguous) or \
           (order == 'F' and not _data.flags.f_contiguous):
            _data = nlcpy.array(_data, dtype=dtype, ndmin=ndmin, order=order, copy=copy)

        _data = ndarray.view(_data, MaskedArray)
        return _data

    def __init__(self, data=None, mask=nomask, dtype=None, copy=False,
                 subok=None, ndmin=0, fill_value=None, keep_mask=True,
                 hard_mask=None, shrink=True, order=None):
        if hasattr(data, '_mask'):
            self._mask = data._mask
        else:
            self._mask = nomask

        self._sharedmask = True
        if mask is nomask:
            if not keep_mask:
                if shrink:
                    self._mask = nomask
                else:
                    self._mask = nlcpy.zeros(self.shape, dtype=MaskType)
            elif isinstance(data, (tuple, list)):
                try:
                    mask = nlcpy.array(
                        [getmaskarray(nlcpy.asanyarray(m, dtype=self.dtype))
                         for m in data], dtype=MaskType)
                except ValueError:
                    mask = nomask
                if mask.any():
                    self._mask = mask
                self._sharedmask = False
            else:
                self._sharedmask = not copy
                if copy:
                    self._mask = self._mask.copy()
                    if getmask(data) is not nomask:
                        data._mask.shape = data.shape
        else:
            if mask is True:
                mask = nlcpy.ones(self.shape, dtype=MaskType)
            elif mask is False:
                mask = nlcpy.zeros(self.shape, dtype=MaskType)
            else:
                mask = nlcpy.array(mask, copy=copy, dtype=MaskType)

            if mask.shape != self.shape:
                (nd, nm) = (self.size, mask.size)
                if nm == 1:
                    mask = nlcpy.resize(mask, self.shape)
                elif nm == nd:
                    mask = nlcpy.reshape(mask, self.shape)
                else:
                    msg = "Mask and data not compatible: data size is %i, " + \
                          "mask size is %i."
                    raise MaskError(msg % (nd, nm))
                copy = True

            if self._mask is nomask:
                self._mask = mask
                self._sharedmask = not copy
            else:
                if not keep_mask:
                    self._mask = mask
                    self._sharedmask = not copy
                else:
                    self._mask = nlcpy.logical_or(mask, self._mask)
                    self._sharedmask = False
        if isinstance(self._mask, nlcpy.ndarray):
            if self.venode != self._mask.venode:
                raise ValueError(
                    'Mask and data exist different VE node: mask on {}, data on {}'
                    .format(self._mask.venode, self.venode)
                )
        if fill_value is None:
            fill_value = getattr(data, '_fill_value', None)
        if fill_value is not None:
            self._fill_value = _check_fill_value(fill_value, self.dtype)
        else:
            self._fill_value = None
        if hard_mask is None:
            self._hardmask = getattr(data, '_hardmask', False)
        else:
            self._hardmask = hard_mask
        self._baseclass = ndarray

    def _update_from(self, obj):
        """
        Copies some attributes of obj to self.
        """
        if isinstance(obj, ndarray):
            _baseclass = type(obj)
        else:
            _baseclass = ndarray
        _optinfo = {}
        _optinfo.update(getattr(obj, '_optinfo', {}))
        _optinfo.update(getattr(obj, '_basedict', {}))
        if not isinstance(obj, MaskedArray):
            _optinfo.update(getattr(obj, '__dict__', {}))
        _dict = dict(_fill_value=getattr(obj, '_fill_value', None),
                     _hardmask=getattr(obj, '_hardmask', False),
                     _sharedmask=getattr(obj, '_sharedmask', False),
                     _baseclass=getattr(obj, '_baseclass', _baseclass),
                     _optinfo=_optinfo,
                     _basedict=_optinfo)
        self.__dict__.update(_dict)
        self.__dict__.update(_optinfo)
        return

    def view(self, dtype=None, type=None, fill_value=None):
        """ Returns a view of the MaskedArray data.

        Parameters
        ----------
        dtype : data-type or ndarray sub-class, optional
            Data-type descriptor of the returned view, e.g., float32 or int32.
            The default, None, results in the view having the same data-type
            as `a`. As with :func:`nlcpy.ndarray.view`, dtype can also be specified as
            an ndarray sub-class, which then specifies the type of the
            returned object (this is equivalent to setting the ``type``.
        type : Python type, optional
            Type of the returned view, either ndarray or a subclass.  The
            default None results in type preservation.
        fill_value : scalar, optional
            The value to use for invalid entries (None by default).
            If None, then this argument is inferred from the passed `dtype`, or
            in its absence the original array, as discussed in the notes below.

        See Also
        --------
        nlcpy.ndarray.view : Equivalent method on ndarray object.

        Notes
        -----
        ``a.view()`` is used two different ways:
        ``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
        of the array's memory with a different data-type.  This can cause a
        reinterpretation of the bytes of memory.
        ``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
        returns an instance of `ndarray_subclass` that looks at the same array
        (same shape, dtype, etc.)  This does not cause a reinterpretation of the
        memory.
        If `fill_value` is not specified, but `dtype` is specified (and is not
        an ndarray sub-class), the `fill_value` of the MaskedArray will be
        reset. If neither `fill_value` nor `dtype` are specified (or if
        `dtype` is an ndarray sub-class), then the fill value is preserved.
        Finally, if `fill_value` is specified, but `dtype` is not, the fill
        value is set to the specified value.
        For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
        bytes per entry than the previous dtype (for example, converting a
        regular array to a structured array), then the behavior of the view
        cannot be predicted just from the superficial appearance of ``a`` (shown
        by ``print(a)``). It also depends on exactly how ``a`` is stored in
        memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
        defined as a slice or transpose, etc., the view may give different
        results.
        """
        if type is None and not (dtype is ndarray or dtype in ndarray.__subclasses__()):
            type = MaskedArray
        v = super().view(dtype=dtype, type=type)
        if not hasattr(v, '_mask'):
            return v
        mask = self._mask
        if fill_value is None:
            fill_value = self._fill_value
        if isinstance(mask, ndarray):
            mask = mask.reshape(v.shape)
        if self.dtype != v.dtype:
            v._fill_value = None
        elif isinstance(fill_value, ndarray):
            v._fill_value = fill_value.copy()
        else:
            v._fill_value = fill_value
        v._mask = mask
        v._sharedmask = True
        return v

    def __getitem__(self, indx):
        dout = self.data[indx]
        _mask = self._mask

        def _scalar_heuristic(arr, elem):
            if type(arr).__getitem__ == ndarray.__getitem__ and arr.size > 1:
                return False
            return None

        if _mask is not nomask:
            mout = _mask[indx]
            scalar_expected = not (isinstance(mout, ndarray) and mout.ndim > 0)
        else:
            mout = nomask
            scalar_expected = _scalar_heuristic(self.data, mout)
            if scalar_expected is None:
                scalar_expected = not isinstance(getmaskarray(self)[indx], ndarray)

        if scalar_expected:
            return masked if mout else dout.view(MaskedArray)
        else:
            dout = dout.view(MaskedArray)
            dout._update_from(self)
            # Update the mask if needed
            if mout is not nomask:
                # set shape to match that of data; this is needed for matrices
                dout._mask = mout.reshape(dout.shape)
                dout._sharedmask = True
                # Note: Don't try to check for m.any(), that'll take too long
        return dout

    def __setitem__(self, indx, value):
        """ x.__setitem__(i, y) is equivalent to x[i] = y.

        Set item described by index. If value is masked, masks those
        locations.
        """
        _data = self._data
        _mask = self._mask
        _dtype = _data.dtype

        if value is masked:
            # The mask wasn't set: create a full version.
            if _mask is nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
            # Now, set the mask to its value.
            _mask[indx] = True
            return

        # Get the _data part of the new value
        dval = getattr(value, '_data', value)
        # Get the _mask part of the new value
        mval = getmask(value)
        if _mask is nomask:
            # Set the data, then the mask
            _data[indx] = dval
            if mval is not nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
                _mask[indx] = mval
        elif not self._hardmask:
            # Set the data, then the mask
            if isinstance(indx, MaskedArray):
                _data[indx.data] = dval
                _mask[indx.data] = mval
            else:
                _data[indx] = dval
                _mask[indx] = mval
        elif hasattr(indx, 'dtype') and (indx.dtype == MaskType):
            indx = indx * nlcpy.logical_not(_mask)
            _data[indx] = dval
        else:
            mindx = mask_or(_mask[indx], mval, copy=True)
            dindx = self._data[indx]
            if dindx.size > 1:
                nlcpy.copyto(dindx, dval, where=nlcpy.logical_not(mindx))
            elif mindx is nomask:
                dindx = dval
            _data[indx] = dindx
            _mask[indx] = mindx
        return

    @property
    def shape(self):
        return super().shape

    @shape.setter
    def shape(self, shape):
        super(MaskedArray, type(self)).shape.__set__(self, shape)
        if getmask(self) is not nomask:
            self._mask.shape = self.shape

    def __setmask__(self, mask, copy=False):
        idtype = self.dtype
        current_mask = self._mask
        if mask is masked:
            mask = True

        if current_mask is nomask:
            # Make sure the mask is set
            # Just don't do anything if there's nothing to do.
            if mask is nomask:
                return
            current_mask = self._mask = make_mask_none(self.shape, idtype)

        # Hardmask: don't unmask the data
        if self._hardmask:
            current_mask |= mask
        # Softmask: set everything to False
        # If it's obviously a compatible scalar, use a quick update
        # method.
        elif isinstance(mask, (int, float, nlcpy.bool_, nlcpy.number)):
            current_mask[...] = mask
        # Otherwise fall back to the slower, general purpose way.
        else:
            mask = nlcpy.asarray(mask, dtype=nlcpy.bool_)
            current_mask = mask[tuple([slice(0, i) for i in self.shape])]

        # Reshape if needed
        if current_mask.shape:
            current_mask.shape = self.shape
        self._mask = current_mask
        return

    _set_mask = __setmask__

    @property
    def mask(self):
        return self._mask.view()

    @mask.setter
    def mask(self, value):
        self.__setmask__(value)

    def harden_mask(self):
        """ Forces the mask to hard.

        Whether the mask of a masked array is hard or soft is determined by
        its :attr:`hardmask` property. `harden_mask` sets
        `hardmask` to ``True``.

        See Also
        --------
        nlcpy.ma.MaskedArray.hardmask
        """
        self._hardmask = True
        return self

    def soften_mask(self):
        """ Forces the mask to soft.

        Whether the mask of a masked array is hard or soft is determined by
        its :attr:`hardmask` property. `soften_mask` sets
        `hardmask` to ``False``.

        See Also
        --------
        nlcpy.ma.MaskedArray.hardmask
        """
        self._hardmask = False
        return self

    @property
    def hardmask(self):
        """ Hardness of the mask.

        If True, masked values cannot be unmasked.
        """
        return self._hardmask

    def unshare_mask(self):
        """ Copies the mask and set the sharedmask flag to False.

        Whether the mask is shared between masked arrays can be seen from
        the `sharedmask` property. `unshare_mask` ensures the mask is not shared.
        A copy of the mask is only made if it was shared.

        See Also
        --------
        sharedmask
        """
        if self._sharedmask:
            self._mask = self._mask.copy()
            self._sharedmask = False
        return self

    @property
    def sharedmask(self):
        """ Share status of the mask (read-only). """
        return self._sharedmask

    def shrink_mask(self):
        """ Reduces a mask to nomask when possible.

        Examples
        --------
        >>> import nlcpy as vp
        >>> x = vp.ma.array([[1, 2], [3, 4]], mask=[0]*4)
        >>> x.mask
        array([[False, False],
               [False, False]])
        >>> x.shrink_mask()
        masked_array(
          data=[[1, 2],
                [3, 4]],
          mask=False,
          fill_value=999999)
        >>> x.mask
        False
        """
        self._mask = _shrink_mask(self._mask)
        return self

    @property
    def baseclass(self):
        """ Class of the underlying data (read-only). """
        return self._baseclass

    def _get_data(self):
        """Returns the underlying data, as a view of the masked array.

        The type of the data can be accessed through the :attr:`baseclass`
        attribute.
        """
        return super().view()

    _data = property(fget=_get_data)
    data = property(fget=_get_data)

    @property
    def fill_value(self):
        """The filling value of the masked array is a scalar.

        When setting, None will set to a default based on the data type.

        Examples
        --------
        >>> import nlcpy as vp
        >>> for dt in [vp.int32, vp.int64, vp.float64, vp.complex128]:
        ...     vp.ma.array([0, 1], dtype=dt).get_fill_value()
        ...
        999999
        999999
        1e+20
        (1e+20+0j)
        >>> x = vp.ma.array([0, 1.], fill_value=-vp.inf)
        >>> x.fill_value
        -inf
        >>> x.fill_value = vp.pi
        >>> x.fill_value
        3.141592653589793

        Reset to default:

        >>> x.fill_value = None
        >>> x.fill_value
        1e+20
        """
        if self._fill_value is None:
            self._fill_value = _check_fill_value(None, self.dtype)
        if isinstance(self._fill_value, (ndarray, numpy.ndarray)):
            return self._fill_value[()]
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value):
        target = _check_fill_value(value, self.dtype)
        if target.ndim != 0:
            warnings.warn(
                "Non-scalar arrays for the fill value are deprecated. Use "
                "arrays with scalar values instead. The filled function "
                "still supports any array as `fill_value`.",
                DeprecationWarning, stacklevel=2)

        if self._fill_value is None:
            self._fill_value = target
        else:
            self._fill_value[()] = target

    # kept for compatibility
    get_fill_value = fill_value.fget
    set_fill_value = fill_value.fset

    def filled(self, fill_value=None):
        """Returns a copy of self, with masked values filled with a given value.

        **However**, if there are no masked values to fill, self will be
        returned instead as an ndarray.

        Parameters
        ----------
        fill_value : array_like, optional
            The value to use for invalid entries. Can be scalar or non-scalar.
            If non-scalar, the resulting ndarray must be broadcastable over
            input array. Default is None, in which case, the `fill_value`
            attribute of the array is used instead.

        Returns
        -------
        filled_array : ndarray
            A copy of ``self`` with invalid entries replaced by *fill_value*
            (be it the function argument or the attribute of ``self``), or
            ``self`` itself as an ndarray if there are no invalid entries to
            be replaced.

        Notes
        -----
        The result is **not** a MaskedArray!

        Examples
        --------
        >>> import nlcpy as vp
        >>> x = vp.ma.array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999)
        >>> x.filled()
        array([   1,    2, -999,    4, -999])
        >>> x.filled(fill_value=1000)
        array([   1,    2, 1000,    4, 1000])
        >>> type(x.filled())
        <class 'nlcpy.core.core.ndarray'>
        """
        m = self._mask
        if m is nomask:
            return self._data

        if fill_value is None:
            fill_value = self.fill_value
        else:
            fill_value = _check_fill_value(fill_value, self.dtype)

        if not m.any():
            return self._data
        else:
            result = self._data.copy('K')
            try:
                nlcpy.copyto(result, fill_value, where=m)
            except (TypeError, AttributeError):
                fill_value = nlcpy.narray(fill_value, dtype=object)
                d = result.astype(object)
                result = nlcpy.choose(m, (d, fill_value))
            except IndexError:
                # ok, if scalar
                if self._data.shape:
                    raise
                elif m:
                    result = nlcpy.array(fill_value, dtype=self.dtype)
                else:
                    result = self._data
        return result

    def __str__(self):
        return self.get().__str__()

    def __repr__(self):
        """
        Literal string representation.
        """
        return self.get().__repr__()

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __iadd__(self, other):
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask:
                self._mask = m.copy()
        else:
            if m is not nomask:
                self._mask += m
        if numpy.isscalar(other):
            if self._mask is not nomask and self._mask.any():
                other = numpy.array(
                    other, dtype=numpy.result_type(self.dtype, type(other)))
            else:
                other = numpy.result_type(self.dtype, type(other)).type(other)
            where = nlcpy.logical_not(self._mask)
            nlcpy.add(self._data, other, where=where, out=self._data)
        else:
            self._data.__iadd__(
                nlcpy.where(self._mask, self.dtype.type(0), getdata(other)))
        return self

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __isub__(self, other):
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask:
                self._mask = m.copy()
        else:
            if m is not nomask:
                self._mask += m
        if numpy.isscalar(other):
            if self._mask is not nomask and self._mask.any():
                other = numpy.array(
                    other, dtype=numpy.result_type(self.dtype, type(other)))
            else:
                other = numpy.result_type(self.dtype, type(other)).type(other)
            where = nlcpy.logical_not(self._mask)
            nlcpy.subtract(self._data, other, where=where, out=self._data)
        else:
            self._data.__isub__(
                nlcpy.where(self._mask, self.dtype.type(0), getdata(other)))
        return self

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __imul__(self, other):
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask:
                self._mask = m.copy()
        else:
            if m is not nomask:
                self._mask += m
        if numpy.isscalar(other):
            if self._mask is not nomask and self._mask.any():
                other = numpy.array(
                    other, dtype=numpy.result_type(self.dtype, type(other)))
            else:
                other = numpy.result_type(self.dtype, type(other)).type(other)
            where = nlcpy.logical_not(self._mask)
            nlcpy.multiply(self._data, other, where=where, out=self._data)
        else:
            self._data.__imul__(
                nlcpy.where(self._mask, self.dtype.type(1), getdata(other)))
        return self

    def __div__(self, other):
        return divide(self, other)

    def __idiv__(self, other):
        other_data = getdata(other)
        dom_mask = _DomainSafeDivide().__call__(self._data, other_data)
        other_mask = getmask(other)
        new_mask = mask_or(other_mask, dom_mask)
        # The following 3 lines control the domain filling
        if dom_mask.any():
            (_, fval) = ufunc_fills[nlcpy.divide]
            other_data = nlcpy.where(dom_mask, fval, other_data)
        self._mask |= new_mask
        self._data.__idiv__(
            nlcpy.where(self._mask, self.dtype.type(1), other_data))
        return self

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __itruediv__(self, other):
        other_data = getdata(other)
        dom_mask = _DomainSafeDivide().__call__(self._data, other_data)
        other_mask = getmask(other)
        new_mask = mask_or(other_mask, dom_mask)
        # The following 3 lines control the domain filling
        if dom_mask.any():
            (_, fval) = ufunc_fills[nlcpy.true_divide]
            other_data = nlcpy.where(dom_mask, fval, other_data)
        self._mask |= new_mask
        self._data.__itruediv__(
            nlcpy.where(self._mask, self.dtype.type(1), other_data))
        return self

    @property
    def imag(self):
        """The imaginary part of the masked array.

        This property is a view on the imaginary part of this `MaskedArray`.

        See Also
        --------
        real

        Examples
        --------
        >>> import nlcpy as vp
        >>> x = vp.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.imag
        masked_array(data=[1.0, --, 1.6],
                     mask=[False,  True, False],
               fill_value=1e+20)
        """
        result = self._data.imag.view(type(self))
        result.__setmask__(self._mask)
        result._sharedmask = False
        return result

    get_imag = imag.fget

    @property
    def real(self):
        """The real part of the masked array.

        This property is a view on the real part of this `MaskedArray`.

        See Also
        --------
        imag

        Examples
        --------
        >>> import nlcpy as vp
        >>> x = vp.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.real
        masked_array(data=[1.0, --, 3.45],
                     mask=[False,  True, False],
               fill_value=1e+20)
        """
        result = self._data.real.view(type(self))
        result.__setmask__(self._mask)
        result._sharedmask = False
        return result

    get_real = real.fget

    def ravel(self, order='C'):
        """Returns a 1D version of self, as a view.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            The elements of `a` are read using this index order. 'C' means to
            index the elements in C-like order, with the last axis index
            changing fastest, back to the first axis index changing slowest.
            'F' means to index the elements in Fortran-like index order, with
            the first index changing fastest, and the last index changing
            slowest. Note that the 'C' and 'F' options take no account of the
            memory layout of the underlying array, and only refer to the order
            of axis indexing.  'A' means to read the elements in Fortran-like
            index order if `m` is Fortran *contiguous* in memory, C-like order
            otherwise.  'K' means to read the elements in the order they occur
            in memory, except for reversing the data when strides are negative.
            By default, 'C' index order is used.

        Returns
        -------
        MaskedArray
            Output view is of shape ``(self.size,)`` (or
            ``(nlcpy.ma.product(self.shape),)``).

        Restriction
        -----------
        * If order == 'K': *NotImplementedError* occurs.

        Examples
        --------
        >>> import nlcpy as vp
        >>> x = vp.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.ravel()
        masked_array(data=[1, --, 3, --, 5, --, 7, --, 9],
                     mask=[False,  True, False,  True, False,  True, False,  True,
                           False],
               fill_value=999999)
        """
        r = ndarray.ravel(self._data, order=order).view(type(self))
        r._update_from(self)
        if self._mask is not nomask:
            r._mask = self._mask.ravel(order=order).reshape(r.shape)
        else:
            r._mask = nomask
        return r

    def reshape(self, *shape, order='C'):
        """ Gives a new shape to the array without changing its data.

        Returns a masked array containing the same data, but with a new shape.
        The result is a view on the original array; if this is not possible, a
        ValueError is raised.

        Parameters
        ----------
        shape : int or tuple of ints
            The new shape should be compatible with the original shape. If an
            integer is supplied, then the result will be a 1-D array of that
            length.
        order : {'C', 'F'}, optional
            Determines whether the array data should be viewed as in C
            (row-major) or FORTRAN (column-major) order.

        Returns
        -------
        reshaped_array : array
            A new view on the array.

        See Also
        --------
        reshape : Equivalent function in the masked array module.
        nlcpy.ndarray.reshape : Equivalent method on ndarray object.
        nlcpy.reshape : Equivalent function in the NLCPy module.

        Notes
        -----
        The reshaping operation cannot guarantee that a copy will not be made,
        to modify the shape in place, use ``a.shape = s``

        Examples
        --------
        >>> import nlcpy as vp
        >>> x = vp.ma.array([[1,2],[3,4]], mask=[1,0,0,1])
        >>> x
        masked_array(
          data=[[--, 2],
                [3, --]],
          mask=[[ True, False],
                [False,  True]],
          fill_value=999999)
        >>> x = x.reshape((4,1))
        >>> x
        masked_array(
          data=[[--],
                [2],
                [3],
                [--]],
          mask=[[ True],
                [False],
                [False],
                [ True]],
          fill_value=999999)
        """
        result = self.data.reshape(*shape, order=order).view(type(self))
        result._update_from(self)
        if self._mask is not nomask:
            result._mask = self._mask.reshape(*shape, order=order)
        return result

    def resize(self, newshape, refcheck=True, order=False):
        """
        .. warning::
            This method does nothing, except raise a ValueError exception. A
            masked array does not own its data and therefore cannot safely be
            resized in place. Use the :func:`nlcpy.ma.resize` function instead.

        This method is difficult to implement safely and may be deprecated in
        future releases of NLCPy.
        """
        # Note : the 'order' keyword looks broken, let's just drop it
        errmsg = "A masked array does not own its data "\
                 "and therefore cannot be resized.\n" \
                 "Use the nlcpy.ma.resize function instead."
        raise ValueError(errmsg)

    def all(self, axis=None, out=None, keepdims=nlcpy._NoValue):
        """
        Masked version of all is not implemented yet.
        """
        raise NotImplementedError('masked version of all is not implemented yet')

    def any(self, axis=None, out=None, keepdims=False):
        """
        Masked version of any is not implemented yet.
        """
        raise NotImplementedError('masked version of any is not implemented yet')

    def argmax(self, axis=None, out=None):
        """
        Masked version of argmax is not implemented yet.
        """
        raise NotImplementedError('masked version of argmax is not implemented yet')

    def argmin(self, axis=None, out=None):
        """
        Masked version of argmin is not implemented yet.
        """
        raise NotImplementedError('masked version of argmin is not implemented yet')

    def argsort(self, axis=-1, kind=None, order=None):
        """
        Masked version of argsort is not implemented yet.
        """
        raise NotImplementedError('masked version of argsort is not implemented yet')

    def clip(self, a_min, a_max, out=None, **kwargs):
        """
        Masked version of clip is not implemented yet.
        """
        raise NotImplementedError('masked version of clip is not implemented yet')

    def conj(self, out=None, where=True, casting='same_kind', order='K',
             dtype=None, subok=False):
        """
        Masked version of conj is not implemented yet.
        """
        raise NotImplementedError('masked version of conj is not implemented yet')

    def conjugate(self, out=None, where=True, casting='same_kind', order='K',
                  dtype=None, subok=False):
        """
        Masked version of conjugate is not implemented yet.
        """
        raise NotImplementedError('masked version of conjugate is not implemented yet')

    def cumsum(self, axis=None, dtype=None, out=None):
        """
        Masked version of cumsum is not implemented yet.
        """
        raise NotImplementedError('masked version of cumsum is not implemented yet')

    def dot(self, b, out=None):
        """
        Masked version of dot is not implemented yet.
        """
        raise NotImplementedError('masked version of dot is not implemented yet')

    def max(self, axis=None, out=None, keepdims=False, initial=nlcpy._NoValue,
            where=True):
        """
        Masked version of max is not implemented yet.
        """
        raise NotImplementedError('masked version of max is not implemented yet')

    def min(self, axis=None, out=None, keepdims=False, initial=nlcpy._NoValue,
            where=True):
        """
        Masked version of min is not implemented yet.
        """
        raise NotImplementedError('masked version of min is not implemented yet')

    def mean(self, axis=None, dtype=None, out=False, keepdims=nlcpy._NoValue):
        """
        Masked version of mean is not implemented yet.
        """
        raise NotImplementedError('masked version of mean is not implemented yet')

    def nonzero(self):
        """
        Masked version of nonzero is not implemented yet.
        """
        raise NotImplementedError('masked version of nonzero is not implemented yet')

    def prod(self, axis=None, dtype=None, out=None, keepdims=False,
             initial=nlcpy._NoValue, where=True):
        """
        Masked version of prod is not implemented yet.
        """
        raise NotImplementedError('masked version of prod is not implemented yet')

    def ptp(self, axis=None, out=None, keepdims=nlcpy._NoValue):
        """
        Masked version of ptp is not implemented yet.
        """
        raise NotImplementedError('masked version of ptp is not implemented yet')

    def sort(self, axis=-1, kind=None, order=None):
        """
        Masked version of sort is not implemented yet.
        """
        raise NotImplementedError('masked version of sort is not implemented yet')

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=nlcpy._NoValue):
        """
        Masked version of std is not implemented yet.
        """
        raise NotImplementedError('masked version of std is not implemented yet')

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=0,
            where=True):
        """
        Masked version of sum is not implemented yet.
        """
        raise NotImplementedError('masked version of sum is not implemented yet')

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=nlcpy._NoValue):
        """
        Masked version of var is not implemented yet.
        """
        raise NotImplementedError('masked version of var is not implemented yet')

    def take(self, indices, axis=None, out=None, mode='raise'):
        if mode != 'raise':
            raise NotImplementedError('mode is not supported yet')
        (_data, _mask) = (self._data, self._mask)
        cls = type(self)
        # Make sure the indices are not masked
        maskindices = getmask(indices)
        if maskindices is not nomask:
            indices = indices.filled(0)
        # Get the data, promoting scalars to 0d arrays with [...] so that
        # .view works correctly
        if out is None:
            out = _data.take(indices, axis=axis)[...].view(cls)
        elif isinstance(out, MaskedArray):
            nlcpy.take(_data, indices, axis=axis, mode=mode, out=out._data)
        else:
            nlcpy.take(_data, indices, axis=axis, mode=mode, out=out)

        # Get the mask
        if isinstance(out, MaskedArray):
            if _mask is nomask:
                outmask = maskindices
            else:
                outmask = _mask.take(indices, axis=axis)
                outmask |= maskindices
            out.__setmask__(outmask)
        # demote 0d arrays back to scalars, for consistency with ndarray.take
        return out[()]

    # Array methods
    copy = _arraymethod('copy')
    diagonal = _arraymethod('diagonal')
    flatten = _arraymethod('flatten')
    repeat = _arraymethod('repeat')
    squeeze = _arraymethod('squeeze')
    swapaxes = _arraymethod('swapaxes')
    T = property(fget=lambda self: self.transpose())
    transpose = _arraymethod('transpose')

    def tolist(self, fill_value=None):
        """ Returns the data portion of the masked array as a hierarchical Python list.

        Data items are converted to the nearest compatible Python type.
        Masked values are converted to `fill_value`. If `fill_value` is None,
        the corresponding entries in the output list will be ``None``.

        Parameters
        ----------
        fill_value : scalar, optional
            The value to use for invalid entries. Default is None.

        Returns
        -------
        result : list
            The Python list representation of the masked array.

        Examples
        --------
        >>> import nlcpy as vp
        >>> x = vp.ma.array([[1,2,3], [4,5,6], [7,8,9]], mask=[0] + [1,0]*4)
        >>> x.tolist()
        [[1, None, 3], [None, 5, None], [7, None, 9]]
        >>> x.tolist(-999)
        [[1, -999, 3], [-999, 5, -999], [7, -999, 9]]
        """
        return self.get().tolist(fill_value)

    def tobytes(self, fill_value=None, order='C'):
        """ Returns the array data as a string containing the raw bytes in the array.

        The array is filled with a fill value before the string conversion.

        Parameters
        ----------
        fill_value : scalar, optional
            Value used to fill in the masked values. Default is None, in which
            case `MaskedArray.fill_value` is used.
        order : {'C','F','A'}, optional
            Order of the data item in the copy. Default is 'C'.
            - 'C'   -- C order (row major).
            - 'F'   -- Fortran order (column major).
            - 'A'   -- Any, current order of array.
            - None  -- Same as 'A'.

        See Also
        --------
        nlcpy.ndarray.tobytes : Constructs Python bytes containing the raw data
            bytes in the array.
        tolist : Returns the data portion of the masked array as a hierarchical
            Python list.

        Notes
        -----
        As for :func:`nlcpy.ndarray.tobytes`, information about the shape, dtype, etc.,
        but also about `fill_value`, will be lost.

        Examples
        --------
        >>> import nlcpy as vp
        >>> x = vp.ma.array(vp.array([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
        >>> x.tobytes()
        b'\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00?B\\x0f\\x00\\x00\\x00\\x00\\x00?B\\x0f\\x00\\x00\\x00\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00'
        """
        return self.get().tobytes(fill_value, order)

    def astype(self, dtype, order='K', casting=None, subok=None, copy=True):
        if order is None:
            order = 'K'
        order_bchar = internal._normalize_order(order)
        order_char = chr(order_bchar)
        if casting is not None:
            raise NotImplementedError('casting is not supported yet')
        if subok is not None:
            raise NotImplementedError('subok is not supported yet')

        dtype = _dtype.get_dtype(dtype)
        if dtype == self.dtype:
            if not copy and (
                    order_char == 'K' or
                    order_char == 'A' and (
                        self._c_contiguous or self._f_contiguous) or
                    order_char == 'C' and self._c_contiguous or
                    order_char == 'F' and self._f_contiguous):
                return self

        order_bchar = core._update_order_char(self, order_bchar)
        order_char = chr(order_bchar)
        newarray = array(
            self.data, dtype=dtype, order=order_char,
            fill_value=self._fill_value, mask=self._mask)
        return newarray

    # -------------------------------------------------------------------------
    # nlcpy original attributes and methods
    # -------------------------------------------------------------------------
    def get(self, order='C'):
        data = self._data.get(order)
        mask = self._mask
        fill_value = self._fill_value
        if isinstance(mask, ndarray):
            mask = mask.get(order)
        if isinstance(fill_value, ndarray):
            fill_value = fill_value.get(order)
        hardmask = self._hardmask
        return numpy.ma.array(
            data, mask=mask, fill_value=fill_value, hard_mask=hardmask)

    def __getattr__(self, attr):
        raise AttributeError(
            "'nlcpy.ma.MaskedArray' object has no attribute '{}'.".format(attr))


masked_array = MaskedArray


class _MaskedUFunc:
    def __init__(self, ufunc):
        self.f = ufunc

    def __str__(self):
        return f"Masked version of {self.f}"


class _DomainSafeDivide:
    """
    Define a domain for safe division.
    """

    def __init__(self, tolerance=None):
        self.tolerance = tolerance

    def __call__(self, a, b):
        # Delay the selection of the tolerance to here in order to reduce numpy
        # import times. The calculation of these parameters is a substantial
        # component of numpy's import time.
        if self.tolerance is None:
            self.tolerance = numpy.finfo(float).tiny
        # don't call ma ufuncs from __array_wrap__ which would fail for scalars
        a, b = nlcpy.asarray(a), nlcpy.asarray(b)
        return nlcpy.absolute(a) * self.tolerance >= nlcpy.absolute(b)


class _MaskedBinaryOperation(_MaskedUFunc):
    def __init__(self, mbfunc, fillx=0, filly=0):
        super().__init__(mbfunc)
        self.fillx = fillx
        self.filly = filly
        ufunc_domain[mbfunc] = None
        ufunc_fills[mbfunc] = (fillx, filly)

    def __call__(self, a, b, *args, **kwargs):
        (da, db) = (getdata(a), getdata(b))
        (ma, mb) = (getmask(a), getmask(b))
        if ma is nomask:
            if mb is nomask:
                m = nomask
            else:
                m = nlcpy.logical_or(getmaskarray(a), mb)
        elif mb is nomask:
            m = nlcpy.logical_or(ma, getmaskarray(b))
        else:
            m = nlcpy.logical_or(ma, mb)
        result = self.f(da, db, *args, **kwargs)
        if not numpy.isscalar(a):
            nlcpy.copyto(result, da, where=m)
        if not result.ndim:
            return masked if m else result

        masked_result = array(result, mask=m)
        if isinstance(a, MaskedArray):
            masked_result._update_from(a)
        elif isinstance(b, MaskedArray):
            masked_result._update_from(b)
            masked_result._sharedmask = False
        return masked_result


add = _MaskedBinaryOperation(ufunc_op.add)
subtract = _MaskedBinaryOperation(ufunc_op.subtract)
multiply = _MaskedBinaryOperation(ufunc_op.multiply)


class _DomainedBinaryOperation(_MaskedUFunc):
    """Defines binary operations that have a domain, like divide.

    They have no reduce, outer or accumulate.

    Parameters
    ----------
    mbfunc : function
        The function for which to define a masked version. Made available
        as ``_DomainedBinaryOperation.f``.
    domain : class instance
        Default domain for the function. Should be one of the ``_Domain*``
        classes.
    fillx : scalar, optional
        Filling value for the first argument, default is 0.
    filly : scalar, optional
        Filling value for the second argument, default is 0.
    """

    def __init__(self, dbfunc, domain, fillx=0, filly=0):
        """abfunc(fillx, filly) must be defined.
           abfunc(x, filly) = x for all x to enable reduce.
        """
        super(_DomainedBinaryOperation, self).__init__(dbfunc)
        self.domain = domain
        self.fillx = fillx
        self.filly = filly
        ufunc_domain[dbfunc] = domain
        ufunc_fills[dbfunc] = (fillx, filly)

    def __call__(self, a, b, *args, **kwargs):
        "Execute the call behavior."
        # Get the data
        (da, db) = (getdata(a), getdata(b))
        # Get the result
        result = self.f(da, db, *args, **kwargs)
        # Get the mask as a combination of the source masks and invalid
        m1 = getmask(a)
        m2 = getmask(b)
        if m1 is nomask and m2 is nomask:
            m = nlcpy.zeros_like(result, dtype=numpy.bool_)
            m1 = m
            m2 = m
        else:
            m = nlcpy.empty_like(result, dtype=numpy.bool_)
            if m1 is nomask:
                m1 = m2
            elif m2 is nomask:
                m2 = m1
        request._push_request(
            "nlcpy_domain_mask",
            "mask_op",
            (m1, m2, result, m),
        )
        # Apply the domain
        domain = ufunc_domain.get(self.f, None)
        if domain is not None:
            m |= domain(da, db)
        # Take care of the scalar case first
        if not m.ndim:
            if m:
                return masked
            else:
                return result
        # When the mask is True, put back da if possible
        # any errors, just abort; impossible to guarantee masked values
        if type(a) is MaskedArray:
            try:
                nlcpy.copyto(result, 0, casting='unsafe', where=m)
            # avoid using "*" since this may be overlaid
                masked_da = nlcpy.multiply(m, da)
            # only add back if it can be cast safely
                if numpy.can_cast(masked_da.dtype, result.dtype, casting='safe'):
                    result += masked_da
            except Exception:
                pass

        # Transforms to a (subclass of) MaskedArray
        masked_result = result.view(MaskedArray)
        masked_result._mask = m
        if isinstance(a, MaskedArray):
            masked_result._update_from(a)
        elif isinstance(b, MaskedArray):
            masked_result._update_from(b)
            masked_result._sharedmask = False
        return masked_result


divide = _DomainedBinaryOperation(ufunc_op.divide, _DomainSafeDivide(), 0, 1)
true_divide = _DomainedBinaryOperation(ufunc_op.true_divide, _DomainSafeDivide(), 0, 1)


def filled(a, fill_value=None):
    """Returns input as an array with masked data replaced by a fill value.

    If `a` is not a `MaskedArray`, `a` itself is returned.
    If `a` is a `MaskedArray` and `fill_value` is None, `fill_value` is set to
    ``a.fill_value``.

    Parameters
    ----------
    a : MaskedArray or array_like
        An input object.
    fill_value : array_like, optional.
        Can be scalar or non-scalar. If non-scalar, the
        resulting filled array should be broadcastable
        over input array. Default is None.

    Returns
    -------
    a : ndarray
        The filled array.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.ma.array(vp.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                   [1, 0, 0],
    ...                                                   [0, 0, 0]])
    >>> x.filled()
    array([[999999,      1,      2],
           [999999,      4,      5],
           [     6,      7,      8]])
    >>> x.filled(fill_value=333)
    array([[333,   1,   2],
           [333,   4,   5],
           [  6,   7,   8]])
    >>> x.filled(fill_value=vp.arange(3))
    array([[0, 1, 2],
           [0, 4, 5],
           [6, 7, 8]])
    """
    if hasattr(a, 'filled'):
        return a.filled(fill_value)
    elif isinstance(a, ndarray):
        return a
    else:
        return nlcpy.array(a)


def _check_fill_value(fill_value, ndtype):
    """ Private function validating the given `fill_value` for the given dtype.

    If fill_value is None, it is set to the default corresponding to the dtype.
    If fill_value is not None, its value is forced to the given dtype.
    The result is always a 0d array.
    """
    ndtype = nlcpy.dtype(ndtype)
    if fill_value is None:
        fill_value = numpy.array(default_fill_value(ndtype))
    else:
        try:
            if numpy.isscalar(fill_value):
                fill_value = numpy.array(fill_value, copy=False, dtype=ndtype)
            else:
                fill_value = nlcpy.array(fill_value, copy=False, dtype=ndtype)
        except (OverflowError, ValueError) as e:
            err_msg = "Cannot convert fill_value %s to dtype %s"
            raise TypeError(err_msg % (fill_value, ndtype)) from e
    return fill_value


def _get_dtype_of(obj):
    """ Convert the argument for *_fill_value into a dtype """
    if isinstance(obj, nlcpy.dtype):
        return obj
    elif hasattr(obj, 'dtype'):
        return obj.dtype
    else:
        return nlcpy.asanyarray(obj).dtype


def default_fill_value(obj):
    """Returns the default fill value for the argument object.

    The default filling value depends on the datatype of the input
    array or the type of the input scalar:

       ========  ========
       datatype  default
       ========  ========
       bool      True
       int       999999
       float     1.e20
       complex   1.e20+0j
       ========  ========

    For structured types, a structured scalar is returned, with each field the
    default fill value for its type.

    For subarray types, the fill value is an array of the same size containing
    the default scalar fill value.

    Parameters
    ----------
    obj : ndarray, dtype or scalar
        The array data-type or scalar for which the default fill value
        is returned.

    Returns
    -------
    fill_value : scalar
        The default fill value.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.ma.default_fill_value(1)
    999999
    >>> vp.ma.default_fill_value(vp.array([1.1, 2., vp.pi]))
    1e+20
    >>> vp.ma.default_fill_value(vp.dtype(complex))
    (1e+20+0j)
    """
    dtype = _get_dtype_of(obj)
    return default_filler.get(dtype.kind, '?')


def getdata(a, subok=None):
    """Returns the data of a masked array as an ndarray.

    Returns the data of `a` (if any) as an ndarray if `a` is a ``MaskedArray``,
    else return `a` as a ndarray or subclass (depending on `subok`) if not.

    Parameters
    ----------
    a : array_like
        Input ``MaskedArray``, alternatively a ndarray or a subclass thereof.
    subok : bool
        Whether to force the output to be a `pure` ndarray (False) or to
        return a subclass of ndarray if appropriate.
        subok=True is Not Implemented.

    See Also
    --------
    getmask : Returns the mask of a masked array, or nomask.
    getmaskarray : Returns the mask of a masked array, or full array of False.

    Examples
    --------
    >>> import nlcpy.ma as ma
    >>> mask = [[False, True], [False, False]]
    >>> a = ma.array([[1, 2], [3, 4]], mask=mask, fill_value=2)
    >>> a
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=2)
    >>> ma.getdata(a)
    array([[1, 2],
           [3, 4]])

    Equivalently use the ``MaskedArray`` `data` attribute.

    >>> a.data
    array([[1, 2],
           [3, 4]])
    """
    if subok is not None:
        raise NotImplementedError('subok is not implemented yet')
    subok = False

    if numpy.isscalar(a):
        return a

    try:
        data = a._data
    except AttributeError:
        data = nlcpy.array(a, copy=False, subok=subok)
    return data


def getmask(a):
    """Returns the mask of a masked array, or nomask.

    Returns the mask of `a` as an ndarray if `a` is a `MaskedArray` and the
    mask is not `nomask`, else return `nomask`. To guarantee a full array
    of booleans of the same shape as a, use `getmaskarray`.

    Parameters
    ----------
    a : array_like
        Input `MaskedArray` for which the mask is required.

    See Also
    --------
    getdata : Returns the data of a masked array as an ndarray.
    getmaskarray : Returns the mask of a masked array, or full array of False.

    Examples
    --------
    >>> import nlcpy.ma as ma
    >>> mask=[[False, True], [False, False]]
    >>> a = ma.array([[1, 2],[3, 4]], mask=mask, fill_value=2)
    >>> a
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=2)
    >>> ma.getmask(a)
    array([[False,  True],
           [False, False]])

    Equivalently use the `MaskedArray` `mask` attribute.

    >>> a.mask
    array([[False,  True],
           [False, False]])

    Result when mask == `nomask`

    >>> b = ma.masked_array([[1,2],[3,4]])
    >>> b
    masked_array(
      data=[[1, 2],
            [3, 4]],
      mask=False,
      fill_value=999999)
    >>> ma.nomask
    False
    >>> ma.getmask(b) == ma.nomask
    True
    >>> b.mask == ma.nomask
    True
    """
    return getattr(a, '_mask', nomask)


def _shrink_mask(m):
    """
    Shrinks a mask to nomask if possible
    """
    if not m.any():
        return nomask
    else:
        return m


def make_mask(m, copy=False, shrink=True, dtype=MaskType):
    """Creates a boolean mask from an array.

    Returns `m` as a boolean mask, creating a copy if necessary or requested.
    The function can accept any sequence that is convertible to integers,
    or ``nomask``.  Does not require that contents must be 0s and 1s, values
    of 0 are interpreted as False, everything else as True.

    Parameters
    ----------
    m : array_like
        Potential mask.
    copy : bool, optional
        Whether to return a copy of `m` (True) or `m` itself (False).
    shrink : bool, optional
        Whether to shrink `m` to ``nomask`` if all its values are False.
    dtype : dtype, optional
        Data-type of the output mask. By default, the output mask has a
        dtype of MaskType (bool). If the dtype is flexible, each field has
        a boolean dtype. This is ignored when `m` is ``nomask``, in which
        case ``nomask`` is always returned.

    Returns
    -------
    result : ndarray
        A boolean mask derived from `m`.

    Examples
    --------
    >>> import nlcpy as vp
    >>> import nlcpy.ma as ma
    >>> m = [True, False, True, True]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True])
    >>> m = [1, 0, 1, 1]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True])
    >>> m = [1, 0, 2, -3]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True])

    Effect of the `shrink` parameter.

    >>> m = vp.zeros(4)
    >>> m
    array([0., 0., 0., 0.])
    >>> ma.make_mask(m)
    False
    >>> ma.make_mask(m, shrink=False)
    array([False, False, False, False])
    """
    if m is nomask:
        return nomask

    # Fill the mask in case there are missing data; turn it into an ndarray.
    result = nlcpy.array(filled(m, True), copy=copy, dtype=MaskType)
    # Bas les masques !
    if shrink:
        result = _shrink_mask(result)
    return result


def make_mask_none(newshape, dtype=None):
    """Returns a boolean mask of the given shape, filled with False.

    This function returns a boolean ndarray with all entries False, that can
    be used in common mask manipulations. If a complex dtype is specified, the
    type of each field is converted to a boolean type.

    Parameters
    ----------
    newshape : tuple
        A tuple indicating the shape of the mask.
    dtype : {None, dtype}, optional
        If None, use a MaskType instance. Otherwise, use a new datatype with
        the same fields as `dtype`, converted to boolean types.

    Returns
    -------
    result : ndarray
        An ndarray of appropriate shape and dtype, filled with False.

    See Also
    --------
    make_mask : Creates a boolean mask from an array.

    Examples
    --------
    >>> import nlcpy.ma as ma
    >>> ma.make_mask_none((3,))
    array([False, False, False])
    """
    return nlcpy.zeros(newshape, dtype=MaskType)


def mask_or(m1, m2, copy=False, shrink=True):
    """Combines two masks with the ``logical_or`` operator.

    The result may be a view on `m1` or `m2` if the other is `nomask`
    (i.e. False).

    Parameters
    ----------
    m1, m2 : array_like
        Input masks.
    copy : bool, optional
        If copy is False and one of the inputs is `nomask`, return a view
        of the other input mask. Defaults to False.
    shrink : bool, optional
        Whether to shrink the output to `nomask` if all its values are
        False. Defaults to True.

    Returns
    -------
    mask : output mask
        The result masks values that are masked in either `m1` or `m2`.

    Raises
    ------
    ValueError
        If `m1` and `m2` have different flexible dtypes.

    Examples
    --------
    >>> import nlcpy as vp
    >>> m1 = vp.ma.make_mask([0, 1, 1, 0])
    >>> m2 = vp.ma.make_mask([1, 0, 0, 0])
    >>> vp.ma.mask_or(m1, m2)
    array([ True,  True,  True, False])
    """

    if (m1 is nomask) or (m1 is False):
        dtype = getattr(m2, 'dtype', MaskType)
        return make_mask(m2, copy=copy, shrink=shrink, dtype=dtype)
    if (m2 is nomask) or (m2 is False):
        dtype = getattr(m1, 'dtype', MaskType)
        return make_mask(m1, copy=copy, shrink=shrink, dtype=dtype)
    if m1 is m2 and is_mask(m1):
        return m1
    (dtype1, dtype2) = (getattr(m1, 'dtype', None), getattr(m2, 'dtype', None))
    if dtype1 != dtype2:
        raise ValueError("Incompatible dtypes '%s'<>'%s'" % (dtype1, dtype2))
    return make_mask(nlcpy.logical_or(m1, m2), copy=copy, shrink=shrink)


def getmaskarray(arr):
    """Returns the mask of a masked array, or full boolean array of False.

    Returns the mask of `arr` as an ndarray if `arr` is a `MaskedArray` and
    the mask is not `nomask`, else return a full boolean array of False of
    the same shape as `arr`.

    Parameters
    ----------
    arr : array_like
        Input `MaskedArray` for which the mask is required.

    See Also
    --------
    getmask : Returns the mask of a masked array, or nomask.
    getdata : Returns the data of a masked array as an ndarray.

    Examples
    --------
    >>> import nlcpy.ma as ma
    >>> mask = [[False, True], [False, False]]
    >>> a = ma.array([[1, 2],[3, 4]], mask=mask, fill_value=2)
    >>> a
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=2)
    >>> ma.getmaskarray(a)
    array([[False,  True],
           [False, False]])

    Result when mask == ``nomask``

    >>> b = ma.masked_array([[1,2],[3,4]])
    >>> b
    masked_array(
      data=[[1, 2],
            [3, 4]],
      mask=False,
      fill_value=999999)
    >>> ma.getmaskarray(b)
    array([[False, False],
           [False, False]])
    """
    mask = getmask(arr)
    if mask is nomask:
        mask = make_mask_none(nlcpy.shape(arr), getattr(arr, 'dtype', None))
    return mask


def is_mask(m):
    """Returns True if m is a valid, standard mask.

    This function does not check the contents of the input, only that the
    type is MaskType. In particular, this function returns False if the
    mask has a flexible dtype.

    Parameters
    ----------
    m : array_like
        Array to test.

    Returns
    -------
    result : bool
        True if `m.dtype.type` is MaskType, False otherwise.

    Examples
    --------
    >>> import nlcpy as vp
    >>> import nlcpy.ma as ma
    >>> mask = [True, False, True, False, False]
    >>> m = ma.array([0, 1, 0, 2, 3], mask=mask, fill_value=0)
    >>> m
    masked_array(data=[--, 1, --, 2, 3],
                 mask=[ True, False,  True, False, False],
           fill_value=0)
    >>> ma.is_mask(m)
    False
    >>> ma.is_mask(m.mask)
    True

    Input must be an ndarray (or have similar attributes)
    for it to be considered a valid mask.

    >>> m = [False, True, False]
    >>> ma.is_mask(m)
    False
    >>> m = vp.array([False, True, False])
    >>> m
    array([False,  True, False])
    >>> ma.is_mask(m)
    True
    """
    try:
        return m.dtype.type is MaskType
    except AttributeError:
        return False


def array(data, dtype=None, copy=False, order=None,
          mask=nomask, fill_value=None, keep_mask=True, hard_mask=False,
          shrink=True, subok=None, ndmin=0):
    """ Shortcut to MaskedArray.

    The options are in a different order for convenience and backwards
    compatibility.
    """
    return MaskedArray(data, mask=mask, dtype=dtype, copy=copy,
                       subok=subok, keep_mask=keep_mask,
                       hard_mask=hard_mask, fill_value=fill_value,
                       ndmin=ndmin, shrink=shrink, order=order)


def take(a, indices, axis=None, out=None, mode='raise'):
    """
    """
    a = masked_array(a)
    return a.take(indices, axis=axis, out=out, mode=mode)


def transpose(a, axes=None):
    """ Permutes the dimensions of an array.

    This function is exactly equivalent to :func:`nlcpy.transpose`.

    See Also
    --------
    nlcpy.transpose : Equivalent function in top-level NLCPy module.

    Examples
    --------
    >>> import nlcpy as vp
    >>> import nlcpy.ma as ma
    >>> mask = [[False, False], [False, True]]
    >>> x = ma.array([[0, 1], [2, 3]], mask=mask)
    >>> x
    masked_array(
      data=[[0, 1],
            [2, --]],
      mask=[[False, False],
            [False,  True]],
      fill_value=999999)
    >>> ma.transpose(x)
    masked_array(
      data=[[0, 2],
            [1, --]],
      mask=[[False, False],
            [False,  True]],
      fill_value=999999)
    """
    # We can't use 'frommethod', as 'transpose' doesn't take keywords
    try:
        return a.transpose(axes)
    except AttributeError:
        return ndarray(a, copy=False).transpose(axes).view(MaskedArray)


def reshape(a, new_shape, order='C'):
    """ Returns an array containing the same data with a new shape.

    Refer to :func:`MaskedArray.reshape` for full documentation.

    See Also
    --------
    MaskedArray.reshape : equivalent function
    """
    # We can't use 'frommethod', it whine about some parameters. Dmmit.
    try:
        return a.reshape(new_shape, order=order)
    except AttributeError:
        _tmp = ndarray(a, copy=False).reshape(new_shape, order=order)
        return _tmp.view(MaskedArray)


def resize(x, new_shape):
    """ Returns a new masked array with the specified size and shape.

    This is the masked equivalent of the :func:`nlcpy.resize` function. The new
    array is filled with repeated copies of `x` (in the order that the
    data are stored in memory). If `x` is masked, the new array will be
    masked, and the new mask will be a repetition of the old one.

    See Also
    --------
    nlcpy.resize : Equivalent function in the top level NLCPy module.

    Examples
    --------
    >>> import nlcpy as vp
    >>> import nlcpy.ma as ma
    >>> a = ma.array([[1, 2] ,[3, 4]])
    >>> a[0, 1] = ma.masked
    >>> a
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=999999)
    >>> ma.resize(a, (3, 3))
    masked_array(
      data=[[1, --, 3],
            [4, 1, --],
            [3, 4, 1]],
      mask=[[False,  True, False],
            [False, False,  True],
            [False, False, False]],
      fill_value=999999)

    A MaskedArray is always returned, regardless of the input type.

    >>> a = vp.array([[1, 2] ,[3, 4]])
    >>> ma.resize(a, (3, 3))
    masked_array(
      data=[[1, 2, 3],
            [4, 1, 2],
            [3, 4, 1]],
      mask=False,
      fill_value=999999)
    """
    # We can't use _frommethods here, as N.resize is notoriously whiny.
    m = getmask(x)
    if m is not nomask:
        m = nlcpy.resize(m, new_shape)
    data = x if isinstance(x, ndarray) else x._data
    result = nlcpy.resize(data, new_shape).view(MaskedArray)
    result._sharedmask = False
    if result.ndim:
        result._mask = m
    return result


def asarray(a, dtype=None, order=None):
    """ Converts the input to a masked array of the given data-type.

    No copy is performed if the input is already an `ndarray`. If `a` is
    a subclass of `MaskedArray`, a base class `MaskedArray` is returned.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to a masked array. This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists, ndarrays and masked arrays.
    dtype : dtype, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major ('C') or column-major ('FORTRAN') memory
        representation.  Default is 'C'.

    Returns
    -------
    out : MaskedArray
        Masked array interpretation of `a`.

    See Also
    --------
    asanyarray : Converts the input to a masked array, conserving subclasses.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(10.).reshape(2, 5)
    >>> x
    array([[0., 1., 2., 3., 4.],
           [5., 6., 7., 8., 9.]])
    >>> vp.ma.asarray(x)
    masked_array(
      data=[[0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.]],
      mask=False,
      fill_value=1e+20)
    >>> type(np.ma.asarray(x))
    <class 'nlcpy.ma.core.MaskedArray'>
    """
    order = order or 'C'
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True, order=order)


def asanyarray(a, dtype=None):
    """ Converts the input to a masked array, conserving subclasses.

    If `a` is a subclass of `MaskedArray`, its class is conserved.
    No copy is performed if the input is already an `ndarray`.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.
    dtype : dtype, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major ('C') or column-major ('FORTRAN') memory
        representation.  Default is 'C'.

    Returns
    -------
    out : MaskedArray
        MaskedArray interpretation of `a`.

    See Also
    --------
    asarray : Converts the input to a masked array of the given data-type.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.arange(10.).reshape(2, 5)
    >>> x
    array([[0., 1., 2., 3., 4.],
           [5., 6., 7., 8., 9.]])
    >>> vp.ma.asanyarray(x)
    masked_array(
      data=[[0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.]],
      mask=False,
      fill_value=1e+20)
    >>> type(vp.ma.asanyarray(x))
    <class 'nlcpy.ma.core.MaskedArray'>
    """
    if isinstance(a, MaskedArray) and (dtype is None or dtype == a.dtype):
        return a
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True)


def copy(a, order='K'):
    """ Returns a copy of the array.

    Refer to :func:`nlcpy.ma.MaskedArray.copy` for full documentation.
    """
    marr = asanyarray(a)
    return marr.copy(order)


def diagonal(a, offset=0, axis1=0, axis2=1):
    """ Returns specified diagonals.

    Refer to :func:`nlcpy.ma.MaskedArray.diagonal` for full documentation.
    """
    marr = asanyarray(a)
    return marr.diagonal(offset, axis1, axis2)


def harden_mask(a):
    """ Forces the mask to hard.

    Refer to :func:`nlcpy.ma.MaskedArray.harden_mask` for full documentation.
    """
    marr = asanyarray(a)
    return marr.harden_mask()


def ravel(a, order='C'):
    """ Returns a contiguous flattened array.

    Refer to :func:`nlcpy.ma.MaskedArray.ravel` for full documentation.
    """
    marr = asanyarray(a)
    return marr.ravel(order)


def repeat(a, repeats, axis=None):
    """ Repeats elements of an array.

    Refer to :func:`nlcpy.ma.MaskedArray.repeat` for full documentation.
    """
    marr = asanyarray(a)
    return marr.repeat(repeats, axis)


def soften_mask(a):
    """ Forces the mask to soft.

    Refer to :func:`nlcpy.ma.MaskedArray.soften_mask` for full documentation.
    """
    marr = asanyarray(a)
    return marr.soften_mask()


def swapaxes(a, axis1, axis2):
    """ Interchanges two axes of an array.

    Refer to :func:`nlcpy.swapaxes` for full documentation.
    """
    marr = asanyarray(a)
    return marr.swapaxes(axis1, axis2)

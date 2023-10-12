#
# * The source code in this file is based on the soure code of CuPy.
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

# distutils: language = c++

import ctypes
import warnings
import time
import atexit

from libc.stdint cimport *
from libcpp.vector cimport vector
from cpython.object cimport *

import nlcpy
from nlcpy.veo cimport _veo
from nlcpy.core import flags
from nlcpy.core cimport internal
from nlcpy.core cimport vememory
from nlcpy.core cimport manipulation
from nlcpy.core cimport sorting
from nlcpy.core cimport indexing
from nlcpy.core cimport searching
from nlcpy.core cimport math
from nlcpy.core cimport dtype as _dtype
from nlcpy.core import error
from nlcpy.ufuncs import operations as ufunc_op
from nlcpy.request.ve_kernel cimport NLCPY_MAXNDIM
from nlcpy.error_handler import error_handler
from nlcpy.statistics.order import ptp
from nlcpy.statistics.average import mean
from nlcpy.statistics.average import var
from nlcpy.statistics.average import std
from nlcpy.sca.description cimport description
from nlcpy.venode._venode cimport VE
from nlcpy.venode._venode cimport _find_venode_from_proc_handle
from nlcpy._environment import _is_numpy_wrap_enabled
import numpy
cimport numpy as cnp


cdef class ndarray:
    """N-dimensional array class for VE.

    An array object represents a multidimensional, homogeneous array of fixed-size items.
    An associated data-type object describes the format of each element in the array (its
    byte-order, how many bytes it occupies in memory, whether it is an integer, a
    floating point number, or something else, etc.) Arrays should be constructed using
    (refer to the See Also section below). The parameters given here refer to a low-level
    method (ndarray(...)) for instantiating an array.

    Parameters
    ----------
    shape : tuple of ints
        Shape of created array.
    dtype : dtype, optional
        Any object that can be interpreted as a numpy or nlcpy data type.
    strides : tuple of ints
        Strides of data in memory.
    order : {'C','F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.
    veo_hmem : int, optional
        VEO heterogeneous memory.

    See Also
    --------
    array : Constructs an array.
    zeros : Creates an array, each element of
        which is zero.
    empty : Creates an array, but leave its
        allocated memory unchanged.

    Examples
    --------

    This example illustrate the low-level ndarray constructor. Refer to the See Also
    section above for easier ways of constructing an ndarray.

    >>> import nlcpy as vp
    >>> vp.ndarray(shape=(2,2), dtype=float, order='F') # doctest: +SKIP
    array([[0., 0.],     # may vary
           [0., 0.]])

    """

    def __init__(self, shape, dtype=float, strides=None, order='C', veo_hmem=None):
        cdef int64_t itemsize
        cdef tuple _shape = internal.get_size(shape)
        del shape

        if len(_shape) > NLCPY_MAXNDIM:
            raise ValueError('maximum supported dimension for an ndarray is %d, '
                             'found %d' % (NLCPY_MAXNDIM, len(_shape)))

        cdef int order_char = (
            b'C' if order is None
            else internal._normalize_order(order)
        )

        # set shape
        self._shape.reserve(len(_shape))
        for x in _shape:
            if x < 0:
                raise ValueError('negative dimensions are not allowed')
            self._shape.push_back(x)

        # dtype check
        if dtype in _dtype._nlcpy_not_supported_type_set:
            raise TypeError("dtype \'%s\' is not supported." % dtype)

        # set dtype
        # If input dtype character is 'q' or 'Q',
        # NLCPy converts the dtype into 'l' or 'L'.
        self.dtype, self.itemsize = _dtype.get_dtype_with_itemsize(dtype)

        # set strides
        if strides is not None:
            self._set_shape_and_strides(self._shape, strides, True, True)
        elif order_char == b'C':
            self._set_contiguous_strides(self.itemsize, True)
        elif order_char == b'F':
            self._set_contiguous_strides(self.itemsize, False)
        else:
            raise ValueError('Unsupported order \'%s\'' % order)

        if veo_hmem is None:
            # set address with allocating memory.
            self._venode = VE()
            self._is_pool, self.veo_hmem, self.ve_adr = vememory._alloc_mem(
                self.nbytes, self._venode)
            self._is_view = False
        else:
            # set address without allocating memory.
            if not _veo.VEO_HMEM.is_ve_addr(veo_hmem):
                raise MemoryError('veo_hmem was not allocated by veo_alloc_hmem().')
            self._venode = _find_venode_from_proc_handle(
                _veo.VEO_HMEM.get_proc_handle_from_hmem(veo_hmem))
            if self._venode is None:
                raise MemoryError('veo_hmem exists on unknown veo process.')
            self._is_pool = False
            self.veo_hmem = veo_hmem
            self.ve_adr = _veo.VEO_HMEM.get_hmem_addr(veo_hmem)
            self._is_view = True
            self._owndata = False

    def __dealloc__(self):
        if _exit_mode:
            return
        if self.veo_hmem != 0L:
            if self.base is None:
                # double check to avoid double free
                try:
                    if not self._is_view and self._venode.connected:
                        vememory._free_mem(self.veo_hmem, self._is_pool, self._venode)
                except AttributeError:
                    pass
                finally:
                    self.veo_hmem = 0L
                    self.ve_adr = 0L

    # -------------------------------------------------------------------------
    # getitem / setitem methods
    # -------------------------------------------------------------------------
    def __getitem__(self, slices):
        return indexing._ndarray_getitem(self, slices)

    def __setitem__(self, slices, value):
        indexing._ndarray_setitem(self, slices, value)

    # -------------------------------------------------------------------------
    # string representations
    # -------------------------------------------------------------------------
    def __repr__(self):
        return repr(self.get())

    def __str__(self):
        return str(self.get())

    # -------------------------------------------------------------------------
    # nlcpy.ndarray's attributes
    # -------------------------------------------------------------------------

    def __bool__(self):
        if self.size > 1:
            raise ValueError('The truth value of an array with more than'
                             'one element is ambiguous. Use a.any() or a.all()')
        return True if self.get() else False

    def __iter__(self):
        if self._shape.size() == 0:
            raise TypeError('iteration over a 0-d array')
        return (self[i] for i in range(self._shape[0]))

    def __len__(self):
        if self._shape.size() == 0:
            raise TypeError('len() of unsized object')
        return self._shape[0]

    def __int__(self):
        return int(self.get())

    def __float__(self):
        return float(self.get())

    def __complex__(self):
        return complex(self.get())

    def __index__(self):
        return self.get().__index__()

    def __reduce__(self):
        return array, (self.get(),)

    def __array__(self, dtype=None):
        if dtype is None or self.dtype == dtype:
            return self.get()
        else:
            return self.get().astype(dtype)

    __array_priority__ = 1.0

    property shape:
        """
        Lengths of each axis.
        """
        def __get__(self):
            return tuple(self._shape)

        def __set__(self, newshape):
            manipulation._ndarray_shape_setter(self, newshape)

    property strides:
        """
        Strides of each axis in bytes.
        """
        def __get__(self):
            return tuple(self._strides)

    property _ve_array:
        def __get__(self):
            buf = self._venode.request_manager._create_ve_array_buffer(self)
            return _veo.OnStack(buf, inout=_veo.INTENT_IN)

    property venode:
        """
        VENode object that ndarray exists on.
        """
        def __get__(self):
            return self._venode

    @property
    def __ve_array_interface__(self):
        """
        VE array interface for interoperating Python VE libraries.
        """
        _rm = self._venode.request_manager
        _rm._flush(sync=True)

        cdef dict desc = {
            'shape': self.shape,
            'typestr': self.dtype.str,
            'descr': self.dtype.descr,
        }
        desc['typestr'] = desc['typestr'][:2] + str(self.itemsize)
        desc['descr'] = [(desc['descr'][0][0], desc['typestr'])]
        cdef int ver = 1
        desc['version'] = ver
        if self._c_contiguous:
            desc['strides'] = None
        else:
            desc['strides'] = self.strides
        if self.size > 0:
            desc['data'] = (self.veo_hmem, False)
        else:
            desc['data'] = (0, False)
        desc['veo_ctxt'] = <uint64_t>self._venode.ctx.thr_ctxt
        return desc

    @property
    def ndim(self):
        """
        Number of dimensions.
        """
        return len(self._shape)

    @property
    def nbytes(self):
        """
        Total number of bytes for all elements.
        """
        if self.dtype is numpy.dtype('bool'):
            return self.size * numpy.dtype('i4').itemsize
        return self.size * self.itemsize

    @property
    def T(self):
        """
        Transposed array.
        """
        if self.ndim < 2:
            return self
        else:
            return manipulation._T(self)

    @property
    def flags(self):
        """
        Information about the memory layout of the array.
        """
        return flags.Flags(self._c_contiguous, self._f_contiguous, self.base is None)

    # -------------------------------------------------------------------------
    # ufunc operations
    # -------------------------------------------------------------------------

    def __pos__(x):
        return ufunc_op.positive(x)

    def __neg__(x):
        return ufunc_op.negative(x)

    def __invert__(x):
        return ufunc_op.invert(x)

    def __abs__(x):
        return ufunc_op.absolute(x)

    def __add__(x, y):
        if hasattr(y, '__radd__'):
            x_priority = getattr(x, '__array_priority__', -1)
            y_priority = getattr(y, '__array_priority__', -1)
            if y_priority > x_priority > 0:
                return y.__radd__(x)
        return ufunc_op.add(x, y)

    def __sub__(x, y):
        if hasattr(y, '__rsub__'):
            x_priority = getattr(x, '__array_priority__', -1)
            y_priority = getattr(y, '__array_priority__', -1)
            if y_priority > x_priority > 0:
                return y.__rsub__(x)
        return ufunc_op.subtract(x, y)

    def __mul__(x, y):
        if isinstance(y, description):
            return y.__mul__(x)
        if hasattr(y, '__rmul__'):
            x_priority = getattr(x, '__array_priority__', -1)
            y_priority = getattr(y, '__array_priority__', -1)
            if y_priority > x_priority > 0:
                return y.__rmul__(x)
        return ufunc_op.multiply(x, y)

    def __truediv__(x, y):
        if hasattr(y, '__rtruediv__'):
            x_priority = getattr(x, '__array_priority__', -1)
            y_priority = getattr(y, '__array_priority__', -1)
            if y_priority > x_priority > 0:
                return y.__rtruediv__(x)
        return ufunc_op.true_divide(x, y)

    def __floordiv__(x, y):
        return ufunc_op.floor_divide(x, y)

    def __mod__(x, y):
        return ufunc_op.mod(x, y)

    def __pow__(x, y, z):
        if z is None:
            return ufunc_op.power(x, y)
        else:
            u = ufunc_op.power(x, y)
            return ufunc_op.mod(u, z)

    def __and__(x, y):
        return ufunc_op.bitwise_and(x, y)

    def __xor__(x, y):
        return ufunc_op.bitwise_xor(x, y)

    def __or__(x, y):
        return ufunc_op.bitwise_or(x, y)

    def __rshift__(x, y):
        return ufunc_op.right_shift(x, y)

    def __lshift__(x, y):
        return ufunc_op.left_shift(x, y)

    def __matmul__(x, y):
        return ufunc_op.matmul(x, y)

    def __iadd__(x, y):
        if hasattr(y, '_data'):
            y = y._data
        return ufunc_op.add(x, y, out=x)

    def __isub__(x, y):
        if hasattr(y, '_data'):
            y = y._data
        return ufunc_op.subtract(x, y, out=x)

    def __imul__(x, y):
        if hasattr(y, '_data'):
            y = y._data
        return ufunc_op.multiply(x, y, out=x)

    def __itruediv__(x, y):
        if hasattr(y, '_data'):
            y = y._data
        return ufunc_op.true_divide(x, y, out=x)

    def __ifloordiv__(x, y):
        return ufunc_op.floor_divide(x, y, out=x)

    def __imod__(x, y):
        return ufunc_op.mod(x, y, out=x)

    def __ipow__(x, y):
        return ufunc_op.power(x, y, out=x)

    def __iand__(x, y):
        return ufunc_op.bitwise_and(x, y, out=x)

    def __ixor__(x, y):
        return ufunc_op.bitwise_xor(x, y, out=x)

    def __ior__(x, y):
        return ufunc_op.bitwise_or(x, y, out=x)

    def __irshift__(x, y):
        return ufunc_op.right_shift(x, y, out=x)

    def __ilshift__(x, y):
        return ufunc_op.left_shift(x, y, out=x)

    def __imatmul__(x, y):
        raise TypeError("In-place matrix multiplication is not (yet) "
                        "supported. Use 'a = a @ b' instead of 'a @= b'.")

    def __richcmp__(x, y, int op):
        if x is None or y is None:
            return
        if op == Py_LT:  # __lt__()
            return ufunc_op.less(x, y)
        if op == Py_LE:  # __le__()
            return ufunc_op.less_equal(x, y)
        if op == Py_EQ:  # __eq__()
            return ufunc_op.equal(x, y)
        if op == Py_NE:  # __ne__()
            return ufunc_op.not_equal(x, y)
        if op == Py_GT:  # __gt__()
            return ufunc_op.greater(x, y)
        if op == Py_GE:  # __ge__()
            return ufunc_op.greater_equal(x, y)
        return RuntimeError

    @property
    def real(self):
        """
        Real part.
        """
        return math._ndarray_real_getter(self)

    @real.setter
    def real(self, value):
        math._ndarray_real_setter(self, value)

    @property
    def imag(self):
        """
        Imaginary part.
        """
        return math._ndarray_imag_getter(self)

    @imag.setter
    def imag(self, value):
        math._ndarray_imag_setter(self, value)

    def clip(self, a_min, a_max, out=None, **kwargs):
        where = kwargs.pop('where', True)
        casting = kwargs.pop('casting', 'same_kind')
        order = kwargs.pop('order', 'K')
        dtype = kwargs.pop('dtype', None)
        return math._ndarray_clip(self, a_min, a_max, out, dtype, order, where, casting)

    # -------------------------------------------------------------------------
    # shape manipulation
    # -------------------------------------------------------------------------
    def reshape(self, *shape, order='C'):
        """Returns an array containing the same data with a new shape.

        Refer to :func:`nlcpy.reshape` for full documentation.

        Note
        ----
        Unlike the free function :func:`nlcpy.reshape`, this method on
        :func:`ndarray.reshape` allows the elements of the shape parameter
        to be passed in as separate arguments.
        For example, ``a.reshape(10, 11)`` is equivalent to ``a.reshape((10, 11))``.

        See Also
        --------
        nlcpy.reshape : Equivalent function.
        """
        return manipulation._ndarray_reshape(self, shape, order)

    def ravel(self, order='C'):
        """Returns a flattened array.

        Refer to :func:`nlcpy.ravel` for full documentation.

        See Also
        --------
        nlcpy.ravel : Equivalent function.

        """
        return manipulation._ndarray_ravel(self, order)

    def resize(self, *new_shape, refcheck=True):
        """Changes shape and size of array in-place.

        Parameters
        ----------
        new_shape : tuple of ints, or n ints
            Shape of resized array.
        refcheck : bool, optional
            If False, reference count will not be checked. Default is True.

        Returns
        -------
        None

        Note
        ----
        This reallocates space for the data area if necessary.

        Only contiguous arrays (data elements consecutive in memory) can be resized.

        The purpose of the reference count check is to make sure you do not use this
        array as a buffer for another Python object and then reallocate the memory.
        However, reference counts can increase in other ways so if you are sure that you
        have not shared the memory for this array with another Python object, then you
        may safely set `refcheck` to False.

        See Also
        --------
        nlcpy.resize : Returns a new array with the specified shape.

        Examples
        --------
        Shrinking an array: array is flattened (in the order that the data are stored in
        memory), resized, and reshaped:

        >>> import nlcpy as vp
        >>> a = vp.array([[0, 1], [2, 3]], order='C')
        >>> a.resize((2, 1))
        >>> a
        array([[0],
               [1]])
        >>> a = vp.array([[0, 1], [2, 3]], order='F')
        >>> a.resize((2, 1))
        >>> a
        array([[0],
               [2]])

        Enlarging an array: as above, but missing entries are filled with zeros:

        >>> b = vp.array([[0, 1], [2, 3]])
        >>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
        >>> b
        array([[0, 1, 2],
               [3, 0, 0]])

        Referencing an array prevents resizing...

        >>> c = a
        >>> a.resize((1, 1))   # doctest: +SKIP
        Traceback (most recent call last):
        ...
        ValueError: cannot resize an array that references or is referenced ...

        Unless *refcheck* is False:

        >>> a.resize((1, 2), refcheck=False)
        >>> a
        array([[0, 2]])
        >>> c
        array([[0, 2]])
        """
        return manipulation._ndarray_resize(self, new_shape, refcheck)

    def transpose(self, *axes):
        """Returns a view of the array with axes transposed.

        For a 1-D array this has no effect, as a transposed vector is simply the same
        vector. To convert a 1-D array into a 2D column vector, an additional dimension
        must be added. a[:, nlcpy.newaxis] achieves this. For a 2-D array, this is a
        standard matrix transpose. For an n-D array, if axes are given, their order
        indicates how the axes are permuted (see Examples). If axes are not provided and
        ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
        ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

        Parameters
        ----------
        axes : None, tuple of ints, or n ints
            - None or no argument: reverses the order of the axes.
            - tuple of ints: *i* in the *j-th* place in the tuple means *a's* *i-th*
              axis becomes *a.transpose()'s* *j-th* axis.
            - *n* ints: same as an n-tuple of the same ints (this form is intended
              simply as a "convenience" alternative to the tuple form)

        Returns
        -------
        out : ndarray
            View of *a*, with axes suitably permuted.

        See Also
        --------
        nlcpy.ndarray.reshape : Returns an array containing
            the same data with a new shape.

        Examples
        --------
        >>> import nlcpy as vp
        >>> a = vp.array([[1, 2], [3, 4]])
        >>> a
        array([[1, 2],
               [3, 4]])
        >>> a.transpose()
        array([[1, 3],
               [2, 4]])
        >>> a.transpose((1, 0))
        array([[1, 3],
               [2, 4]])
        >>> a.transpose(1, 0)
        array([[1, 3],
               [2, 4]])

        """
        if len(axes) == 1:
            axes = axes[0]
        elif axes == ():
            axes = None
        return manipulation._ndarray_transpose(self, axes)

    def flatten(self, order='C'):
        """Returns a copy of the array collapsed into one dimension.

        Parameters
        ----------
        order : {'C','F','A','K'}, optional
            'C' means to flatten in row-major (C-style) order. 'F' means to flatten in
            column-major (Fortran-style) order. 'A' means to flatten in column-major
            order if a is Fortran contiguous in memory, row-major order otherwise. 'K'
            means to flatten a in the order the elements occur in memory. The default is
            'C'.

        Returns
        -------
        out : ndarray
            A copy of the input array, flattened to one dimension.

        Restriction
        -----------
        * If order = 'F' : *NotImplementedError* occurs.
        * | If order = 'A' or 'K' : *NotImplementedError* occurs when *a* is using
          |                         Fortran-style order

        See Also
        --------
        nlcpy.ravel : Returns a flattened array.

        Examples
        --------
        >>> import nlcpy as vp
        >>> a = vp.array([[1,2], [3,4]])
        >>> a.flatten()
        array([1, 2, 3, 4])

        """
        return manipulation._ndarray_flatten(self, order)

    def squeeze(self, axis=None):
        """Removes single-dimensional entries from the shape of an array.

        Refer to :func:`nlcpy.squeeze` for full documentation.

        See Also
        --------
        nlcpy.squeeze : Equivalent function.

        """
        return manipulation._ndarray_squeeze(self, axis=axis)

    def repeat(self, repeats, axis=None):
        """Repeats elements of an array.

        Refer to :func:`nlcpy.repeat` for full documentation.

        See Also
        --------
            nlcpy.repeat : Equivalent function.

        """
        return manipulation._ndarray_repeat(self, repeats, axis)

    def swapaxes(self, axis1, axis2):
        """Returns a view of the array with axis1 and axis2 interchanged.

        Refer to :func:`nlcpy.swapaxes` for full documentation.

        See Also
        --------
            nlcpy.swapaxes : Equivalent function.

        """
        return manipulation._ndarray_swapaxes(self, axis1, axis2)
    # -------------------------------------------------------------------------
    # sorting, searching, counting
    # -------------------------------------------------------------------------
    cpdef sort(self, axis=-1, kind=None, order=None):
        """Sorts an array in-place.

        Refer to :func:`nlcpy.sort` for full documentation.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. Default is -1, which means sort along the last
            axis.
        kind : {'stable'}, optional
            Sorting algorithm.
        order : str or list of str, optional
            Not implemented.

        Restriction
        -----------
        * *kind* is not None and not 'stable' : NotImplementedError occurs.
        * *order* is not None : NotImplementedError occurs.

        See Also
        --------
        nlcpy.sort : Returns a sorted copy of an array.
        nlcpy.argsort : Returns the indices that would sort an array.

        Examples
        --------
        >>> import nlcpy as vp
        >>> a = vp.array([[1,4], [3,1]])
        >>> a.sort(axis=1)
        >>> a
        array([[1, 4],
               [1, 3]])
        >>> a.sort(axis=0)
        >>> a
        array([[1, 3],
               [1, 4]])
        """
        msg = None
        if kind is not None and kind is not 'stable':
            msg = 'kind only supported \'stable\'.'
        if order is not None:
            msg = 'order is not implemented.'
        if self.dtype.kind in ('c',):
            msg = 'Unsupported dtype \'%s\'' % self.dtype
        if msg is not None:
            try:
                f = numpy.ndarray.sort
                return nlcpy._make_wrap_method(f, self)(axis, kind, order)
            except NotImplementedError:
                raise NotImplementedError(msg)
        sorting._ndarray_sort(self, axis, kind=kind, order=order)

    cpdef ndarray argsort(self, axis=-1, kind=None, order=None):
        """Returns the indices that would sort this array.

        Refer to :func:`nlcpy.argsort` for full documentation.

        See Also
        --------
        nlcpy.argsort : Equivalent function.

        """
        msg = None
        if kind is not None and kind is not 'stable':
            msg = 'kind only supported \'stable\'.'
        if order is not None:
            msg = 'order is not implemented.'
        if self.dtype.kind in ('c',):
            msg = 'Unsupported dtype \'%s\'' % self.dtype
        if msg is not None:
            try:
                f = numpy.ndarray.argsort
                return nlcpy._make_wrap_method(f, self)(axis, kind, order)
            except NotImplementedError:
                raise NotImplementedError(msg)
        return sorting._ndarray_argsort(self, axis, kind=kind, order=order)

    # -------------------------------------------------------------------------
    # array conversion
    # -------------------------------------------------------------------------
    cpdef tolist(self):
        """Returns the array as an ``a.ndim``-levels deep nested list of Python scalars.

        Returns a copy of the array data as a (nested) Python list. Data items are
        converted to the nearest compatible builtin Python type, via the item function.
        If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will not be
        a list at all, but a simple Python scalar.

        Parameters
        ----------
        none

        Returns
        -------
        y : object, or list of object, or list of list of object, or...
            The possibly nested list of array elements.

        Note
        ----
        The array may be recreated via ``a = nlcpy.array(a.tolist())``, although this may
        sometimes lose precision.

        Examples
        --------
        For a 1D array, ``a.tolist()`` is almost the same as ``list(a)``:

        >>> import nlcpy as vp
        >>> a = vp.array([1, 2])
        >>> list(a)
        [array(1), array(2)]
        >>> a.tolist()
        [1, 2]

        However, for a 2D array, ``tolist`` applies recursively

        >>> a = vp.array([[1, 2], [3, 4]])
        >>> list(a)
        [array([1, 2]), array([3, 4])]
        >>> a.tolist()
        [[1, 2], [3, 4]]

        The base case for this recursion is a 0D array:

        >>> a = vp.array(1)
        >>> list(a)
        Traceback (most recent call last):
         ...
        TypeError: iteration over a 0-d array
        >>> a.tolist()
        1
        """
        return self.get().tolist()

    cpdef ndarray view(self, dtype=None, object type=None):
        """Returns a new view of array with the same data.

        Parameters
        ----------
        dtype : dtype, optional
            Data-type descriptor of the returned view, e.g., float32 or int64. The
            default, None, results in the view having the same data-type as *a.*
        type : Python type, optional
            Type of the returned view, e.g., ndarray or matrix. Again, omission of
            the parameter results in type preservation.

        Note
        ----
        ``a.view()`` is used two different ways:

        ``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view of the
        array's memory with a different dtype. This can cause a reinterpretation of the
        bytes of memory.

        For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of bytes per
        entry than the previous dtype (for example, converting a regular array to a
        structured array), then the behavior of the view cannot be predicted just from
        the superficial appearance of ``a`` (shown by ``print(a)``). It also depends on
        exactly how ``a`` is stored in memory. Therefore if a is C-ordered versus
        fortran-ordered, versus defined as a slice or transpose, etc., the view may give
        different results.

        Examples
        --------
        Viewing array data using a different dtype:

        >>> import nlcpy as vp
        >>> x = vp.array([(1, 2)], dtype=vp.int32)
        >>> y = x.view(dtype=vp.int64)
        >>> y
        array([[8589934593]])

        Making changes to the view changes the underlying array

        >>> x = vp.array([(1, 2),(3,4)], dtype=vp.int32)
        >>> xv = x.view(dtype=vp.int32).reshape(-1,2)
        >>> xv[0,1] = 20
        >>> x
        array([[ 1, 20],
               [ 3,  4]], dtype=int32)
        """
        cdef Py_ssize_t ndim
        if dtype is ndarray or dtype in ndarray.__subclasses__():
            if type is not None:
                raise ValueError("Cannot specify output type twice.")
            type = dtype
            dtype = None
        v = self._view(self._shape, self._strides, False, False, True,
                       dtype=dtype, type=type)
        return v

    cpdef ndarray copy(self, order='C'):
        """Returns a copy of the array.

        Parameters
        ----------
        order : {'C','F','A','K'}, optional
            Controls the memory layout of the copy. 'C' means C-order, 'F' means F-order,
            'A' means 'F' if a is Fortran contiguous, 'C' otherwise. 'K' means match the
            layout of a as closely as possible. (Note that this function and nlcpy.copy
            are very similar, but have different default values for their order=
            arguments.)

        See Also
        --------
        nlcpy.copy : Equivalent function.

        """
        cdef int order_char = (
            b'C' if order is None
            else internal._normalize_order(order)
        )
        # when order_char is 'A' or 'K', update order_char
        order_char = _update_order_char(self, order_char)

        out = ndarray(self.shape, dtype=self.dtype, order=chr(order_char))
        self._venode.request_manager._push_request(
            "nlcpy_copy",
            "creation_op",
            (self, out),
        )
        return out

    cpdef ndarray astype(self, dtype, order='K', casting=None, subok=None, copy=True):
        """Returns a copy of the array, casts to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        order : {'C','F','A','K'}, optional
            Controls the memory layout order of the result. 'C' means C order, 'F' means
            Fortran order, 'A' means 'F' order if all the arrays are Fortran contiguous,
            'C' order otherwise, and 'K' means as close to the order the array elements
            appear in memory as possible. Default is 'K'.
        casting : str, optional
            This argument is not supported. The default is None.
        subok : bool, optional
            This argument is not supported. The default is None.
        copy : bool, optional
            By default, astype always returns a newly allocated array. If this is set to
            false, and the dtype, order requirements are satisfied, the input array is
            returned instead of a copy.

        Returns
        -------
        arr_t : ndarray
            Unless copy is False and the other conditions for returning the input array
            are satisfied (see description for copy input parameter), arr_t is a new
            array of the same shape as the input array, with dtype, order given by dtype,
            order.

        Examples
        --------
        >>> import nlcpy as vp
        >>> x = vp.array([1, 2, 2.5])
        >>> x
        array([1. , 2. , 2.5])
        >>> x.astype(int)
        array([1, 2, 2])

        """
        if casting is not None:
            try:
                f = numpy.ndarray.astype
                return nlcpy._make_wrap_method(f, self)(
                    dtype, order, casting, False, copy)
            except NotImplementedError:
                raise NotImplementedError('casting is not supported yet')
        if subok is not None:
            raise NotImplementedError('subok is not supported yet')

        if order is None:
            order = 'K'
        cdef int order_char = internal._normalize_order(order)

        dtype = _dtype.get_dtype(dtype)
        if dtype == self.dtype:
            if not copy and (
                    order_char == b'K' or
                    order_char == b'A' and (
                        self._c_contiguous or self._f_contiguous) or
                    order_char == b'C' and self._c_contiguous or
                    order_char == b'F' and self._f_contiguous):
                return self

        order_char = _update_order_char(self, order_char)

        if order_char != b'K':
            newarray = ndarray(self.shape, dtype=dtype, order=chr(order_char))

        if self.dtype.kind == 'c' and newarray.dtype.kind not in 'bc':
            warnings.warn(
                'Casting complex values to real discards the imaginary part',
                numpy.ComplexWarning)

        if self.size == 0:
            pass
        else:
            self._venode.request_manager._push_request(
                "nlcpy_copy",
                "creation_op",
                (self, newarray),
            )
        return newarray

    cpdef fill(self, value):
        """Fills the array with a scalar value.

        Parameters
        ----------
        value : scalar
            All elements of the ndarray will be assigned this value.

        Examples
        --------
        >>> import nlcpy as vp
        >>> a = vp.array([1, 2])
        >>> a.fill(0)
        >>> a
        array([0, 0])
        >>> a = vp.empty(2)
        >>> a.fill(1)
        >>> a
        array([1., 1.])

        """
        manipulation._fill_kernel(self, value)

    def tobytes(self, order='C'):
        """Constructs Python bytes containing the raw data bytes in the array.

        Constructs Python bytes showing a copy of the raw contents of data memory. The
        bytes object can be produced in either 'C' or 'Fortran', or 'Any' order (the
        default is 'C'-order). 'Any' order means C-order unless the F_CONTIGUOUS flag in
        the array is set, in which case it means 'Fortran' order.

        Parameters
        ----------
        order : order : {'C', 'F', None}, optional
            Order of the data for multidimensional arrays: C, Fortran, or the same as for
            the original array.

        Returns
        -------
        s : bytes
            Python bytes exhibiting a copy of *a's* raw data.

        Examples
        --------
        >>> import nlcpy as vp
        >>> x = vp.array([[0, 1], [2, 3]], dtype='<u4')
        >>> x.tobytes()
        b'\\x00\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x02\\x00\\x00\\x00\\x03\\x00\\x00\\x00'
        >>> x.tobytes('C') == x.tobytes()
        True
        >>> x.tobytes('F')
        b'\\x00\\x00\\x00\\x00\\x02\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x03\\x00\\x00\\x00'

        """
        # ndarray.get() is numpy.ndarray value.
        return self.get().tobytes(order)

    # -------------------------------------------------------------------------
    # item selection and manipulation
    # -------------------------------------------------------------------------
    cpdef ndarray take(self, indices, axis=None, out=None, mode='wrap'):
        """Takes elements from an array along an axis.

        Refer to :func:`nlcpy.take` for full documentation.

        See Also
        --------
        nlcpy.take : Equivalent function.

        """
        if mode != 'wrap':
            try:
                f = numpy.ndarray.take
                return nlcpy._make_wrap_method(f, self)(indices, axis, out, mode)
            except NotImplementedError:
                raise NotImplementedError('mode is not supported yet')
        return indexing._ndarray_take(self, indices, axis, out)

    cpdef ndarray diagonal(self, offset=0, axis1=0, axis2=1):
        """Returns specified diagonals.

        Refer to nlcpy.diagonal for full documentation.

        See Also
        --------
        nlcpy.diagonal : Equivalent function.

        """
        return indexing._ndarray_diagonal(self, offset, axis1, axis2)

    # -------------------------------------------------------------------------
    # statistics
    # -------------------------------------------------------------------------
    cpdef ndarray max(self, axis=None, out=None, keepdims=False,
                      initial=nlcpy._NoValue, where=True):
        """Returns the maximum along a given axis.

        Refer to :func:`nlcpy.amax` for full documentation.

        See Also
        --------
        nlcpy.amax : Equivalent function.

        """
        return ufunc_op.maximum.reduce(
            self, axis=axis, out=out, initial=initial,
            dtype=None, keepdims=keepdims, where=where
        )

    cpdef ndarray min(self, axis=None, out=None, keepdims=False,
                      initial=nlcpy._NoValue, where=True):
        """Returns the minimum along a given axis.

        Refer to :func:`nlcpy.amin` for full documentation.

        See Also
        --------
        nlcpy.amin : Equivalent function.

        """
        return ufunc_op.minimum.reduce(
            self, axis=axis, out=out, initial=initial,
            dtype=None, keepdims=keepdims, where=where
        )

    # -------------------------------------------------------------------------
    # item selection and searching
    # -------------------------------------------------------------------------
    cpdef ndarray argmax(self, axis=None, out=None):
        """Returns indices of the maximum values along the given axis.

        Refer to :func:`nlcpy.argmax` for full documentation.

        See Also
        --------
        nlcpy.argmax : Equivalent function.

        """
        return searching.argmax(self, axis, out)

    cpdef ndarray argmin(self, axis=None, out=None):
        """Returns indices of the minimum values along the given axis.

        Refer to :func:`nlcpy.argmin` for full documentation.

        See Also
        --------
        nlcpy.argmin : Equivalent function.

        """
        return searching.argmin(self, axis, out)

    cpdef nonzero(self):
        """Returns the indices of the elements that are non-zero.

        Refer to :func:`nlcpy.nonzero` for full documentation.

        See Also
        --------
        nlcpy.nonzero : Equivalent function.

        """
        return searching.nonzero(self)

    # -------------------------------------------------------------------------
    # logic functions
    # -------------------------------------------------------------------------
    cpdef all(self, axis=None, out=None, keepdims=False):
        """Returns True if all elements evaluate to True.

        Refer to :func:`nlcpy.all` for full documentation.

        See Also
        --------
        nlcpy.all : Equivalent function.

        """
        if keepdims is nlcpy._NoValue:
            keepdims = False
        ret = nlcpy.logical_and.reduce(
            self, axis=axis, out=out, keepdims=keepdims)
        return ret

    cpdef any(self, axis=None, out=None, keepdims=False):
        """Returns True if any elements evaluate to True.

        Refer to :func:`nlcpy.any` for full documentation.

        See Also
        --------
        nlcpy.any : Equivalent function.

        """
        if keepdims is nlcpy._NoValue:
            keepdims = False
        ret = nlcpy.logical_or.reduce(
            self, axis=axis, out=out, keepdims=keepdims)
        return ret

    # -------------------------------------------------------------------------
    #  statistics methods
    # -------------------------------------------------------------------------
    def ptp(self, axis=None, out=None, keepdims=nlcpy._NoValue):
        """Range of values (maximum - minimum) along an axis.

        Refer to :func:`nlcpy.ptp` for full documentation.

        See Also
        --------
        nlcpy.ptp : Equivalent function.

        """
        ret = nlcpy.statistics.order.ptp(self, axis, out, keepdims)
        return ret

    def mean(self, axis=None, dtype=None, out=None, keepdims=nlcpy._NoValue):
        """Computes the arithmetic mean along the specified axis.

        Returns the average of the array elements. The average is taken over the
        flattened array by default, otherwise over the specified axis. float64
        intermediate and return values are used for integer inputs.

        Refer to :func:`nlcpy.mean` for full documentation.

        See Also
        --------
        nlcpy.mean : Equivalent function.

        """
        ret = nlcpy.statistics.average.mean(self, axis, dtype, out, keepdims)
        return ret

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=nlcpy._NoValue):
        """Computes the variance along the specified axis.

        Returns the variance of the array elements, a measure of the spread of a
        distribution. The variance is computed for the flattened array by default,
        otherwise over the specified axis.

        Refer to :func:`nlcpy.var` for full documentation.

        See Also
        --------
        nlcpy.var : Equivalent function.

        """
        ret = nlcpy.statistics.average.var(self, axis, dtype, out, ddof, keepdims)
        return ret

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=nlcpy._NoValue):
        """Computes the standard deviation along the specified axis.

        Returns the standard deviation, a measure of the spread of a distribution, of the
        array elements. The standard deviation is computed for the flattened array by
        default, otherwise over the specified axis.

        Refer to :func:`nlcpy.std` for full documentation.

        See Also
        --------
        nlcpy.std : Equivalent function.

        """
        ret = nlcpy.statistics.average.std(self, axis, dtype, out, ddof, keepdims)
        return ret

    def conj(self, out=None, where=True, casting='same_kind',
             order='K', dtype=None, subok=False):
        """Function that operates element by element on whole arrays.

        See Also
        --------
            nlcpy.conj : Equivalent function.
        """
        ret = nlcpy.ufunc_op.conj(self, out=out, where=where,
                                  casting=casting, order=order, dtype=dtype, subok=subok)
        return ret

    def conjugate(self, out=None, where=True, casting='same_kind',
                  order='K', dtype=None, subok=False):
        """Function that operates element by element on whole arrays.

        See Also
        --------
            nlcpy.conjugate : Equivalent function.
        """
        ret = nlcpy.ufunc_op.conjugate(self, out=out, where=where,
                                       casting=casting, order=order, dtype=dtype,
                                       subok=subok)
        return ret

    def dot(self, b, out=None):
        """Computes a dot product of two arrays.

        See Also
        --------
            nlcpy.dot : Equivalent function.

        """
        ret = nlcpy.dot(self, b, out=out)
        return ret

    def cumsum(self, axis=None, dtype=None, out=None):
        """Returns the cumulative sum of the elements along a given axis.

        See Also
        --------
            nlcpy.cumsum : Equivalent function.
        """
        ret = nlcpy.cumsum(self, axis=axis, dtype=dtype, out=out)
        return ret

    def prod(self, axis=None, dtype=None, out=None, keepdims=False,
             initial=nlcpy._NoValue, where=True):
        """Returns the product of the array elements over the given axis.

        See Also
        --------
            nlcpy.prod : Equivalent function.
        """
        ret = nlcpy.prod(self, axis=axis, dtype=dtype, out=out,
                         keepdims=keepdims, initial=initial, where=where)
        return ret

    def sum(self, axis=None, dtype=None, out=None, keepdims=False,
            initial=0, where=True):
        """Returns the sum of the array elements over the given axis.

        See Also
        --------
            nlcpy.sum : Equivalent function.
        """
        ret = nlcpy.sum(self, axis=axis, dtype=dtype, out=out,
                        keepdims=keepdims, initial=initial, where=where)
        return ret

    # -------------------------------------------------------------------------
    # nlcpy original attributes and methods
    # -------------------------------------------------------------------------
    cpdef get(self, order='C', out=None):
        """Returns a NumPy array on VH that copied from VE.

        Parameters
        ----------
        order : {'C','F','A'}, optional
            Controls the memory layout order of the result. The default is 'C'.
            The ``order`` will be ignored if ``out`` is specified.

        """

        venode = self._venode
        if out is not None:
            if not isinstance(out, numpy.ndarray):
                raise TypeError('Only numpy.ndarray can be obtained from '
                                'nlcpy.ndarray')
            if self.dtype != out.dtype:
                raise TypeError(
                    '{} array cannot be obtained from {} array'.format(
                        out.dtype, self.dtype))
            if self.shape != out.shape:
                raise ValueError(
                    'Shape mismatch. Expected shape: {}, '
                    'actual shape: {}'.format(self.shape, out.shape))
            if not (out.flags.c_contiguous and self._c_contiguous or
                    out.flags.f_contiguous and self._f_contiguous):
                prev_ve = VE()
                try:
                    venode.use()
                    if out.flags.c_contiguous:
                        a_ve = nlcpy.asarray(self, order='C')
                    elif out.flags.f_contiguous:
                        a_ve = nlcpy.asarray(self, order='F')
                    else:
                        raise RuntimeError(
                            '`out` cannot be specified when copying to '
                            'non-contiguous ndarray')
                finally:
                    prev_ve.use()
            else:
                a_ve = self
            a_cpu = out
        else:
            if self.size == 0:
                return numpy.ndarray(self._shape, dtype=self.dtype)

            order = order.upper()
            if order == 'A':
                if self._f_contiguous:
                    order = 'F'
                else:
                    order = 'C'
            if not (order == 'C' and self._c_contiguous or
                    order == 'F' and self._f_contiguous):
                prev_ve = VE()
                try:
                    venode.use()
                    if order == 'C':
                        a_ve = nlcpy.asarray(self, order='C')
                    elif order == 'F':
                        a_ve = nlcpy.asarray(self, order='F')
                    else:
                        raise ValueError('unsupported order: {}'.format(order))
                finally:
                    prev_ve.use()
            else:
                a_ve = self
            a_cpu = numpy.empty(self._shape, dtype=self.dtype, order=order)

        venode.request_manager._flush(sync=True)

        if self.dtype is numpy.dtype('bool'):
            a_cpu_tmp = numpy.empty_like(a_cpu, dtype='i4')
            venode.proc.read_mem(a_cpu_tmp.data, a_ve.ve_adr, a_ve.nbytes)
            a_cpu[...] = a_cpu_tmp.astype('bool')
        else:
            venode.proc.read_mem(a_cpu.data, a_ve.ve_adr, a_ve.nbytes)
        return a_cpu

    cpdef set(self, a_cpu):
        """Copies an array on the host memory to :class:`nlcpy.ndarray`.

        Parameters
        ----------
        a_cpu : numpy.ndarray
            The source array on the host memory.

        """

        if not isinstance(a_cpu, numpy.ndarray):
            raise TypeError('Only numpy.ndarray can be set to nlcpy.ndarray')
        if self.dtype != a_cpu.dtype:
            raise TypeError('{} array cannot be set to {} array'.format(
                a_cpu.dtype, self.dtype))
        if self.shape != a_cpu.shape:
            raise ValueError(
                'Shape mismatch. Old shape: {}, new shape: {}'.format(
                    self.shape, a_cpu.shape))
        if self._c_contiguous:
            a_cpu = numpy.ascontiguousarray(a_cpu)
        elif self._f_contiguous:
            a_cpu = numpy.asfortranarray(a_cpu)
        else:
            raise RuntimeError('Cannot set to non-contiguous array')

        if a_cpu.dtype is numpy.dtype('bool'):
            a_cpu = a_cpu.astype('i4')

        venode = self._venode
        venode.request_manager._flush(sync=True)
        venode.proc.write_mem(self.ve_adr, a_cpu.data, self.nbytes)

    cdef ndarray _view(
            self,
            const vector[Py_ssize_t]& shape,
            const vector[Py_ssize_t]& strides,
            bint update_c_contiguity,
            bint update_f_contiguity,
            bint mem_check=True,
            object dtype=None,
            object type=None,
            int64_t offset=0L):
        cdef ndarray v
        cdef Py_ssize_t ndim
        if shape.size() > NLCPY_MAXNDIM:
            raise ValueError('maximum supported dimension for an ndarray is %d, '
                             'found %d' % (NLCPY_MAXNDIM, shape.size()))
        if type not in (None, ndarray):
            v = ndarray.__new__(type)
            v.__init__()
            if type is nlcpy.ma.MaskedArray:
                v._sharedmask = False
        else:
            v = ndarray.__new__(ndarray)
        v.dtype, v.itemsize = (self.dtype, self.itemsize) if dtype is None \
            else _dtype.get_dtype_with_itemsize(dtype)
        v.base = self.base if self.base is not None else self
        v._c_contiguous = self._c_contiguous
        v._f_contiguous = self._f_contiguous
        v._is_view = True
        v._set_shape_and_strides(
            shape, strides, update_c_contiguity, update_f_contiguity)
        v._owndata = False
        v._venode = self._venode
        v.node_id = self.node_id

        v.veo_hmem = self.veo_hmem + offset
        v.ve_adr = self.ve_adr + offset
        v._is_pool = self._is_pool

        if dtype is None:
            return v

        v_is = v.itemsize
        self_is = self.dtype.itemsize
        if v_is == self_is:
            return v

        if mem_check:
            ndim = self._shape.size()
            if ndim == 0 and mem_check:
                raise ValueError(
                    "Changing the dtype of a 0d array is only supported if "
                    "the itemsize is unchanged")
            if not self._c_contiguous:
                raise ValueError(
                    "To change to a dtype of a different size, the array must "
                    "be C-contiguous")
            v._shape[ndim - 1] = v._shape[ndim - 1] * self_is // v_is
            v._strides[ndim - 1] = v._strides[ndim - 1] * v_is // self_is
            v.size = v.size * self_is // v_is
        v.itemsize = v_is
        return v

    cpdef _set_contiguous_strides(
            self,
            int64_t itemsize,
            bint is_c_contiguous
    ):
        self.size = internal.set_contiguous_strides(
            self._shape,
            self._strides,
            itemsize,
            is_c_contiguous
        )
        if is_c_contiguous:
            self._c_contiguous = True
            self._update_f_contiguity()
        else:
            self._f_contiguous = True
            self._update_c_contiguity()

    cpdef _update_contiguity(self):
        self._update_c_contiguity()
        self._update_f_contiguity()

    cpdef _set_shape_and_strides(
            self,
            const vector[Py_ssize_t]& shape,
            const vector[Py_ssize_t]& strides,
            bint update_c_contiguity,
            bint update_f_contiguity
    ):
        if shape.size() != strides.size():
            raise ValueError('len(shape) != len(strides)')
        self._shape = shape
        self._strides = strides
        self.size = internal.prod(shape)
        if update_c_contiguity:
            self._update_c_contiguity()
        if update_f_contiguity:
            self._update_f_contiguity()

    # if array's data locates c_contiguous, ndarray._c_contiguous is set to True.
    cpdef _update_c_contiguity(self):
        if self.size == 0:
            self._c_contiguous = True
            return
        self._c_contiguous = internal.get_c_contiguity(
            self._shape,
            self._strides,
            self.itemsize
        )
    cpdef _update_f_contiguity(self):
        cdef Py_ssize_t i, count
        cdef vector[Py_ssize_t] rev_shape, rev_strides
        if self.size == 0:
            self._f_contiguous = True
            return
        if self._c_contiguous:
            count = 0
            for i in self._shape:
                if i == 1:
                    count += 1
            # if all shape is '1', ndarray is f contiguous
            self._f_contiguous = (<Py_ssize_t>self._shape.size()) - count <= 1
            return
        rev_shape.assign(self._shape.rbegin(), self._shape.rend())
        rev_strides.assign(self._strides.rbegin(), self._strides.rend())
        self._f_contiguous = internal.get_c_contiguity(
            rev_shape, rev_strides, self.itemsize)

    # -------------------------------------------------------------------------
    # numpy wrap
    # -------------------------------------------------------------------------
    def __getattr__(self, attr):
        if attr[:2] == '__' or attr == 'setflags':
            raise AttributeError(
                "'nlcpy.core.core.ndarray' object has no attribute '{}'.".format(attr))
        if not type(self) is ndarray:
            return self.__getattr__(attr)
        try:
            f = getattr(numpy.ndarray, attr)
        except AttributeError as _err:
            raise AttributeError(
                "'nlcpy.core.core.ndarray' object has no attribute '{}'."
                .format(attr)) from _err
        if not callable(f):
            raise AttributeError(
                "'nlcpy.core.core.ndarray' object has no attribute '{}'.".format(attr))
        return nlcpy._make_wrap_method(f, self)


# -------------------------------------------------------------------------
# Array creation routine
# -------------------------------------------------------------------------
cpdef ndarray array(obj, dtype=None, bint copy=True, order='K',
                    bint subok=False, Py_ssize_t ndmin=0):
    cdef Py_ssize_t ndim
    cdef ndarray a, src
    cdef int nbytes
    cdef int nd
    cdef void *data

    if subok:
        raise NotImplementedError

    if ndmin > NLCPY_MAXNDIM:
        raise ValueError("ndmin bigger than allowable number of dimensions "
                         "NLCPY_MAXDIMS (=%d)" % NLCPY_MAXNDIM)

    if order is None:
        order = 'K'

    if isinstance(obj, ndarray):
        if dtype is None:
            dtype = obj.dtype
        a = obj.astype(dtype, order=order, copy=copy)

        if ndmin > a.ndim:
            if a is obj:
                a = a.view()
            # expands shape
            a.shape = (1,) * (ndmin - a.ndim) + a.shape
    elif hasattr(obj, '__ve_array_interface__'):
        a = _array_from_ve_array_interface(obj, dtype, copy, order, subok, ndmin)
    elif isinstance(obj, numpy.ndarray):
        if order is None or order in 'kK':
            if obj.flags['F_CONTIGUOUS'] and not obj.flags["C_CONTIGUOUS"]:
                order = 'F'
            else:
                order = 'C'
        else:
            order = ('F' if order in 'Ff' else 'C')
        if dtype is None:
            dtype = obj.dtype
        a = _send_object_to_ve(obj, dtype, order, ndmin)
    else:
        shape, elem_type, elem_dtype = _get_concat_shape(obj)
        if shape is not None and shape[-1] != 0:
            # obj is a non-empty sequence of ndarrays which share same shape
            # and dtype

            # resulting array is C order unless 'F' is explicitly specified
            # (i.e., it ignores order of element arrays in the sequence)
            order = (
                'F'
                if order is not None and len(order) >= 1 and order[0] in 'Ff'
                else 'C')
            ndim = len(shape)
            if ndmin > ndim:
                shape = (1,) * (ndmin - ndim) + shape

            if dtype is None:
                dtype = elem_dtype
            # Note: dtype might not be numpy.dtype in this place

            if issubclass(elem_type, numpy.ndarray):
                # obj is Seq[numpy.ndarray]
                a = _send_object_to_ve(obj, dtype, order, ndmin)
            elif issubclass(elem_type, ndarray):
                # obj is Seq[nlcpy.ndarray]
                lst = _flatten_list(obj)

                a = (nlcpy.concatenate(lst, 0)
                     .reshape(shape)
                     .astype(dtype, order=order, copy=False))
            else:
                # should not be reached here
                assert issubclass(elem_type, (numpy.ndarray, ndarray))
        else:
            # obj is:
            # - scalar or sequence of scalar
            # - empty sequence or sequence with elements whose shapes or
            #   dtypes are unmatched
            # - other types
            if obj is None:
                raise NotImplementedError("creating ndarray from None "
                                          "object is not implemented.")
            order = ('F' if order is not None and order in 'Ff' else 'C')
            a = _send_object_to_ve(obj, dtype, order, ndmin)
    return a


cdef ndarray _array_from_ve_array_interface(
        obj, dtype, bint copy, order, bint subok, Py_ssize_t ndim):
    cdef dict vai
    cdef tuple data
    cdef str typestr
    cdef tuple shape
    cdef tuple strides
    cdef list descr
    cdef uint64_t veo_hmem
    cdef bint readonly = 0
    cdef Py_ssize_t s, size = 1
    cdef Py_ssize_t itemsize = 1
    cdef char typekind = c'u'
    try:
        vai = obj.__ve_array_interface__
    except AttributeError:
        raise RuntimeError("missing VE array interface")
    # mandatory
    data = vai['data']
    typestr = vai['typestr']
    shape = tuple(vai['shape'])
    # optional
    strides = None if vai.get('strides') is None else tuple(vai.get('strides'))
    descr = vai.get('descr')
    mask = vai.get('mask')
    veo_hmem, readonly = data
    for s in shape:
        size *= s
    if mask is not None:
        raise NotImplementedError(
            "__ve_array_interface__: "
            "cannot handle masked arrays"
        )
    if size < 0:
        raise ValueError(
            "__ve_array_interface__: "
            "buffer with negative size (shape:%s, size:%d)"
            % (shape, size)
        )
    return ndarray(shape, dtype=typestr, strides=strides, order=None, veo_hmem=veo_hmem)


cdef tuple _get_concat_shape(object obj):
    # Returns a tuple of the following:
    # 1. concatenated shape if it can be converted to a single CuPy array by
    #    just concatenating it (i.e., the object is a (nested) sequence only
    #    which contains NumPy/CuPy array(s) with same shape and dtype).
    #    Returns None otherwise.
    # 2. type of the first item in the object
    # 3. dtype if the object is an array
    if isinstance(obj, (list, tuple)):
        return _get_concat_shape_impl(obj)
    return (None, None, None)


cdef tuple _get_concat_shape_impl(object obj):
    cdef obj_type = type(obj)
    if issubclass(obj_type, (numpy.ndarray, ndarray)):
        # obj.shape is () when obj.ndim == 0
        return (obj.shape, obj_type, obj.dtype)
    if isinstance(obj, (list, tuple)):
        shape = None
        typ = None
        dtype = None
        for elem in obj:
            # Find the head recursively if obj is a nested built-in list
            elem_shape, elem_type, elem_dtype = _get_concat_shape_impl(elem)

            # Use shape of the first element as the common shape.
            if shape is None:
                shape = elem_shape
                typ = elem_type
                dtype = elem_dtype

            # `elem` is not concatable or the shape and dtype does not match
            # with siblings.
            if (elem_shape is None
                    or shape != elem_shape
                    or dtype != elem_dtype):
                return (None, obj_type, None)

        if shape is None:
            shape = ()
        return (
            (len(obj),) + shape,
            typ,
            dtype)
    # scalar or object
    return (None, obj_type, None)


cdef list _flatten_list(object obj):
    ret = []
    if isinstance(obj, (list, tuple)):
        for elem in obj:
            ret += _flatten_list(elem)
        return ret
    if isinstance(obj, ndarray):
        if type(obj) is nlcpy.ma.MaskedArray:
            obj = obj.data
        if obj.ndim == 0:
            # convert each scalar (0-dim) ndarray to 1-dim
            obj = nlcpy.expand_dims(obj, 0)
    return [obj]


cdef ndarray _send_object_to_ve(obj, dtype, order, Py_ssize_t ndmin):
    a_cpu = numpy.array(obj, dtype=dtype, copy=False, order=order, ndmin=ndmin)
    a_dtype = a_cpu.dtype

    if a_dtype.char not in '?bhilqBHILQefdFD':
        raise NotImplementedError('Unsupported dtype \'%s\'' % a_dtype)

    if a_cpu.dtype is numpy.dtype('bool'):
        a_cpu = a_cpu.astype(dtype='i4')

    cdef ndarray a_ve = ndarray(
        a_cpu.shape,
        dtype=a_dtype,
        strides=a_cpu.strides,
        order=order
    )
    vememory._write_mem(a_cpu, a_ve.ve_adr, a_ve.nbytes, a_ve._venode)
    return a_ve


cpdef int _update_order_char(ndarray a, int order_char):
    # update order_char based on array contiguity when order is 'A' or 'K'
    if order_char == b'A':
        if a._f_contiguous:
            order_char = b'F'
        else:
            order_char = b'C'
    elif order_char == b'K':
        if a._f_contiguous:
            order_char = b'F'
        else:
            order_char = b'C'
    return order_char


# -------------------------------------------------------------------------
# memory check
# -------------------------------------------------------------------------
def may_share_memory(a, b, max_work=None):
    """Determine if two arrays might share memory.

    A return of True does not necessarily mean that the two arrays share any element.
    It just means that they might.

    Parameters
    ----------
    a, b : ndarray
        Input arrays
    max_work : int, optional
        Effort to spend on solving the overlap problem.
        Please note that the current version supports only when max_work is None.

    Returns
    -------
    out : bool
        Dictionary containing the old settings.

    Restriction
    -----------
    - If *max_work* is not None, *NotImplementedError* occurs.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.may_share_memory(vp.array([1,2]), vp.array([5,8,9]))
    False
    >>> x = vp.zeros([3, 4])
    >>> vp.may_share_memory(x[:,0], x[:,1])
    True

    """
    if isinstance(a, ndarray) and isinstance(b, ndarray):
        if max_work is not None:
            raise NotImplementedError("Only supported for `max_work` "
                                      "is `None`")
        if a.base is not None:
            aid = id(a.base)
        else:
            aid = id(a)
        if b.base is not None:
            bid = id(b.base)
        else:
            bid = id(b)
        if aid == bid:
            return True
        return False

    else:
        return numpy.may_share_memory(a, b, max_work)


# -------------------------------------------------------------------------
# nlcpy original routines
# -------------------------------------------------------------------------
cpdef ndarray argument_conversion(x):
    if not isinstance(x, ndarray) and x is not None:
        x = array(x)
    return x

cpdef check_fpe_flags(fpe_flags, reqnames):
    if (fpe_flags | 0x00000000):
        hnd = error_handler.geterr()
        if (fpe_flags & 0x00000020) != 0:
            mes = "divide by zero encountered in any of (" + reqnames._uniq_str() + ")"
            f = hnd['divide']
            if (f == 'ignore'):
                pass
            elif (f == 'warn'):
                warnings.warn(mes, RuntimeWarning)
            elif (f == 'raise'):
                raise FloatingPointError(mes)
            elif (f == 'print'):
                print(' RuntimeWarning: %s' % mes)

        if (fpe_flags & 0x00000010) != 0:
            mes = "overflow encountered in any of (" + reqnames._uniq_str() + ")"
            f = hnd['over']
            if (f == 'ignore'):
                pass
            elif (f == 'warn'):
                warnings.warn(mes, RuntimeWarning)
            elif (f == 'raise'):
                raise FloatingPointError(mes)
            elif (f == 'print'):
                print(' RuntimeWarning: %s' % mes)

        if (fpe_flags & 0x00000008) != 0:
            mes = "underflow encountered in any of (" + reqnames._uniq_str() + ")"
            f = hnd['under']
            if (f == 'ignore'):
                pass
            elif (f == 'warn'):
                warnings.warn(mes, RuntimeWarning)
            elif (f == 'raise'):
                raise FloatingPointError(mes)
            elif (f == 'print'):
                print(' RuntimeWarning: %s' % mes)

        # if (fpe_flags & 0x00000004) != 0:
        #     mes = "fixed-point overflow encountered in " + __name__
        #     warnings.warn(mes, RuntimeWarning)

        if (fpe_flags & 0x00000002) != 0:
            mes = "invalid value encountered in any of (" + reqnames._uniq_str() + ")"
            f = hnd['invalid']
            if (f == 'ignore'):
                pass
            elif (f == 'warn'):
                warnings.warn(mes, RuntimeWarning)
            elif (f == 'raise'):
                raise FloatingPointError(mes)
            elif (f == 'print'):
                print(' RuntimeWarning: %s' % mes)


cpdef get_nlcpy_maxndim():
    return NLCPY_MAXNDIM


cdef _exit_mode = False


@atexit.register
def _finalize():
    global _exit_mode
    _exit_mode = True

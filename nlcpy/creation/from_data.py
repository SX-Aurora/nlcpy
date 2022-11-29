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
import nlcpy
from nlcpy.core import core
from nlcpy.core import internal
from nlcpy import ndarray
from nlcpy.wrapper.numpy_wrap import numpy_wrap

# ----------------------------------------------------------------------------
# create arrays from existing data
# see: https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html
# ----------------------------------------------------------------------------


def array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    """Creates an array.

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an object whose \\__array\\__
        method returns an array, or any (nested) sequence.
    dtype : dtype, optional
        The desired dtype for the array. If not given, then the type will be determined
        as the minimum type required to hold the objects in the sequence. This argument
        can only be used to 'upcast' the array. For downcasting, use the .astype(t)
        method.
    copy : bool, optional
        If true (default), then the object is copied. Otherwise, a copy will only be made
        if \\__array\\__ returns a copy, if object is a nested sequence, or if a copy is
        needed to satisfy any of the other requirements (*dtype*, *order*, etc.).
    order : {'K', 'A', 'C', 'F'}, optional
        Specify the memory layout of the array. If object is not an array,
        the newly created array will be in C order (row major) unless 'F'
        is specified, in which case it will be in Fortran order (column major).
        If object is an array the following holds.

        .. csv-table::
            :header: "order", "no copy", "copy=True"
            :widths: 5, 7, 15

            "'K'", "unchanged", "| F & C order preserved,
            | otherwise most similar order"
            "'A'", "unchanged", "| F order if input is F
            | and not C, otherwise C order"
            "'C'", "C order", "C order"
            "'F'", "F order", "F order"

        When ``copy=False`` and a copy is made for other reasons, the result
        is the same as if ``copy=True``, with some exceptions for A, see the
        Notes section. The default order is 'K'.

    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise, the returned array
        will be forced to be a base-class array (default). subok=True is Not Implemented.
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array should have.
        Ones will be prepended to the shape as needed to meet this requirement.

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.

    Restriction
    -----------
    * object[i].shape not equals to object[j].shape for some i,j :
      *NotImplementedError* occurs.

    See Also
    --------
    empty_like : Returns a new array with the same
        shape and type as a given array.
    ones_like : Returns an array of ones with
        the same shape and type as a given array.
    zeros_like : Returns an array of zeros
        with the same shape and type as a given array.
    full_like : Returns a full array
        with the same shape and type as a given array.
    empty : Returns a new array of given
        shape and type, without initializing entries.
    ones : Returns a new array of given
        shape and type, filled with ones.
    zeros : Returns a new array of given
        shape and type, filled with zeros.
    full : Returns a new array of given
        shape and type, filled with fill_value.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.array([1, 2, 3])
    array([1, 2, 3])

    Upcasting:

    >>> vp.array([1, 2, 3.0])
    array([1., 2., 3.])

    More than one dimension:

    >>> vp.array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

    Minimum dimensions 2:

    >>> vp.array([1, 2, 3], ndmin=2)
    array([[1, 2, 3]])

    Type provided:

    >>> vp.array([1, 2, 3], dtype=complex)
    array([1.+0.j, 2.+0.j, 3.+0.j])

    """
    if subok is not False:
        raise NotImplementedError('subok in array is not implemented yet.')
    return core.array(object, dtype, copy, order, subok, ndmin)


def copy(a, order='K', subok=False):
    """Returns an array copy of the given object.

    Parameters
    ----------
    a : array_like
        Input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the copy. 'C' means C-order, 'F' means F-order, 'A'
        means 'F' if *a* is Fortran contiguous, 'C' otherwise. 'K' means match the layout
        of *a* as closely as possible. (Note that this function and :func:`ndarray.copy`
        are very similar, but have different default values for their order= arguments.)
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the returned array
        will be forced to be a base-class array (defaults to False).

    Returns
    -------
    arr : ndarray
        Array interpretation of *a*.

    Note
    ----
    This is equivalent to:

    >>> import nlcpy as vp
    >>> a = vp.array([1, 2, 3])
    >>> vp.array(a, copy=True)
    array([1, 2, 3])

    Examples
    --------
    Create an array x, with a reference y and a copy z:

    >>> import nlcpy as vp
    >>> x = vp.array([1, 2, 3])
    >>> y = x
    >>> z = vp.copy(x)

    Note that when we modify x, y changes, but not z:

    >>> x[0] = 10
    >>> x[0] == y[0]
    array(True)
    >>> x[0] == z[0]
    array(False)
    """
    if subok is not False:
        raise NotImplementedError('subok in array is not implemented yet.')
    a = nlcpy.asanyarray(a)
    return a.copy(order=order)


def asarray(a, dtype=None, order=None):
    """Converts the input to an array.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array. This includes scalars,
        lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    dtype : dtype, optional
        By default, the dtype is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major (C-style) or column-major (Fortran-style) memory
        representation. Defaults to 'C'.

    Returns
    -------
    out : ndarray
        Array interpretation of *a*. No copy is performed if the input is already an
        ndarray with matching and order. If *a* is a subclass of ndarray, a base class
        ndarray is returned.

    See Also
    --------
    asanyarray : Converts the input to an array, but pass ndarray
        subclasses through.

    Examples
    --------

    Convert a list into an array:

    >>> import nlcpy as vp
    >>> a = [1, 2]
    >>> vp.asarray(a)
    array([1, 2])

    Existing arrays are not copied:

    >>> a = vp.array([1, 2])
    >>> vp.asarray(a) is a
    True

    If *dtype* is set, array is copied only if dtype does not match:

    >>> a = vp.array([1, 2], dtype=vp.float32)
    >>> vp.asarray(a, dtype=vp.float32) is a
    True
    >>> vp.asarray(a, dtype=vp.float64) is a
    False

    """
    if type(a) is ndarray:
        if dtype is None and order is None:
            return a
        elif dtype is not None and order is None:
            if a.dtype == numpy.dtype(dtype):
                return a
        elif dtype is None and order is not None:
            order_char = internal._normalize_order(order)
            order_char = chr(core._update_order_char(a, order_char))
            if order_char == 'C' and a._c_contiguous:
                return a
            if order_char == 'F' and a._f_contiguous:
                return a
        else:
            order_char = internal._normalize_order(order)
            order_char = chr(core._update_order_char(a, order_char))
            if a.dtype == numpy.dtype(dtype) and \
                (order_char == 'C' and a._c_contiguous
                 or order_char == 'F' and a._f_contiguous):
                return a
    return core.array(a, dtype=dtype, order=order)


def asanyarray(a, dtype=None, order=None):
    """Converts the input to an array, but passes ndarray subclasses through.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array. This includes scalars,
        lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    dtype : dtype, optional
        By default, the dtype is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major (C-stype) or column-major (Fortran-style) memory
        representation. Defaults to 'C'.

    Returns
    -------
    out : ndarray or an ndarray subclass
        Array interpretation of *a*. If *a* is a subclass of ndarray, it is returned
        as-is and no copy is performed.

    See Also
    --------
    asarray : Converts the input to an array.

    Examples
    --------

    Convert a list into an array:

    >>> import nlcpy as vp
    >>> a = [1, 2]
    >>> vp.asanyarray(a)
    array([1, 2])

    """
    if isinstance(a, ndarray):
        if dtype is None and order is None:
            return a
        elif dtype is not None and order is None:
            if a.dtype == numpy.dtype(dtype):
                return a
        elif dtype is None and order is not None:
            order_char = internal._normalize_order(order)
            order_char = chr(core._update_order_char(a, order_char))
            if order_char == 'C' and a._c_contiguous:
                return a
            if order_char == 'F' and a._f_contiguous:
                return a
        else:
            order_char = internal._normalize_order(order)
            order_char = chr(core._update_order_char(a, order_char))
            if a.dtype == numpy.dtype(dtype) and \
                (order_char == 'C' and a._c_contiguous
                 or order_char == 'F' and a._f_contiguous):
                return a
    return core.array(a, dtype=dtype, order=order)


@numpy_wrap
def fromfile(file, dtype=float, count=-1, sep='', offset=0):
    """Constructs an array from data in a text or binary file.

    A highly efficient way of reading binary data with a known data-type, as well as
    parsing simply formatted text files. Data written using the *tofile* method can be
    read using this function.

    Parameters
    ----------
    file : file or str or pathlib.Path
        Open file object or filename.
    dtype : dtype, optional
        Data type of the returned array. For binary files, it is used to determine
        the size and byte-order of the items in the file.
    count : int, optional
        Number of items to read. ``-1`` means all items (i.e., the complete file).
    sep : str, optional
        Separator between items if file is a text file. Empty ("") separator means
        the file should be treated as binary. Spaces (" ") in the separator match
        zero or more whitespace characters. A separator consisting only of spaces
        must match at least one whitespace.
    offset : int, optional
        The offset (in bytes) from the file's current position. Defaults to 0. Only
        permitted for binary files.

    Returns
    -------
    out : ndarray
        Data read from the file.

    Note
    ----
    Do not rely on the combination of *tofile* and :func:`fromfile` for data storage,
    as the binary files generated are not platform independent. In particular, no
    byte-order or data-type information is saved. Data can be stored in the platform
    independent ``.npy`` format using save and load instead.

    See Also
    --------
    loadtxt : Loads data from a text file.
    load : Loads arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

    Examples
    --------
    >>> import numpy as np
    >>> import nlcpy as vp

    Construct an ndarray:

    >>> x = np.random.uniform(0, 1, 5)
    >>> x  # doctest: +SKIP
    array([0.61878546, 0.87721538, 0.92901071, 0.87754926, 0.07167856]) # random

    Save the raw data to disk:

    >>> import tempfile
    >>> fname = tempfile.mkstemp()[1]
    >>> x.tofile(fname)

    Read the raw data from disk:

    >>> vp.fromfile(fname)   # doctest: +SKIP
    array([0.61878546, 0.87721538, 0.92901071, 0.87754926, 0.07167856]) # random

    The recommended way to store and load data:

    >>> np.save(fname, x)
    >>> vp.load(fname + '.npy')  # doctest: +SKIP
    array([0.61878546, 0.87721538, 0.92901071, 0.87754926, 0.07167856]) # random

    """
    raise NotImplementedError('fromfile is not implemented yet.')


@numpy_wrap
def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes',
            max_rows=None):
    """Loads data from a text file.

    Each row in the text file must have the same number of values.

    Parameters
    ----------
    fname : file or str or pathlib.Path
        File, filename, or generator to read. If the filename extension is ``.gz`` or
        ``.bz2``, the file is first decompressed. Note that generators should return
        byte strings for Python 3k.
    dtype : dtype, optional
        Data-type of the resulting array; default: float.
    comments : str or sequence of str, optional
        The characters or list of characters used to indicate the start of a comment.
        None implies no comments. For backwards compatibility, byte strings will be
        decoded as 'latin1'. The default is '#'.
    delimiter : str, optional
        The string used to separate values. For backwards compatibility, byte strings
        will be decoded as 'latin1'. The default is whitespace.
    converters : dict, optional
        A dictionary mapping column number to a function that will parse the column
        string into the desired value. E.g., if column 0 is a date string:
        ``converters = {0: datestr2num}``. Converters can also be used to provide a
        default value for missing data:
        ``converters = {3: lambda s: float(s.strip() or 0)}``. Default: None.
    skiprows : int, optional
        Skip the first skiprows lines, including comments; default: 0.
    usecols : int or sequence, optional
        Which columns to read, with 0 being the first. For example, ``usecols = (1,4,5)``
        will extract the 2nd, 5th and 6th columns. The default, None, results in all
        columns being read.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be unpacked
        using ``x, y, z = loadtxt(...)``. Default is False.
    ndmin : int, optional
        The returned array will have at least *ndmin* dimensions. Otherwise
        mono-dimensional axes will be squeezed. Legal values: 0 (default), 1 or 2.
    encoding : str, optional
        Encoding used to decode the inputfile. Does not apply to input streams. The
        special value 'bytes' enables backward compatibility workarounds that ensures
        you receive byte arrays as results if possible and passes 'latin1' encoded
        strings to converters.
        Override this value to receive unicode arrays and pass strings as input to
        converters. If set to None the system default is used. The default value is
        'bytes'.
    max_rows : int, optional
        Read *max_rows* lines of content after *skiprows* lines.
        The default is to read all the lines.

    Returns
    -------
    out : ndarray
        Data read from the text file.

    Note
    ----
    This function aims to be a fast reader for simply formatted files.
    The strings produced by the Python float.hex method can be used as input for
    floats.

    See Also
    --------
    load : Loads arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

    Examples
    --------
    >>> import nlcpy as vp
    >>> from io import StringIO   # StringIO behaves like a file object
    >>> c = StringIO(u"0 1\\n2 3")
    >>> vp.loadtxt(c)
    array([[0., 1.],
           [2., 3.]])
    >>> c = StringIO(u"1,0,2\\n3,0,4")
    >>> x, y = vp.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
    >>> x
    array([1., 3.])
    >>> y
    array([2., 4.])

    """
    raise NotImplementedError('loadtxt is not implemented yet.')

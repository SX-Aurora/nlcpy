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
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#     THE SOFTWARE.
#
import nlcpy
from nlcpy.wrapper.numpy_wrap import numpy_wrap


class NpzFile(object):

    def __init__(self, npz_file):
        self.npz_file = npz_file
        self.files = npz_file.files

    def __enter__(self):
        self.npz_file.__enter__()
        return self

    def __exit__(self, typ, val, traceback):
        self.npz_file.__exit__(typ, val, traceback)

    def __getitem__(self, key):
        arr = self.npz_file[key]
        return nlcpy.array(arr)

    def close(self):
        self.npz_file.close()


@numpy_wrap
def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    """Loads arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

    .. Warning::
        Loading files that contain object arrays uses the ``pickle`` module, which is
        not secure against erroneous or maliciously constructed data. Consider passing
        ``allow_pickle=False`` to load data that is known not to contain object arrays
        for the safer handling of untrusted sources.

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
        The file to read. File-like objects must support the ``seek()`` and ``read()``
        methods. Pickled files require that the file-like object support the
        ``readline()`` method as well.
    mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, memory-map the file to construct an intermediate
        :obj:`numpy.ndarray` object and create :obj:`nlcpy.ndarray` from it.
    allow_pickle : bool, optional
        Allow loading pickled object arrays stored in npy files. Reasons for
        disallowing pickles include security, as loading pickled data can execute
        arbitrary code. If pickles are disallowed, loading object arrays will fail.
        Default: False
    fix_imports : bool, optional
        Only useful when loading Python 2 generated pickled files on Python 3, which
        includes npy/npz files containing object arrays. If *fix_imports* is True,
        pickle will try to map the old Python 2 names to the new names used in Python
        3.
    encoding : str, optional
        What encoding to use when reading Python 2 strings. Only useful when loading
        Python 2 generated pickled files in Python 3, which includes npy/npz files
        containing object arrays. Values other than 'latin1', 'ASCII', and 'bytes'
        are not allowed, as they can corrupt numerical data. Default: 'ASCII'

    Returns
    -------
    result : ndarray, tuple, dict, etc.
        Data stored in the file. For ``.npz`` files, the returned instance of NpzFile
        class must be closed to avoid leaking file descriptors.

    Note
    ----
    - If the file contains pickle data, then whatever object is stored in the pickle
      is returned.
    - If the file is a ``.npy`` file, then a single array is returned.
    - If the file is a ``.npz`` file, then a dictionary-like object is returned,
      containing {filename: array} key-value pairs, one for each file in the archive.
    - If the file is a ``.npz`` file, the returned value supports the context manager
      protocol in a similar fashion to the open function::

          with load('foo.npz') as data:
              a = data['a']

      The underlying file descriptor is closed when exiting the 'with' block.

    See Also
    --------
    loadtxt : Loads data from a text file.

    Examples
    --------
    Store data to disk, and load it again:

    >>> import nlcpy as vp
    >>> vp.save('/tmp/123', vp.array([[1, 2, 3], [4, 5, 6]]))
    >>> vp.load('/tmp/123.npy')
    array([[1, 2, 3],
           [4, 5, 6]])

    Store compressed data to disk, and load it again:

    >>> a=vp.array([[1, 2, 3], [4, 5, 6]])
    >>> b=vp.array([1, 2])
    >>> vp.savez('/tmp/123.npz', a=a, b=b)
    >>> data = vp.load('/tmp/123.npz')
    >>> data['a']
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> data['b']
    array([1, 2])
    >>> data.close()

    """
    # use numpy_wrap
    raise NotImplementedError('load is not implemented yet.')


@numpy_wrap
def save(file, arr, allow_pickle=True, fix_imports=True):
    """Saves an array to a binary file in NumPy ``.npy`` format.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        File or filename to which the data is saved. If file is a file-object,
        then the filename is unchanged. If file is a string or Path, a ``.npy``
        extension will be appended to the file name if it does not already have one.
    arr : array_like
        Array data to be saved.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for disallowing pickles
        include security (loading pickled data can execute arbitrary code) and
        portability (pickled objects may not be loadable on different Python
        installations, for example if the stored objects require libraries that are not
        available, and not all pickled data is compatible between Python 2 and
        Python 3). Default: True
    fix_imports : bool, optional
        Only useful in forcing objects in object arrays on Python 3 to be pickled in a
        Python 2 compatible way. If fix_imports is True, pickle will try to map the new
        Python 3 names to the old module names used in Python 2, so that the pickle data
        stream is readable with Python 2.

    See Also
    --------
    savez : Saves several arrays into a single file in uncompressed ``.npz`` format.
    savetxt : Saves an array to a text file.
    load : Loads arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

    Note
    ----
    For a description of the ``.npy`` format, see `numpy.lib.format.
    <https://numpy.org/doc/1.17/reference/generated/numpy.lib.format.html#module-numpy.lib.format>`_

    Examples
    --------
    >>> import nlcpy as vp
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()

    >>> x = vp.arange(10)
    >>> vp.save(outfile, x)

    >>> _ = outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> vp.load(outfile)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """
    # use numpy_wrap
    raise NotImplementedError('save is not implemented yet.')


@numpy_wrap
def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='n', header='', footer='',
            comments='# ', encoding=None):
    """Saves an array to a text file.

    Parameters
    ----------
    fname : filename or file handle
        If the filename ends in ``.gz``, the file is automatically saved in compressed
        gzip format. loadtxt understands gzipped files transparently.
    x : 1D or 2D array_like
        Data to be saved to a text file.
    fmt : str or sequence of strs, optional
        A single format (%10.5f), a sequence of formats, or a multi-format string, e.g.
        'Iteration %d – %10.5f', in which case *delimiter* is ignored.
        For complex *X*, the legal options for *fmt* are:

        - a single specifier, *fmt='%.4e'*, resulting in numbers formatted like
          *'(%s+%sj)' % (fmt, fmt)*
        - a full string specifying every real and imaginary part,
          e.g. *'%.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'* for 3 columns
        - a list of specifiers, one per column - in this case, the real and imaginary
          part must have separate specifiers, e.g. *['%.3e + %.3ej', '(%.15e%+.15ej)']*
          for 2 columns
    delimiter : str, optional
        String or character separating columns.
    newline : str, optional
        String or character separating lines.
    header : str, optional
        String that will be written at the beginning of the file.
    footer : str, optional
        String that will be written at the end of the file.
    comments : str, optional
        String that will be prepended to the ``header`` and ``footer`` strings,
        to mark them as comments. Default: '# ', as expected by e.g. ``numpy.loadtxt``.
    encoding : {None, str}, optional
        Encoding used to encode the outputfile. Does not apply to output streams.
        If the encoding is something other than 'bytes' or 'latin1' you will not be
        able to load the file in NumPy versions < 1.14. Default is 'latin1'.

    See Also
    --------
    save : Saves an array to a binary file in NumPy ``.npy`` format.
    savez : Saves several arrays into a single file in uncompressed ``.npz`` format.
    savez_compressed : Saves several arrays into a single file in compressed
                       ``.npz`` format.

    Note
    ----
    Further explanation of the fmt parameter (``%[flag]width[.precision]specifier``):

    **flags:**
        ``-`` : left justify

        ``+`` : Forces to precede result with + or -.

        ``0`` : Left pad the number with zeros instead of space (see width).

    **width:**
        Minimum number of characters to be printed.
        The value is not truncated if it has more characters.

    **precision:**
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print after
          the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.

    **specifiers:**
        ``c`` : character

        ``d`` or ``i`` : signed decimal integer

        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.

        ``f`` : decimal floating point

        ``g,G`` : use the shorter of e,E or f

        ``o`` : signed octal

        ``s`` : string of characters

        ``u`` : unsigned decimal integer

        ``x,X`` : unsigned hexadecimal integer

    This explanation of ``fmt`` is not complete, for an exhaustive specification
    see `Format Specification Mini-Language
    <https://docs.python.org/3/library/string.html#format-specification-mini-language>`_
    , Python Documentation.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = y = z = vp.arange(0.0, 5.0, 1.0)
    >>> vp.savetxt('test.out', x, delimiter=',')   # X is an array
    >>> vp.savetxt('test.out', (x, y, z))   # x,y,z equal sized 1D arrays
    >>> vp.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation
    """
    # use numpy_wrap
    raise NotImplementedError('savetxt is not implemented yet.')


@numpy_wrap
def savez(file, *args, **kwds):
    """Saves several arrays into a single file in uncompressed ``.npz`` format.

    If arguments are passed in with no keywords, the corresponding variable names,
    in the ``.npz`` file, are 'arr_0', 'arr_1', etc. If keyword arguments are given,
    the corresponding variable names, in the ``.npz`` file will match the keyword names.

    Parameters
    ----------
    file : str or file
        Either the file name (string) or an open file (file-like object) where the data
        will be saved. If file is a string or a Path, the ``.npz`` extension will be
        appended to the file name if it is not already there.
    args : Arguments, optional
        Arrays to save to the file. Since it is not possible for Python to know the
        names of the arrays outside savez, the arrays will be saved with names "arr_0",
        "arr_1", and so on. These arguments can be any expression.
    kwds : keyword arguments, optional
        Arrays to save to the file.
        Arrays will be saved in the file with the keyword names.

    See Also
    --------
    save : Saves an array to a binary file in NumPy ``.npy`` format.
    savetxt : Saves an array to a text file.
    savez_compressed : Saves several arrays into a single file in compressed
                       ``.npz`` format.

    Note
    ----
    The ``.npz`` file format is a zipped archive of files named after the variables
    they contain. The archive is not compressed and each file in the archive contains
    one variable in ``.npy`` format. For a description of the ``.npy`` format,
    see `numpy.lib.format.
    <https://numpy.org/doc/1.17/reference/generated/numpy.lib.format.html#module-numpy.lib.format>`_

    When opening the saved ``.npz`` file with :func:`load` a *NpzFile* object is
    returned.
    This is a dictionary-like object which can be queried for its list of arrays (with
    the ``.files`` attribute), and for the arrays themselves.

    Examples
    --------
    >>> import nlcpy as vp
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = vp.arange(10)
    >>> y = vp.sin(x)

    Using savez with \\*args, the arrays are saved with default names.

    >>> vp.savez(outfile, x, y)
    >>> _ = outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> npzfile = vp.load(outfile)
    >>> npzfile.files
    ['arr_0', 'arr_1']
    >>> npzfile['arr_0']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Using savez with \\**kwds, the arrays are saved with the keyword names.

    >>> outfile = TemporaryFile()
    >>> vp.savez(outfile, x=x, y=y)
    >>> _ = outfile.seek(0)
    >>> npzfile = vp.load(outfile)
    >>> sorted(npzfile.files)
    ['x', 'y']
    >>> npzfile['x']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """
    # use numpy_wrap
    raise NotImplementedError('savez is not implemented yet.')


@numpy_wrap
def savez_compressed(file, *args, **kwds):
    """ Saves several arrays into a single file in compressed ``.npz`` format.

    If keyword arguments are given, then filenames are taken from the keywords.
    If arguments are passed in with no keywords, then stored file names are arr_0,
    arr_1, etc.

    Parameters
    ----------
    file : str or file
        Either the file name (string) or an open file (file-like object) where the data
        will be saved. If file is a string or a Path, the ``.npz`` extension will be
        appended to the file name if it is not already there.
    args : Arguments, optional
        Arrays to save to the file. Since it is not possible for Python to know the
        names of the arrays outside :func:`savez`, the arrays will be saved with
        names "arr_0", "arr_1", and so on. These arguments can be any expression.
    kwds : keyword arguments
        Arrays to save to the file.
        Arrays will be saved in the file with the keyword names.

    See Also
    --------
    save : Saves an array to a binary file in NumPy ``.npy`` format.
    savetxt : Saves an array to a text file.
    savez : Saves several arrays into a single file in uncompressed ``.npz`` format.
    load : Loads arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

    Note
    ----
    The ``.npz`` file format is a zipped archive of files named after the variables
    they contain. The archive is compressed with ``zipfile.ZIP_DEFLATED`` and each file
    in the archive contains one variable in ``.npy`` format. For a description of the
    ``.npy`` format, see `numpy.lib.format.
    <https://numpy.org/doc/1.17/reference/generated/numpy.lib.format.html#module-numpy.lib.format>`_

    When opening the saved ``.npz`` file with :func:`load` a *NpzFile* object is
    returned.
    This is a dictionary-like object which can be queried for its list of arrays (with
    the ``.files`` attribute), and for the arrays themselves.

    Examples
    --------
    >>> import nlcpy as vp
    >>> from nlcpy import testing
    >>> test_array = vp.random.rand(3, 2)
    >>> test_vector = vp.random.rand(4)
    >>> vp.savez_compressed('/tmp/123', a=test_array, b=test_vector)
    >>> loaded = vp.load('/tmp/123.npz')
    >>> vp.testing.assert_array_equal(test_array, loaded['a'])
    >>> vp.testing.assert_array_equal(test_vector, loaded['b'])
    """
    # use numpy_wrap
    raise NotImplementedError('savez_compressed is not implemented yet.')

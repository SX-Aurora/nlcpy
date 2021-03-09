#
# * The source code in this file is based on the soure code of CuPy.
#
# # NLCPy License #
#
#     Copyright (c) 2020-2021 NEC Corporation
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
        includes npy/npz files containing object arrays. If ``fix_imports`` is True,
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

    >>> import numpy as np
    >>> import nlcpy as vp
    >>> np.save('./123', np.array([[1, 2, 3], [4, 5, 6]]))
    >>> vp.load('./123.npy')
    array([[1, 2, 3],
           [4, 5, 6]])

    Store compressed data to disk, and load it again:

    >>> a=np.array([[1, 2, 3], [4, 5, 6]])
    >>> b=np.array([1, 2])
    >>> np.savez('./123.npz', a=a, b=b)
    >>> data = vp.load('./123.npz')
    >>> data['a']
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> data['b']
    array([1, 2])
    >>> data.close()

    """
    raise NotImplementedError

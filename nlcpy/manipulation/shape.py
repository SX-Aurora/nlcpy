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
import nlcpy


# ----------------------------------------------------------------------------
# Array manipulation routines
# see: https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html
# ----------------------------------------------------------------------------
def reshape(a, newshape, order='C'):
    """Gives a new shape to an array without changing its data.

    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    newshape : int or sequence of ints
        The new shape should be compatible with the original shape. If an integer, then
        the result will be a 1-D array of that length. One shape dimension can be -1. In
        this case, the value is inferred from the length of the array and remaining
        dimensions.
    order : {'C', 'F', 'A'}, optional
        Read the elements of *a* using this index order, and place the elements into the
        reshaped array using this index order. 'C' means to read / write the elements
        using C-like index order, with the last axis index changing fastest, back to the
        first axis index changing slowest. 'F' means to read / write the elements using
        Fortran-like index order, with the first index changing fastest, and the last
        index changing slowest. Note that the 'C' and 'F' options take no account of the
        memory layout of the underlying array, and only refer to the order of indexing.
        'A' means to read / write the elements in Fortran-like index order if *a* is
        Fortran *contiguous* in memory, C-like order otherwise.

    Returns
    -------
    reshaped_array : ndarray
        This will be a new view object if possible; otherwise, it will be a copy. Note
        there is no guarantee of the *memory* *layout* (C- or Fortran- contiguous) of the
        returned array.

    Note
    ----
    It is not always possible to change the shape of an array without copying the data.
    If you want an error to be raised when the data is copied, you should assign the new
    shape to the shape attribute of the array:

    >>> import nlcpy as vp
    >>> a = vp.zeros((10, 2))

    A transpose makes the array non-contiguous

    >>> b = a.T

    Taking a view makes it possible to modify the shape without modifying
    the initial object.

    >>> c = b.view()
    >>> c.shape = (20)     # doctest: +SKIP
    Traceback (most recent call last):
       ...
    AttributeError: incompatible shape for a non-contiguous array

    The `order` keyword gives the index ordering both for `fetching` the values from `a,`
    and then `placing` the values into the output array. For example, let's say you have
    an array:

    >>> a = vp.arange(6).reshape((3, 2))
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5]])

    You can think of reshaping as first raveling the array (using the given index order),
    then inserting the elements from the raveled array into the new array using the same
    kind of index ordering as was used for the raveling.

    >>> vp.reshape(a, (2, 3)) # C-like index ordering
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> vp.reshape(vp.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> vp.reshape(a, (2, 3), order='F') # Fortran-like index ordering
    array([[0, 4, 3],
           [2, 1, 5]])
    >>> vp.reshape(vp.ravel(a, order='F'), (2, 3), order='F')
    array([[0, 4, 3],
           [2, 1, 5]])

    Examples
    --------
    >>> import nlcpy as vp
    >>> a = vp.array([[1,2,3], [4,5,6]])
    >>> vp.reshape(a, 6)
    array([1, 2, 3, 4, 5, 6])
    >>> vp.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
    array([[1, 2],
           [3, 4],
           [5, 6]])

    """
    a = nlcpy.asanyarray(a)
    return a.reshape(newshape, order=order)


def ravel(a, order='C'):
    """Returns a contiguous flattened array.

    A 1-D array, containing the elements of the
    input, is returned. A copy is made only if needed.

    Parameters
    ----------
    a : array_like
        Input array. The elements in *a* are read in the order specified by *order*, and
        packed as a 1-D array.
    order : {'C', 'F', 'A', 'K'}, optional
        The elements of *a* are read using this index order. 'C' means to index the
        elements in row-major, C-style order, with the last axis index changing fastest,
        back to the first axis index changing slowest. 'F' means to index the elements in
        column-major, Fortran-style order, with the first index changing fastest, and the
        last index changing slowest. Note that the 'C' and 'F' options take no account of
        the memory layout of the underlying array, and only refer to the order of axis
        indexing. 'A' means to read the elements in Fortran-like index order if *a* is
        Fortran *contiguous* in memory, C-like order otherwise. 'K' means to read the
        elements in the order they occur in memory, except for reversing the data when
        strides are negative. By default, 'C' index order is used.

    Returns
    -------
    y : ndarray
        y is an array of the same subtype as *a*, with shape ``(a.size,)``. Note that
        matrices are special cased for backward compatibility, if *a* is an ndarray, then
        y is a 1-D ndarray.

    Restriction
    -----------
    * If order == 'K': *NotImplementedError* occurs.

    Note
    ----
    In row-major, C-style order, in two dimensions, the row index varies the slowest, and
    the column index the quickest. This can be generalized to multiple dimensions, where
    row-major order implies that the index along the first axis varies slowest, and the
    index along the last quickest. The opposite holds for column-major, Fortran-style
    index ordering.
    When a view is desired in as many cases as possible, ``arr.reshape(-1)`` may be
    preferable.

    Examples
    --------
    It is equivalent to reshape(-1, order=order).

    >>> import nlcpy as vp
    >>> x = vp.array([[1, 2, 3], [4, 5, 6]])
    >>> vp.ravel(x)
    array([1, 2, 3, 4, 5, 6])
    >>> vp.ravel(x, order='F')
    array([1, 4, 2, 5, 3, 6])
    >>> x.reshape(-1)
    array([1, 2, 3, 4, 5, 6])
    >>> a = vp.arange(12).reshape(2,3,2); a
    array([[[ 0,  1],
            [ 2,  3],
            [ 4,  5]],
    <BLANKLINE>
           [[ 6,  7],
            [ 8,  9],
            [10, 11]]])
    >>> a.ravel(order='C')
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    """
    a = nlcpy.asanyarray(a)
    return a.ravel(order)

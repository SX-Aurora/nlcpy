Multi-Dimensional Array (ndarray)
=================================

:class:`nlcpy.ndarray` is the NLCPy counterpart of NumPy :class:`numpy.ndarray`.
It provides an intuitive interface for a fixed-size multidimensional array which resides
in a VE.

For the basic concept of ``ndarray``, please refer to the `NumPy documentation <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_.

ndarray class
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   nlcpy.ndarray


Array Indexing
--------------

Arrays can be indexed using an extended Python slicing syntax, array[selection].
For the basic concept of indexing arrays, please refer to the
`NumPy Array Indexing <https://numpy.org/doc/stable/reference/arrays.indexing.html#arrays-indexing>`_.

Differences from NumPy
----------------------

* **Out-of-bounds indices**

    NLCPy handles out-of-bounds indices differently by default from NumPy
    when using integer array indexing.
    NumPy handles them by raising an error, but NLCPy wraps around them.

    .. code-block:: python

        >>> import numpy as np
        >>> import nlcpy as vp
        >>> nx = np.arange(3)
        >>> nx[[0, 1, 5]]
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        IndexError: index 5 is out of bounds for axis 0 with size 3
        >>> vx = vp.arange(3)
        >>> vx[[0, 1, 5]]
        array([0, 1, 2])

* **Multiple boolean indices**

    NLCPy does not support slices that consists of more than one boolean arrays.

.. seealso::

    :ref:`Notices and Restrictions <notices>`


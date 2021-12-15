.. _label_ma:

===========
MaskedArray
===========

:class:`nlcpy.ma.MaskedArray` is the NLCPy counterpart of NumPy :class:`numpy.ma.MaskedArray`.

For the basic concept of ``MaskedArray``, please refer to the `NumPy documentation <https://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_.

.. attention::
    In the current NLCPy, masked version operations are only supported by four arithmetic operations in nlcpy.ma. Other functions may raise TypeError or treat MaskedArray as ndarray.

MaskedArray class
=================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   nlcpy.ma.MaskedArray


Masked Array Operations
=======================

Creation
--------

From existing data
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.ma.masked_array
    nlcpy.ma.copy
    nlcpy.ma.array

Inspecting the array
--------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.ma.getmask
    nlcpy.ma.getmaskarray
    nlcpy.ma.getdata
    nlcpy.ma.is_mask

Manipulating a MaskedArray
--------------------------

Changing the shape
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.ma.ravel
    nlcpy.ma.reshape
    nlcpy.ma.resize

Modifying axes
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.ma.swapaxes
    nlcpy.ma.transpose

Operations on masks
-------------------

Creating a mask
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.ma.make_mask
    nlcpy.ma.make_mask_none

Modifying a mask
^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.ma.harden_mask
    nlcpy.ma.soften_mask

Conversion operations
---------------------

> to a ndarray
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.ma.filled

Filling a masked array
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.ma.default_fill_value

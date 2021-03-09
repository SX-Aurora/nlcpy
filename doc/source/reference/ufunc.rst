.. _ufuncs:

Universal Functions (ufunc)
===========================

.. contents:: :local:

A universal function (or :class:`nlcpy.ufunc` for short) is a function that operates on ndarrays in an element-by-element fashion, supporting array broadcasting, type casting, and several other standard features. That is, a ufunc is a "vectorized" wrapper for a function that takes a fixed number of specific inputs and produces a fixed number of specific outputs.

As with NumPy, Broadcasting rules are applied to input arrays of the universal functions of NLCPy.


Broadcasting
------------

Each universal function takes array inputs and produces array outputs by performing the core function element-wise on the inputs (where an element is generally a scalar, but can be a vector or higher-order sub-array for generalized ufuncs). Standard broadcasting rules are applied so that inputs not sharing exactly the same shapes can still be usefully operated on. Broadcasting can be understood by four rules:

1. All input arrays with ndim smaller than the input array of largest ndim, have 1's prepended to their shapes.
2. The size in each dimension of the output shape is the maximum of all the input sizes in that dimension.
3. An input can be used in the calculation if its size in a particular dimension either matches the output size in that dimension, or has value exactly 1.
4. If an input has a dimension size of 1 in its shape, the first data entry in that dimension will be used for all calculations along that dimension. In other words, the stepping machinery of the ufunc will simply not step along that dimension (the stride will be 0 for that dimension).


Examples
^^^^^^^^

If a.shape is (5,1), b.shape is (1,6), c.shape is (6,) and d.shape is () so that d is a scalar, then a, b, c, and d are all broadcastable to dimension (5,6); and

* a acts like a (5,6) array where a[:,0] is broadcast to the other columns,
* b acts like a (5,6) array where b[0,:] is broadcast to the other rows,
* c acts like a (1,6) array and therefore like a (5,6) array where c[:] is broadcast to every row, and finally,
* d acts like a (5,6) array where the single value is repeated.


.. _available_ufuncs:

Available Ufuncs
----------------

The following tables show Universal functions(ufuncs) provided by NLCPy.

Math Operations
^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.add
    nlcpy.subtract
    nlcpy.multiply
    nlcpy.matmul
    nlcpy.divide
    nlcpy.logaddexp
    nlcpy.logaddexp2
    nlcpy.true_divide
    nlcpy.floor_divide
    nlcpy.negative
    nlcpy.positive
    nlcpy.power
    nlcpy.remainder
    nlcpy.mod
    nlcpy.fmod
    nlcpy.absolute
    nlcpy.fabs
    nlcpy.rint
    nlcpy.heaviside
    nlcpy.conj
    nlcpy.conjugate
    nlcpy.exp
    nlcpy.exp2
    nlcpy.log
    nlcpy.log2
    nlcpy.log10
    nlcpy.log1p
    nlcpy.expm1
    nlcpy.sqrt
    nlcpy.square
    nlcpy.cbrt
    nlcpy.reciprocal

Trigonometric Functions
^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.sin
    nlcpy.cos
    nlcpy.tan
    nlcpy.arcsin
    nlcpy.arccos
    nlcpy.arctan
    nlcpy.arctan2
    nlcpy.hypot
    nlcpy.sinh
    nlcpy.cosh
    nlcpy.tanh
    nlcpy.arcsinh
    nlcpy.arccosh
    nlcpy.arctanh
    nlcpy.degrees
    nlcpy.radians
    nlcpy.deg2rad
    nlcpy.rad2deg

Bit-Twiddling Functions
^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.bitwise_and
    nlcpy.bitwise_or
    nlcpy.bitwise_xor
    nlcpy.invert
    nlcpy.left_shift
    nlcpy.right_shift

Comparison Functions
^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.greater
    nlcpy.greater_equal
    nlcpy.less
    nlcpy.less_equal
    nlcpy.not_equal
    nlcpy.equal
    nlcpy.logical_and
    nlcpy.logical_or
    nlcpy.logical_xor
    nlcpy.logical_not
    nlcpy.maximum
    nlcpy.minimum
    nlcpy.fmax
    nlcpy.fmin

Floating Point Functions
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.isfinite
    nlcpy.isinf
    nlcpy.isnan
    nlcpy.signbit
    nlcpy.copysign
    nlcpy.sign
    nlcpy.nextafter
    nlcpy.spacing
    nlcpy.ldexp
    nlcpy.floor
    nlcpy.ceil
    nlcpy.trunc

Methods
-------

In NLCPy, :ref:`ufuncs <available_ufuncs>` are instances of the ``nlcpy.ufunc`` class.
``nlcpy.ufunc`` have four methods. However, these methods only make
sense on scalar ufuncs that take two input arguments and return one output argument.
Attempting to call these methods on other ufuncs will cause a *ValueError*.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.ufunc.reduce
    nlcpy.ufunc.accumulate
    nlcpy.ufunc.reduceat
    nlcpy.ufunc.outer

The current version of NLCPy does not provide ``nlcpy.ufunc.at()``, which is supported
by NumPy.


.. _optional_keyword_arg:

Optional Keyword Arguments
--------------------------

All ufuncs take optional keyword arguments. Most of these represent advanced usage and will not typically be used.

**out**

    A location into which the result is stored.
    If provided, it must have a shape that the
    inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned.
    A tuple (possible only as a keyword argument)
    must have length equal to the number of outputs.

**where**

    This condition is broadcast over the input. At locations where the condition is True,
    the *out* array will be set to the ufunc result.
    Elsewhere, the *out* array will retain its original value.
    Note that if an uninitialized *out* array is created via the default
    ``out=None``, locations within it where the condition is False will remain uninitialized.

**casting**

    Controls what kind of data casting may occur.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind, like float64 to float32, are allowed.

    NLCPy does NOT support 'unsafe', which is supported in NumPy.

**order**

    Specifies the calculation iteration order/memory layout of the output array. Defaults to 'K'.
    'C' means the output should be C-contiguous, 'F' means F-contiguous, 'A'
    means F-contiguous if the inputs are F-contiguous and not also not C-contiguous,
    C-contiguous otherwise, and 'K' means to match the element ordering of the inputs as closely as possible.

**dtype**

    Overrides the dtype of the calculation and output arrays.

**subok**

    Not implemented in NLCPy.

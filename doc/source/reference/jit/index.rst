.. _label_jit:
.. module:: nlcpy.jit
    :noindex:

========================
Just-In-Time Compilation
========================

Overview
========
NLCPy provides a Just-In-Time(JIT) compilation functionality.
By using this, you can customize your C/C++/Fortran kernel directly on your Python program.

Here is a simple example.

.. code-block:: python

    >>> import nlcpy
    >>> from nlcpy import ve_types
    >>>
    >>> ve_lib = nlcpy.jit.CustomVELibrary(
    ...     code=r'''
    ...         int ve_add(double *px, double *py, double *pz, int n) {
    ...             #pragma omp parallel for
    ...             for (int i = 0; i  < n; i++) pz[i] = px[i] + py[i];
    ...             return 0;
    ...         }
    ... '''
    ... )
    >>> ve_add = ve_lib.get_function(
    ...     've_add',
    ...     args_type=(ve_types.uint64, ve_types.uint64, ve_types.uint64, ve_types.int32),
    ...     ret_type=ve_types.int32
    ... )
    >>>
    >>> x = nlcpy.arange(10., dtype='f8')
    >>> y = nlcpy.arange(10., dtype='f8')
    >>> z = nlcpy.empty(10, dtype='f8')
    >>> ret = ve_add(x.ve_adr, y.ve_adr, z.ve_adr, z.size, sync=True)
    >>> z
    array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.])
    >>> ret
    0
    >>>
    >>> nlcpy.jit.unload_library(ve_lib)

To compile and execute the VE function from raw source code:

#. Define source code in Python code as a string.
#. Pass the string source into ``code`` argument of :class:`nlcpy.jit.CustomVELibrary`.
#. Call :meth:`CustomVELibrary.get_function` to get a function symbol.
#. Execute the object that is returned from :meth:`CustomVELibrary.get_function` as a function.
#. Unload the shared library if needed.

Coding Guide
============

.. toctree::
    :maxdepth: 1

    basic_usage
    advanced
    notices


APIs
====

Customize VE Library and Kernel
-------------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.jit.CustomVELibrary
    nlcpy.jit.CustomVEKernel

Helper Routines
---------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.jit.get_default_cflags
    nlcpy.jit.get_default_ldflags
    nlcpy.jit.unload_library

.. _label_ve_types_consts:

Constants
---------

.. data:: nlcpy.ve_types.void

    :Returns:

        **'void'** : string

.. data:: nlcpy.ve_types.char

    :Returns:

        **'char'** : string


.. data:: nlcpy.ve_types.int32

    :Returns:

        **'int32_t'** : string

.. data:: nlcpy.ve_types.int64

    :Returns:

        **'int64_t'** : string

.. data:: nlcpy.ve_types.uint32

    :Returns:

        **'uint32_t'** : string

.. data:: nlcpy.ve_types.uint64

    :Returns:

        **'uint64_t'** : string

.. data:: nlcpy.ve_types.float32

    :Returns:

        **'float'** : string

.. data:: nlcpy.ve_types.float64

    :Returns:

        **'double'** : string

.. data:: nlcpy.ve_types.void_p

    :Returns:

        **'void *'** : string

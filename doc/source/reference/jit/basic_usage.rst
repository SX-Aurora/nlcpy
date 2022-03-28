.. module:: nlcpy.jit
    :noindex:

===========
Basic Usage
===========

This section describes how to customize a VE function.

.. contents:: :local:


Constructing Custom VE Library
==============================

There are two ways to construct a custom VE library:

**A. Compiling and Loading From Source Code**

    #. Define source code as a Python string.

        ::

            >>> c_src=r'''
            ...     int ve_add(double *px, double *py, double *pz, int n) {
            ...         #pragma omp parallel for
            ...         for (int i = 0; i  < n; i++) pz[i] = px[i] + py[i];
            ...         return 0;
            ...     }
            ... '''

        .. note::
            When using the tripple quotes ``"""`` or ``'''``, always prepend them by
            ``r`` such that the content is interpreted as raw string.
            Otherwise the escaped characters will be interpreted and spoil the
            source code.

    #. Compile and Load from the Python string.

        ::

            >>> ve_lib = nlcpy.jit.CustomVELibrary(code=c_src)


**B. Loading From Pre-built Binary**

    ::

        >>> ve_lib = nlcpy.jit.CustomVELibrary(path='/path/to/existing/library.so')

If you need complicated build rules, we recommend using case ``B``.

.. seealso::
    :class:`nlcpy.jit.CustomVELibrary`


Getting a Symbol of a Function
==============================

You need to call :meth:`CustomVELibrary.get_function` to get a symbol of a function on VE.

::

    >>> from nlcpy import ve_types
    >>> ve_add = ve_lib.get_function(
    ...     've_add',
    ...     args_type=(ve_types.uint64, ve_types.uint64, ve_types.uint64, ve_types.int32),
    ...     ret_type=ve_types.int32
    ... )

``ve_add`` is the instance of :class:`CustomVEKernel` class.

The tuple of character string element or the tuple of constant element of the ``nlcpy.ve_types``
can be passed into the argument ``args_type``.
Also, the character string or the constant of the ``nlcpy.ve_types`` can be passed
into the argument ``ret_type``.

The supported data-types and its corresponding notations are as follows:

    .. csv-table::
        :header: "data-type description", "character strings", "constants of nlcpy.ve_types"

        void, 'void', nlcpy.ve_types.void
        8-bit character, 'char', nlcpy.ve_types.char
        32-bit signed integer, 'int32_t', nlcpy.ve_types.int32
        64-bit signed integer, 'int64_t', nlcpy.ve_types.int64
        32-bit unsigned integer, 'uint32_t', nlcpy.ve_types.uint32
        64-bit unsigned integer, 'uint64_t', nlcpy.ve_types.uint64
        32-bit floating-point real, 'float', nlcpy.ve_types.float32
        64-bit floating-point real, 'double', nlcpy.ve_types.float64
        void pointer, 'void \*', nlcpy.ve_types.void_p

.. note::
    For the complex data-type, it is necessary to transfer data as :class:`nlcpy.ndarray` or
    ``nlcpy.veo.OnStack``.
    Please refer to the :ref:`Advanced Topics <label_advanced_topics>`.

.. seealso::
    :meth:`nlcpy.jit.CustomVELibrary.get_function`
.. seealso::
    :class:`nlcpy.jit.CustomVEKernel`

Execution
=========

Here, you can execute the VE function.

::

    >>> x = nlcpy.arange(10., dtype='f8')
    >>> y = nlcpy.arange(10., dtype='f8')
    >>> z = nlcpy.empty(10, dtype='f8')
    >>> ret = ve_add(x.ve_adr, y.ve_adr, z.ve_adr, z.size, sync=True)
    >>> z
    array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.])
    >>> ret
    0

Just only pass the attribute ``ndarray.ve_adr`` into arguments,
the VE function can get the pointer of the array.

.. note::
    If you pass the argument ``sync=False``, the return value will be None.

.. note::
    You can invoke the VE function without recompiling by calling
    :meth:`CustomVEKernel.__call__` repeatedly.

.. seealso::
    :meth:`nlcpy.jit.CustomVEKernel.__call__`

Finalization (Optional)
=======================

If needed, you can unload the shared library.

::

    >>> nlcpy.jit.unload_library(ve_lib)

.. admonition:: Restriction

    Please avoid unloading the shared library linked with FTRACE.
    Otherwise, SIGSEGV may occur.

.. seealso::
    :func:`nlcpy.jit.unload_library`


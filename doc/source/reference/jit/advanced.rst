.. _label_advanced_topics:
.. module:: nlcpy.jit
    :noindex:

===============
Advanced Topics
===============

This section describes a advanced usage of the JIT compilation functionality.

.. contents:: :local:

Compiling from C++ Source
=========================

* **Definition of the source**

    The entry routine should be defined with ``extern "C"``.

    ::

        >>> cpp_src=r'''
        ... extern "C" {
        ...     int ve_add(double *px, double *py, double *pz, int n) {
        ...         #pragma omp parallel for
        ...         for (int i = 0; i  < n; i++) pz[i] = px[i] + py[i];
        ...         return 0;
        ...     }
        ... }
        ... '''

* **Compilation**

    Pass ``'nc++'`` or ``'/opt/nec/ve/bin/nc++'`` into the ``compiler`` argument.

    ::

        >>> ve_lib = nlcpy.jit.CustomVELibrary(code=cpp_src, compiler='nc++')

* **Getting the function symbol**

    This is the same procedure as the case that compile from C source.

    ::

        >>> ve_add = ve_lib.get_function(
        ...     've_add',
        ...     args_type=(ve_types.uint64, ve_types.uint64, ve_types.uint64, ve_types.int32),
        ...     ret_type=ve_types.int32
        ... )


Compiling from Fortran Source
=============================

* **Definition of the source**

    ::

        >>> f_src = r"""
        ... integer(kind=4) function ve_add(px, py, pz, n)
        ...     integer(kind=4), value :: n
        ...     double precision :: px(n), py(n), pz(n)
        ...     !$omp parallel do
        ...     do i=1, n
        ...         pz(i) = px(i) + py(i)
        ...     end do
        ...     ve_add = 0
        ... end
        ... """


* **Compilation**

    Pass ``'nfort'`` or ``'/opt/nec/ve/bin/nfort'`` into the ``compiler`` argument.

    ::

        >>> ve_lib = nlcpy.jit.CustomVELibrary(code=f_src, compiler='nfort')

* **Getting the subroutine or function symbol**

    You should add ``_`` to the end of the function name.

    ::

        >>> ve_add = ve_lib.get_function(
        ...     've_add_',
        ...     args_type=(ve_types.uint64, ve_types.uint64, ve_types.uint64, ve_types.int32),
        ...     ret_type=ve_types.int32
        ... )

.. note::
    If you want to pass a scalar value as a parameter of VE Fortran function/subroutine,
    please define the Fortran parameter with ``VALUE`` attribute.

    Alternatively, you can use ``nlcpy.veo.OnStack`` to pass a parameter as a stack.
    In this case, you should define the Fortran parameter without ``VALUE`` attribute.

    .. seealso::
        :ref:`Transferring Buffer Data to VE <label_transferring_buffer>`.


.. note::

    The Fortran source code is generated internally with the suffix ``.f03``

    * To preprocess before the compilation, please specify ``'-fpp'`` option into the ``cflags``.
    * To enable source code to be described in fixed form, please specify
      ``'-ffixed-form'`` option into the ``cflags``.

    For details, please refer to the Fortran Compiler User's Guide from
    `here <https://www.hpc.nec/documentation>`_.


Transferring Ndarray Attributes to VE
=====================================

NLCPy provides a easy way to access :class:`nlcpy.ndarray` attributes from VE side.

.. admonition:: Restriction

    Accessing :class:`nlcpy.ndarray` attributes from Fortran has not supported yet.

* **Definition of the source**

    You should include ``nlcpy.h`` in C/C++ source and use ``ve_array`` structure.

    Here is a example of 2-D element-wise addition.

    ::

        >>> c_src=r'''
        ... #include <nlcpy.h>
        ...
        ... void ve_add(ve_array *x, ve_array *y, ve_array *z) {
        ...     /* get a pointer */
        ...     double *px = (double *)x->ve_adr;
        ...     double *py = (double *)y->ve_adr;
        ...     double *pz = (double *)z->ve_adr;
        ...     /* get an each stride of an array index */
        ...     uint64_t ix0 = x->strides[x->ndim-1] / x->itemsize;
        ...     uint64_t ix1 = x->strides[x->ndim-2] / x->itemsize;
        ...     uint64_t iy0 = y->strides[y->ndim-1] / y->itemsize;
        ...     uint64_t iy1 = y->strides[y->ndim-2] / y->itemsize;
        ...     uint64_t iz0 = z->strides[z->ndim-1] / z->itemsize;
        ...     uint64_t iz1 = z->strides[z->ndim-2] / z->itemsize;
        ...     /* execute element-wise addition */
        ...     #pragma omp parallel for
        ...     for (int i = 0; i  < z->shape[z->ndim-2]; i++) {
        ...         for (int j = 0; j < z->shape[z->ndim-1]; j++) {
        ...             pz[i*iz1 + j*iz0] = px[i*ix1 + j*ix0] + py[i*iy1 + j*iy0];
        ...         }
        ...     }
        ... }
        ... '''

    For details of the C-structure, please refer to the :ref:`C Interfaces <label_c_interface>`.

* **Compilation**

    By default, only pass source code into the ``code`` argument.

    ::

        >>> ve_lib = nlcpy.jit.CustomVELibrary(code=c_src)

    If you specify ``cflags`` argument, it is necessary to include
    the header file path that can be retrieved from :func:`nlcpy.get_include`.

* **Getting the function symbol**

    Pass ``'void *'`` or ``ve_types.void_p`` into the ``args_type`` elements that corresponding to the
    ``ve_array`` structure in the VE side argument.

    ::

        >>> ve_add = ve_lib.get_function(
        ...     've_add',
        ...     args_type=(ve_types.void_p, ve_types.void_p, ve_types.void_p),
        ...     ret_type=ve_types.void
        ... )

* **Execution**

    Pass a :class:`nlcpy.ndarray` object into the argument of the
    :meth:`CustomVEKernel.__call__`.

    ::

        >>> x = nlcpy.arange(20, dtype='f8').reshape((4, 5))
        >>> y = nlcpy.arange(20, dtype='f8').reshape((4, 5))
        >>> z = nlcpy.empty((4, 5), dtype='f8')
        >>> ve_add(x, y, z)
        >>> print(z)
        [[ 0.  2.  4.  6.  8.]
         [10. 12. 14. 16. 18.]
         [20. 22. 24. 26. 28.]
         [30. 32. 34. 36. 38.]]

.. _label_c_interface:

C Interfaces
============

.. c:struct:: ve_array

    The ``ve_array`` C-structure contains the required information for a :class:`nlcpy.ndarray`.
    All instances of a :class:`nlcpy.ndarray` will have this structure.

    The members of the ``ve_array`` are as follows:

    .. c:member:: uint64_t ve_adr

        The address point to the first element of the array.

    .. c:member:: uint64_t ndim

        The number of dimensions in the array.

    .. c:member:: uint64_t size

        The total size of the array.

    .. c:member:: uint64_t shape[NLCPY_MAXNDIM]

        The shapes of the array.
        An array of integers providing the shape in each dimension.

        Given a :class:`nlcpy.ndarray` from ``nlcpy.empty((3, 4, 5), dtype='f8')``,
        the shape of C-structer is::

            ve_array.shape[0]               : 3
            ve_array.shape[1]               : 4
            ve_array.shape[2]               : 5
            ve_array.shape[3]               : undifiend
            ...
            ve_array.shape[NLCPY_MAXNDIM-1] : undifiend

    .. c:member:: uint64_t strides[NLCPY_MAXNDIM]

        The strides of the array.
        An array of integers providing the number of bytes that must be
        skipped to get to the next element in that dimension.

        Given a :class:`nlcpy.ndarray` from ``nlcpy.empty((3, 4, 5), dtype='f8')``,
        the strides of C-structer is::

            ve_array.strides[0]               : 160
            ve_array.strides[1]               :  40
            ve_array.strides[2]               :   8
            ve_array.strides[3]               : undifiend
            ...
            ve_array.strides[NLCPY_MAXNDIM-1] : undifiend

    .. c:member:: uint64_t dtype

        The data type of the array.
        The correspondence values is below:

        .. code-block:: c

            enum ve_dtype {
                ve_bool = 0,
                ve_i8   = 1,
                ve_u8   = 2,
                ve_i16  = 3,
                ve_u16  = 4,
                ve_i32  = 5,
                ve_u32  = 6,
                ve_i64  = 7,
                ve_u64  = 8,
                ve_f16  = 23,
                ve_f32  = 11,
                ve_f64  = 12,
                ve_c64  = 14,
                ve_c128 = 15,
            };

        This enum data can be defined by ``nlcpy.h``.

    .. c:member:: uint64_t itemsize

        The number of bytes for one element of the array.

    .. c:member:: uint64_t is_c_contiguous

        Whether the array is C-style contiguous order or not.
        ``1`` means yes, ``0`` means no.

    .. c:member:: uint64_t is_f_contiguous

        Whether the array is Fortran-style contiguous order or not.
        ``1`` means yes, ``0`` means no.


.. _label_transferring_buffer:

Transferring Buffer Data to VE
==============================

Python objects that support the buffer interface can be transferred to the
VE arguments by using ``nlcpy.veo.OnStack``.

::

    >>> from nlcpy import veo
    >>> import numpy
    >>>
    >>> src = r'''
    ... #include <stdint.h>
    ... void onstack_test(int32_t *a, float *b) {
    ...     b[0] = (float)(a[0] + a[1]);
    ... }
    ... '''
    >>> ve_lib = nlcpy.jit.CustomVELibrary(code=src)
    >>> test = ve_lib.get_function(
    ...     'onstack_test',
    ...     args_type=(ve_types.void_p, ve_types.void_p),
    ...     ret_type=ve_types.void
    ... )
    >>>
    >>> a = numpy.array([1, 2], dtype='i4')
    >>> b = numpy.empty(1, dtype='f4')
    >>> test(
    ...     veo.OnStack(a, inout=veo.INTENT_IN),
    ...     veo.OnStack(b, inout=veo.INTENT_OUT),
    ...     sync=True
    ... )
    >>> b
    array([3.], dtype=float32)

.. seealso::
    For details of ``OnStack``, please refer to the
    `py-veo project <https://github.com/SX-Aurora/py-veo>`_.

Customizing Compiler Options
============================

Cflags and ldflags can be customized from a tuple of string elements.

::

    >>> ve_lib = nlcpy.jit.CustomVELibrary(
    ...     code=something,
    ...     cflags=nlcpy.jit.get_default_cflags(openmp=False, opt_level=3) + ('-mvector-packed', '-ffast-math'),
    ...     ldflags=nlcpy.jit.get_default_ldflags(openmp=False) + ('-L', 'your/library/path', '-lsomething'),
    ... )

FTRACE can be enabled from the ``ftrace`` argument.

::

    >>> ve_lib = nlcpy.jit.CustomVELibrary(
    ...     code=something,
    ...     ftrace=True,
    ... )

You can also use NLC routines just by enabling  the ``use_nlc`` argument.

::

    >>> ve_lib = nlcpy.jit.CustomVELibrary(
    ...     code=r'''
    ...         #include <asl.h>
    ...         asl_int_t call_dbgmsm(double *ab, asl_int_t *ipvt, asl_int_t lna,
    ...                               asl_int_t n, int64_t m) {
    ...             return ASL_dbgmsm(ab, lna, n, m, ipvt);
    ...         }
    ... ''',
    ...     use_nlc=True,
    ... )

.. note::

    When enabling the ``use_nlc`` flag, the following libraries will be linked internally:

        * libasl_openmp_i64
        * libaslfftw3_i64
        * liblapack_i64
        * libblas_openmp_i64
        * libsca_openmp_i64
        * libheterosolver_openmp_i64
        * libsblas_openmp_i64
        * libcblas_i64

.. note::

    Only the OpenMP & 64bit integer version of the NLC can be used.

.. note::

    If you enable the ``use_nlc`` flag with Fortran source, you should add the
    option ``'-fdefault-integer=8'`` to the ``cflags``.

.. note::

    If you use ASL Unified Interface, you should not call following functions
    because there will be internally called at the beginning/end of the NLCPy process.

    * ``asl_library_initialize()``
    * ``asl_library_finalize()``

.. seealso::
    For the notices of compiler options, please refer to the
    `aveo documentation <https://www.hpc.nec/documents/veos/en/aveo/index.html>`_.


.. _label_callback:

Callback Setting
================

The Python function set into the ``callback`` argument
will be executed when the result of the VE function will be retrieved.
The callback function should take a one argument that is corresponding to the return value of the
VE function.

::

    >>> def callback(err):
    ...     # do something
    ...     return


Here, we show a simple example that uses a callback function.

The following code will be used for the example:

::

    >>> import string
    >>> err = {
    ...     'ERR_OK': 0,
    ...     'ERR_MEMORY': 1,
    ...     'ERR_NDIM': 2,
    ...     'ERR_DTYPE': 3,
    ...     'ERR_CONTIGUOUS': 4,
    ... }
    >>>
    >>> temp = string.Template(r'''
    ... #include <nlcpy.h>
    ... #include <stdlib.h>
    ...
    ... uint64_t callback_test(ve_array *x) {
    ...     double *px = (double *)x->ve_adr;
    ...     if (px == NULL) return ${ERR_MEMORY};
    ...     if (x->ndim != 1) return ${ERR_NDIM};
    ...     if (x->dtype != ve_f64) return ${ERR_DTYPE};
    ...     if (! (x->is_c_contiguous & x->is_f_contiguous)) return ${ERR_CONTIGUOUS};
    ...     /* do something here */
    ...     return ${ERR_OK};
    ... }
    ... ''')
    >>> src = temp.substitute(err)
    >>> print(src)

    #include <nlcpy.h>
    #include <stdlib.h>

    uint64_t callback_test(ve_array *x) {
        double *px = (double *)x->ve_adr;
        if (px == NULL) return 1;
        if (x->ndim != 1) return 2;
        if (x->dtype != ve_f64) return 3;
        if (x->is_c_contiguous & x->is_f_contiguous) return 4;
        /* do something here */
        return 0;
    }

Prepare the executable object:

::

    >>> ve_lib = nlcpy.jit.CustomVELibrary(code=src)
    >>> callback_test = ve_lib.get_function(
    ...     'callback_test',
    ...     args_type=(ve_types.void_p,),
    ...     ret_type=ve_types.uint64
    ... )

Define the callback function:

::

    >>> def err_print(retval):
    ...     # reverse lookup
    ...     for k, v in err.items():
    ...         if retval == v:
    ...             print(k)
    ...             return
    ...     raise Exception

Execute some patterns with the callback function:

::

    >>> x = nlcpy.arange(9, dtype='f8')
    >>> callback_test(x, callback=err_print)
    >>> nlcpy.request.flush()
    ERR_OK
    >>>
    >>> x = nlcpy.arange(9, dtype='f8').reshape(3,3)
    >>> callback_test(x, callback=err_print)
    >>> nlcpy.request.flush()
    ERR_NDIM
    >>>
    >>> x = nlcpy.arange(9, dtype='f4')
    >>> callback_test(x, callback=err_print)
    >>> nlcpy.request.flush()
    ERR_DTYPE
    >>>
    >>> x = nlcpy.arange(9, dtype='f8')[::2]
    >>> callback_test(x, callback=err_print)
    >>> nlcpy.request.flush()
    ERR_CONTIGUOUS

.. note::

    When you enable ``sync`` flag, the return value of the VE function can be
    retrieved from the return value of :meth:`CustomVEKernel.__call__`.

    ::

        >>> x = nlcpy.arange(9, dtype='f8')
        >>> callback_test(x, sync=True)
        0


Logging Compiler Output
=======================

* **Logging to standard output**

::

    >>> import sys
    >>> ve_lib = nlcpy.jit.CustomVELibrary(code=src, log_stream=sys.stdout)

* **Logging to file stream**

::

    >>> with open('./compiler.log', 'w') as fs:
    ...     ve_lib = nlcpy.jit.CustomVELibrary(code=src, log_stream=fs)

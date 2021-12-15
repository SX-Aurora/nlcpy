.. module:: nlcpy.jit
    :noindex:

=================================
Notices Regarding JIT Compilation
=================================

* To pass a complex data into VE arguments, it is necessary to transfer data as
  :class:`nlcpy.ndarray` or ``nlcpy.veo.OnStack``.
  Please refer to the :ref:`Advanced Topics <label_advanced_topics>`.

* You can invoke the VE function without recompiling by calling
  :meth:`CustomVEKernel.__call__` repeatedly.

* Only the OpenMP & 64bit integer version of the NLC can be used.

* When you use ASL Unified Interface, you should not call following functions
  because there will be internally called at the beginning/end of the NLCPy process.

  - ``asl_library_initialize()``
  - ``asl_library_finalize()``

* Please avoid unloading the shared library linked with FTRACE.
  Otherwise, SIGSEGV may occur.

* When you use NLCPy with Jupyter Notebook or Jupyter Lab,
  the browser cannot display stdout/stderr output from the VE side.

.. _execution:

Execution
=========

.. contents:: :local:
   :depth: 1


Parallel Execution
------------------

Python scripts using NLCPy can gain performance when it is executed in parallel by multithreading on a Vector Engine(VE).
The default number of parallel threads can be specified by the environment variable ``VE_OMP_NUM_THREADS`` at the time of execution.
If the ``VE_OMP_NUM_THREADS`` is undefined or has an invalid value, the maximum number of the available CPU cores on the system is set.

Examples are shown below.

* Interactive mode:

    ::

        $ VE_OMP_NUM_THREADS=4 python
        >>> import nlcpy

    Subsequent computations will be performed by 4 parallel threads on a VE.

* Non-interactive mode:

    ::

        $ VE_OMP_NUM_THREADS=4 python example.py

    Computations in example.py will be performed by 4 parallel threads on a VE.


When you want to specify VE Node number, you should use the environment variable ``VE_NODE_NUMBER`` as follows.

* Interactive mode:

    ::

        $ VE_NODE_NUMBER=1 VE_OMP_NUM_THREADS=4 python
        >>> import nlcpy

    Subsequent computations will be performed by 4 parallel threads on VE Node #1.

* Non-interactive mode:

    ::

        $ VE_NODE_NUMBER=1 VE_OMP_NUM_THREADS=4 python example.py

    Computations in example.py will be performed by 4 parallel threads on VE Node #1.

.. note::
    If you execute two or more NLCPy programs on a VE, context switching occurs.
    In such a case, the performance becomes significantly slower.


.. _label_multiple_ves:

Using Multiple VEs
------------------

NLCPy provides two ways to use multiple VEs.

1. `mpi4py-ve <https://github.com/SX-Aurora/mpi4py-ve>`_

    mpi4py-ve is a Message Passing Interface (MPI) Python library for SX-Aurora TSUBASA.
    It provides point-to-point and collective communication operations among processes.
    The available objects for MPI communication are `numpy.ndarray` and `nlcpy.ndarray`.
    Note that mpi4py-ve requires NEC MPI runtime packages.

    For details, please refer to the `mpi4py-ve project <https://github.com/SX-Aurora/mpi4py-ve>`_

2. :ref:`VE Device Management <label_venode>`

    You can select execution VE device in a Python script by using "with" context manager.

    ::

        import nlcpy
        with nlcpy.venode.VE(0):
            # do something on VE#0
            x_ve0 = nlcpy.arange(10)
        with nlcpy.venode.VE(1):
            # do something on VE#1
            y_ve1 = nlcpy.arange(10)
        # transfer x_ve0 to VE#1
        x_ve1 = nlcpy.venode.transfer_array(x_ve0, nlcpy.venode.VE(1))
        with nlcpy.venode.VE(1):
            # do something on VE#1
            z_ve1 = x_ve1 + y_ve1

    Note that :func:`nlcpy.venode.transfer_array` cannot yet transfer data directly between VEs, in other words, the data transfer between VEs has to go through VH.
    If you need to transfer large data, we recommend using mpi4py-ve.

    By default, NLCPy creates a VE process on only physical VE#0 when executing ``import nlcpy``.
    Other VE processes will be created at first call of :meth:`nlcpy.venode.VENode.__enter__`, :meth:`nlcpy.venode.VENode.use`, or :meth:`nlcpy.venode.VENode.apply`.
    Please note that creation of VE process takes few seconds.
    If you want to create processes on some VEs at ``import nlcpy``, you can set an environment variable ``VE_NLCPY_NODELIST=0,1,...``.
    NLCPy creates processes at ``import nlcpy`` for the physical VEs corresponding to the IDs set by ``VE_NLCPY_NODELIST``.

    When environment variable ``VE_NLCPY_NODELIST`` is set, it may be different from the argument id of :func:`nlcpy.venode.VE` to
    the physical VE id.

    Example of VE device mapping when environment variable is set by ``VE_NLCPY_NODELIST=1,2`` is the following:

    ::

        $ VE_NLCPY_NODELIST=1,2 python
        >>> import nlcpy
        >>> nlcpy.venode.VE(0)
        <VE node logical_id=0, physical_id=1>
        >>> nlcpy.venode.VE(1)
        <VE node logical_id=1, physical_id=2>


.. _label_fast_math:

Optimization for Mathematical Functions
---------------------------------------

When the environment variable ``VE_NLCPY_FAST_MATH`` is set to ``yes`` or ``YES``,
NLCPy uses shared object ``libnlcpy_ve_kernel_fast_math.so`` for VE.
By default, ``VE_NLCPY_FAST_MATH`` is not set.
The shared objects ( ``libnlcpy_ve_kernel_fast_math.so`` ) have been compiled with the following optimization options in NEC C/C++ compiler.

* *-ffast-math*

    Uses fast scalar version mathematical functions outside of vectorized loops.

* *-mno-vector-intrinsic-check*

    | Disable vectorized mathematical functions to check the value ranges of arguments.
    | The target mathematical functions of this option are as follows:
    | ``acos``, ``acosh``, ``asin``, ``atan``, ``atan2``, ``atanh``, ``cos``, ``cosh``, ``cotan``, ``exp``, ``exp10``, ``exp2``, ``expm1``, ``log10``, ``log2``, ``log``, ``pow``, ``sin``, ``sinh``, ``sqrt``, ``tan``, ``tanh``

* *-freciprocal-math*

    Allows change an expression ``x/y`` to ``x * (1/y)``.

* *-mvector-power-to-explog*

    Allows to replace ``pow(R1,R2)`` in a vectorized loop with ``exp(R2*log(R1))``.
    ``powf()`` is replaced, too.
    By the replacement, the execution time would be shortened, but numerical error occurs rarely in the calculation.

* *-mvector-low-precise-divide-function*

    Allows to use low precise version for vector floating divide operation.
    It is faster than the normal precise version but the result may include at most one bit numerical error in mantissa.

These optimizations cause side-effects.
For example, ``nan`` or ``inf`` might not be obtained correctly.

You can set ``VE_NLCPY_FAST_MATH`` as follows:

* Interactive mode:

    ::

        $ VE_NLCPY_FAST_MATH=yes python
        >>> import nlcpy

* Non-interactive mode:

    ::

        $ VE_NLCPY_FAST_MATH=yes python example.py


.. _label_mempool:

Memory Pool Management
----------------------

NLCPy reduces overhead of VE memory allocation by reusing pre-allocated memory (memory pool) without calling malloc and free.
You can control amount of memory pool by an environment variable ``VE_NLCPY_MEMPOOL_SIZE``.
The default value is set 1 GB.
For usage of this variable, please refer to the :ref:`Environment Variables <label_envs>`

In some cases, setting this variable to larger value than 1 GB may improve performance.
However, an out of memory error may be caused by memory fragmentation.

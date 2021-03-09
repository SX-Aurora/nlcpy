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


Optimization for Mathematical Functions
---------------------------------------

When the environment variable ``VE_NLCPY_FAST_MATH`` is set to ``yes`` or ``YES``,
NLCPy uses shared objects ``nlcpy_ve_kernel_fast_math.so`` for VE.
By default, ``VE_NLCPY_FAST_MATH`` is not set.
The shared objects ( ``nlcpy_ve_kernel_fast_math.so`` ) have been compiled with the following optimization options in NEC C/C++ compiler.

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

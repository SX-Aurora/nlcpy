.. _label_envs:

Environment Vaiables
====================

.. envvar:: VE_NLCPY_NODELIST

    Sets the execution VE node ids.

    Example when using VE 0,1,2:

    ::

        $ export VE_NLCPY_NODELIST=0,1,2


.. envvar:: VE_NLCPY_MEMPOOL_SIZE

    Default: ``1G``

    Sets the upper limit of memory amount that can be allocated from memory pool.
    You can specify the suffixes B, b, K, k, M, m, G, or g.

    ================= =================
    ================= =================
    ``B`` or ``b``    Bytes
    ``K`` or ``k``    Kilobytes
    ``M`` or ``m``    Megabytes
    ``G`` or ``g``    Gigabytes
    ================= =================

    If you don't specify any suffixes, the value is represented in Kilobytes.

    Example that sets pool size 10 gigabytes:

    ::

        $ export VE_NLCPY_MEMPOOL_SIZE=10G

.. envvar:: VE_NLCPY_FAST_MATH

    Default: ``NO``

    If set to ``YES`` or ``yes``, NLCPy uses shared object ``libnlcpy_ve_kernel_fast_math.so`` for VE.
    For details, please see :ref:`Optimization for Mathematical Functions <label_fast_math>`

.. envvar:: VE_NLCPY_ENABLE_NUMPY_WRAP

    Default: ``YES``

    If set to ``NO`` or ``no``, NLCPy raises `Exception` when replacing NLCPy function and method to
    NumPy's one.

.. envvar:: VE_OMP_NUM_THREADS

    Default: The number of VE cores.

    The number of OpenMP parallel threads.

    ::

        $ export VE_OMP_NUM_THREADS=4

.. envvar:: VE_NODE_NUMBER

    Default: ``0``

    The node number of VE to be executed.
    Note that if both ``VE_NLCPY_NODELIST`` and ``VE_NODE_NUMBER`` are set, ``VE_NLCPY_NODELIST`` takes precedence.

    ::

        $ export VE_NODE_NUMBER=1

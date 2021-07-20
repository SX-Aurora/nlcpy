.. _label_sca_basic:

Basic Usage
===========

This section describes how to call the SCA interface in your program.

Overview
--------

The procedure using the SCA interface consists of the following five steps.

1. :ref:`label_sca_creation_desc`
2. :ref:`label_sca_definition_desc`
3. :ref:`label_sca_creation_kernel`
4. :ref:`label_sca_execution_kernel`
5. :ref:`label_sca_destruction_kernel`

Here is a simple example that adds adjacent elements of an one-dimensional array ``xin``, whose size is 10, and stores the result to ``xout``.

.. code-block:: python

    >>> import nlcpy
    >>> xin = nlcpy.arange(10, dtype='f4')
    >>> xout = nlcpy.zeros_like(xin)
    >>> dxin, dxout = nlcpy.sca.create_descriptor((xin, xout)) # Creation of Stencil Descriptor
    >>> desc_i = dxin[-1] + dxin[0] + dxin[1] # Definition of Stencil Description
    >>> desc_o = dxout[0]
    >>> kern = nlcpy.sca.create_kernel(desc_i, desc_o=desc_o) # Creation of Kernel
    >>> res = kern.execute() # Execution of Kernel
    >>> res
    array([ 0., 3., 6., 9., 12., 15., 18., 21., 24., 0.], dtype=float32)
    >>> nlcpy.sca.destroy_kernel(kern) # Destruction of Kernel

.. _label_sca_creation_desc:

Creation of Stencil Descriptor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A stencil descriptor can be created from a :class:`nlcpy.ndarray` used in the stencil computation.
The stencil descriptor is a Python object that can represent a stencil shape and is associated with the :class:`nlcpy.ndarray`.

In [1]:

.. code-block:: python

    >>> xin = nlcpy.arange(10, dtype='f4')
    >>> xout = nlcpy.zeros_like(xin)
    >>> dxin, dxout = nlcpy.sca.create_descriptor((xin, xout))

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.sca.create_descriptor

.. _label_sca_definition_desc:

Definition of Stencil Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Elements of stencil descriptor described above can concretely define a stencil description, which means "stencil shape".
The stencil description can be denoted by relative indices of the stencil descriptor.
The following example defines that adds adjacent elements for each element of a one-dimensional array:

In [2]:

.. code-block:: python

    >>> desc_i = dxin[-1] + dxin[0] + dxin[1]
    >>> desc_i

Out[2]:

.. code-block:: python

    stencil description
      in_0[0, 0, 0, -1] +
      in_0[0, 0, 0, 0] +
      in_0[0, 0, 0, 1]

    assigned arrays
      in_0: shape=(10,), dtype=float32 array

    computation size
      nx = 8, ny = 1, nz = 1, nw = 1

For details of how to set coefficients for the input description, please see :ref:`label_sca_coef`.

You can also define the output description if you need.
The output description is useful when you specify an array offset for the output.
For details of the array offset, please see :ref:`label_sca_offset`.

In [3]:

.. code-block:: python

    >>> desc_o = dxout[0]
    >>> desc_o

Out[3]:

.. code-block:: python

    stencil description
      in_0[0, 0, 0, 0]

    assigned arrays
      in_0: shape=(10,), dtype=float32 array

    computation size
      nx = 10, ny = 1, nz = 1, nw = 1

.. _label_sca_creation_kernel:

Creation of Kernel
^^^^^^^^^^^^^^^^^^
After defining the stencil description, you can create a SCA kernel, which is an instruction sequence required for computations defined by the stencil description. :func:`nlcpy.sca.create_kernel()` dynamically generates the instruction sequence, stores it into the memory on VE, and returns the object of the SCA kernel.

In [4]:

.. code-block:: python

    >>> kern = nlcpy.sca.create_kernel(desc_i, desc_o=desc_o)

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.sca.create_kernel

.. _label_sca_execution_kernel:

Execution of Kernel
^^^^^^^^^^^^^^^^^^^

After the creation of the SCA kernel, you can execute the SCA kernel.

In [5]:

.. code-block:: python

    >>> res = kern.execute()

In [6]:

.. code-block:: python

    >>> res

Out[6]:

.. code-block:: python

    array([ 0.,  3.,  6.,  9., 12., 15., 18., 21., 24.,  0.], dtype=float32)

If you specify desc_o as a keyword argument to :func:`nlcpy.sca.create_kernel()`, the :class:`nlcpy.ndarray` returned by :func:`nlcpy.sca.kernel.kernel.execute()` is identical to the :class:`nlcpy.ndarray` which is associated with ``desc_o``. The IDs of them are the same.


In [7]:

.. code-block:: python

    >>> id(res) == id(xout)

Out[7]:

.. code-block:: python

    True

.. _label_sca_destruction_kernel:

Destruction of Kernel
^^^^^^^^^^^^^^^^^^^^^

The destruction of the SCA kernel can be done as follows:

In [8]:

.. code-block:: python

    >>> nlcpy.sca.destroy_kernel(kern)

Even if you do not explicitly destroy the SCA kernel, it will be automatically destroyed by the garbage collector when there are no more references to the SCA kernel.
However, for programs where the reference to the SCA kernel remains to the end, it may squeeze memory, so it is recommended to destroy the SCA kernel properly when it is no longer used.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.sca.destroy_kernel


Speedup Method (TIPS)
---------------------

Stride Adjustment
^^^^^^^^^^^^^^^^^

Please use :func:`nlcpy.sca.convert_optimized_array()` to gain maximal performance.
This function converts ndarrays into optimized ndarrays, whose strides are adjusted to improve performance.
It is highly recommended to use this function from a performance standpoint, although it is not necessary to use it.
Note that :func:`nlcpy.sca.convert_optimized_array()` returns a copy of the input :class:`nlcpy.ndarray`, not a view. So, memory area of the returned :class:`nlcpy.ndarray` is different from that of the input :class:`nlcpy.ndarray`.

In [9]:

.. code-block:: python

    >>> import nlcpy
    >>> x = nlcpy.random.rand(1000, 1000)
    >>> x_opt = nlcpy.sca.convert_optimized_array(x, dtype='f8')
    >>> x.strides

Out[9]:

.. code-block:: python

    (8000, 8)

In [10]:

.. code-block:: python

    >>> x_opt.strides

Out[10]:

.. code-block:: python

    (8008, 8)

In [11]:

.. code-block:: python

    >>> nlcpy.all(x == x_opt)

Out[11]:

.. code-block:: python

    array(True)


.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.sca.convert_optimized_array
    nlcpy.sca.create_optimized_array

Kernel Reuse
^^^^^^^^^^^^

To gain maximal performance, it is strongly recommended to reuse the created SCA kernel if the stencil description or the coefficients of the stencil kernel is unchanged.
If you repeat to create SCA kernels, your program will not be able to obtain sufficient performance because the cost of the creating a SCA kernel is not so small compared to executing the kernel.

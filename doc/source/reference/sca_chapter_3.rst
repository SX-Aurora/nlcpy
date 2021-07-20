.. _label_sca_advanced:

Advanced Usage
==============
 
This section describes some details on the use of the SCA interface.

- :ref:`label_sca_coef`
- :ref:`label_sca_for_loop`
- :ref:`label_sca_auto_cre`
- :ref:`label_sca_offset`
- :ref:`label_sca_indexing`
- :ref:`label_sca_multi`
- :ref:`label_sca_complex`

.. _label_sca_coef:

Applying Coefficient
--------------------

    In the SCA interface, coefficients of stencil computation can be specified as a scalar or an array.
    For details of the coefficients, see also :ref:`label_sca_concept`.

    1. A scalar constant is set as the coefficient:

        You can set the coefficient as a scalar constant. Below is an easy example:

        In [1]:

        .. code-block:: python

            >>> x = nlcpy.arange(10.)
            >>> dx = nlcpy.sca.create_descriptor(x)
            >>> desc = 0.5 * dx[-1] - 1.5 * dx[1]
            >>> kern = nlcpy.sca.create_kernel(desc)
            >>> kern.execute()
 
        Out[1]:

        .. code-block:: python

            array([  0.,  -3.,  -4.,  -5.,  -6.,  -7.,  -8.,  -9., -10.,   0.])
 
    2. An element of an :class:`nlcpy.ndarray` is set as the coefficient:

        You can also set the coefficients as an element of an :class:`nlcpy.ndarray`:

        In [2]:

        .. code-block:: python

            >>> coef = nlcpy.array([0.5, 1.5])
            >>> desc = coef[0] * dx[-1] - coef[1] * dx[1]
            >>> kern = nlcpy.sca.create_kernel(desc)
            >>> kern.execute()

        Out[2]:

        .. code-block:: python

            array([  0.,  -3.,  -4.,  -5.,  -6.,  -7.,  -8.,  -9., -10.,   0.])
 
        Here, the operators with a description  (``dx[-1]`` or ``dx[1]``) are overloaded.
        The overloaded operators hold the address of the coefficient ``coef``, not the value of that.
        So if the value of the coefficient is updated, the SCA kernel refers to the value at the time.
        In other words, if the coefficients are given as an :class:`nlcpy.ndarray` object,
        the updated values can be used without recreating a SCA kernel.

        In [3]:

        .. code-block:: python

            # Execute after updating the coefficients
            >>> coef *= 2
            >>> kern.execute()

        Out[3]:

        .. code-block:: python

            array([  0.,  -6.,  -8., -10., -12., -14., -16., -18., -20.,   0.])

        .. note::

            If the operators with a description refers to one or more temporary elements, the SCA kernel does NOT 
            use the updated values after :func:`nlcpy.sca.create_kernel()` is called
            because the SCA kernel refers to each address of the temporary elements, not each value of those.
            Below is an easy example that ``1.0 * coef[0]`` and ``1.0 * coef[1]`` are specified as the 
            temporary elements.

            .. code-block:: python

                >>> coef = nlcpy.array([0.5, 1.5])
                >>> desc = 1.0 * coef[0] * dx[-1] - 1.0 * coef[1] * dx[1]
                >>> kern = nlcpy.sca.create_kernel(desc)
                >>> kern.execute()
                array([  0.,  -3.,  -4.,  -5.,  -6.,  -7.,  -8.,  -9., -10.,   0.])

                # Execute after updating the coefficients
                >>> coef *= 2
                >>> kern.execute()
                array([  0.,  -3.,  -4.,  -5.,  -6.,  -7.,  -8.,  -9., -10.,   0.]) # old coef used

            Similarly, if ``coef`` is set as ``numpy.ndarray``, updating ``coef`` does not affect the results
            because numpy.ndarray is converted into a temporaly :class:`nlcpy.ndarray` in :func:`nlcpy.sca.create_kernel()`.

            .. code-block:: python

                >>> coef = numpy.array([0.5, 1.5])
                >>> desc = coef[0] * dx[-1] - coef[1] * dx[1]
                >>> kern = nlcpy.sca.create_kernel(desc)

                # Execute after updating the coefficients
                >>> coef *= 2
                >>> kern.execute()
                array([  0.,  -3.,  -4.,  -5.,  -6.,  -7.,  -8.,  -9., -10.,   0.]) # old coef used

    3. An :class:`nlcpy.ndarray` is set as the coefficient:

        You can also set the coefficients as an :class:`nlcpy.ndarray`:

        In [4]:

        .. code-block:: python

            >>> coef = nlcpy.arange(10.)
            >>> desc = coef * dx[0]
            >>> kern = nlcpy.sca.create_kernel(desc)
            >>> kern.execute()

        Out[4]:

        .. code-block:: python

            array([ 0.,  1.,  4.,  9., 16., 25., 36., 49., 64., 81.])
 

        In[4] is similar to In[2]. The overloaded operator with the description ``dx[0]`` 
        hold the address of ``coef``, not the values of that.
        So when the values of ``coef`` are updated, the SCA kernel refers to the values at the time.

        In [5]:

        .. code-block:: python

            # Execute after updating the coefficients
            >>> coef *= 10
            >>> kern.execute()

        Out[5]:

        .. code-block:: python

            array([  0.,  10.,  40.,  90., 160., 250., 360., 490., 640., 810.])

        .. note::

            If the operators with a description refers to one or more temporary arrays, the SCA kernel does NOT 
            use the updated values after :func:`nlcpy.sca.create_kernel()` is called
            because the SCA kernel refers to each address of the temporary arrays, not each value of those.
            Below is an easy example that ``1.0 * coef`` is specified as a temporary array.

            .. code-block:: python

                >>> coef = nlcpy.arange(10.)
                >>> desc = 1.0 * coef * dx[0]
                >>> kern = nlcpy.sca.create_kernel(desc)
                >>> kern.execute()
                array([ 0.,  1.,  4.,  9., 16., 25., 36., 49., 64., 81.])

                # Execute after updating the coefficients
                >>> coef *= 10
                >>> kern.execute()
                array([ 0.,  1.,  4.,  9., 16., 25., 36., 49., 64., 81.])  # old coef used

            Similarly, if ``coef`` is set as ``numpy.ndarray``, updating ``coef`` does not affect the results
            because numpy.ndarray is converted into a temporaly :class:`nlcpy.ndarray` in :func:`nlcpy.sca.create_kernel()`.

.. _label_sca_for_loop:

Definition of Stencil Description Using for Loop
------------------------------------------------

You can also use "for-loop" in a definition of a stencil description. For example, if you want to perform a stencil computation consisting of 10 elements, it is possible to define using for-loop as follows:

In [6]:

.. code-block:: python

    >>> x = nlcpy.arange(30, dtype='f4')
    >>> dx = nlcpy.sca.create_descriptor(x)
    >>> desc = nlcpy.sca.empty_description()
    >>> for i in range(-10, 1):
    >>>     desc += dx[i]
    >>> nlcpy.sca.create_kernel(desc).execute()
 
Out[6]:

.. code-block:: python

    array([  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  55.,
            66.,  77.,  88.,  99., 110., 121., 132., 143., 154., 165., 176.,
           187., 198., 209., 220., 231., 242., 253., 264.], dtype=float32)
 
In the above example, :func:`nlcpy.sca.empty_description` is used to create an empty stencil description, and 10 neighboring stencil descriptoions are added to it.
Depending on the application, it is possible to create more complex stencil descriptions.
The following is an example of creating the 3x3 XY planar stencil description.
 
In [7]:

.. code-block:: python

    >>> x = nlcpy.arange(25, dtype='f4').reshape(5,5)
    >>> dx = nlcpy.sca.create_descriptor(x)
    >>> desc = nlcpy.sca.empty_description()
    >>> for i in range(-1, 2):
    >>>     for j in range(-1, 2):
    >>>         desc += dx[i, j]
    >>> nlcpy.sca.create_kernel(desc).execute()
 
Out[7]:

.. code-block:: python

    array([[  0.,   0.,   0.,   0.,   0.],
           [  0.,  54.,  63.,  72.,   0.],
           [  0.,  99., 108., 117.,   0.],
           [  0., 144., 153., 162.,   0.],
           [  0.,   0.,   0.,   0.,   0.]], dtype=float32)


.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.sca.empty_description  
 
.. _label_sca_auto_cre:

Automatic Creation of Output Array
------------------------------------
If the keyword argument ``desc_o`` of :func:`nlcpy.sca.create_kernel` is omitted, an output :class:`nlcpy.ndarray` is automatically created.

In [8]:

.. code-block:: python

    >>> nlcpy.sca.create_kernel(desc_i).execute()

Out[8]:

.. code-block:: python

    array([ 0.,  3.,  6.,  9., 12., 15., 18., 21., 24.,  0.], dtype=float32)
 
.. _label_sca_offset:

Offset Adjustment for Output Array
----------------------------------

The offset of the output :class:`nlcpy.ndarray` can be adjusted by the keyword argument ``desc_o`` of :func:`nlcpy.sca.create_kernel`. For example, if you want to set the offset to -1, ``desc_o`` is specified as follows:

In [9]:

.. code-block:: python

    >>> xout[...] = 0
    >>> nlcpy.sca.create_kernel(desc_i, desc_o=dxout[-1]).execute()
 
Out[9]:

.. code-block:: python

    array([ 3.,  6.,  9., 12., 15., 18., 21., 24.,  0.,  0.], dtype=float32)
 
If you want to set the offset to 1, ``desc_o`` is specified as follows:
In [10]:

.. code-block:: python

    >>> xout[...] = 0
    >>> nlcpy.sca.create_kernel(desc_i, desc_o=dxout[1]).execute()
 
Out[10]:

.. code-block:: python

   array([ 0.,  0.,  3.,  6.,  9., 12., 15., 18., 21., 24.], dtype=float32)

.. note::

   Note that an IndexError occurs if the offset position is an out-of-range reference.

   .. code-block:: python

       >>> xout[...] = 0
       >>> nlcpy.sca.create_kernel(desc_i, dxout[-2]).execute()

       ...
       IndexError

.. _label_sca_indexing:

Simplified Indexing
-------------------
Indexing for stencil elements can be simplified by using ``Ellipsis (...)`` or ``Slice(None) (:)``.
Any axis specified by ``...`` or ``:`` is interpreted as that indexing starts at 0.
 
In [11]:

.. code-block:: python

    >>> dx = nlcpy.sca.create_descriptor(nlcpy.random.rand(5,5,5,5))
    >>> dx[...]
 
Out[11]:

.. code-block:: python

    stencil description
      in_0[0, 0, 0, 0]
    assigned arrays
      in_0: shape=(5, 5, 5, 5), dtype=float64 array
    computation size
      nx = 5, ny = 5, nz = 5, nw = 5
 
In [12]:

.. code-block:: python

    >>> dx[..., 1]
 
Out[12]:

.. code-block:: python

    stencil description
      in_0[0, 0, 0, 1]
    assigned arrays
      in_0: shape=(5, 5, 5, 5), dtype=float64 array
    computation size
      nx = 4, ny = 5, nz = 5, nw = 5
 
In [13]:

.. code-block:: python

    >>> dx[..., 1, :]
 
Out[13]:

.. code-block:: python

    stencil description
      in_0[0, 0, 1, 0]
    assigned arrays
      in_0: shape=(5, 5, 5, 5), dtype=float64 array
    computation size
      nx = 5, ny = 4, nz = 5, nw = 5
 
In [14]:

.. code-block:: python

    >>> dx[:, 1, ...]
 
Out[14]:

.. code-block:: python

    stencil description
      in_0[0, 1, 0, 0]
    assigned arrays
      in_0: shape=(5, 5, 5, 5), dtype=float64 array
    computation size
      nx = 5, ny = 5, nz = 4, nw = 5
 
.. _label_sca_multi:

Stencil Calculation Using Multiple Ndarrays
-------------------------------------------
Even if there are two or more ``ndarrays`` used for stencil computations, the same procedure can be used.

In [15]:

.. code-block:: python

    >>> x = nlcpy.arange(5, dtype='f4')
    >>> y = nlcpy.arange(5, 0, -1, dtype='f4')
    >>> dx, dy = nlcpy.sca.create_descriptor((x, y))
    >>> nlcpy.sca.create_kernel(dx[-1] + dy[1]).execute()
 
Out[15]:

.. code-block:: python

    array([0., 3., 3., 3., 0.], dtype=float32)

When shapes are different between multiple ``ndarrays``, the narrowest range is selected as the effective range.
Below is an example of ``x.shape = (5, 5)``, ``y.shape = (4, 6)``
 
In [16]:

.. code-block:: python

    >>> x = nlcpy.arange(5 * 5, dtype='f4').reshape(5, 5)
    >>> y = nlcpy.arange(4 * 6, dtype='f4').reshape(4, 6)
    >>> dx, dy = nlcpy.sca.create_descriptor((x, y))
    >>> desc = dx[-1, 0] + dx[1, 0] + dx[0, -1] + dx[0, 1] + dy[-1, 0] + dy[1, 0] + dy[0, -1] + dy[0,1]
    >>> nlcpy.sca.create_kernel(desc).execute()
 
Out[16]:

.. code-block:: python

    array([[  0.,   0.,   0.,   0.,   0.],
           [  0.,  52.,  60.,  68.,   0.],
           [  0.,  96., 104., 112.,   0.],
           [  0.,   0.,   0.,   0.,   0.]], dtype=float32)


In the above example
    - Effective range of x: [1:4, 1:4]

    - Effective range of y: [1:3, 1:5]

Therefore, the effective range of the output :class:`nlcpy.ndarray` becomes ``[1: 3, 1: 4]``, which is the minimum range for each axis.
In this example, the output is automatically created, but when specifying the output destination :class:`nlcpy.ndarray`, the following condition must be satisfied.
 
    - | out.shape[i] >= min(in1.shape[i], in2.shape[i], ..., inN.shape[i])
      | i: 0 to n (For n-dimensional arrays)

When the output array is automatically created, an :class:`nlcpy.ndarray` whose shape meets above condition is returned by :func:`nlcpy.sca.create_kernel`.

.. _label_sca_complex:

Stencil Calculation for Complex Types
-------------------------------------

The SCA interface does not support complex number types. When performing stencil computations for complex numbers, use :obj:`nlcpy.ndarray.real` and :obj:`nlcpy.ndarray.imag` as follows:


In [17]:

.. code-block:: python

    >>> xin = nlcpy.arange(10) + nlcpy.arange(10, 0, -1) * 1j
    >>> xout = nlcpy.zeros_like(xin)
    >>> dRe, dIm = nlcpy.sca.create_descriptor((xin.real, xin.imag))
    >>> xout.real = nlcpy.sca.create_kernel(dRe[-1] + dRe[1]).execute()
    >>> xout.imag = nlcpy.sca.create_kernel(dIm[-1] + dIm[1]).execute()
    >>> xout
 
Out[17]:

.. code-block:: python

    array([ 0. +0.j,  2.+18.j,  4.+16.j,  6.+14.j,  8.+12.j, 10.+10.j,
           12. +8.j, 14. +6.j, 16. +4.j,  0. +0.j])
 

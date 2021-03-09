.. _notices:

Notices and Restrictions
========================

.. contents:: :local:

This page describes notices and restrictions which are common to NLCPy functions.

Notices
-------

1. To use NLCPy in your Python scripts, the package ``nlcpy`` must be imported. For more details, see :ref:`Basic Usage <basic_usage>`.

2. To reduce overhead between Vector Host and Vector Engine, NLCPy adopts the lazy evaluation, which means that values are not calculated until they are required. So the position of warnings where your Python script raised may not be accurate. For details, see :ref:`Lazy Evaluation <lazy>`.

    * Example:

        .. code-block:: python

            # sample.py
            import nlcpy as vp
            a = vp.divide(1, 0) # divide by zero warning
            b = a + 1
            print(b)
    
    * Results:

        ::

            $ python sample.py
            sample.py:5: RuntimeWarning: divide by zero encountered in nlcpy.core.core
              print(b)
            inf

3. NLCPy API is based on NumPy one. However, there are some differences due to performance reasons. For example, when NumPy function returns a scalar value, NLCPy function returns it as a 0-dimension array.

    .. doctest::

        >>> import numpy, nlcpy
        >>> numpy.add(1,2)         # returns a scalar
        3
        >>> nlcpy.add(1,2)         # returns an array
        array(3)
        >>> nlcpy.add(1,2).ndim    # print the dimention of nlcpy.add(1.2)
        0

4. Vector Host (x86) supports denormal numbers, whereas Vector Engine does NOT support it. So, if denormal numbers are caluculated in NLCPy functions, they are rounded to zero.

    .. doctest::

        >>> import numpy, nlcpy
        >>> x=numpy.array([-1.e-310, +1.e-310])    # denormal numbers in IEEE754 double precision format.
        >>> x
        array([-1.e-310,  1.e-310])
        >>>
        >>> numpy.add(x,0)   # doctest: +SKIP
        array([-1.e-310,  1.e-310])                # works on Vector Host
        >>>
        >>> nlcpy.add(x,0)                         # works on Vector Engine
        array([0., 0.])

5. NumPy functions run on an x86 Node (VH). On the other hand, most of NLCPy functions offload automatically input ndarrays to a Vector Engine (VE), and then run on the VE. So, as computational cost becomes smaller than the offloading cost, NLCPy performance decreases. In this case, please use NumPy.

Restrictions
------------

Here is a list of restrictions which are common to NLCPy functions.
Besides these restrictions, there are some individual restrictions.
Please see also the item of "Restrictions" in the detailed description of each function.

1. Data type, which is called "dtype", can be specified for NLCPy functions like NumPy ones. However, the current version of NLCPy supports only the following dtypes:

    ============================== =============================== ================
    data-type                      dtype                           character code
    ============================== =============================== ================
    bool                           'bool'                          '?'
    32-bit signed integer          'int32', 'i4'                   'i'
    64-bit signed integer          'int64', 'i8', int              'l', 'q'
    32-bit unsigned integer        'uint32', 'u4'                  'I'
    64-bit unsigned integer        'uint64', 'u8', uint            'L', 'Q'
    32-bit floating-point real     'float32', 'f4'                 'f'
    64-bit floating-point real     'float64', 'f8', 'float'        'd'
    32-bit floating-point complex  'complex64', 'c8'               'F'
    64-bit floating-point complex  'complex128', 'c16', 'complex'  'D'
    ============================== =============================== ================

    Each dtype has character codes that identify it.
    In NLCPy, the character code 'q' and 'Q' are internally converted to 'l' and 'L', respectively.
    The dtypes and character codes other than described above are not supported yet. 
    In addition, the current version does not support a structured data type, which contains above dtypes.

    Please note that there are functions which can not even support above dtypes.
    For example, the complex version of :func:`nlcpy.mean()` does not support.

    .. doctest::

        >>> import nlcpy
        >>> nlcpy.mean(nlcpy.array([1,2,3],dtype='complex64'))   # doctest: +SKIP
        NotImplementedError: dtype=complex64 not supported


2. If the unsupported dtype appears in the parameter list or the return type for NLCPy function, *TypeError* occurs. In case a NumPy function treats float16 type internally, the corresponding NLCPy function treats it as float32. Similarly, int8, int16, uint8, or uint16 is treated as int32 or uint32 during calculations. In such case the return value of NLCPy differs from that of NumPy.

    .. doctest::

        >>> import numpy
        >>>
        >>> # numpy.divide.accumulate() treats 1e-8 as float16.
        >>> numpy.divide.accumulate([1e-8], dtype='bool')   
        array([0.], dtype=float16)
        >>>
        >>> # 1e-8 is internally rounded to zero in float16 type,
        >>> # so the boolean result becomes False.
        >>> numpy.divide.accumulate([1e-8], dtype='bool', out=numpy.array([True]))
        array([False])

    .. doctest::

        >>> import nlcpy
        >>>
        >>> # NLCPy does not support float16, so TypeError occors.
        >>> nlcpy.divide.accumulate([1e-8], dtype='bool') # doctest: +SKIP 
        TypeError: not support for float16.
        >>>
        >>> # 1e-8 is treated as float32, so the boolean result becomes True.
        >>> nlcpy.divide.accumulate([1e-8], dtype='bool', out=nlcpy.array([True]))
        array([ True])



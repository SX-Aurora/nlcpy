.. _interoperability:

Interoperability between NumPy and NLCPy
========================================

NLCPy provides a subset of NumPy's API.
However, NLCPy ndarray (:class:`nlcpy.ndarray`) implements *__array__* method.
It enables you to pass NLCPy ndarray to a NumPy function.

* Example 1:

.. doctest::

        >>> import nlcpy
        >>> import numpy
        >>>
        >>> x = nlcpy.arange(3)
        >>> y = nlcpy.arange(3)
        >>> numpy.meshgrid(x, y)
        [array([[0, 1, 2],
               [0, 1, 2],
               [0, 1, 2]]), array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]])]

Please note that `numpy.meshgrid <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>`_ is the NumPy function, whereas ``x`` and ``y`` are both NLCPy ndarrays.

.. note::
    NumPy functions run on an x86 Node (VH). On the other hand, most of NLCPy functions
    offload automatically input ndarrays to a Vector Engine (VE), and then run on the VE.

| You can also pass a NLCPy ndarray to a function of other packages which uses ``numpy.ndarray``.
| Here are easy examples of Matplotlib and Pandas.

* Example 2 (Matplotlib):

.. plot::
    :align: center

    >>> import nlcpy as vp
    >>> import matplotlib.pylab as plt
    >>>
    >>> x = vp.linspace(-vp.pi, vp.pi, 201)
    >>> type(x)
    <class 'nlcpy.core.core.ndarray'>
    >>>
    >>> plt.plot(x, vp.sin(x)) # doctest: +SKIP
    ... # doctest: +ELLIPSIS
    >>> plt.xlabel('Angle [rad]') # doctest: +SKIP
    ... # doctest: +ELLIPSIS
    >>> plt.ylabel('sin(x)') # doctest: +SKIP
    ... # doctest: +ELLIPSIS
    >>> plt.axis('tight') # doctest: +SKIP
    ... # doctest: +ELLIPSIS
    >>> plt.show()

* Example 3 (Pandas):

.. doctest::

    >>> import nlcpy as vp
    >>> import pandas as pd
    >>>
    >>> x = vp.random.rand(3,3)
    >>> pd.DataFrame(x, index=list('abc'), columns=list('ABC'))   # doctest: +SKIP
              A                      B                   C
    a  0.6575677252840251     0.7966675218194723  0.5927528077736497
    b  0.1310200293082744  0.0033949704375118017  0.4242931657936424
    c   0.343795241555199       0.88619629223831  0.9364728704094887

If you convert between :class:`nlcpy.ndarray` and ``numpy.ndarray``,
you should use :func:`nlcpy.asarray` and `numpy.asarray() <https://numpy.org/doc/stable/reference/generated/numpy.asarray.html>`_.

In Example 4, :func:`nlcpy.asarray` transfers a NumPy ndarray from VH to VE, and then the ndarray is represented as a NLCPy ndarray.
Conversely, `numpy.asarray() <https://numpy.org/doc/stable/reference/generated/numpy.asarray.html>`_ transfers a ndarray from VE to VH.

* Example 4:

.. doctest::

    >>> x = numpy.arange(5)
    >>> type(x)
    <class 'numpy.ndarray'>
    >>>
    >>> x = nlcpy.asarray(x)  # converts from numpy.ndarray to nlcpy.core.core.ndarray
    >>> type(x)
    <class 'nlcpy.core.core.ndarray'>
    >>>
    >>> x = numpy.asarray(x)  # converts from nlcpy.core.core.ndarray to numpy.ndarray
    >>> type(x)
    <class 'numpy.ndarray'>
    >>>
    >>> x
    array([0, 1, 2, 3, 4])

In addition, :meth:`nlcpy.ndarray.get()` returns a NumPy ndarray whose data is transferred from VE to VH.

* Example 5:

.. doctest::

    >>> x = nlcpy.arange(5)
    >>> type(x)
    <class 'nlcpy.core.core.ndarray'>
    >>>
    >>> y = x.get()  # converts from nlcpy.core.core.ndarray to numpy.ndarray.
    >>> type(y)
    <class 'numpy.ndarray'>
    >>>
    >>> y
    array([0, 1, 2, 3, 4])


.. note::

    When both :class:`nlcpy.ndarray` and ``numpy.ndarray`` are passed to a NLCPy function,
    the function returns the result as :class:`nlcpy.ndarray`. Conversely, a NumPy
    function returns the result as ``numpy.ndarray``.

    .. doctest::

        >>> import numpy, nlcpy
        >>> x = nlcpy.arange(10)
        >>> y = numpy.arange(10)
        >>>
        >>> type(x+y)   # doctest: +SKIP
        <class 'nlcpy.core.core.ndarray'>   # ndarray of nlcpy
        >>>
        >>> type(nlcpy.add(x,y))   # doctest: +SKIP
        <class 'nlcpy.core.core.ndarray'>   # ndarray of nlcpy
        >>>
        >>> type(numpy.add(x,y))   # doctest: +SKIP
        <class 'numpy.ndarray'>             # ndarray of numpy


.. _label_auto_replace:

Auto Replacing to NumPy
-----------------------

Since v2.2.0, NLCPy automatically replace almost functions and methods that implemented not yet to the NumPy's one.

Before v2.2.0:

    ::

        >>> import nlcpy
        >>> nlcpy.nancumsum(nlcpy.array(1))
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          AttributeError: module 'nlcpy' has no attribute 'nancumsum'

Since v2.2.0:

    ::

        >>> import nlcpy
        >>> nlcpy.nancumsum(nlcpy.array(1))
        array([1])
        # 1. transfer argument value to VH
        # 2. execute NumPy's function
        # 3. transfer result value to VE

If you want to limit this feature, please set an environment variable ``VE_NLCPY_ENABLE_NUMPY_WRAP`` to ``NO`` or ``no``.
The default is ``YES``.

    ::

        $ VE_NLCPY_ENABLE_NUMPY_WRAP=NO python
        >>> import nlcpy
        >>> nlcpy.nancumsum(nlcpy.array(1))
        ...
        NotImplementedError: nancumsum is not implemented yet.

.. note::
    When using Python3.6, this feature does not work for module level functions; such as ``nlcpy.xxx``, ``nlcpy.linalg.xxx``, or ``nlcpy.random.xxx``.

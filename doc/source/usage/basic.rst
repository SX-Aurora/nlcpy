.. _basic_usage:

Basic Usage
===========

.. contents:: :local:


Preparation
-----------

NLCPy uses the functions of `NEC Numeric Library collection (NLC) <https://sxauroratsubasa.sakura.ne.jp/documents/sdk/SDK_NLC/UsersGuide/main/en/>`_.
Before importing this package, you need to execute the environment setup script ``nlcvars.sh``
or ``nlcvars.csh`` once in advance.

* When using ``sh`` or its variant:

::

    $ source /opt/nec/ve/bin/nlcvars.sh

* When using ``csh`` or its variant:

::

    % source /opt/nec/ve/bin/nlcvars.csh


Preparation with specifying the version of NEC Numeric Library Collection
-------------------------------------------------------------------------

If you want to specify the version of the NEC Numeric Library Collection, specify the version in the following manner before executing.

**For VE30**

* When using ``sh`` or its variant:

::

    $ source /opt/nec/ve3/nlc/X.X.X/bin/nlcvars.sh

* When using ``csh`` or its variant:

::

    % source /opt/nec/ve3/nlc/X.X.X/bin/nlcvars.csh

**For VE20, VE10, or VE10E**

* When using ``sh`` or its variant:

::

    $ source /opt/nec/ve/nlc/X.X.X/bin/nlcvars.sh

* When using ``csh`` or its variant:

::

    % source /opt/nec/ve/nlc/X.X.X/bin/nlcvars.csh

Here, **X.X.X** denotes the version number of NEC Numeric Library Collection.


Supported Python Versions
-------------------------

NLCPy is available from Python version 3.6, 3.7, or 3.8.

.. note::

    - NLCPy is not supported for Python version 3.5 or earlier.

    - This manual is intended for users who will use Python version 3.6, 3.7, or 3.8.


Import Package
--------------

When you use NLCPy in your Python scripts, the package ``nlcpy`` must be imported.

* When running the scripts using NLCPy in interactive mode:

    ::

        $ python
        >>> import nlcpy

* When running the scripts using NLCPy in non-interactive mode:

    ::

        import nlcpy


If you get something similar to the following errors, NLCPy or NEC Numeric Library Collection probably is not installed,
or is not executed the environment setup script mentioned above.
Please see also :ref:`installation guide <installation>`.

::

    import nlcpy
    ...
    ModuleNotFoundError: No module named 'nlcpy.core.core'

, or

::

    import nlcpy
    ...
    RuntimeError: veo_load_library 'b'/path_to_nlcpy/lib/nlcpy_ve_kernel_no_fast_math.so'' failed

After you import ``nlcpy`` successfully in Python scripts,
the scripts can use :class:`nlcpy.ndarray` of
NLCPy and functions described in :ref:`Reference <nlcpy_reference>` .

An easy example of NLCPy script is shown below:

.. doctest::

    >>> import nlcpy as vp
    >>> vp.add(1.0, 4.0)
    array(5.)
    >>> x1 = vp.arange(9.0).reshape((3, 3))
    >>> x2 = vp.arange(3.0)
    >>> vp.add(x1, x2)
    array([[ 0.,  2.,  4.],
           [ 3.,  5.,  7.],
           [ 6.,  8., 10.]])

In addition, the current version of NLCPy provides the
following operators of the :class:`nlcpy.ndarray` class:

================================ =========================================================
================================ =========================================================
Assignment operator              ``=``
Arithmetic operators             ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``
Arithmetic assignment operators  ``+=``, ``-=``, ``*=``, ``/=``, ``//=``, ``%=``, ``**=``
Matrix multiplication operator   ``@``
Comparison operators             ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``
Bitwise operators                ``&``, ``|``, ``^``, ``~``
Bitwise assignment operators     ``&=``, ``|=``, ``^=``
Bit-shift operators              ``<<``, ``>>``
Bit-shift assignment operators   ``<<=``, ``>>=``
Logical operators                ``and``, ``or``, ``xor``, ``not``
================================ =========================================================


.. note::

    In-place matrix multiplication operator ``@=`` is not implemented yet.

.. _installation:

Installation Guide
==================

This page describes installation of NLCPy.

.. attention::
    - Since April 2021, NLCPy has been provided as a software of NEC SDK (NEC Software
      Development Kit for Vector Engine). If NEC SDK on your machine has been properly
      installed or updated after that, NLCPy is available by using ``/usr/bin/python3``
      command, and the installation described in this page is not needed.
      However, when you use another Python command such as ``/usr/local/bin/python3``
      and `python3` in a virtual environment, please install NLCPy from a wheel
      package or source files described in this page.

    .. seealso::
        `SX-Aurora TSUBASA Installation Guide
        <https://www.hpc.nec/documents/guide/pdfs/InstallationGuide_E.pdf>`_

    - The libraries of NLCPy are located in the following directory after NEC SDK
      is installed or updated::

          /opt/nec/ve/nlcpy/X.X.X/lib/python36/nlcpy

      Here, X.X.X denotes the version of NLCPy.
      If you install or update NEC SDK, the directory of the latest version of
      NLCPy is added in Python module search path. When you use a specific
      version of NLCPy, the evironmental variable **PYTHONPATH** must be set as
      follows::

          $ export PYTHONPATH=/opt/nec/ve/nlcpy/X.X.X/lib/python36/

Requirements
------------

Before the installation of NLCPy, the following components are required to be
installed on your x86 Node of SX-Aurora TSUBASA.

* | `NEC SDK <https://www.hpc.nec/documents/guide/pdfs/InstallationGuide_E.pdf>`_

    - required NEC C/C++ compiler version: >= 3.2.1
    - required NLC version: >= 2.3.0

* | `VEOS <https://www.hpc.nec/documentation/>`_

    - required version: >= 2.8.1

* | `Python <https://www.python.org/>`_

    - required version: 3.6, 3.7, or 3.8

* | `NumPy <https://www.numpy.org/>`_

    - required version: v1.17, v1.18, v1.19, or v1.20


Install from wheel
------------------

You can install NLCPy by executing either of following commands.

* Install from PyPI

  ::

      $ pip install nlcpy


* Install from your local computer

    1. Download the wheel package from `GitHub <https://github.com/SX-Aurora/nlcpy/>`_.
    2. Put the wheel package to your any directory.
    3. Install the local wheel package via pip command.

       ::

           $ pip install <path_to_wheel>

The shared objects for Vector Engine, which are included in the wheel package, are compiled and tested by using NEC C/C++ Version 3.2.1 and NumPy v1.19.2.


Install from source (with building)
-----------------------------------

Before building source files, please install following packages.

::

    $ pip install numpy cython wheel

And, entering the following commands in the environment where `gcc` and `ncc` commands are available.

::

    $ git clone https://github.com/SX-Aurora/nlcpy.git
    $ cd nlcpy
    $ pip install .


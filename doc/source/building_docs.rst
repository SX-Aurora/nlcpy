.. _building_docs:

Building the Documentation
==========================

This page describes how to build this documentation.


Requirements
------------

Before building the documents, the following packages are required to be installed on your x86 Node of SX-Aurora TSUBASA:

* `Sphinx <https://www.sphinx-doc.org/en/master/>`_
* `Read the Docs Sphinx Theme <https://sphinx-rtd-theme.readthedocs.io/en/latest/#>`_
* `Matplotlib <https://matplotlib.org/>`_
* `NumPy <https://numpy.org/>`_

Building
--------

Before building the documents, please download the package from GitHub::

    $ git clone https://github.com/SX-Aurora/nlcpy.git

And, entering the following commands::

    $ cd nlcpy
    $ sh build_inplace.sh
    $ cd doc
    $ make html

The documentation is created in the following directory::

    $ ls build/html/{en, ja}

.. _nlcpy_linalg:

Linear Algebra
==============

The following table shows linear algebra routines provided by NLCPy.

Matrix and Vector Products
--------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.dot
    nlcpy.inner
    nlcpy.outer
    nlcpy.matmul


Decompositions
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.linalg.svd
    nlcpy.linalg.cholesky
    nlcpy.linalg.qr


Matrix Eigenvalues
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.linalg.eig
    nlcpy.linalg.eigh
    nlcpy.linalg.eigvals
    nlcpy.linalg.eigvalsh


Norms and Other Numbers
-----------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.linalg.norm


Solving Equations and Inverting Matrices
----------------------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.linalg.solve
    nlcpy.linalg.lstsq
    nlcpy.linalg.inv


Exceptions
----------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.linalg.LinAlgError


.. _linalg_several_matrices_at_once:

Linear Algebra on Several Matrices at Once
------------------------------------------

Several of the linear algebra routines listed above are able to compute results for several matrices at once, if they are stacked into the same array.
This is indicated in the documentation via input parameter specifications such as ``a : (..., M, M) array_like``.
This means that if for instance given an input array ``a.shape == (N, M, M)``, it is interpreted as a "stack" of N matrices, each of size M-by-M.
Similar specification applies to return values, for instance the inverse has ``ainv : (..., M, M)`` and will in this case return an array of shape ``inv(a).shape == (N,M,M)``.
This generalizes to linear algebra operations on higher-dimensional arrays: the last 1 or 2 dimensions of a multidimensional array are interpreted as vectors or matrices, as appropriate for each operation.

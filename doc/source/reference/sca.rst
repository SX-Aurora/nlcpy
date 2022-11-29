.. _label_sca_top:
.. module:: nlcpy.sca

==================================
Stencil Code Accelerator Interface
==================================


Coding Guide
==========================
The Stencil Code Accelerator (SCA) interface accelerates stencil computations, a processing pattern frequently used in image processing, scientific or engineering simulations, deep learning, and so on.
The SCA interface enables Python scripts to call functions provided by `SCA <https://sxauroratsubasa.sakura.ne.jp/documents/sdk/SDK_NLC/UsersGuide/sca/c/en/index.html>`_ of `NEC Numeric Library Collection <https://sxauroratsubasa.sakura.ne.jp/documents/sdk/SDK_NLC/UsersGuide/main/en/index.html>`_, which is highly optimized for Vector Engine of SX-Aurora TSUBASA.

.. toctree::
   :maxdepth: 1

   sca_chapter_1.rst
   sca_chapter_2.rst
   sca_chapter_3.rst
   sca_chapter_4.rst
   sca_chapter_5.rst


Functions
=========

Creation of Stencil Descriptor
------------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.sca.create_descriptor
    nlcpy.sca.empty_description

Creation of Kernel
------------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.sca.create_kernel

Execution of Kernel
------------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.sca.kernel.kernel.execute

Destruction of Kernel
------------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.sca.destroy_kernel

Stride Adjustment
------------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.sca.convert_optimized_array
    nlcpy.sca.create_optimized_array

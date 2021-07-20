Notice Regarding SCA Interface
===============================

    - The SCA interface supports up to 4 dimension :class:`nlcpy.ndarray`. It can handle stencil computations of arbitrary shape (axial, planar, diagonal, and its combination).
    - In terms of dtype (data type), the current version of the SCA interface supports only ``float32`` and ``float64``.

        - As for integer types (int, uint, etc.), convert dtype into ``float32`` or ``float64``.
        - As for complex types, see :ref:`label_sca_complex`.

    - :func:`nlcpy.sca.create_kernel` dynamically generates a instruction sequence required to perform stencil computations on VE. This generating cost is not small compared to :func:`nlcpy.sca.kernel.kernel.execute`, so the overall performance gets better as the number of kernel executions per kernel generation increase.
    - An output :class:`nlcpy.ndarray` may not be shared with an input :class:`nlcpy.ndarray`. Otherwise, unpredictable results occurs.


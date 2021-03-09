.. _nlcpy_constants:

Constants
=========

NLCPy includes several constants:

.. data:: nlcpy.inf

    IEEE 754 floating point representation of positive infinity.

    :Returns:

        **y** : float
            A floating point representation of positive infinity.

    .. seealso::

        :obj:`isinf`
        :obj:`isnan`
        :obj:`isfinite`

    .. note::

        NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
        This means that Not a Number is not equivalent to infinity.
        Also that positive infinity is not equivalent to negative infinity.
        But infinity is equivalent to positive infinity.

        :data:`Inf`, :data:`Infinity`, :data:`PINF` and :data:`infty` are aliases for :data:`inf`.


.. data:: nlcpy.Inf

    IEEE 754 floating point representation of positive infinity.

    Use :data:`inf` because :data:`Inf`, :data:`Infinity`, :data:`PINF` and :data:`infty` are aliases for inf.
    For more details, see :data:`inf`.

    .. seealso::

        :data:`inf`


.. data:: nlcpy.Infinity

    IEEE 754 floating point representation of positive infinity.

    Use :data:`inf` because :data:`Inf`, :data:`Infinity`, :data:`PINF` and :data:`infty` are aliases for inf.
    For more details, see :data:`inf`.

    .. seealso::

        :data:`inf`


.. data:: nlcpy.PINF

    IEEE 754 floating point representation of positive infinity.

    Use :data:`inf` because :data:`Inf`, :data:`Infinity`, :data:`PINF` and :data:`infty` are aliases for inf.
    For more details, see :data:`inf`.

    .. seealso::

        :data:`inf`



.. data:: nlcpy.infty

    IEEE 754 floating point representation of positive infinity.

    Use :data:`inf` because :data:`Inf`, :data:`Infinity`, :data:`PINF` and :data:`infty` are aliases for inf.
    For more details, see :data:`inf`.

    .. seealso::

        :data:`inf`


.. data:: nlcpy.NINF

    IEEE 754 floating point representation of negative infinity.

    :Returns:

        **y** : float
            A floating point representation of negative infinity.

    .. seealso::

        :obj:`isinf`
        :obj:`isnan`
        :obj:`isfinite`

    .. note::

        NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
        This means that Not a Number is not equivalent to infinity.
        Also that positive infinity is not equivalent to negative infinity.
        But infinity is equivalent to positive infinity.


.. data:: nlcpy.nan

    IEEE 754 floating point representation of Not a Number (NaN).

    :Returns:

        **y** : float
            A floating point representation of Not a Number.

    .. seealso::

        :obj:`isnan`
        :obj:`isfinite`

    .. note::

        NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
        This means that Not a Number is not equivalent to infinity.
        :data:`NaN` and :data:`NAN` are aliases of :data:`nan`.


.. data:: nlcpy.NAN

    IEEE 754 floating point representation of Not a Number (NaN).

    :data:`NaN` and :data:`NAN` are equivalent definitions of :data:`nan`.
    Please use :data:`nan` instead of :data:`NAN`.

    .. seealso::

        :data:`nan`


.. data:: nlcpy.NaN

    IEEE 754 floating point representation of Not a Number (NaN).

    :data:`NaN` and :data:`NAN` are equivalent definitions of :data:`nan`.
    Please use :data:`nan` instead of :data:`NaN`.

    .. seealso::

        :data:`nan`


.. data:: nlcpy.NZERO

    IEEE 754 floating point representation of negative zero.

    :Returns:

        **y** : float
            A floating point representation of negative zero.

    .. seealso::

        :data:`PZERO`
        :obj:`isinf`
        :obj:`isnan`
        :obj:`isfinite`

    .. note::

        NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
        Negative zero is considered to be a finite number.

    .. rubric:: Examples

    >>> import nlcpy as vp
    >>> vp.NZERO
    -0.0
    >>> vp.PZERO
    0.0

    >>> vp.isfinite([vp.NZERO])
    array([ True])
    >>> vp.isnan([vp.NZERO])
    array([False])
    >>> vp.isinf([vp.NZERO])
    array([False])


.. data:: nlcpy.PZERO

    IEEE 754 floating point representation of positive zero.

    :Returns:

        **y** : float
            A floating point representation of positive zero.

    .. seealso::

        :data:`NZERO`
        :obj:`isinf`
        :obj:`isnan`
        :obj:`isfinite`

    .. note::

        NLCPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
        Positive zero is considered to be a finite number.

    .. rubric:: Examples

    >>> import nlcpy as vp
    >>> vp.PZERO
    0.0
    >>> vp.NZERO
    -0.0

    >>> vp.isfinite([vp.PZERO])
    array([ True])
    >>> vp.isnan([vp.PZERO])
    array([False])
    >>> vp.isinf([vp.NZERO])
    array([False])


.. data:: nlcpy.e

    Euler's constant, base of natural logarithms, Napier's constant.

    ``e = 2.71828182845904523536028747135266249775724709369995...``

    .. seealso::

        :obj:`exp`
        :obj:`log`


.. data:: nlcpy.euler_gamma

    ``γ = 0.5772156649015328606065120900824024310421...``


.. data:: nlcpy.pi

    ``pi = 3.1415926535897932384626433...``

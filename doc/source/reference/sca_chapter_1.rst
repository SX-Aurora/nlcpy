.. _label_sca_concept:

Concept of Stencil Computation
==============================

Before using the SCA interface, it is necessary to understand some fundamental concepts of the stencil computation.

A stencil consists of stencil elements. Each stencil element has a relative location from the stencil center,
a coefficient, and a reference to an input array. The mathematical expression of the stencil is shown below.

.. math::
   Stencil(i_{\rm x},i_{\rm y},i_{\rm z},i_{\rm w}) = \sum_{k} c^{(k)} d^{(k)}[i_{\rm x}+l_{\rm x}^{(k)},i_{\rm y}+l_{\rm y}^{(k)},i_{\rm z}+l_{\rm z}^{(k)},i_{\rm w}+l_{\rm w}^{(k)}]


Here, :math:`(l_{\rm x}^{(k)},l_{\rm y}^{(k)},l_{\rm z}^{(k)},l_{\rm w}^{(k)})` denotes the relative location of the kth stencil element, :math:`c^{(k)}` denotes the coefficient of the kth stencil element, and :math:`d^{(k)}` denotes the input array which the kth stencil element refers to.

In the SCA interface, you can specify either a scalar or an array as the coefficients :math:`c^{(k)}`.

pcg-stein documentation
=======================

Implements preconditioned conjugate gradient (PCG) methods to solve the system

.. math::

   K x = y

where :math:`K` is a symmetric PSD Gram matrix produced by a Stein kernel.

.. toctree::
   :maxdepth: 4
   :caption: API Reference

   pcg_stein
   .. pcg_stein.kernel
   .. pcg_stein.pcg
   .. pcg_stein.precon

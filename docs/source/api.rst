=============
API Reference
=============

This page groups the public API by task. The same objects are usually imported
from the top-level package:

.. code-block:: python

   import softmobility as sm

Body geometry and mobility
--------------------------

.. autosummary::
   :toctree: autosummary

   softmobility.classes.sphere.Sphere
   softmobility.classes.sphereassembly.SphereAssembly
   softmobility.classes.softbody.SoftBody
   softmobility.classes.flexiblefiber.FlexibleFiber

Inputs and flows
----------------

.. autosummary::
   :toctree: autosummary

   softmobility.classes.inputs.Scalar
   softmobility.classes.inputs.Field
   softmobility.classes.inputs.Flow
   softmobility.classes.inputs.constant_scalar
   softmobility.classes.inputs.oscillating_scalar
   softmobility.classes.inputs.gravity_field
   softmobility.classes.inputs.rotating_magnetic_field
   softmobility.classes.inputs.oscillating_magnetic_field
   softmobility.classes.inputs.no_flow
   softmobility.classes.inputs.shear_flow
   softmobility.classes.inputs.rotating_flow
   softmobility.classes.inputs.extensional_flow
   softmobility.classes.inputs.taylor_green_flow

Simulation and optimization
---------------------------

.. autosummary::
   :toctree: autosummary

   softmobility.classes.solver.FlowBodyRollout
   softmobility.classes.solver.FlowBodyOptimizer

Rotation utilities
------------------

These helpers are exposed at the package top level
(``sm.rotation_matrix``, ``sm.rescale_orientation``, ``sm.compute_bortz_operator``)
and are mostly useful for advanced users working directly with Rodrigues vectors.

.. autosummary::
   :toctree: autosummary

   softmobility.rescale_orientation
   softmobility.compute_bortz_operator
   softmobility.rotation_matrix

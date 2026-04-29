========
Overview
========

SoftMobility represents a deformable body as an assembly of spheres whose
geometry depends on degrees of freedom and design parameters. The library then
computes reduced mobility tensors for Stokes-flow dynamics, integrates the body
trajectory in a prescribed flow, and exposes the calculation to JAX
transformations such as ``jax.jit`` and ``jax.grad``.

Conceptual model
----------------

A model has three groups of variables:

``dofs``
    Dynamic degrees of freedom that describe deformation, such as spring
    extension, hinge angle, or local orientation.

``design``
    Fixed morphology or material parameters, such as sphere radii, distances,
    or stiffnesses. These are the natural variables for design optimization.

``inputs``
    External or active controls. Three-component input names ending in
    ``0``, ``1``, and ``2`` are grouped as vector fields; other input names are
    treated as scalar controls.

For each sphere, positions and orientations are expressed in the body frame.
During a rollout, the body has a lab-frame position, a Rodrigues orientation
vector, and the current values of the internal degrees of freedom.

Array conventions
-----------------

The main array shapes are:

.. list-table::
   :header-rows: 1

   * - Quantity
     - Shape
     - Meaning
   * - ``position``
     - ``(3,)``
     - Lab-frame position of the body reference point.
   * - ``orientation``
     - ``(3,)``
     - Rodrigues vector for body orientation.
   * - ``dofs``
     - ``(Ndof,)``
     - Internal configuration variables.
   * - ``design``
     - ``(Ndesign,)``
     - Fixed design variables.
   * - ``inputs``
     - ``(Ninput,)``
     - Field components followed by scalar inputs.
   * - ``positions``
     - ``(n_steps, 3)``
     - Trajectory returned by ``FlowBodyRollout.rollout``.

Workflow
--------

The usual workflow is:

1. Define the geometry with ``SphereAssembly`` or a YAML description.
2. Promote the assembly to ``SoftBody`` to compute hydrodynamic mobility.
3. Create fields, scalar controls, and background flows.
4. Run ``FlowBodyRollout.rollout`` to obtain trajectories.
5. Optionally optimize design variables with ``FlowBodyOptimizer``.

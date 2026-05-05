========
Overview
========

SoftMobility represents a deformable body as an assembly of spheres whose
geometry depends on degrees of freedom and design parameters. The library then
computes reduced mobility tensors for Stokes-flow dynamics, integrates the body
trajectory in a prescribed flow, and exposes the calculation to JAX
transformations such as ``jax.jit`` and ``jax.grad``.

Physical model
--------------

All flows are assumed to be in the **Stokes (creeping-flow) regime**: inertia is
negligible and the fluid responds instantaneously to forces. In this regime the
velocity of each sphere is linearly related to the forces and torques acting on
it through the **hydrodynamic mobility tensor**.

SoftMobility uses the **Rotne–Prager–Yamakawa (RPY) approximation** to build
that tensor. The self-mobility of sphere *i* (radius *a*\:sub:`i`, viscosity
*μ* = 1) is the standard Stokes result:

.. math::

   \boldsymbol{\mu}^{tt}_{ii} = \frac{1}{6\pi a_i}\mathbf{I}, \qquad
   \boldsymbol{\mu}^{rr}_{ii} = \frac{1}{8\pi a_i^3}\mathbf{I}.

The cross-mobility between spheres *i* and *j* separated by **r** = **x**\
:sub:`i` − **x**\:sub:`j` (magnitude *R*, unit vector **r̂**) uses the RPY
correction, which regularises the Oseen tensor for overlapping or nearly
touching spheres and guarantees a positive-definite mobility matrix.

The grand mobility matrix (size 6*N* × 6*N* for *N* spheres) is assembled from
these blocks and then projected onto the reduced set of body degrees of freedom
by a kinematic Jacobian.

Orientation convention
----------------------

Body orientation is stored as a **Rodrigues vector** **p** = θ **n̂**, where
**n̂** is the unit rotation axis and θ ∈ [0, π) is the rotation angle. The
vector norm encodes the magnitude of the rotation:

- **p** = **0** means no rotation (identity).
- ‖\ **p**\ ‖ = π/2 is a 90° rotation about **n̂**.

When numerical integration drives ‖\ **p**\ ‖ beyond π, ``rescale_orientation``
maps it back to the equivalent vector with ‖\ **p**\ ‖ < π, avoiding
representation singularities. The body-frame-to-lab-frame rotation matrix for a
given Rodrigues vector is computed by ``rotation_matrix``.

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

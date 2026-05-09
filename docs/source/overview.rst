========
Overview
========

SoftMobility represents a deformable body as an assembly of spheres whose
geometry depends on a small number of generalized coordinates: a body-frame
position and orientation, plus a set of internal degrees of freedom that
parameterize the shape change. The library assembles the configuration-
dependent hydrodynamic tensors of the assembly, integrates the resulting
ordinary differential equation in a prescribed background flow, and exposes
the whole calculation to JAX transformations such as ``jax.jit`` and
``jax.grad``.

The notation used throughout the documentation mirrors the *Soft Mobility
Theory* manuscript shipped under ``manuscript/Article3.pdf``; see that
document for the complete derivation.

Physical model
--------------

The fluid is in the **Stokes (creeping-flow) regime**, :math:`\mathrm{Re}\ll 1`.
Inertia is negligible and the fluid responds instantaneously to the forces
applied on the body. Because the body size :math:`L` is small compared to
the smallest length scale of the background flow, the background velocity
:math:`\boldsymbol{u}^\infty(\boldsymbol{r})` is linearized about a
reference point :math:`\boldsymbol{r}_0` attached to the body,

.. math::

   \boldsymbol{u}^\infty(\boldsymbol{r}) \approx
       \boldsymbol{u}_0^\infty
     + \boldsymbol{\omega}_0^\infty \times (\boldsymbol{r}-\boldsymbol{r}_0)
     + \boldsymbol{E}_0^\infty \cdot (\boldsymbol{r}-\boldsymbol{r}_0),

with :math:`\boldsymbol{u}_0^\infty` the translational velocity,
:math:`\boldsymbol{\omega}_0^\infty = \tfrac{1}{2}\nabla\times\boldsymbol{u}^\infty`
the angular velocity, and :math:`\boldsymbol{E}_0^\infty` the symmetric
traceless rate-of-strain tensor of the background flow at
:math:`\boldsymbol{r}_0`.

The body is modelled as an assembly of :math:`N` rigid spheres connected by
elastic and rigid links. The :math:`i`-th sphere has radius :math:`a_i`,
body-frame center :math:`\boldsymbol{R}_i`, and Rodrigues orientation
:math:`\boldsymbol{\Theta}_i`; its body-frame six-component velocity is
:math:`\boldsymbol{V}_i = [\boldsymbol{U}_i, \boldsymbol{\Omega}_i]`. The
hydrodynamic interaction between spheres is captured by the
**Rotne–Prager–Yamakawa (RPY) approximation**, which provides a closed-form
:math:`6N\times 6N` grand resistance matrix :math:`\boldsymbol{R}` and a
:math:`6N\times 3\times 3` stresslet resistance tensor
:math:`\boldsymbol{R}_S` for assemblies of spheres of possibly different
radii. The full RPY expressions, including off-diagonal blocks and Faxén
corrections, can be found in Wajnryb *et al.* 2013 and Cichocki *et al.*
2021. **Spheres must not overlap**: the implementation assumes
:math:`R_{ij} > a_i + a_j` for every pair and produces unphysical results
otherwise. Use :py:meth:`softmobility.SoftBody.validate_no_overlap` to
check the geometry before simulation.

Generalized coordinates
-----------------------

The body's frame has lab-frame position :math:`\boldsymbol{r}_0` and
Rodrigues orientation :math:`\boldsymbol{\theta}_0`, gathered into the
six-component pose :math:`\boldsymbol{x}_0 = [\boldsymbol{r}_0,
\boldsymbol{\theta}_0]`. The internal deformation is described by
:math:`N_Q` degrees of freedom :math:`\boldsymbol{Q}` (such as spring
extensions or hinge angles). The complete generalized coordinates of size
:math:`N_q = 6 + N_Q` are

.. math::

   \boldsymbol{q} = [\boldsymbol{x}_0,\, \boldsymbol{Q}].

The associated generalized velocity is :math:`\boldsymbol{p} =
[\boldsymbol{u}_0,\, \boldsymbol{\omega}_0,\, \dot{\boldsymbol{Q}}]`,
with :math:`\boldsymbol{v}_0 = [\boldsymbol{u}_0, \boldsymbol{\omega}_0]`
the lab-frame six-component velocity of the body frame. The relation
between :math:`\boldsymbol{p}` and :math:`\dot{\boldsymbol{q}}` involves a
Bortz operator :math:`\boldsymbol{B}(\boldsymbol{\theta}_0)` that accounts
for the non-commutativity of finite rotations.

Soft mobility equation
----------------------

After projection onto the generalized coordinates by the assembly Jacobian
:math:`\boldsymbol{J} = \partial\boldsymbol{v}/\partial\boldsymbol{p}` of
size :math:`6N\times N_q`, and elimination of the constraint forces, the
dynamics reduce to the **soft mobility equation**

.. math::

   \begin{pmatrix}
     \boldsymbol{u}_0 - \boldsymbol{u}_0^\infty \\
     \boldsymbol{\omega}_0 - \boldsymbol{\omega}_0^\infty \\
     \dot{\boldsymbol{Q}}
   \end{pmatrix}
   = \boldsymbol{M}_K\cdot\boldsymbol{Q}
   + \boldsymbol{M}_H\cdot\boldsymbol{H}
   + \boldsymbol{C}_E:\boldsymbol{E}_0^\infty
   - \boldsymbol{\Pi}\cdot\boldsymbol{V}_\mathrm{act},

where the right-hand side is expressed in the body frame. The four reduced
tensors all depend on the current deformation state :math:`\boldsymbol{Q}`:

- :math:`\boldsymbol{M}_K` (size :math:`N_q\times N_Q`) — elastic mobility,
  acting on the linear restoring force :math:`\boldsymbol{f}_K =
  \boldsymbol{C}_K\cdot\boldsymbol{Q}`.
- :math:`\boldsymbol{M}_H` (size :math:`N_q\times 3`) — body-force
  mobility, acting on a uniform external field
  :math:`\boldsymbol{H}` (e.g. gravity :math:`\boldsymbol{g}` or a magnetic
  field :math:`\boldsymbol{B}`).
- :math:`\boldsymbol{C}_E` (size :math:`N_q\times 3\times 3`) — coupling
  to the background rate-of-strain.
- :math:`\boldsymbol{\Pi}` — the projection operator that eliminates the
  constraint forces; its action on the active velocity
  :math:`\boldsymbol{V}_\mathrm{act}` accounts for prescribed kinematic
  drives.

When the flow and external fields are known in the lab frame, they must
first be rotated into the body frame using the rotation matrix
:math:`\boldsymbol{\mathcal{R}}_0(\boldsymbol{\theta}_0)` from the
Euler–Rodrigues formula. Combined with :math:`\boldsymbol{p} =
\boldsymbol{B}^{-1}\cdot\dot{\boldsymbol{q}}`, the soft mobility equation
is a first-order ODE :math:`\dot{\boldsymbol{q}} =
\boldsymbol{f}(\boldsymbol{q},t)` that can be integrated forward in time.

Orientation convention
----------------------

Both the body frame (:math:`\boldsymbol{\theta}_0`) and individual sphere
orientations (:math:`\boldsymbol{\Theta}_i`) are stored as **Rodrigues
vectors**: a three-component vector :math:`\boldsymbol{p} = \theta\,
\hat{\boldsymbol{n}}` whose direction :math:`\hat{\boldsymbol{n}}` is the
unit rotation axis and whose magnitude :math:`\theta` is the rotation
angle.

- :math:`\boldsymbol{p} = \boldsymbol{0}` is the identity rotation.
- :math:`\|\boldsymbol{p}\| = \pi/2` is a 90° rotation about
  :math:`\hat{\boldsymbol{n}}`.

When numerical integration drives :math:`\|\boldsymbol{p}\|` beyond
:math:`\pi`, ``rescale_orientation`` maps it back to the equivalent vector
with :math:`\|\boldsymbol{p}\| < \pi`, avoiding representation
singularities. The body-frame-to-lab-frame rotation matrix is computed by
``rotation_matrix``.

Variable groups
---------------

A soft body partitions its parameters into three groups:

``dofs``
    The generalized coordinates :math:`\boldsymbol{Q}` describing
    deformation (spring extension, hinge angle, local orientation, …).
    They evolve during a simulation.

``design``
    Fixed morphological or material parameters (sphere radii, rest
    lengths, stiffnesses). They are constant during a given simulation
    and are the natural targets for design optimization.

``inputs``
    External or active controls. Three-component input names ending in
    ``0``, ``1``, ``2`` (e.g. ``gravity0``, ``gravity1``, ``gravity2``)
    are treated as components of a vector field
    :math:`\boldsymbol{H}`; single-component names are treated as scalar
    controls. The distinction matters because vector fields rotate when
    changing reference frame and scalars do not.

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
     - Lab-frame position :math:`\boldsymbol{r}_0` of the body reference
       point.
   * - ``orientation``
     - ``(3,)``
     - Rodrigues vector :math:`\boldsymbol{\theta}_0` for the body frame.
   * - ``dofs``
     - ``(Ndof,)``
     - Internal coordinates :math:`\boldsymbol{Q}`.
   * - ``design``
     - ``(Ndesign,)``
     - Fixed design variables.
   * - ``inputs``
     - ``(Ninput,)``
     - Vector-field components followed by scalar inputs.
   * - ``positions``
     - ``(n_steps, 3)``
     - Trajectory returned by ``sm.FlowBodyRollout.rollout``.

Workflow
--------

The usual workflow is:

1. Define the geometry with ``sm.SphereAssembly`` or a YAML description.
2. Promote the assembly to ``sm.SoftBody``: this builds the assembly
   Jacobian :math:`\boldsymbol{J}`, the RPY grand resistance
   :math:`\boldsymbol{R}`, and the reduced soft mobility tensors
   (:math:`\boldsymbol{M}_K`, :math:`\boldsymbol{M}_H`,
   :math:`\boldsymbol{C}_E`, :math:`\boldsymbol{\Pi}`).
3. Create fields, scalar controls, and background flows.
4. Run ``sm.FlowBodyRollout.rollout`` to integrate
   :math:`\dot{\boldsymbol{q}} = \boldsymbol{f}(\boldsymbol{q},t)` in
   time.
5. Optionally optimize design variables with ``sm.FlowBodyOptimizer`` by
   differentiating the rollout end-to-end through ``jax.grad``.

Flexible fibers
---------------

For chains of identical beads with rigid bonds and a linear bending
elasticity, use :class:`softmobility.FlexibleFiber`. It implements the
Joint Model of Delmotte *et al.* 2015 (Fig. 3, Eqs. 2–4): bead positions
are parameterized by bead orientations through the recurrence
:math:`\boldsymbol{R}_{i+1} = \boldsymbol{R}_i + (a + \varepsilon
g)(\boldsymbol{p}_i + \boldsymbol{p}_{i+1})`, so the rigid-bond
constraint is satisfied by construction (no Lagrange multipliers).
Bending elasticity is the discrete biharmonic of the orientation
:math:`\boldsymbol{Q}`, and gravity is registered automatically as a
3-component field input. Both planar (``planar=True``, one angle per
bead) and full 3-D (Rodrigues vector per bead) variants are available.

.. code-block:: python

   import softmobility as sm

   fiber = sm.FlexibleFiber(n_beads=20, radius=0.5, bending_rigidity=1.0, mass=0.1)
   rollout = sm.FlowBodyRollout(
       soft_body=fiber,
       flow=sm.no_flow(),
       input_map={"gravity": sm.gravity_field(g=9.81)},
   )

=====
Usage
=====

This page gives a compact, end-to-end example. For conceptual background, see
:doc:`overview`; for the complete object list, see :doc:`api`.

Basic Import
------------

Start by importing the main classes:

.. code-block:: python

    import softmobility as sm

Creating a Soft Body
--------------------

``sm.SoftBody`` can be created from a YAML file path or from a YAML string. The
YAML parser detects degrees of freedom, design variables, and inputs from the
symbols used in sphere expressions.

.. code-block:: python

    import softmobility as sm

    yaml_text = """
    dof_names:
      - x
    design_names:
      - radius
      - length
      - k
    defaults:
      x0: 0.1
      radius: 0.25
      length: 1.0
      k: 1.0
    spheres:
      - radius: radius
        position: [-length / 2, 0, 0]
        orientation: [0, x0, 0]
        torque: [0, -k * x0, 0]
      - radius: radius
        position: [length / 2, 0, 0]
        orientation: [0, -x0, 0]
        torque: [0, k * x0, 0]
    """

    body = sm.SoftBody(yaml_text, verbose=False)
    print(body.dof_variables, body.design_variables)

Working with Flows
------------------

Define fluid flows using the ``Flow`` class:

.. code-block:: python

    import softmobility as sm
    
    # Create a shear flow
    flow = sm.shear_flow(shear_rate=1.0)
    
    # Or no flow for stationary fluid
    flow = sm.no_flow()

Running Simulations
-------------------

Use ``sm.FlowBodyRollout`` to simulate the body trajectory:

.. code-block:: python

    import jax.numpy as jnp
    import softmobility as sm

    rollout = sm.FlowBodyRollout(body, flow)
    positions, orientations, dofs = rollout.rollout(
        dt=0.01,
        n_steps=100,
        init_position=jnp.zeros(3),
        init_orientation=jnp.zeros(3),
    )

Optimization
------------

Optimization is expressed through an objective function of ``(rollout,
design)``:

.. code-block:: python

    import optax
    import softmobility as sm

    def final_height(rollout, design):
        positions, _, _ = rollout.rollout(dt=0.01, n_steps=100, design=design)
        return positions[-1, 2]

    optimizer = sm.FlowBodyOptimizer(rollout, final_height, optax.adam(1e-3))
    optimized_design = optimizer.run(
        init_design=body.design_defaults,
        n_steps=200,
        maximize=True,
    )

Example Notebooks
-----------------

The tutorials are grouped into three layers. Numbering reflects the layer
(0X = library introduction, 1X = validation against published results,
2X = original case studies).

**Library introduction (0X)**

- ``01_assembly_creation.ipynb`` — methods to create a sphere assembly
- ``02_rigid_mobility.ipynb`` — mobility properties of a rigid sphere assembly
- ``03_soft_mobility_simulation.ipynb`` — soft mobility tensors and
  simulation of a trajectory
- ``04_optimization.ipynb`` — optimization principles
- ``05_figure_styling.ipynb`` — paper-figure aesthetics with ``figstyle``

**Validation cases (1X)**

- ``11_sinking_rigid_body.ipynb`` — sinking trajectory of a rigid body
- ``12_flexible_fiber_2d.ipynb`` — 2-D flexible fiber in shear and gravity
  (Delmotte et al. 2015)
- ``13_rotating_fiber_3d.ipynb`` — 3-D filament: bending and rotational
  relaxation (Coq et al. 2008; Wiggins et al. 1998)
- ``14_jeffery_rigid.ipynb`` — Jeffery orbits of a rigid body
- ``15_three_sphere_swimmer.ipynb`` — three-sphere swimmer
  (Najafi & Golestanian 2004)

**Original case studies (2X)**

- ``21_jeffery_soft.ipynb`` — Jeffery orbit of a one-DOF deformable body
- ``22_soft_surfer.ipynb`` — soft surfer in Taylor–Green vortices

API Reference
-------------

For detailed API documentation, see :doc:`api`.

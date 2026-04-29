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
    from softmobility import Sphere, SphereAssembly, SoftBody
    from softmobility import FlowBodyRollout, FlowBodyOptimizer

Creating a Soft Body
--------------------

``SoftBody`` can be created from a YAML file path or from a YAML string. The
YAML parser detects degrees of freedom, design variables, and inputs from the
symbols used in sphere expressions.

.. code-block:: python

    from softmobility import SoftBody

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

    body = SoftBody(yaml_text, verbose=False)
    print(body.dof_variables, body.design_variables)

Working with Flows
------------------

Define fluid flows using the Flow classes:

.. code-block:: python

    from softmobility import shear_flow, no_flow
    
    # Create a shear flow
    flow = shear_flow(shear_rate=1.0)
    
    # Or no flow for stationary fluid
    flow = no_flow()

Running Simulations
-------------------

Use ``FlowBodyRollout`` to simulate the body trajectory:

.. code-block:: python

    import jax.numpy as jnp
    from softmobility import FlowBodyRollout

    rollout = FlowBodyRollout(body, flow)
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
    from softmobility import FlowBodyOptimizer

    def final_height(rollout, design):
        positions, _, _ = rollout.rollout(dt=0.01, n_steps=100, design=design)
        return positions[-1, 2]

    optimizer = FlowBodyOptimizer(rollout, final_height, optax.adam(1e-3))
    optimized_design = optimizer.run(
        init_design=body.design_defaults,
        n_steps=200,
        maximize=True,
    )

Example Notebooks
-----------------

For more detailed examples, see the tutorial notebooks:

- ``01_tutorial_sphere_assemblies.ipynb`` - Basic sphere assembly usage
- ``02_rigid_assembly.ipynb`` - Rigid body simulations  
- ``03_freefall_trajectories.ipynb`` - Freefall dynamics
- ``04_Jeffery_orbits.ipynb`` - Jeffery orbits in flow
- ``06_three_sphere_swimmer.ipynb`` - Three-sphere swimmer model
- ``07_soft_surfer.ipynb`` - Soft surfer simulations

API Reference
-------------

For detailed API documentation, see :doc:`api`.

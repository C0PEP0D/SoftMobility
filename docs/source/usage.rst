=====
Usage
=====

SoftMobility is a Python library for simulating soft body dynamics and fluid mechanics problems. This guide provides an overview of how to use the main classes and features.

Basic Import
------------

Start by importing the main classes:

.. code-block:: python

    import softmobility as sm
    from softmobility import Sphere, SphereAssembly, SoftBody
    from softmobility import FlowBodyRollout, FlowBodyOptimizer

Creating Sphere Assemblies
--------------------------

The ``SphereAssembly`` class allows you to create assemblies of spheres with configurable degrees of freedom:

.. code-block:: python

    # Create a simple two-sphere assembly
    assembly = SphereAssembly.from_yaml('''
    dof_names:
        - length
        - angle
    
    defaults:
        length: 1.0
        angle: 0.0
    
    spheres:
      - radius: 1.0
        position: [-1, 0, 0]
        orientation: [0, 0, angle]
      - radius: 1.0  
        position: [1, 0, 0]
        orientation: [0, 0, angle]
    ''')

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

Use the solver classes to run simulations:

.. code-block:: python

    # Create a rollout for simulating dynamics
    rollout = FlowBodyRollout(assembly, flow)
    
    # Run a simulation
    results = rollout.run(initial_state, time_steps=100)

Optimization
------------

The library also supports optimization:

.. code-block:: python

    optimizer = FlowBodyOptimizer(assembly, flow)
    optimized_design = optimizer.optimize(objective_function)

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

For detailed API documentation, see the :ref:`autosummary` section.
==========
Simulation
==========

``FlowBodyRollout`` integrates a ``SoftBody`` in a background ``Flow``. It is
pure-functional: the rollout method returns trajectory arrays and does not
mutate the body, flow, or input objects.

Minimal rollout
---------------

.. code-block:: python

   import jax.numpy as jnp
   from softmobility import SoftBody, FlowBodyRollout, no_flow

   yaml_text = """
   dof_names: [x]
   design_names: [radius, length, k]
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
   rollout = FlowBodyRollout(body, no_flow())

   positions, orientations, dofs = rollout.rollout(
       dt=0.01,
       n_steps=100,
       init_position=jnp.zeros(3),
       init_orientation=jnp.zeros(3),
   )

Inputs during a rollout
-----------------------

If a body uses field or scalar inputs, pass an ``input_map`` whose keys match
the base input names detected in the geometry:

.. code-block:: python

   from softmobility import FlowBodyRollout, gravity_field, no_flow

   input_map = {"gravity": gravity_field(g=9.81)}
   rollout = FlowBodyRollout(body, no_flow(), input_map=input_map)

Returned values
---------------

``rollout.rollout`` returns ``(positions, orientations, dofs)``. Each entry
contains one row per time step. Initial conditions are not prepended to the
returned arrays; the first row is the state after the first integration step.

The method is designed for JAX transformations. Keep ``n_steps`` static when
jitting a function that calls it.

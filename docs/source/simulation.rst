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

JAX notes
---------

SoftMobility is built on JAX. A few things to keep in mind if you are new to
JAX:

- **Static shapes**: ``n_steps`` must be a Python integer (not a traced array).
  If you wrap the rollout in ``jax.jit``, pass ``n_steps`` as a static argument
  or fix it at definition time.
- **No Python control flow over traced arrays**: conditionals on JAX arrays
  inside a jitted function must use ``jax.lax.cond``, not ``if``.
- **CPU by default**: JAX runs on CPU unless a GPU/TPU build is installed. See
  the :doc:`installation` page for platform-specific setup.
- **Compilation on first call**: ``jax.jit``-compiled functions are traced and
  compiled the first time they are called with a given input signature. Expect a
  delay on the first call; subsequent calls are fast.

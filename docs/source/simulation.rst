==========
Simulation
==========

``sm.FlowBodyRollout`` integrates a ``sm.SoftBody`` in a background ``sm.Flow``. It is
pure-functional: the rollout method returns trajectory arrays and does not
mutate the body, flow, or input objects.

Minimal rollout
---------------

.. code-block:: python

   import jax.numpy as jnp
   import softmobility as sm

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

   body = sm.SoftBody(yaml_text, verbose=False)
   rollout = sm.FlowBodyRollout(body, sm.no_flow())

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

   import softmobility as sm

   input_map = {"gravity": sm.gravity_field(g=9.81)}
   rollout = sm.FlowBodyRollout(body, sm.no_flow(), input_map=input_map)

Returned values
---------------

``rollout.rollout`` returns ``(positions, orientations, dofs)``. Each entry
contains one row per time step. Initial conditions are not prepended to the
returned arrays; the first row is the state after the first integration step.

Time-integration scheme
-----------------------

``rollout.rollout`` accepts a ``scheme`` keyword argument selecting the
time-stepper:

- ``"rk4"`` (default) — classical 4-stage Runge–Kutta with the Bortz
  operator recomputed at every stage. Converges as :math:`O(dt^4)`.
- ``"rk2"`` — explicit midpoint method with the Bortz operator recomputed
  at the predicted half-step. Converges as :math:`O(dt^2)`.

RK4 costs roughly twice as much per step as RK2 but is typically orders of
magnitude more accurate at any non-trivial tolerance, so it is the
recommended default. Pass ``scheme="rk2"`` only if you need the cheaper
per-step cost and your tolerance is loose.

.. code-block:: python

   positions, orientations, dofs = rollout.rollout(
       dt=0.01, n_steps=100, scheme="rk2",   # opt-in to the cheaper scheme
   )

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

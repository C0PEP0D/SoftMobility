======
Inputs
======

SoftMobility separates inputs from geometry. ``Scalar``, ``Field``, and
``Flow`` wrap user callables and optional parameters in a JAX-compatible form.

Scalars
-------

A ``Scalar`` returns one value from ``value(pos, time)``. It is useful for
active forcing, prescribed controls, or time-dependent coefficients.

.. code-block:: python

   from softmobility import constant_scalar, oscillating_scalar

   import jax.numpy as jnp

   constant = constant_scalar(2.0)
   signal = oscillating_scalar(amplitude=1.0, omega=3.0, phase=0.0)
   pos = jnp.zeros(3)
   value = signal.value(pos, time=0.5)

Fields
------

A ``Field`` returns a vector of shape ``(3,)`` from ``vector(pos, time)``.
Common constructors include gravity and simple magnetic-field signals:

.. code-block:: python

   from softmobility import gravity_field, rotating_magnetic_field

   gravity = gravity_field(g=9.81)
   magnetic = rotating_magnetic_field(amp_x=1.0, amp_y=0.5, omega=2.0)

When a YAML model uses variables such as ``gravity0``, ``gravity1``, and
``gravity2``, the rollout expects an input map key named ``"gravity"`` whose
value is a ``Field``.

Flows
-----

A ``Flow`` returns the background fluid velocity. It can also compute the
velocity gradient and decompose it into vorticity and rate-of-strain:

.. code-block:: python

   import jax.numpy as jnp
   from softmobility import shear_flow

   flow = shear_flow(shear_rate=1.0)
   u = flow.velocity(jnp.array([0.0, 2.0, 0.0]))
   grad_u = flow.gradient(jnp.zeros(3))
   omega, strain = flow.omega_rate_of_strain(jnp.zeros(3))

Available flow constructors are ``no_flow``, ``shear_flow``,
``rotating_flow``, ``extensional_flow``, and ``taylor_green_flow``.

Updating parameters
-------------------

Constructors return parametric objects whose parameters can be updated by name:

.. code-block:: python

   flow = shear_flow(1.0)
   flow.update_params(shear_rate=2.0)

Parameter shapes should remain fixed after construction because JAX traces
array shapes.

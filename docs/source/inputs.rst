======
Inputs
======

SoftMobility separates inputs from geometry. ``sm.Scalar``, ``sm.Field``, and
``sm.Flow`` wrap user callables and optional parameters in a JAX-compatible form.

Scalars
-------

A ``sm.Scalar`` returns one value from ``value(pos, time)``. It is useful for
active forcing, prescribed controls, or time-dependent coefficients.

.. code-block:: python

   import softmobility as sm

   import jax.numpy as jnp

   constant = sm.constant_scalar(2.0)
   signal = sm.oscillating_scalar(amplitude=1.0, omega=3.0, phase=0.0)
   pos = jnp.zeros(3)
   value = signal.value(pos, time=0.5)

Fields
------

A ``sm.Field`` returns a vector of shape ``(3,)`` from ``vector(pos, time)``.
Common constructors include gravity and simple magnetic-field signals:

.. code-block:: python

   import softmobility as sm

   gravity = sm.gravity_field(g=9.81)
   magnetic = sm.rotating_magnetic_field(amp_x=1.0, amp_y=0.5, omega=2.0)

When a YAML model uses variables such as ``gravity0``, ``gravity1``, and
``gravity2``, the rollout expects an input map key named ``"gravity"`` whose
value is a ``sm.Field``.

Flows
-----

A ``sm.Flow`` returns the background fluid velocity. It can also compute the
velocity gradient and decompose it into vorticity and rate-of-strain:

.. code-block:: python

   import jax.numpy as jnp
   import softmobility as sm

   flow = sm.shear_flow(shear_rate=1.0)
   u = flow.velocity(jnp.array([0.0, 2.0, 0.0]))
   grad_u = flow.gradient(jnp.zeros(3))
   omega, strain = flow.omega_rate_of_strain(jnp.zeros(3))

Available flow constructors are ``sm.no_flow``, ``sm.shear_flow``,
``sm.rotating_flow``, ``sm.extensional_flow``, and ``sm.taylor_green_flow``.

Updating parameters
-------------------

Constructors return parametric objects whose parameters can be updated by name:

.. code-block:: python

   flow = sm.shear_flow(1.0)
   flow.update_params(shear_rate=2.0)

Parameter shapes should remain fixed after construction because JAX traces
array shapes.

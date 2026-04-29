============
Optimization
============

``FlowBodyOptimizer`` performs gradient-based optimization of design variables.
It uses ``jax.value_and_grad`` internally and accepts any Optax optimizer.

Objective functions
-------------------

An objective is a callable with signature:

.. code-block:: python

   objective(rollout, design) -> scalar

For example, to maximize final vertical displacement:

.. code-block:: python

   import optax
   from softmobility import FlowBodyOptimizer

   def final_height(rollout, design):
       positions, _, _ = rollout.rollout(dt=0.01, n_steps=200, design=design)
       return positions[-1, 2]

   optimizer = FlowBodyOptimizer(
       rollout,
       objective=final_height,
       optimizer=optax.adam(1e-3),
   )
   best_design = optimizer.run(
       init_design=rollout.soft_body.design_defaults,
       n_steps=500,
       maximize=True,
   )

Practical guidance
------------------

Start by evaluating the objective for the default design before optimizing. If
the objective returns ``nan`` or changes discontinuously with small design
perturbations, fix that before running many optimizer steps.

Use ``clip_min`` and ``clip_max`` to keep design variables in a physically
meaningful range. Use ``grad_clip`` if the model has sharp hydrodynamic or
geometric sensitivities.

Experimental reinforcement learning
------------------------------------

``FlowBodyRL`` is an experimental actor-critic-style helper. Its current API is
less mature than ``FlowBodyOptimizer``. Prefer ``FlowBodyOptimizer`` unless you
are developing reinforcement-learning methods around SoftMobility rollouts.

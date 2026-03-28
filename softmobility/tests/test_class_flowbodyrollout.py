import pytest
import jax
import jax.numpy as jnp
import numpy as np
from softmobility import SoftBody, no_flow, gravity_field, constant_scalar
from softmobility.classes.solver import FlowBodyRollout

# --- Setup: plain module-level instances ---
YAML = """
input_names: [gravity, active_force]

# Spheres 
spheres:
  - # Sphere 0 #################
    radius: 1                       
    position: [-1, 0, 0]  
    force: [gravity0, 0 , 0]           
    torque: [0, 0, active_force]

  - # Sphere 1 #################
    radius: 1                 
    position: [1, 0, 0]   
    orientation: [0, 0, dof]    
    force: [-gravity0, 0 , 0]           
    torque: [0, 0, -active_force]
"""

BODY = SoftBody(YAML, verbose=False)
DESIGN = BODY.design_defaults
ROLLOUT = FlowBodyRollout(
    soft_body=BODY,
    flow=no_flow(),
    input_map={"gravity": gravity_field(), "active_force": constant_scalar()},
)

POS = jnp.zeros(3)
ORI = jnp.zeros(3)
DOFS = jnp.zeros(BODY.Ndof)
DT = 0.1
N_STEPS = 2


# --- Tests just use the module-level instances directly ---


def test_velocity_shapes():
    v, omega, ddofs = ROLLOUT.velocity(DESIGN, POS, ORI, DOFS, time=0.0)
    assert v.shape == (3,)
    assert omega.shape == (3,)
    assert ddofs.shape == (BODY.Ndof,)


def test_rollout_shapes():
    positions, orientations, dofs = ROLLOUT.rollout(DT, N_STEPS)
    assert positions.shape == (N_STEPS, 3)
    assert orientations.shape == (N_STEPS, 3)
    assert dofs.shape == (N_STEPS, BODY.Ndof)


def test_jittable():
    jitted = jax.jit(lambda d: ROLLOUT.rollout(DT, N_STEPS, design=d))
    positions, _, _ = jitted(DESIGN)
    assert positions.shape == (N_STEPS, 3)


def test_differentiable():
    def objective(design):
        positions, _, _ = ROLLOUT.rollout(DT, N_STEPS, design=design)
        return positions[-1, 0]

    grad = jax.grad(objective)(DESIGN)
    assert grad.shape == DESIGN.shape
    assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradient: {grad}"


def test_vmappable():
    init_positions = jnp.stack([POS, POS + 1.0])
    batched = jax.vmap(lambda pos: ROLLOUT.rollout(DT, N_STEPS, init_position=pos))
    positions, _, _ = batched(init_positions)
    assert positions.shape == (2, N_STEPS, 3)


if __name__ == "__main__":
    test_velocity_shapes()
    test_rollout_shapes()
    test_jittable()
    test_differentiable()

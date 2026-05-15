import jax
import jax.numpy as jnp
import pytest

from softmobility import SoftBody, constant_scalar, gravity_field, no_flow
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
    v, omega, ddofs = ROLLOUT._velocity(DESIGN, POS, ORI, DOFS, time=0.0)
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


def test_rk4_rollout_shapes():
    positions, orientations, dofs = ROLLOUT.rollout(DT, N_STEPS, scheme="rk4")
    assert positions.shape == (N_STEPS, 3)
    assert orientations.shape == (N_STEPS, 3)
    assert dofs.shape == (N_STEPS, BODY.Ndof)


def test_rk4_jittable_and_differentiable():
    @jax.jit
    def objective(design):
        positions, _, _ = ROLLOUT.rollout(DT, N_STEPS, design=design, scheme="rk4")
        return positions[-1, 0]

    val = objective(DESIGN)
    assert jnp.isfinite(val)
    grad = jax.grad(objective)(DESIGN)
    assert grad.shape == DESIGN.shape
    assert jnp.all(jnp.isfinite(grad))


def test_rk4_more_accurate_than_rk2():
    """RK4 should be strictly more accurate than RK2 at the same dt for smooth
    rotational dynamics. Use a chiral 3-sphere body so the orientation
    actually evolves (the module-level ``BODY`` has perfectly balanced
    torques and would give zero error for both schemes)."""
    yaml = """
    input_names: [gravity]
    spheres:
      - radius: 1.0
        position: [0, 0, 0]
        force: [gravity0, gravity1, gravity2]
      - radius: 0.5
        position: [1.5, 0, 0]
        force: [gravity0, gravity1, gravity2]
      - radius: 0.5
        position: [0, 1.5, 0]
        force: [gravity0, gravity1, gravity2]
    """
    body = SoftBody(yaml, verbose=False)
    rollout = FlowBodyRollout(body, no_flow(), {"gravity": gravity_field(g=10.0)})

    dt = 0.5
    n_steps = 20
    _, ori_ref, _ = rollout.rollout(dt / 8, n_steps * 8, scheme="rk4")
    ori_ref = ori_ref[7::8]
    _, ori_rk2, _ = rollout.rollout(dt, n_steps, scheme="rk2")
    _, ori_rk4, _ = rollout.rollout(dt, n_steps, scheme="rk4")

    err_rk2 = float(jnp.max(jnp.abs(ori_rk2 - ori_ref)))
    err_rk4 = float(jnp.max(jnp.abs(ori_rk4 - ori_ref)))
    assert err_rk2 > 1e-4, f"need a regime where RK2 has error; got {err_rk2:.2e}"
    assert err_rk4 < err_rk2 / 100, (
        f"RK4 ({err_rk4:.2e}) should be at least 100x better than RK2 ({err_rk2:.2e})"
    )


def test_rollout_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unknown integration scheme"):
        ROLLOUT.rollout(DT, N_STEPS, scheme="rk5")


def test_input_map_validation_errors_are_clear():
    with pytest.raises(ValueError, match="Missing Field input 'gravity'"):
        FlowBodyRollout(BODY, no_flow(), input_map={"active_force": constant_scalar()})

    with pytest.raises(TypeError, match="Input 'gravity' expected a Field"):
        FlowBodyRollout(
            BODY,
            no_flow(),
            input_map={"gravity": constant_scalar(), "active_force": constant_scalar()},
        )

    with pytest.raises(ValueError, match="Unexpected input keys"):
        FlowBodyRollout(
            BODY,
            no_flow(),
            input_map={"gravity": gravity_field(), "active_force": constant_scalar(), "extra": constant_scalar()},
        )


if __name__ == "__main__":
    test_velocity_shapes()
    test_rollout_shapes()
    test_jittable()
    test_differentiable()

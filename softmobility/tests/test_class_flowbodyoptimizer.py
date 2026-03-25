# tests/test_class_flowbodyoptimizer.py
import jax
import jax.numpy as jnp
import optax
from softmobility import SoftBody, no_flow, gravity_field, constant_scalar
from softmobility.classes.solver import FlowBodyRollout, FlowBodyOptimizer

# --- Setup ---
YAML = """
input_names: [gravity, active_force]
spheres:
  - radius: 1
    position: [-1, 0, 0]
    force: [gravity0, 0, 0]
    torque: [0, 0, active_force]
  - radius: 1
    position: [1, 0, 0]
    orientation: [0, 0, dof]
    force: [-gravity0, 0, 0]
    torque: [0, 0, -active_force]
"""

BODY = SoftBody(YAML, verbose=False)
ROLLOUT = FlowBodyRollout(
    soft_body=BODY,
    flow=no_flow(),
    input_map={"gravity": gravity_field(), "active_force": constant_scalar()},
)

POS = jnp.ones(3) * 1e-6
ORI = jnp.ones(3) * 1e-6
DOFS = jnp.ones(BODY.Ndof) * 1e-6
DESIGN = BODY.design_defaults
DT = 0.1
N_STEPS = 10


# --- Minimal objective: maximize final X displacement ---
def simple_objective(rollout, design):
    positions, _, _ = rollout.rollout(design, POS, ORI, DOFS, dt=DT, n_steps=N_STEPS)
    return positions[-1, 0]


# --- Tests ---


def test_optimizer_runs():
    """Optimizer completes without error and returns a design array."""
    opt = FlowBodyOptimizer(ROLLOUT, simple_objective)
    result = opt.run(DESIGN, n_steps=5, print_every=5)
    assert result.shape == DESIGN.shape


def test_optimizer_returns_finite_design():
    """Optimal design should never contain NaN or Inf."""
    opt = FlowBodyOptimizer(ROLLOUT, simple_objective)
    result = opt.run(DESIGN, n_steps=10, print_every=10)
    assert jnp.all(jnp.isfinite(result)), f"Non-finite design: {result}"


def test_optimizer_improves_objective():
    """After optimization, objective should be >= initial value (we are maximizing)."""
    opt = FlowBodyOptimizer(ROLLOUT, simple_objective)
    initial_val = float(simple_objective(ROLLOUT, DESIGN))
    result = opt.run(DESIGN, n_steps=20, print_every=20)
    final_val = float(simple_objective(ROLLOUT, result))
    assert final_val >= initial_val - 1e-4, f"Optimizer made things worse: {initial_val:.5f} → {final_val:.5f}"


def test_optimizer_clip():
    """Clipped design should always stay within bounds."""
    opt = FlowBodyOptimizer(ROLLOUT, simple_objective)
    result = opt.run(DESIGN, n_steps=10, print_every=10, clip_min=0.5, clip_max=2.0)
    assert jnp.all(result >= 0.5 - 1e-6), f"Below clip_min: {result}"
    assert jnp.all(result <= 2.0 + 1e-6), f"Above clip_max: {result}"


def test_optimizer_custom_optax():
    """Optimizer works with a user-supplied optax optimizer."""
    opt = FlowBodyOptimizer(ROLLOUT, simple_objective, optimizer=optax.adam(1e-2))
    result = opt.run(DESIGN, n_steps=5, print_every=5)
    assert result.shape == DESIGN.shape
    assert jnp.all(jnp.isfinite(result))


def test_optimizer_minimize():
    """maximize=False should minimize the objective."""
    opt = FlowBodyOptimizer(ROLLOUT, simple_objective)
    initial_val = float(simple_objective(ROLLOUT, DESIGN))
    result = opt.run(DESIGN, n_steps=20, print_every=20, maximize=False)
    final_val = float(simple_objective(ROLLOUT, result))
    assert final_val <= initial_val + 1e-4, f"Minimization made things larger: {initial_val:.5f} → {final_val:.5f}"


def test_optimizer_grad_clip():
    """Gradient clipping should not cause NaN even with aggressive lr."""
    opt = FlowBodyOptimizer(ROLLOUT, simple_objective, optimizer=optax.sgd(1.0))
    result = opt.run(DESIGN, n_steps=10, print_every=10, grad_clip=0.1)
    assert jnp.all(jnp.isfinite(result)), f"NaN with grad clipping: {result}"


if __name__ == "__main__":
    test_optimizer_runs()
    test_optimizer_returns_finite_design()
    test_optimizer_improves_objective()
    test_optimizer_clip()
    test_optimizer_custom_optax()
    test_optimizer_minimize()
    test_optimizer_grad_clip()

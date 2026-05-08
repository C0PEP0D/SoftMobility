"""Tests for the FlexibleFiber subclass of SoftBody."""

import time

import jax.numpy as jnp
import numpy as np
import pytest

import softmobility as sm
from softmobility import FlexibleFiber


def test_construction_3d():
    body = FlexibleFiber(n_beads=5)
    assert body.Nspheres == 5
    assert body.Ndof == 15  # 3 per bead
    assert body.Ndesign == 3  # radius, K_b, mass (no gap)
    assert body.Ninput == 3  # gravity field
    assert body.design_variables == ["radius", "K_b", "mass"]
    assert body.input_variables == ["gravity0", "gravity1", "gravity2"]


def test_construction_planar():
    body = FlexibleFiber(n_beads=5, planar=True)
    assert body.Nspheres == 5
    assert body.Ndof == 5  # 1 per bead
    assert body.dof_variables == [f"theta_{i}" for i in range(5)]


def test_n_beads_validation():
    with pytest.raises(ValueError, match="n_beads"):
        FlexibleFiber(n_beads=1)


def test_straight_equilibrium_positions():
    """Default DOFs ⇒ straight chain along ê_x with bond length exactly 2a."""
    radius = 1.0
    body = FlexibleFiber(n_beads=4, radius=radius)
    bond = 2.0 * radius
    t = jnp.array([0.0])
    expected = jnp.stack([jnp.array([i * bond, 0.0, 0.0]) for i in range(4)])
    actual = jnp.stack([body.spheres[i].position(body.dof_defaults, body.design_defaults, t) for i in range(4)])
    assert jnp.allclose(actual, expected, atol=1e-6)
    body.validate_no_overlap()


def test_planar_bending_kinematics():
    """For planar N=3 with θ_1 = ε, the recurrence gives an analytic answer."""
    n = 3
    radius = 1.0
    body = FlexibleFiber(n_beads=n, radius=radius, planar=True)
    eps = 0.1
    dofs = jnp.array([0.0, eps, 0.0])
    t = jnp.array([0.0])
    a = radius
    # Planar sign convention: p = (cos θ, 0, +sin θ).
    p0 = jnp.array([1.0, 0.0, 0.0])
    p1 = jnp.array([jnp.cos(eps), 0.0, jnp.sin(eps)])
    p2 = jnp.array([1.0, 0.0, 0.0])
    r1_expected = a * (p0 + p1)
    r2_expected = r1_expected + a * (p1 + p2)
    r1 = body.spheres[1].position(dofs, body.design_defaults, t)
    r2 = body.spheres[2].position(dofs, body.design_defaults, t)
    assert jnp.allclose(r1, r1_expected, atol=1e-6)
    assert jnp.allclose(r2, r2_expected, atol=1e-6)


def test_planar_C_K_is_torsional_chain_laplacian():
    """``C_K`` (Ty rows) is the discrete Laplacian on bead orientations
    with coefficient ``K_b / (2a)`` (linearized Gears bending in the
    implicit-DOF parameterization)."""
    n = 5
    K_b = 1.0
    a = 1.0
    coef = K_b / (2.0 * a)
    body = FlexibleFiber(n_beads=n, bending_rigidity=K_b, planar=True)
    C_K = np.asarray(body.grand_C_K())

    M = np.zeros((n, n))
    M[0, 0] = -coef
    M[0, 1] = coef
    for i in range(1, n - 1):
        M[i, i - 1] = coef
        M[i, i] = -2.0 * coef
        M[i, i + 1] = coef
    M[n - 1, n - 2] = coef
    M[n - 1, n - 1] = -coef

    for i in range(n):
        row = i * 6 + 4  # Ty row of bead i
        assert np.allclose(C_K[row], M[i], atol=1e-5), (
            f"row {row} (Ty bead {i}): expected {M[i]}, got {C_K[row]}"
        )


def test_3D_C_H_is_mass_times_identity_on_force():
    """C_H couples only force rows, with each force block = mass · I_3."""
    n = 4
    mass = 2.5
    body = FlexibleFiber(n_beads=n, mass=mass)
    C_H = np.asarray(body.grand_C_H())
    assert C_H.shape == (6 * n, 3)
    for i in range(n):
        force_block = C_H[i * 6 : i * 6 + 3]
        torque_block = C_H[i * 6 + 3 : i * 6 + 6]
        assert np.allclose(force_block, mass * np.eye(3), atol=1e-6)
        assert np.allclose(torque_block, np.zeros((3, 3)), atol=1e-6)


def test_mobility_tensor_symmetric():
    body = FlexibleFiber(n_beads=10)
    M = body.compute_grand_mobility()
    assert M.shape == (60, 60)
    assert jnp.all(jnp.isfinite(M))
    assert jnp.allclose(M, M.T, atol=1e-7)


def test_compute_tensors_runs():
    """End-to-end: compute_tensors should run for a small fiber."""
    body = FlexibleFiber(n_beads=5)
    out = body.compute_tensors()
    assert jnp.all(jnp.isfinite(out.M))
    assert jnp.all(jnp.isfinite(out.M_K))
    assert jnp.all(jnp.isfinite(out.M_H))


def test_construction_speed_n50():
    """Smoke test: construction of N=50 should complete in a sane time.

    Threshold is generous (30 s) because CI runners are noticeably slower and
    noisier than local machines; on an M-series Mac we observe ~3 s. The
    intent is to catch O(N²) regressions, not to lock in absolute timing.
    """
    t0 = time.perf_counter()
    FlexibleFiber(n_beads=50)
    dt = time.perf_counter() - t0
    assert dt < 30.0, f"Construction took {dt:.2f}s, expected <30s"


# ---------------------------------------------------------------------------
# Bending physics (linearized Gears, eq. 32+34)
# ---------------------------------------------------------------------------


def test_straight_no_torque():
    """All-zero DOFs ⇒ every bead torque is zero."""
    n = 6
    body = FlexibleFiber(n_beads=n, planar=True)
    dofs = jnp.zeros(n)
    design = body.design_defaults
    inputs = jnp.zeros(body.Ninput)
    for i in range(n):
        t_i = body.spheres[i].torque(dofs, design, inputs)
        assert jnp.allclose(t_i, 0.0, atol=1e-7)


def test_planar_uniform_progression_no_interior_torque():
    """θ_i = i·δθ (linear progression) ⇒ Laplacian of linear is zero
    ⇒ interior bead torques are zero. Only the end beads carry torque
    (γ_0 = coef·δθ, γ_{N-1} = −coef·δθ)."""
    n = 7
    K_b = 1.0
    a = 1.0
    coef = K_b / (2.0 * a)
    body = FlexibleFiber(n_beads=n, bending_rigidity=K_b, radius=a, planar=True)
    delta = 0.05
    dofs = jnp.array([i * delta for i in range(n)])
    design = body.design_defaults
    inputs = jnp.zeros(body.Ninput)

    torques_y = np.array(
        [float(body.spheres[i].torque(dofs, design, inputs)[1]) for i in range(n)]
    )
    expected = np.zeros(n)
    expected[0] = coef * delta
    expected[-1] = -coef * delta
    np.testing.assert_allclose(torques_y, expected, atol=1e-6)


def test_planar_3D_consistency_around_y():
    """A 3D fiber whose only non-zero rod component is the y component should
    produce the same Ty stencil as the planar fiber."""
    n = 6
    K_b = 1.7
    a = 1.0
    body_planar = FlexibleFiber(n_beads=n, bending_rigidity=K_b, radius=a, planar=True)
    body_3d = FlexibleFiber(n_beads=n, bending_rigidity=K_b, radius=a, planar=False)

    rng = np.random.default_rng(1)
    thetas = rng.uniform(-0.05, 0.05, size=n)
    dofs_planar = jnp.asarray(thetas)
    dofs_3d = jnp.asarray(np.stack([np.zeros(n), thetas, np.zeros(n)], axis=1).reshape(-1))

    inputs = jnp.zeros(body_planar.Ninput)
    for i in range(n):
        t_p = body_planar.spheres[i].torque(dofs_planar, body_planar.design_defaults, inputs)
        t_3 = body_3d.spheres[i].torque(dofs_3d, body_3d.design_defaults, inputs)
        # Planar puts torque on +Ty; 3D should match (Ty component) and have zero Tx, Tz.
        np.testing.assert_allclose(float(t_3[0]), 0.0, atol=1e-6)
        np.testing.assert_allclose(float(t_3[1]), float(t_p[1]), atol=1e-6)
        np.testing.assert_allclose(float(t_3[2]), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Time-varying clamping (FlowBodyRollout)
# ---------------------------------------------------------------------------


def _quiescent_planar_rollout(n=5, K_b=1.0):
    """Helper: return a FlowBodyRollout for a planar fiber in no flow / no gravity."""
    fiber = FlexibleFiber(n_beads=n, bending_rigidity=K_b, mass=0.0, planar=True)
    rollout = sm.FlowBodyRollout(
        soft_body=fiber,
        flow=sm.no_flow(),
        input_map={"gravity": sm.gravity_field(g=0.0)},
    )
    return fiber, rollout


def test_clamp_position_holds():
    """``clamp_position_fn`` must override the body translation at every step."""
    _, rollout = _quiescent_planar_rollout()
    target = jnp.array([0.5, 0.0, 0.0])
    positions, _, _ = rollout.rollout(
        dt=0.01,
        n_steps=20,
        clamp_position_fn=lambda t: target,
    )
    np.testing.assert_allclose(np.asarray(positions), np.tile(target, (20, 1)), atol=1e-7)


def test_clamp_oscillating_dof():
    """Clamping ``θ_1(t) = α₀·sin(ζt)`` must reproduce the prescription element-wise."""
    n = 5
    fiber, rollout = _quiescent_planar_rollout(n=n)
    alpha0 = 0.1
    zeta = 2.0
    mask = jnp.zeros(n, dtype=bool).at[1].set(True)
    dt = 0.01
    n_steps = 30

    def dofs_fn(t):
        return jnp.zeros(n).at[1].set(alpha0 * jnp.sin(zeta * t))

    _, _, dofs_traj = rollout.rollout(
        dt=dt,
        n_steps=n_steps,
        clamp_dofs_mask=mask,
        clamp_dofs_fn=dofs_fn,
    )
    dofs_traj = np.asarray(dofs_traj)
    times = (np.arange(n_steps) + 1) * dt
    expected_theta1 = alpha0 * np.sin(zeta * times)
    np.testing.assert_allclose(dofs_traj[:, 1], expected_theta1, atol=1e-7)


def test_clamp_unaffected_dofs_evolve():
    """Clamping ``θ_0`` only must leave the other DOFs free to evolve under
    bending dynamics — not freeze the entire chain."""
    n = 5
    fiber, rollout = _quiescent_planar_rollout(n=n, K_b=10.0)
    init_dofs = jnp.array([0.0, 0.3, -0.2, 0.1, 0.0])
    mask = jnp.zeros(n, dtype=bool).at[0].set(True)
    _, _, dofs_traj = rollout.rollout(
        dt=0.01,
        n_steps=50,
        init_dofs=init_dofs,
        clamp_dofs_mask=mask,
        clamp_dofs_fn=lambda t: jnp.zeros(n),
    )
    dofs_traj = np.asarray(dofs_traj)
    # θ_0 stays clamped at 0
    np.testing.assert_allclose(dofs_traj[:, 0], 0.0, atol=1e-7)
    # other DOFs change over time (bending relaxation)
    assert np.max(np.abs(dofs_traj[-1, 1:] - init_dofs[1:])) > 1e-3


def test_clamp_dofs_fn_requires_mask():
    """Passing only one of ``clamp_dofs_fn`` / ``clamp_dofs_mask`` raises."""
    _, rollout = _quiescent_planar_rollout()
    with pytest.raises(ValueError, match="clamp_dofs_fn and clamp_dofs_mask"):
        rollout.rollout(dt=0.01, n_steps=2, clamp_dofs_fn=lambda t: jnp.zeros(5))

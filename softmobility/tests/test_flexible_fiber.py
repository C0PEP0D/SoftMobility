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
    # Sphere 0 has no per-bead orientation DOF (its tangent is identified
    # with the body frame's ê_x), and the chain has no torsional DOF
    # (the x-component of each Rodrigues vector is structurally zero).
    # So Ndof = 2 · (N - 1).
    assert body.Ndof == 8
    assert body.Ndesign == 4  # radius, K_b, mass, kappa_0 (no gap)
    assert body.Ninput == 3  # gravity field
    assert body.design_variables == ["radius", "K_b", "mass", "kappa_0"]
    assert body.input_variables == ["gravity0", "gravity1", "gravity2"]
    # DOF naming: theta_{i}_y, theta_{i}_z for distal beads only.
    assert body.dof_variables == [
        f"theta_{i}_{c}" for i in range(1, 5) for c in ("y", "z")
    ]


def test_construction_planar():
    body = FlexibleFiber(n_beads=5, planar=True)
    assert body.Nspheres == 5
    # Same convention: only spheres 1 .. N-1 carry orientation DOFs.
    assert body.Ndof == 4
    assert body.dof_variables == [f"theta_{i}" for i in range(1, 5)]


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
    """Bond length is exactly 2a in any bent configuration; bond direction
    is the normalized average of the two adjacent bead tangents."""
    n = 3
    radius = 1.0
    body = FlexibleFiber(n_beads=n, radius=radius, planar=True)
    eps = 0.1
    # DOFs are theta_1, theta_2 (sphere 0's orientation is structurally
    # zero and not a DOF). Pick theta_1 = eps, theta_2 = 0 so bead 1's
    # tangent is tilted but bead 2's tangent is along ê_x.
    dofs = jnp.array([eps, 0.0])
    t = jnp.array([0.0])
    a = radius
    # Planar sign convention: p = (cos θ, 0, +sin θ).
    p0 = jnp.array([1.0, 0.0, 0.0])
    p1 = jnp.array([jnp.cos(eps), 0.0, jnp.sin(eps)])
    p2 = jnp.array([1.0, 0.0, 0.0])

    def _bond(p_a, p_b):
        s = p_a + p_b
        return 2.0 * a * s / jnp.linalg.norm(s)

    r0 = body.spheres[0].position(dofs, body.design_defaults, t)
    r1 = body.spheres[1].position(dofs, body.design_defaults, t)
    r2 = body.spheres[2].position(dofs, body.design_defaults, t)

    assert jnp.allclose(r0, jnp.zeros(3), atol=1e-6)
    assert jnp.allclose(r1 - r0, _bond(p0, p1), atol=1e-6)
    assert jnp.allclose(r2 - r1, _bond(p1, p2), atol=1e-6)
    # Both bonds have length exactly 2a regardless of bending.
    assert jnp.isclose(jnp.linalg.norm(r1 - r0), 2.0 * a, atol=1e-6)
    assert jnp.isclose(jnp.linalg.norm(r2 - r1), 2.0 * a, atol=1e-6)


def test_bent_bond_length_invariant():
    """Bond length stays at 2a for a wide range of joint angles (planar)."""
    n = 6
    radius = 1.0
    body = FlexibleFiber(n_beads=n, radius=radius, planar=True)
    t = jnp.array([0.0])
    for kappa in (0.05, 0.2, 0.5, 1.0):  # up to ~57° per joint
        # DOFs cover beads 1 .. N-1; bead 0's tangent is structurally
        # along ê_x. Linear progression θ_i = i·κ then becomes
        # dofs = (κ, 2κ, ..., (N-1)·κ).
        dofs = kappa * jnp.arange(1, n, dtype=jnp.float32)
        positions = jnp.stack(
            [body.spheres[i].position(dofs, body.design_defaults, t) for i in range(n)]
        )
        bonds = jnp.linalg.norm(positions[1:] - positions[:-1], axis=1)
        assert jnp.allclose(bonds, 2.0 * radius, atol=1e-5), (
            f"κ={kappa}: bonds={bonds}"
        )


def test_planar_C_K_is_torsional_chain_laplacian():
    """``C_K`` (Ty rows) is the discrete Laplacian on bead orientations
    with coefficient ``K_b / (2a)`` (linearized Gears bending in the
    implicit-DOF parameterization).

    Sphere 0 has no orientation DOF (its tangent is identified with the
    body frame's ê_x), so the DOF vector is ``Q = (θ_1, …, θ_{N-1})``
    and ``C_K`` has ``N`` rows but only ``N - 1`` columns. The expected
    rows are therefore the columns 1.. of the full Laplacian on
    ``(θ_0=0, θ_1, …, θ_{N-1})``.
    """
    n = 5
    K_b = 1.0
    a = 1.0
    coef = K_b / (2.0 * a)
    body = FlexibleFiber(n_beads=n, bending_rigidity=K_b, planar=True)
    C_K = np.asarray(body.grand_C_K())

    # Full Laplacian on (θ_0, …, θ_{N-1}); we keep all N rows but drop
    # the first column (θ_0 is structurally zero, not a DOF).
    M_full = np.zeros((n, n))
    M_full[0, 0] = -coef
    M_full[0, 1] = coef
    for i in range(1, n - 1):
        M_full[i, i - 1] = coef
        M_full[i, i] = -2.0 * coef
        M_full[i, i + 1] = coef
    M_full[n - 1, n - 2] = coef
    M_full[n - 1, n - 1] = -coef
    M_dofs = M_full[:, 1:]  # drop the θ_0 column → shape (n, n-1)

    assert C_K.shape[1] == n - 1
    for i in range(n):
        row = i * 6 + 4  # Ty row of bead i
        assert np.allclose(C_K[row], M_dofs[i], atol=1e-5), (
            f"row {row} (Ty bead {i}): expected {M_dofs[i]}, got {C_K[row]}"
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
    dofs = jnp.zeros(n - 1)
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
    # Sphere 0 has no DOF; θ_0 = 0 implicitly. The linear progression
    # θ_i = i·δθ then becomes dofs = (δθ, 2δθ, …, (N-1)·δθ).
    dofs = jnp.array([i * delta for i in range(1, n)])
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
    """A 3D fiber whose only non-zero distal-bead rod component is along
    ``ê_y`` should produce the same Ty stencil as the planar fiber."""
    n = 6
    K_b = 1.7
    a = 1.0
    body_planar = FlexibleFiber(n_beads=n, bending_rigidity=K_b, radius=a, planar=True)
    body_3d = FlexibleFiber(n_beads=n, bending_rigidity=K_b, radius=a, planar=False)

    rng = np.random.default_rng(1)
    # DOFs cover beads 1 .. N-1. Match conventions: planar packs N-1
    # scalars (θ_i around ê_y); 3D packs (θ_i^y, θ_i^z) per distal bead
    # — set θ_i^z = 0 so the chain stays in the xz-plane.
    thetas = rng.uniform(-0.05, 0.05, size=n - 1)
    dofs_planar = jnp.asarray(thetas)
    dofs_3d = jnp.asarray(
        np.stack([thetas, np.zeros(n - 1)], axis=1).reshape(-1)
    )

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
    """Clamping ``θ_1(t) = α₀·sin(ζt)`` must reproduce the prescription
    element-wise. ``θ_1`` is now ``dofs[0]`` (sphere 0 has no DOF, so
    the planar DOF vector is ``(θ_1, …, θ_{N-1})``)."""
    n = 5
    fiber, rollout = _quiescent_planar_rollout(n=n)
    alpha0 = 0.1
    zeta = 2.0
    ndof = n - 1
    mask = jnp.zeros(ndof, dtype=bool).at[0].set(True)
    dt = 0.01
    n_steps = 30

    def dofs_fn(t):
        return jnp.zeros(ndof).at[0].set(alpha0 * jnp.sin(zeta * t))

    _, _, dofs_traj = rollout.rollout(
        dt=dt,
        n_steps=n_steps,
        clamp_dofs_mask=mask,
        clamp_dofs_fn=dofs_fn,
    )
    dofs_traj = np.asarray(dofs_traj)
    times = (np.arange(n_steps) + 1) * dt
    expected_theta1 = alpha0 * np.sin(zeta * times)
    np.testing.assert_allclose(dofs_traj[:, 0], expected_theta1, atol=1e-7)


def test_clamp_unaffected_dofs_evolve():
    """Clamping ``θ_1`` only must leave the other DOFs free to evolve
    under bending dynamics — not freeze the entire chain. (After the
    refactor, sphere 0 has no DOF; the first free DOF is ``θ_1`` at
    ``dofs[0]``.)"""
    n = 5
    fiber, rollout = _quiescent_planar_rollout(n=n, K_b=10.0)
    init_dofs = jnp.array([0.3, -0.2, 0.1, 0.0])  # θ_1, …, θ_4
    ndof = n - 1
    mask = jnp.zeros(ndof, dtype=bool).at[0].set(True)
    _, _, dofs_traj = rollout.rollout(
        dt=0.01,
        n_steps=50,
        init_dofs=init_dofs,
        clamp_dofs_mask=mask,
        clamp_dofs_fn=lambda t: jnp.zeros(ndof),
    )
    dofs_traj = np.asarray(dofs_traj)
    # θ_1 stays clamped at 0
    np.testing.assert_allclose(dofs_traj[:, 0], 0.0, atol=1e-7)
    # other DOFs change over time (bending relaxation)
    assert np.max(np.abs(dofs_traj[-1, 1:] - init_dofs[1:])) > 1e-3


def test_clamp_dofs_fn_requires_mask():
    """Passing only one of ``clamp_dofs_fn`` / ``clamp_dofs_mask`` raises."""
    _, rollout = _quiescent_planar_rollout()
    with pytest.raises(ValueError, match="clamp_dofs_fn and clamp_dofs_mask"):
        rollout.rollout(dt=0.01, n_steps=2, clamp_dofs_fn=lambda t: jnp.zeros(5))


# ---------------------------------------------------------------------------
# Clamped-anchor mobility (Article3.tex appendix `app:clamped_anchor`)
# ---------------------------------------------------------------------------


def test_rollout_clamped_anchor_static_relaxation():
    """Static anchor at the origin, body orientation pinned at zero,
    initial DOFs perturbed — the final ``max|Q|`` must be at least
    half the initial value lower (the chain has measurably relaxed)."""
    n = 4
    K_b = 30.0
    a = 1.0
    init_max = 0.05
    fiber, rollout = _quiescent_planar_rollout(n=n, K_b=K_b)
    init_dofs = jnp.array([init_max, -0.04, 0.03])  # θ_1, θ_2, θ_3
    dt = 0.05 * (2 * a) ** 4 / K_b
    n_steps = 1500  # ~1.5 τ_1 → factor-4 decay, well below init_max/2

    def anchor_pos(t):
        return jnp.zeros(3)

    def anchor_vel(t):
        return jnp.zeros(6)

    _, _, dofs_traj, _ = rollout.rollout_clamped_anchor(
        dt=dt,
        n_steps=n_steps,
        anchor_position_fn=anchor_pos,
        anchor_velocity_fn=anchor_vel,
        init_dofs=init_dofs,
    )
    dofs_traj = np.asarray(dofs_traj)
    final_max = float(np.max(np.abs(dofs_traj[-1])))
    assert final_max < init_max / 2, (
        f"chain did not relax: initial max|Q| = {init_max}, "
        f"final max|Q| = {final_max}"
    )


def test_rollout_clamped_anchor_rotation_drives_dofs():
    """A rotating anchor must drive the DOFs to a non-zero, body-frame
    steady state — in contrast to the post-step kinematic-clamp path
    which leaves the DOFs at exactly zero."""
    n = 4
    K_b = 30.0
    a = 1.0
    # 3-D fiber: the rotation axis (lab ê_x) is perpendicular to the
    # local bending plane, so the chain has DOFs that respond to it.
    fiber = FlexibleFiber(
        n_beads=n, radius=a, bending_rigidity=K_b, mass=0.0, planar=False
    )
    rollout = sm.FlowBodyRollout(
        soft_body=fiber,
        flow=sm.no_flow(),
        input_map={"gravity": sm.gravity_field(g=0.0)},
    )
    psi = 0.2
    # Sp⁴ = L³ ζ γ⊥ / K_b ≈ 9 with these numbers — modest deformation
    # well inside the linearisation regime.
    zeta = 0.1
    dt = 0.05 * (2 * a) ** 4 / K_b
    n_steps = 1500  # rotating actuation reaches body-frame steady state within ~1τ

    def anchor_pos(t):
        return jnp.zeros(3)

    omega_lab = jnp.array([zeta, 0.0, 0.0])

    def anchor_vel(t):
        return jnp.concatenate([jnp.zeros(3), omega_lab])

    init_orientation = jnp.array([0.0, psi, 0.0])

    _, _, dofs_traj, f_0_traj = rollout.rollout_clamped_anchor(
        dt=dt,
        n_steps=n_steps,
        anchor_position_fn=anchor_pos,
        anchor_velocity_fn=anchor_vel,
        init_orientation=init_orientation,
    )
    dofs_traj = np.asarray(dofs_traj)
    f_0_traj = np.asarray(f_0_traj)
    # The clamped formulation must produce non-zero deformation. (The
    # broken post-step kinematic-clamp path used to leave dofs at zero.)
    assert np.max(np.abs(dofs_traj[-1])) > 1e-4, (
        "rotating actuation did not drive any chain deformation"
    )
    # And a non-zero anchor reaction force to hold the rotating chain
    # against viscous drag.
    assert np.max(np.abs(f_0_traj[-1])) > 1e-4


def test_rollout_clamped_anchor_unknown_scheme_raises():
    _, rollout = _quiescent_planar_rollout()
    with pytest.raises(ValueError, match="Unknown integration scheme"):
        rollout.rollout_clamped_anchor(
            dt=0.01,
            n_steps=2,
            anchor_position_fn=lambda t: jnp.zeros(3),
            anchor_velocity_fn=lambda t: jnp.zeros(6),
            scheme="euler",
        )


# ---------------------------------------------------------------------------
# Analytical validation: Euler–Bernoulli tip deflection pins k_t = K_b/(2a)
# ---------------------------------------------------------------------------


def test_static_cantilever_matches_euler_bernoulli():
    """Clamped planar cantilever under uniform transverse gravity must
    relax to the static bead-chain equilibrium obtained from
    ``γ_int + γ_ext = 0``, which in the small-angle limit reduces to a
    tridiagonal linear system with the bending coefficient
    ``k_t = K_b/(2a)``. We solve that system in-test to get the
    discrete reference, then run the rollout to steady state and
    require the tip deflection to match within 5 %. Any factor-of-2
    error in ``k_t`` would scale every DOF (and hence ``δ_tip``) by
    2× and trip the assertion.

    Sanity-check: the discrete prediction agrees with the continuum
    Euler–Bernoulli answer ``q · L⁴ / (8 · K_b)`` up to the expected
    ``O((a/L)²)`` discretisation residual.
    """
    n = 5
    K_b = 30.0
    a = 1.0
    m = 1.0
    g = 1e-3  # small enough to stay in the small-angle (linear) regime
    L = (n - 1) * 2.0 * a
    q = m * g / (2.0 * a)
    alpha = 2.0 * m * g * a**2 / K_b

    # Discrete static balance on the bead-orientation DOFs
    # (θ_1, …, θ_{N-1}): K_lap · θ = α · (2N - 2j - 1), where K_lap is
    # the (N-1)×(N-1) discrete Laplacian with a clamped row at j=0
    # (interior stencil) and the free-tip row [1, -1] at j=N-2.
    K_lap = np.zeros((n - 1, n - 1))
    for j in range(n - 1):
        K_lap[j, j] = -2.0
        if j > 0:
            K_lap[j, j - 1] = 1.0
        if j < n - 2:
            K_lap[j, j + 1] = 1.0
    K_lap[n - 2, n - 2] = -1.0  # free-tip boundary
    rhs = alpha * np.array(
        [2 * n - 2 * (j + 1) - 1 for j in range(n - 1)], dtype=float
    )
    theta_discrete = np.linalg.solve(K_lap, rhs)
    # Planar-code kinematic recurrence (p_z = +sin θ_i ⇒
    # z_{i+1} − z_i = +a · (θ_i + θ_{i+1}), with θ_0 = 0):
    delta_discrete = abs(
        a * (2.0 * np.sum(theta_discrete[:-1]) + theta_discrete[-1])
    )
    delta_continuum = q * L**4 / (8.0 * K_b)

    # Discrete ↔ continuum stay within the O((a/L)²) discretisation
    # residual — guards against subtle changes in the kinematics.
    assert abs(delta_discrete - delta_continuum) / delta_continuum < 0.4, (
        f"discrete {delta_discrete:.4g} and continuum {delta_continuum:.4g} "
        "tip deflections disagree by more than 40 % — kinematics or "
        "γ_ext projection has drifted."
    )

    fiber = FlexibleFiber(
        n_beads=n, radius=a, bending_rigidity=K_b, mass=m, planar=True
    )
    rollout = sm.FlowBodyRollout(
        soft_body=fiber,
        flow=sm.no_flow(),
        input_map={"gravity": sm.gravity_field(g=g)},
    )

    dt = 0.05 * (2.0 * a) ** 4 / K_b
    n_steps = 24000  # ≳ 4 τ_1 of slowest-mode relaxation for N=5 (τ_1 ∝ L⁴)

    _, _, dofs_traj, _ = rollout.rollout_clamped_anchor(
        dt=dt,
        n_steps=n_steps,
        anchor_position_fn=lambda t: jnp.zeros(3),
        anchor_velocity_fn=lambda t: jnp.zeros(6),
    )
    final_dofs = jnp.asarray(dofs_traj[-1])

    tip_pos = fiber.spheres[n - 1].position(
        final_dofs, fiber.design_defaults, jnp.array([0.0])
    )
    delta_sim = float(jnp.abs(tip_pos[2]))

    rel_err = abs(delta_sim - delta_discrete) / delta_discrete
    assert rel_err < 0.05, (
        f"sim tip deflection {delta_sim:.6g} differs from the discrete "
        f"static-balance prediction {delta_discrete:.6g} by "
        f"{rel_err * 100:.1f} % (> 5 %); k_t = K_b/(2a) prefactor "
        "is likely wrong"
    )


# ---------------------------------------------------------------------------
# Linear-bending validation: see drafts/linear_curvature_findings.md
# ---------------------------------------------------------------------------


def _bending_eigendecomp(fiber):
    """Eigendecompose ``J = M_K[6:, :]`` at the straight reference.

    Returns ``(eigvals, right_eigvecs, left_eigvecs)`` with modes sorted
    by ``|Re(eigval)|`` ascending (slowest mode first). ``J`` is real
    and non-symmetric, but its spectrum is real (the linearised
    soft-mobility problem is similar to a SPD operator), so we drop the
    tiny imaginary parts after sorting.
    """
    tensors = fiber.compute_tensors(
        jnp.zeros(fiber.Ndof), fiber.design_defaults, jnp.array([0.0])
    )
    J = np.asarray(tensors.M_K[6:, :])
    eigvals, eigvecs = np.linalg.eig(J)
    order = np.argsort(np.abs(eigvals.real))
    eigvals = eigvals[order].real
    eigvecs = eigvecs[:, order].real
    W = np.linalg.inv(eigvecs)  # rows are left eigenvectors
    return eigvals, eigvecs, W


def _fit_log_decay(t, signal):
    """Least-squares fit of ``log|signal| = log|s0| - λ·t``.

    Drops samples below ``1e-9 × max|signal|`` (noise floor) so a mode
    that decays through many e-folds is fit on the well-resolved
    portion of its trajectory. Returns ``(λ, R²)``.
    """
    abs_s = np.abs(signal)
    mask = abs_s > 1e-9 * abs_s.max()
    tf = t[mask]
    log_y = np.log(abs_s[mask])
    A = np.vstack([tf, np.ones_like(tf)]).T
    coef, *_ = np.linalg.lstsq(A, log_y, rcond=None)
    slope, _ = coef
    y_pred = A @ coef
    ss_res = float(np.sum((log_y - y_pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(-slope), float(r_sq)


def test_linear_bending_eigenmode_decay():
    """Each mode of the body-frame DOF mobility operator
    ``J = M_K[6:, :]`` must decay exponentially at exactly the rate set
    by its eigenvalue when initialised as a pure right-eigenvector
    perturbation. Validates internal consistency of the linearised
    bending torque and the soft-mobility tensor builder at the straight
    reference. Background sweep: ``drafts/linear_curvature_findings.md``.
    """
    fiber, rollout = _quiescent_planar_rollout(n=3, K_b=10.0)
    eigvals, eigvecs, W = _bending_eigendecomp(fiber)
    dt = 0.05 / abs(eigvals.min())  # stability w.r.t. fastest mode
    # ~2 e-folds of the slowest mode is more than enough for a clean
    # exponential fit; faster modes decay through tens of e-folds in the
    # same window, and the noise-floor mask in _fit_log_decay clips the
    # asymptotic floor.
    n_steps = max(200, int(np.ceil(2.0 / abs(eigvals[0]) / dt)))
    times = (np.arange(n_steps) + 1) * dt

    for k in range(1, fiber.Ndof + 1):
        v_right = eigvecs[:, k - 1]
        w_left = W[k - 1, :]
        init_dofs = jnp.asarray(0.05 * v_right / np.max(np.abs(v_right)))
        _, _, dofs_traj = rollout.rollout(
            dt=dt, n_steps=n_steps, init_dofs=init_dofs
        )
        proj = np.asarray(dofs_traj) @ w_left
        lam_sim, r_sq = _fit_log_decay(times, proj)
        lam_th = abs(eigvals[k - 1])
        assert r_sq > 0.999, f"k={k}: exponential fit R² = {r_sq:.5f} (< 0.999)"
        rel_err = abs(lam_sim - lam_th) / lam_th
        assert rel_err < 0.01, (
            f"k={k}: λ_sim = {lam_sim:.4e} vs λ_th = {lam_th:.4e}, "
            f"rel.err = {rel_err * 100:.2f} % (> 1 %); the linearised "
            "bending torque or the M_K tensor build is wrong at the "
            "straight reference"
        )


def test_linear_bending_amplitude_independence():
    """The linearised bending torque is — by design — exactly
    proportional to the DOFs. Sweeping the slowest-eigenmode initial
    amplitude from 0.05 rad to 1.0 rad must therefore leave the
    decay rate unchanged. Fails if a future Eq.-34 nonlinear-curvature
    swap-in (Delmotte 2015) silently makes the torque
    amplitude-dependent. Background: ``drafts/linear_curvature_findings.md``.
    """
    fiber, rollout = _quiescent_planar_rollout(n=3, K_b=10.0)
    eigvals, eigvecs, W = _bending_eigendecomp(fiber)
    v1 = eigvecs[:, 0]
    w1 = W[0, :]
    lam1 = abs(eigvals[0])
    dt = 0.05 / abs(eigvals.min())
    # ~0.5 e-folds of the slowest mode — short enough that
    # nonlinear mode-coupling at A=1.0 rad can't pollute the
    # projection, long enough for a clean exponential fit.
    n_steps = max(200, int(np.ceil(0.5 / lam1 / dt)))
    times = (np.arange(n_steps) + 1) * dt

    rates = []
    for amp in (0.05, 0.5, 1.0):
        init_dofs = jnp.asarray(amp * v1 / np.max(np.abs(v1)))
        _, _, dofs_traj = rollout.rollout(
            dt=dt, n_steps=n_steps, init_dofs=init_dofs
        )
        proj = np.asarray(dofs_traj) @ w1
        lam, _ = _fit_log_decay(times, proj)
        rates.append(lam)

    spread = max(rates) / min(rates) - 1.0
    assert spread < 0.02, (
        f"decay rates {rates} span more than 2 % "
        f"(max/min - 1 = {spread * 100:.2f} %) — bending torque has "
        "become amplitude-dependent, i.e. no longer linear in the DOFs"
    )


# ---------------------------------------------------------------------------
# Intrinsic curvature: rest state is a uniformly-curved arc
# ---------------------------------------------------------------------------


def test_intrinsic_curvature_equilibrium_is_torque_free_planar():
    """With ``intrinsic_curvature = κ_0``, the uniformly-curved
    configuration ``θ_i = i · 2a · κ_0`` must be the energy minimum —
    i.e. every bead's bending torque vanishes there. Confirms the κ_0
    boundary biases (±K_b·κ_0 on i=0 and i=N-1) are correctly placed
    and the interior κ_0 cancellation is preserved.
    """
    n, a, K_b, kappa_0 = 5, 1.0, 30.0, 0.05
    fiber = FlexibleFiber(
        n_beads=n, radius=a, bending_rigidity=K_b, mass=0.0,
        planar=True, intrinsic_curvature=kappa_0,
    )
    beta = 2.0 * a * kappa_0  # preferred Δθ per bond
    curved_dofs = jnp.asarray([(i + 1) * beta for i in range(n - 1)])
    t = jnp.array([0.0])
    for i in range(n):
        tau = fiber.spheres[i].torque(curved_dofs, fiber.design_defaults, t)
        np.testing.assert_allclose(
            np.asarray(tau), np.zeros(3), atol=1e-5,
            err_msg=f"planar: bead {i} bending torque {np.asarray(tau)} "
            "should vanish at uniformly-curved equilibrium",
        )


def test_intrinsic_curvature_equilibrium_is_torque_free_3d():
    """Same as the planar variant, in 3D mode. Intrinsic curvature is
    around the body ê_y axis, so the curved Rodrigues vector for bead i
    is ``(0, i · 2a · κ_0, 0)``; both unused twist (x) and the
    perpendicular bending (z) components stay zero.
    """
    n, a, K_b, kappa_0 = 4, 1.0, 30.0, 0.05
    fiber = FlexibleFiber(
        n_beads=n, radius=a, bending_rigidity=K_b, mass=0.0,
        planar=False, intrinsic_curvature=kappa_0,
    )
    beta = 2.0 * a * kappa_0
    # 3D DOFs are [θ_1_y, θ_1_z, θ_2_y, θ_2_z, …]; intrinsic curvature is
    # around ê_y, so only the *_y entries pick up the linear progression.
    curved_dofs = jnp.zeros(2 * (n - 1)).at[::2].set(
        jnp.arange(1, n) * beta
    )
    t = jnp.array([0.0])
    for i in range(n):
        tau = fiber.spheres[i].torque(curved_dofs, fiber.design_defaults, t)
        np.testing.assert_allclose(
            np.asarray(tau), np.zeros(3), atol=1e-5,
            err_msg=f"3D: bead {i} bending torque {np.asarray(tau)} "
            "should vanish at uniformly-curved equilibrium",
        )


def test_intrinsic_curvature_default_preserves_existing_behaviour():
    """``intrinsic_curvature=0`` (default) must reproduce the legacy
    bending torque exactly — i.e. the straight configuration is torque-
    free. Guards against accidental sign drift in the κ_0 boundary
    bias terms.
    """
    n, a, K_b = 5, 1.0, 30.0
    fiber = FlexibleFiber(
        n_beads=n, radius=a, bending_rigidity=K_b, mass=0.0, planar=True,
    )
    straight = jnp.zeros(n - 1)
    t = jnp.array([0.0])
    for i in range(n):
        tau = fiber.spheres[i].torque(straight, fiber.design_defaults, t)
        np.testing.assert_allclose(np.asarray(tau), np.zeros(3), atol=1e-6)

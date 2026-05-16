"""Tests for the ExtensibleFiber subclass of SoftBody.

The ExtensibleFiber adds per-bond extension DOFs ``δ_i`` and a linear
axial spring on top of the FlexibleFiber Gears parameterisation. The
tests below mirror the structure of ``test_flexible_fiber.py`` and add
a few ExtensibleFiber-specific checks (DOF layout, bond length tracking
with δ, spring force direction, rigid limit equivalence with
FlexibleFiber).
"""

import time

import jax.numpy as jnp
import numpy as np
import pytest

import softmobility as sm
from softmobility import ExtensibleFiber, FlexibleFiber


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_3d():
    body = ExtensibleFiber(n_beads=5)
    assert body.Nspheres == 5
    # Orientation DOFs: 2 · (N - 1) = 8
    # Extension DOFs:   N - 1 = 4
    # Total: 12
    assert body.Ndof == 12
    assert body.Ndesign == 6  # radius, K_b, gap, k_s, mass, kappa_0
    assert body.Ninput == 3   # gravity field
    assert body.design_variables == ["radius", "K_b", "gap", "k_s", "mass", "kappa_0"]
    assert body.input_variables == ["gravity0", "gravity1", "gravity2"]
    expected_dofs = (
        [f"theta_{i}_{c}" for i in range(1, 5) for c in ("y", "z")]
        + [f"delta_{i}" for i in range(4)]
    )
    assert body.dof_variables == expected_dofs


def test_construction_planar():
    body = ExtensibleFiber(n_beads=5, planar=True)
    assert body.Nspheres == 5
    # Planar: N-1 orientation DOFs + N-1 extension DOFs = 2(N-1)
    assert body.Ndof == 8
    expected_dofs = (
        [f"theta_{i}" for i in range(1, 5)]
        + [f"delta_{i}" for i in range(4)]
    )
    assert body.dof_variables == expected_dofs


def test_n_beads_validation():
    with pytest.raises(ValueError, match="n_beads"):
        ExtensibleFiber(n_beads=1)


def test_negative_gap_rejected():
    with pytest.raises(ValueError, match="gap"):
        ExtensibleFiber(n_beads=3, gap=-0.01)


def test_nonpositive_spring_stiffness_rejected():
    with pytest.raises(ValueError, match="spring_stiffness"):
        ExtensibleFiber(n_beads=3, spring_stiffness=0.0)


# ---------------------------------------------------------------------------
# Equilibrium geometry
# ---------------------------------------------------------------------------


def test_straight_equilibrium_positions():
    """Default DOFs ⇒ straight chain along ê_x with bond length 2a + gap."""
    radius = 1.0
    gap = 0.1
    body = ExtensibleFiber(n_beads=4, radius=radius, gap=gap)
    bond = 2.0 * radius + gap
    t = jnp.array([0.0])
    expected = jnp.stack(
        [jnp.array([i * bond, 0.0, 0.0]) for i in range(4)]
    )
    actual = jnp.stack(
        [
            body.spheres[i].position(body.dof_defaults, body.design_defaults, t)
            for i in range(4)
        ]
    )
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_bent_bond_length_tracking_delta():
    """Bond length is exactly ``2a + gap + δ_i`` for any joint angle."""
    n = 5
    radius = 1.0
    gap = 0.1
    body = ExtensibleFiber(n_beads=n, radius=radius, gap=gap, planar=True)
    eq_bond = 2.0 * radius + gap
    t = jnp.array([0.0])
    n_orient = n - 1

    # Bend with deltas all zero ⇒ bonds at equilibrium length.
    thetas = jnp.array([0.2, -0.1, 0.3, 0.05])
    dofs = jnp.concatenate([thetas, jnp.zeros(n - 1)])
    positions = jnp.stack(
        [body.spheres[i].position(dofs, body.design_defaults, t) for i in range(n)]
    )
    bonds = jnp.linalg.norm(positions[1:] - positions[:-1], axis=1)
    assert jnp.allclose(bonds, eq_bond, atol=1e-5)

    # Set non-zero deltas; each bond length must match 2a + gap + δ_i.
    deltas = jnp.array([0.05, -0.03, 0.02, -0.01])
    dofs = jnp.concatenate([thetas, deltas])
    positions = jnp.stack(
        [body.spheres[i].position(dofs, body.design_defaults, t) for i in range(n)]
    )
    bonds = jnp.linalg.norm(positions[1:] - positions[:-1], axis=1)
    assert jnp.allclose(bonds, eq_bond + deltas, atol=1e-5)


# ---------------------------------------------------------------------------
# Bending physics — same Laplacian as FlexibleFiber, δ-independent
# ---------------------------------------------------------------------------


def test_straight_no_torque():
    """All-zero DOFs ⇒ every bead torque is zero (no bending at rest)."""
    n = 6
    body = ExtensibleFiber(n_beads=n, planar=True)
    dofs = body.dof_defaults
    design = body.design_defaults
    inputs = jnp.zeros(body.Ninput)
    for i in range(n):
        t_i = body.spheres[i].torque(dofs, design, inputs)
        assert jnp.allclose(t_i, 0.0, atol=1e-7)


def test_torque_is_delta_independent():
    """Changing the extension DOFs must not change the bending torque."""
    n = 5
    body = ExtensibleFiber(n_beads=n, planar=True)
    inputs = jnp.zeros(body.Ninput)
    thetas = jnp.array([0.2, 0.4, -0.1, 0.3])
    dofs_a = jnp.concatenate([thetas, jnp.zeros(n - 1)])
    dofs_b = jnp.concatenate([thetas, jnp.array([0.05, -0.03, 0.02, -0.01])])
    for i in range(n):
        t_a = body.spheres[i].torque(dofs_a, body.design_defaults, inputs)
        t_b = body.spheres[i].torque(dofs_b, body.design_defaults, inputs)
        assert jnp.allclose(t_a, t_b, atol=1e-7), f"bead {i}: {t_a} != {t_b}"


def test_planar_torque_matches_flexible_fiber():
    """For δ ≡ 0 the planar bending torque coincides with FlexibleFiber's."""
    n = 5
    K_b = 30.0
    a = 1.0
    ext = ExtensibleFiber(n_beads=n, radius=a, bending_rigidity=K_b, planar=True)
    flex = FlexibleFiber(n_beads=n, radius=a, bending_rigidity=K_b, planar=True)
    inputs_e = jnp.zeros(ext.Ninput)
    inputs_f = jnp.zeros(flex.Ninput)
    thetas = jnp.array([0.05, 0.1, -0.07, 0.02])
    dofs_e = jnp.concatenate([thetas, jnp.zeros(n - 1)])
    dofs_f = thetas
    for i in range(n):
        t_e = ext.spheres[i].torque(dofs_e, ext.design_defaults, inputs_e)
        t_f = flex.spheres[i].torque(dofs_f, flex.design_defaults, inputs_f)
        assert jnp.allclose(t_e, t_f, atol=1e-7), (
            f"bead {i}: ext {t_e} != flex {t_f}"
        )


# ---------------------------------------------------------------------------
# Spring physics
# ---------------------------------------------------------------------------


def test_spring_force_zero_at_equilibrium():
    """With δ ≡ 0, the spring contribution to bead force vanishes."""
    n = 5
    body = ExtensibleFiber(n_beads=n, planar=True)
    inputs = jnp.zeros(body.Ninput)
    dofs = body.dof_defaults
    for i in range(n):
        f_i = body.spheres[i].force(dofs, body.design_defaults, inputs)
        assert jnp.allclose(f_i, 0.0, atol=1e-7)


def test_spring_force_direction_and_magnitude():
    """Single extension δ_0 ≠ 0 ⇒ equal-and-opposite forces along bond 0.

    Bond 0 connects bead 0 to bead 1; with the straight equilibrium the
    bond direction is +ê_x. The expected forces are:
    f_0 = +k_s · δ_0 · ê_x,  f_1 = −k_s · δ_0 · ê_x,  f_{2..N-1} = 0.
    """
    n = 5
    k_s = 100.0
    delta_0 = 0.01
    body = ExtensibleFiber(
        n_beads=n, spring_stiffness=k_s, planar=True
    )
    inputs = jnp.zeros(body.Ninput)
    n_orient = n - 1
    dofs = jnp.concatenate(
        [jnp.zeros(n_orient), jnp.array([delta_0, 0.0, 0.0, 0.0])]
    )
    expected = [
        jnp.array([+k_s * delta_0, 0.0, 0.0]),
        jnp.array([-k_s * delta_0, 0.0, 0.0]),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
    ]
    for i in range(n):
        f_i = body.spheres[i].force(dofs, body.design_defaults, inputs)
        assert jnp.allclose(f_i, expected[i], atol=1e-6), (
            f"bead {i}: got {f_i}, expected {expected[i]}"
        )


def test_spring_force_gravity_superposition():
    """force = gravity + spring; the two are linear and add."""
    n = 3
    mass = 2.0
    k_s = 50.0
    body = ExtensibleFiber(
        n_beads=n, mass=mass, spring_stiffness=k_s, planar=True
    )
    g_vec = jnp.array([0.0, 0.0, -9.81])
    # gravity_field => inputs are body-frame components of g
    inputs = g_vec  # identity for a static frame
    deltas = jnp.array([0.02, -0.01])
    dofs = jnp.concatenate([jnp.zeros(n - 1), deltas])

    # bead 1 should feel gravity + spring(bond 0 on bead 1, bond 1 on bead 1)
    f_1 = body.spheres[1].force(dofs, body.design_defaults, inputs)
    grav = mass * g_vec
    # bond 0: +k_s · δ_0 on bead 0; bead 1 feels the −sign
    # bond 1: +k_s · δ_1 on bead 1
    spring_on_bead_1 = (
        -k_s * deltas[0] * jnp.array([1.0, 0.0, 0.0])
        + k_s * deltas[1] * jnp.array([1.0, 0.0, 0.0])
    )
    assert jnp.allclose(f_1, grav + spring_on_bead_1, atol=1e-6)


# ---------------------------------------------------------------------------
# Coupling matrices and mobility
# ---------------------------------------------------------------------------


def test_C_H_is_mass_times_identity_on_force():
    """Same gravity input convention as FlexibleFiber."""
    n = 4
    mass = 2.5
    body = ExtensibleFiber(n_beads=n, mass=mass)
    C_H = np.asarray(body.grand_C_H())
    assert C_H.shape == (6 * n, 3)
    for i in range(n):
        force_block = C_H[i * 6 : i * 6 + 3]
        torque_block = C_H[i * 6 + 3 : i * 6 + 6]
        assert np.allclose(force_block, mass * np.eye(3), atol=1e-6)
        assert np.allclose(torque_block, np.zeros((3, 3)), atol=1e-6)


def test_C_K_shape_and_finite():
    """C_K has the right shape and finite entries at the default state."""
    n = 5
    body = ExtensibleFiber(n_beads=n, planar=True)
    C_K = np.asarray(body.grand_C_K())
    assert C_K.shape == (6 * n, body.Ndof)
    assert np.all(np.isfinite(C_K))


def test_compute_tensors_runs():
    """End-to-end: compute_tensors should run for a small fiber."""
    body = ExtensibleFiber(n_beads=5, allow_overlap=True)
    out = body.compute_tensors()
    assert jnp.all(jnp.isfinite(out.M))
    assert jnp.all(jnp.isfinite(out.M_K))
    assert jnp.all(jnp.isfinite(out.M_H))


def test_mobility_tensor_symmetric():
    """The grand mobility must remain symmetric (RPY identity)."""
    body = ExtensibleFiber(n_beads=8, allow_overlap=True)
    M = body.compute_grand_mobility()
    assert M.shape == (48, 48)
    assert jnp.all(jnp.isfinite(M))
    assert jnp.allclose(M, M.T, atol=1e-7)


def test_construction_speed_n50():
    """Smoke test: construction of N=50 should complete in a sane time.

    Threshold is generous (30 s) because CI runners are noticeably slower and
    noisier than local machines; on an M-series Mac we observe ~5 s.
    """
    t0 = time.perf_counter()
    ExtensibleFiber(n_beads=50)
    dt = time.perf_counter() - t0
    assert dt < 30.0, f"Construction took {dt:.2f}s, expected <30s"


# ---------------------------------------------------------------------------
# Rigid-limit equivalence with FlexibleFiber
# ---------------------------------------------------------------------------


def test_rigid_limit_matches_flexible_fiber_at_zero_gap():
    """With gap=0 and stiff spring, orientation dynamics match FlexibleFiber.

    The two models have different DOF counts, so we compare the
    orientation-block of ``M_K @ dofs`` at a small bend with the
    extension DOFs pinned to zero. The orientation block must agree
    to better than 1e-6 (single-precision JAX default).
    """
    n = 5
    K_b = 30.0
    a = 1.0
    ext = ExtensibleFiber(
        n_beads=n, radius=a, bending_rigidity=K_b, gap=0.0,
        spring_stiffness=1e6, planar=True, allow_overlap=True,
    )
    flex = FlexibleFiber(n_beads=n, radius=a, bending_rigidity=K_b, planar=True)

    dofs_flex = jnp.array([0.01, 0.02, -0.005, 0.003])
    dofs_ext = jnp.concatenate([dofs_flex, jnp.zeros(n - 1)])

    qdot_flex = flex.compute_tensors().M_K @ dofs_flex
    qdot_ext = ext.compute_tensors(dofs=dofs_ext).M_K @ dofs_ext

    # Orientation-DOF block: indices 6 .. 6+(N-1)
    n_orient = n - 1
    diff = jnp.max(jnp.abs(qdot_flex[6 : 6 + n_orient] - qdot_ext[6 : 6 + n_orient]))
    assert float(diff) < 1e-6, f"orientation block diff = {float(diff):.3e}"


# ---------------------------------------------------------------------------
# Cantilever Euler–Bernoulli benchmark (mirrors test_flexible_fiber.py)
# ---------------------------------------------------------------------------


def test_static_cantilever_matches_euler_bernoulli():
    """Clamped planar cantilever under transverse gravity must relax to the
    static bead-chain equilibrium, with bond length 2a + gap as the
    effective discrete spacing.

    The discrete static balance solves a tridiagonal Laplacian system in
    the orientation DOFs; bond length appears as ``h = (2a + gap) / 2``
    in the small-angle balance (each half-bond contributes a moment arm
    of ``h`` per unit angle). With ``k_s = 10 · K_b / a³`` the spring is
    stable at the chosen ``dt`` and the bond extension stays small
    enough that the bending response is the dominant relaxation.

    Tolerance is 10 % — looser than the 5 % used by FlexibleFiber's
    cantilever test because the finite spring stiffness lets the bond
    length oscillate around equilibrium during the rollout and
    contributes a small residual.
    """
    n = 5
    K_b = 30.0
    a = 1.0
    m = 1.0
    g = 1e-3              # small-angle regime
    gap = 0.1 * a
    bond = 2.0 * a + gap  # equilibrium bond length
    h = bond / 2.0        # half-bond (replaces "a" in the flex case)

    # Discrete static balance, same form as the FlexibleFiber test but
    # parametrised by the bond length.
    alpha = 2.0 * m * g * h**2 / K_b
    K_lap = np.zeros((n - 1, n - 1))
    for j in range(n - 1):
        K_lap[j, j] = -2.0
        if j > 0:
            K_lap[j, j - 1] = 1.0
        if j < n - 2:
            K_lap[j, j + 1] = 1.0
    K_lap[n - 2, n - 2] = -1.0
    rhs = alpha * np.array(
        [2 * n - 2 * (j + 1) - 1 for j in range(n - 1)], dtype=float
    )
    theta_discrete = np.linalg.solve(K_lap, rhs)
    # Kinematic recurrence: z_{i+1} − z_i = +h · (θ_i + θ_{i+1}), θ_0 = 0.
    delta_discrete = abs(
        h * (2.0 * np.sum(theta_discrete[:-1]) + theta_discrete[-1])
    )

    fiber = ExtensibleFiber(
        n_beads=n, radius=a, bending_rigidity=K_b, gap=gap,
        spring_stiffness=10.0 * K_b / a**3, mass=m, planar=True,
        allow_overlap=True,
    )
    rollout = sm.FlowBodyRollout(
        soft_body=fiber,
        flow=sm.no_flow(),
        input_map={"gravity": sm.gravity_field(g=g)},
    )

    dt = 0.05 * (2.0 * a) ** 4 / K_b
    n_steps = 24000

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
    assert rel_err < 0.10, (
        f"sim tip deflection {delta_sim:.6g} differs from the discrete "
        f"static-balance prediction {delta_discrete:.6g} by "
        f"{rel_err * 100:.1f} % (> 10 %); k_t = K_b/(2a) or k_s wiring "
        "is likely wrong."
    )


# ---------------------------------------------------------------------------
# Intrinsic curvature: rest state is a uniformly-curved arc
# ---------------------------------------------------------------------------


def test_intrinsic_curvature_equilibrium_is_torque_free_planar():
    """With ``intrinsic_curvature = κ_0``, the uniformly-curved
    orientation profile ``θ_i = i · 2a · κ_0`` (extensions zero) must
    be torque-free on every bead. The bending elastic energy ignores
    the gap (the elastic "bond" is always ``2a``), so the preferred
    Δθ matches the FlexibleFiber convention.
    """
    n, a, K_b, kappa_0 = 5, 1.0, 30.0, 0.05
    fiber = ExtensibleFiber(
        n_beads=n, radius=a, bending_rigidity=K_b, gap=0.1 * a,
        spring_stiffness=1000.0, mass=0.0, planar=True,
        intrinsic_curvature=kappa_0, allow_overlap=True,
    )
    beta = 2.0 * a * kappa_0
    # Orientation DOFs first (n-1), then extensions (n-1 zeros).
    curved_orient = jnp.asarray([(i + 1) * beta for i in range(n - 1)])
    curved_dofs = jnp.concatenate([curved_orient, jnp.zeros(n - 1)])
    t = jnp.array([0.0])
    for i in range(n):
        tau = fiber.spheres[i].torque(curved_dofs, fiber.design_defaults, t)
        np.testing.assert_allclose(
            np.asarray(tau), np.zeros(3), atol=1e-5,
            err_msg=f"planar: bead {i} bending torque {np.asarray(tau)} "
            "should vanish at uniformly-curved equilibrium",
        )


def test_intrinsic_curvature_equilibrium_is_torque_free_3d():
    """3D variant: intrinsic curvature is around ê_y, so the curved
    Rodrigues vector for bead i is ``(0, i·2a·κ_0, 0)``.
    """
    n, a, K_b, kappa_0 = 4, 1.0, 30.0, 0.05
    fiber = ExtensibleFiber(
        n_beads=n, radius=a, bending_rigidity=K_b, gap=0.1 * a,
        spring_stiffness=1000.0, mass=0.0, planar=False,
        intrinsic_curvature=kappa_0, allow_overlap=True,
    )
    beta = 2.0 * a * kappa_0
    # Orientation DOFs: [θ_1_y, θ_1_z, θ_2_y, θ_2_z, …]; only the *_y
    # entries pick up the linear progression.
    curved_orient = jnp.zeros(2 * (n - 1)).at[::2].set(
        jnp.arange(1, n) * beta
    )
    curved_dofs = jnp.concatenate([curved_orient, jnp.zeros(n - 1)])
    t = jnp.array([0.0])
    for i in range(n):
        tau = fiber.spheres[i].torque(curved_dofs, fiber.design_defaults, t)
        np.testing.assert_allclose(
            np.asarray(tau), np.zeros(3), atol=1e-5,
            err_msg=f"3D: bead {i} bending torque {np.asarray(tau)} "
            "should vanish at uniformly-curved equilibrium",
        )


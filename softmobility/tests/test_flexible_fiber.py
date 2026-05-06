"""Tests for the FlexibleFiber subclass of SoftBody."""

import time

import jax.numpy as jnp
import numpy as np
import pytest

from softmobility import FlexibleFiber


def test_construction_3d():
    body = FlexibleFiber(n_beads=5)
    assert body.Nspheres == 5
    assert body.Ndof == 15  # 3 per bead
    assert body.Ndesign == 4  # radius, gap, K_b, mass
    assert body.Ninput == 3  # gravity field
    assert body.design_variables == ["radius", "gap", "K_b", "mass"]
    assert body.input_variables == ["gravity0", "gravity1", "gravity2"]


def test_construction_planar():
    body = FlexibleFiber(n_beads=5, planar=True)
    assert body.Nspheres == 5
    assert body.Ndof == 5  # 1 per bead
    assert body.dof_variables == [f"theta_{i}" for i in range(5)]


def test_n_beads_validation():
    with pytest.raises(ValueError, match="n_beads"):
        FlexibleFiber(n_beads=1)


def test_gap_ratio_must_be_positive():
    with pytest.raises(ValueError, match="gap_ratio"):
        FlexibleFiber(n_beads=3, gap_ratio=0.0)


def test_straight_equilibrium_positions():
    radius = 1.0
    gap_ratio = 0.05
    body = FlexibleFiber(n_beads=4, radius=radius, gap_ratio=gap_ratio)
    bond = 2.0 * radius * (1.0 + gap_ratio)
    t = jnp.array([0.0])
    expected = jnp.stack([jnp.array([i * bond, 0.0, 0.0]) for i in range(4)])
    actual = jnp.stack([body.spheres[i].position(body.dof_defaults, body.design_defaults, t) for i in range(4)])
    assert jnp.allclose(actual, expected, atol=1e-6)
    body.validate_no_overlap()


def test_planar_bending_kinematics():
    """For planar N=3 with θ_1 = ε, the recurrence gives an analytic answer."""
    n = 3
    radius = 1.0
    gap_ratio = 0.05
    body = FlexibleFiber(n_beads=n, radius=radius, gap_ratio=gap_ratio, planar=True)
    eps = 0.1
    dofs = jnp.array([0.0, eps, 0.0])
    t = jnp.array([0.0])
    bond = radius + gap_ratio * radius
    # p_0 = (1, 0, 0); p_1 = (cos ε, 0, sin ε); p_2 = (1, 0, 0)
    # r_1 = bond * (p_0 + p_1)
    # r_2 = r_1 + bond * (p_1 + p_2)
    p0 = jnp.array([1.0, 0.0, 0.0])
    p1 = jnp.array([jnp.cos(eps), 0.0, jnp.sin(eps)])
    p2 = jnp.array([1.0, 0.0, 0.0])
    r1_expected = bond * (p0 + p1)
    r2_expected = r1_expected + bond * (p1 + p2)
    r1 = body.spheres[1].position(dofs, body.design_defaults, t)
    r2 = body.spheres[2].position(dofs, body.design_defaults, t)
    assert jnp.allclose(r1, r1_expected, atol=1e-6)
    assert jnp.allclose(r2, r2_expected, atol=1e-6)


def test_planar_C_K_is_discrete_biharmonic():
    """C_K should be the discrete biharmonic with coef K_b / L_bond on the y-torque rows."""
    n = 4
    K_b = 1.0
    bond = 2.0 * (1.0 + 0.05)
    coef = K_b / bond
    body = FlexibleFiber(n_beads=n, bending_rigidity=K_b, planar=True)
    C_K = np.asarray(body.grand_C_K())
    # Build expected biharmonic on N rows
    M = np.zeros((n, n))
    M[0, 0] = -coef
    M[0, 1] = coef
    for i in range(1, n - 1):
        M[i, i - 1] = coef
        M[i, i] = -2.0 * coef
        M[i, i + 1] = coef
    M[n - 1, n - 2] = coef
    M[n - 1, n - 1] = -coef
    # Per-bead row layout: [Fx, Fy, Fz, Tx, Ty, Tz]; only Ty (offset 4) carries torque
    for i in range(n):
        row = i * 6 + 4  # Ty row of bead i
        assert np.allclose(C_K[row], M[i], atol=1e-5)


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
    M = body.compute_mobility_tensor()
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

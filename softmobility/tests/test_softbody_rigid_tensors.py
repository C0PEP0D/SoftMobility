"""Tests for ``SoftBody.compute_rigid_tensors``."""

import jax.numpy as jnp
import numpy as np

from softmobility import FlexibleFiber, SoftBody, Sphere


def _rigid_sibling_of_planar_fiber(n_beads: int, radius: float) -> SoftBody:
    """Build a SoftBody with the same equilibrium positions as a planar
    FlexibleFiber but with no DOFs, no design, no inputs."""
    body = SoftBody(verbose=False)
    bond = 2.0 * radius
    for i in range(n_beads):
        body.add_sphere(Sphere(position=[i * bond, 0.0, 0.0], radius=radius))
    return body


def test_rigid_tensors_shapes():
    """Leading dimension of every tensor equals 6; columns match Ndof/Ninput."""
    body = FlexibleFiber(n_beads=3, planar=True)
    n = body.Nspheres
    ndof = body.Ndof
    ninput = body.Ninput

    out = body.compute_rigid_tensors()

    assert out.Mred.shape == (6, 6)
    assert out.M.shape == (6, 6 * n)
    assert out.Pi.shape == (6, 6 * n)
    assert out.C_E.shape == (6, 5)
    assert out.M_K.shape == (6, ndof)
    assert out.M_H.shape == (6, ninput)
    assert out.p_act.shape == (6,)


def test_rigid_tensors_matches_rigid_sibling_assembly():
    """At default DOFs, ``soft.compute_rigid_tensors()`` agrees with
    ``rigid.compute_tensors()`` on tensors whose column dimension is shared
    (Mred, M, Pi, C_E, p_act). M_K / M_H differ because the rigid sibling
    has no DOFs/inputs."""
    radius = 1.0
    body_soft = FlexibleFiber(n_beads=3, planar=True, radius=radius)
    body_rigid = _rigid_sibling_of_planar_fiber(n_beads=3, radius=radius)

    soft = body_soft.compute_rigid_tensors()
    rigid = body_rigid.compute_tensors()

    assert jnp.allclose(soft.Mred, rigid.Mred, atol=1e-10)
    assert jnp.allclose(soft.M, rigid.M, atol=1e-10)
    assert jnp.allclose(soft.Pi, rigid.Pi, atol=1e-10)
    assert jnp.allclose(soft.C_E, rigid.C_E, atol=1e-10)
    assert jnp.allclose(soft.p_act, rigid.p_act, atol=1e-10)


def test_rigid_tensors_all_finite():
    """No NaN/Inf in any returned tensor for a typical configuration."""
    body = FlexibleFiber(n_beads=4, planar=False)
    out = body.compute_rigid_tensors()
    for name in ("M", "Mred", "M_K", "M_H", "C_E", "Pi", "p_act"):
        assert jnp.all(jnp.isfinite(getattr(out, name))), f"{name} has NaN/Inf"


def test_compute_fast_rigid_tensors_matches_eager():
    """JIT-compiled wrapper returns identical values to the eager call."""
    body = FlexibleFiber(n_beads=3, planar=True)
    eager = body.compute_rigid_tensors()
    fast = body.compute_fast_rigid_tensors()
    assert jnp.allclose(eager.Mred, fast.Mred, atol=1e-12)
    assert jnp.allclose(eager.M, fast.M, atol=1e-12)
    assert jnp.allclose(eager.C_E, fast.C_E, atol=1e-12)


def test_rigid_tensors_mred_is_symmetric_positive_definite():
    """``Mred`` is a 6×6 mobility — must be symmetric and positive-definite."""
    body = FlexibleFiber(n_beads=4, planar=True)
    Mred = np.asarray(body.compute_rigid_tensors().Mred)
    assert np.allclose(Mred, Mred.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(Mred)
    assert np.all(eigvals > 0.0), f"Mred has non-positive eigenvalue: {eigvals}"

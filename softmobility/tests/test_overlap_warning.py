"""Tests for the post-rollout overlap warning emitted by
:meth:`SoftBody.scan_trajectory_for_overlap` and auto-called by
:meth:`FlowBodyRollout.rollout`.

Two modes are exercised:

* ``allow_overlap=False`` (default) — far-field-only mobility; the
  warning text flags results as unphysical and points at
  ``allow_overlap=True``.
* ``allow_overlap=True`` — three-regime mobility; the warning text
  identifies which regime (partial overlap or full immersion).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from softmobility import Sphere, SoftBody


@pytest.fixture(autouse=True)
def _reset_warning_state():
    SoftBody.reset_overlap_warnings()
    SoftBody.silence_overlap_warnings(False)
    yield
    SoftBody.reset_overlap_warnings()
    SoftBody.silence_overlap_warnings(False)


def _make_pair(separation: float, r_i: float = 1.0, r_j: float = 1.0,
               allow_overlap: bool = False) -> SoftBody:
    sp = SoftBody(allow_overlap=allow_overlap)
    sp.add_sphere(Sphere(position=[0, 0, 0], radius=r_i))
    sp.add_sphere(Sphere(position=[separation, 0, 0], radius=r_j))
    return sp


def _scan_static_pair(sp: SoftBody) -> list[warnings.WarningMessage]:
    """Call ``scan_trajectory_for_overlap`` on a synthetic 2-step trajectory
    of empty DOFs — exercises the static geometry."""
    dofs_traj = jnp.zeros((2, 0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.scan_trajectory_for_overlap(dofs_traj=dofs_traj)
    return caught


# ---------------------------------------------------------------------------
# allow_overlap=True  → regime-specific warning text
# ---------------------------------------------------------------------------

def test_allow_overlap_true_warns_partial_overlap_text():
    sp = _make_pair(separation=1.5, allow_overlap=True)
    caught = _scan_static_pair(sp)
    assert any("partial-overlap" in str(w.message) for w in caught), (
        [str(w.message) for w in caught]
    )


def test_allow_overlap_true_warns_full_immersion_text():
    sp = _make_pair(separation=0.1, r_i=0.5, r_j=0.1, allow_overlap=True)
    caught = _scan_static_pair(sp)
    assert any("full-immersion" in str(w.message) for w in caught), (
        [str(w.message) for w in caught]
    )


def test_allow_overlap_true_no_warning_on_touching():
    sp = _make_pair(separation=2.0, allow_overlap=True)
    caught = _scan_static_pair(sp)
    overlap = [w for w in caught if "GRPY" in str(w.message)]
    assert overlap == [], [str(w.message) for w in overlap]


# ---------------------------------------------------------------------------
# allow_overlap=False (default) → unified "invalid results" warning text
# ---------------------------------------------------------------------------

def test_default_mode_warns_with_invalid_results_message():
    sp = _make_pair(separation=1.5)  # default allow_overlap=False
    caught = _scan_static_pair(sp)
    msgs = [str(w.message) for w in caught if "allow_overlap" in str(w.message)]
    assert msgs, [str(w.message) for w in caught]
    msg = msgs[0]
    assert "unphysical" in msg or "invalid" in msg
    assert "allow_overlap=True" in msg


def test_default_mode_warning_on_full_immersion_geometry():
    sp = _make_pair(separation=0.1, r_i=0.5, r_j=0.1)
    caught = _scan_static_pair(sp)
    assert any("allow_overlap=True" in str(w.message) for w in caught), (
        [str(w.message) for w in caught]
    )


def test_default_mode_no_warning_on_touching():
    sp = _make_pair(separation=2.0)
    caught = _scan_static_pair(sp)
    overlap = [w for w in caught if "allow_overlap" in str(w.message)]
    assert overlap == []


def test_default_mode_no_warning_on_far_field():
    sp = _make_pair(separation=5.0)
    caught = _scan_static_pair(sp)
    overlap = [w for w in caught if "allow_overlap" in str(w.message)]
    assert overlap == []


# ---------------------------------------------------------------------------
# Shared behaviour: dedup, silence, finiteness, JIT short-circuit
# ---------------------------------------------------------------------------

def test_dedup_within_session_default_mode():
    sp = _make_pair(separation=1.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            sp.scan_trajectory_for_overlap(dofs_traj=jnp.zeros((2, 0)))
    invalid = [w for w in caught if "allow_overlap" in str(w.message)]
    assert len(invalid) == 1


def test_dedup_within_session_allow_overlap_mode():
    sp = _make_pair(separation=1.5, allow_overlap=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            sp.scan_trajectory_for_overlap(dofs_traj=jnp.zeros((2, 0)))
    partial = [w for w in caught if "partial-overlap" in str(w.message)]
    assert len(partial) == 1


def test_silence_suppresses_both_modes():
    SoftBody.silence_overlap_warnings(True)
    try:
        sp1 = _make_pair(separation=1.5)
        caught1 = _scan_static_pair(sp1)
        sp2 = _make_pair(separation=1.5, allow_overlap=True)
        caught2 = _scan_static_pair(sp2)
        overlap = [w for w in caught1 + caught2 if "GRPY" in str(w.message)
                   or "allow_overlap" in str(w.message)]
        assert overlap == []
    finally:
        SoftBody.silence_overlap_warnings(False)


@pytest.mark.parametrize(
    "sep, ri, rj",
    [
        (1.5, 1.0, 1.0),  # partial overlap
        (0.1, 0.5, 0.1),  # full immersion
        (2.0, 1.0, 1.0),  # touching
        (2.5, 1.0, 1.0),  # far-field
    ],
)
def test_compute_grand_mobility_returns_finite_under_overlap(sep, ri, rj):
    """``allow_overlap=True`` must return finite values across regimes."""
    SoftBody.silence_overlap_warnings(True)
    try:
        sp = _make_pair(separation=sep, r_i=ri, r_j=rj, allow_overlap=True)
        mu = np.asarray(sp.compute_grand_mobility())
        assert np.all(np.isfinite(mu))
    finally:
        SoftBody.silence_overlap_warnings(False)


def test_default_mode_far_field_consistent_with_overlap_mode_far_field():
    """In the far-field regime both modes must agree exactly (up to float
    precision) — confirms allow_overlap=False uses the same formula as the
    case_far branch of allow_overlap=True."""
    sp_def = _make_pair(separation=5.0)  # default
    sp_ovl = _make_pair(separation=5.0, allow_overlap=True)
    M_def = np.asarray(sp_def.compute_grand_mobility())
    M_ovl = np.asarray(sp_ovl.compute_grand_mobility())
    assert np.allclose(M_def, M_ovl, atol=1e-6)


def test_default_mode_and_overlap_mode_differ_in_partial_overlap():
    sp_def = _make_pair(separation=1.5)
    sp_ovl = _make_pair(separation=1.5, allow_overlap=True)
    SoftBody.silence_overlap_warnings(True)
    try:
        M_def = np.asarray(sp_def.compute_grand_mobility())
        M_ovl = np.asarray(sp_ovl.compute_grand_mobility())
    finally:
        SoftBody.silence_overlap_warnings(False)
    # Off-diagonal mu_tt block — should differ between far-only and case_medium.
    assert not np.allclose(M_def[0:3, 6:9], M_ovl[0:3, 6:9], atol=1e-6)


def test_direct_compute_grand_mobility_does_not_warn():
    """Direct (non-rollout) calls remain silent in both modes."""
    for ovl in (False, True):
        SoftBody.reset_overlap_warnings()
        sp = _make_pair(separation=1.5, allow_overlap=ovl)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jax.block_until_ready(sp.compute_grand_mobility())
        msgs = [w for w in caught
                if "GRPY mobility entered" in str(w.message)
                or "allow_overlap=True" in str(w.message)]
        assert msgs == [], (ovl, [str(w.message) for w in caught])


def test_scan_skips_silently_under_jit():
    sp = _make_pair(separation=1.5)

    @jax.jit
    def f():
        sp.scan_trajectory_for_overlap(dofs_traj=jnp.zeros((2, 0)))
        return jnp.array(0.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jax.block_until_ready(f())
    overlap = [w for w in caught
               if "GRPY" in str(w.message) or "allow_overlap" in str(w.message)]
    assert overlap == [], [str(w.message) for w in overlap]

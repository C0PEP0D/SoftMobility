"""Flexible fiber as a chain of spheres (Joint Model)."""

import contextlib
import io
import warnings

import jax
import jax.numpy as jnp

from .sphere import Sphere
from .softbody import SoftBody


class FlexibleFiber(SoftBody):
    """
    Chain of identical beads with rigid bonds and a linear bending elasticity.

    Implements the Joint Model of Delmotte et al. 2015 (Fig. 3, Eqs. 2–4) in
    the SoftMobility framework. Bead positions are derived from bead
    orientations via the recurrence ``r_{i+1} = r_i + (a + εg)(p_i +
    p_{i+1})`` where ``p_i = R(θ_i) · ê_x``, so the rigid-bond constraint is
    satisfied by construction — no Lagrange multipliers are needed. Bending
    elasticity uses the discrete biharmonic torque (linear in the orientation
    DOFs).

    Parameters
    ----------
    n_beads : int
        Number of beads in the chain.
    radius : float, default 1.0
        Sphere radius ``a``.
    gap_ratio : float, default 0.05
        Joint gap as a fraction of the radius. The bond half-length is
        ``a + εg`` with ``εg = gap_ratio · a``. A positive gap is required
        because :class:`SoftBody` rejects overlapping spheres; the default
        keeps the geometry valid up to curvature ``κ ≈ 0.31/a``.
    bending_rigidity : float, default 1.0
        Bending modulus ``K_b``. The bending torque on bead ``i`` is
        ``(K_b / L_bond)`` times the discrete second difference of the
        neighbouring orientations.
    mass : float, default 1.0
        Per-bead mass; the gravity force on each bead is ``mass · g``.
    planar : bool, default False
        If True, bending is restricted to the xz-plane and there is one
        scalar angle DOF per bead. If False, each bead has a full 3-vector
        Rodrigues orientation DOF.
    verbose : bool, default False
        If True, prints the per-DOF / per-design / per-input messages
        emitted by :class:`SphereAssembly`.

    Notes
    -----
    DOFs are named ``theta_{i}`` (planar) or ``theta_{i}_x``,
    ``theta_{i}_y``, ``theta_{i}_z`` (3D). All default to zero, giving a
    straight fiber along ``ê_x`` rooted at the origin.

    The gravity field is registered as a 3-D field input named ``gravity``;
    at runtime the solver supplies the body-frame components as
    ``gravity0``, ``gravity1``, ``gravity2``.
    """

    def __init__(
        self,
        n_beads: int,
        radius: float = 1.0,
        gap_ratio: float = 0.05,
        bending_rigidity: float = 1.0,
        mass: float = 1.0,
        planar: bool = False,
        verbose: bool = False,
    ):
        if n_beads < 2:
            raise ValueError("n_beads must be at least 2.")
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        if gap_ratio <= 0.0:
            raise ValueError("gap_ratio must be positive (overlapping spheres are not supported).")
        if bending_rigidity < 0.0:
            raise ValueError("bending_rigidity must be non-negative.")

        # Initialise empty SoftBody. The empty-geometry validation warning is
        # expected at this point and is suppressed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            super().__init__(verbose=False)

        self._n_beads = int(n_beads)
        self._planar = bool(planar)

        if verbose:
            self._build_fiber(radius, gap_ratio, bending_rigidity, mass)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                self._build_fiber(radius, gap_ratio, bending_rigidity, mass)

        # Skip the SoftBody default-geometry validation: it runs the full
        # mobility problem (slow for N≥20 due to O(N²) Python loops in
        # _compute_coupling_with_strain), and FlexibleFiber's straight default
        # configuration is guaranteed valid by construction. Users can call
        # ``validate_no_overlap()`` or ``compute_tensors()`` explicitly.

    @property
    def n_beads(self) -> int:
        """Number of beads in the chain."""
        return self._n_beads

    @property
    def planar(self) -> bool:
        """True if the fiber is restricted to xz-plane bending."""
        return self._planar

    def _build_fiber(self, radius, gap_ratio, bending_rigidity, mass):
        n = self._n_beads
        planar = self._planar

        # ---- Design parameters (insertion order is preserved) ----
        self.add_design("radius", default=float(radius))
        self.add_design("gap", default=float(gap_ratio * radius))
        self.add_design("K_b", default=float(bending_rigidity))
        self.add_design("mass", default=float(mass))
        i_radius = self.design_variables.index("radius")
        i_gap = self.design_variables.index("gap")
        i_K = self.design_variables.index("K_b")
        i_mass = self.design_variables.index("mass")

        # ---- DOFs: one Rodrigues 3-vector per bead (or one scalar in planar) ----
        if planar:
            for i in range(n):
                self.add_dof(f"theta_{i}", default=0.0)
        else:
            for i in range(n):
                self.add_dof(f"theta_{i}_x", default=0.0)
                self.add_dof(f"theta_{i}_y", default=0.0)
                self.add_dof(f"theta_{i}_z", default=0.0)

        # ---- Gravity 3-D field input ----
        self.add_input("gravity", kind="field")
        i_g0 = self.input_variables.index("gravity0")
        i_g1 = self.input_variables.index("gravity1")
        i_g2 = self.input_variables.index("gravity2")

        all_positions = _make_all_positions_callable(n, planar, i_radius, i_gap)
        force_callable = _make_force_callable(i_mass, i_g0, i_g1, i_g2)
        radius_callable = _make_radius_callable(i_radius)

        # ---- Add the spheres ----
        for i in range(n):
            sphere = Sphere(
                radius=radius_callable,
                position=_make_position_callable(i, all_positions),
                orientation=_make_orientation_callable(i, planar),
                force=force_callable,
                torque=_make_torque_callable(i, n, planar, i_radius, i_gap, i_K),
            )
            self.add_sphere(sphere)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rodrigues_to_tangent(rod):
    """Apply rotation R(rod) to ê_x = (1, 0, 0) using Rodrigues' formula.

    ``rod`` is the Rodrigues vector ``θ · k̂`` (axis-angle). Numerically safe
    near ``θ = 0`` via Taylor-series fallbacks for ``sin θ / θ`` and
    ``(1 - cos θ) / θ²``.
    """
    theta_sq = jnp.dot(rod, rod)
    theta = jnp.sqrt(theta_sq + 1e-30)
    cos_t = jnp.cos(theta)

    sinc = jnp.where(theta_sq < 1e-8, 1.0 - theta_sq / 6.0, jnp.sin(theta) / theta)
    one_minus_cos_over_t2 = jnp.where(theta_sq < 1e-8, 0.5 - theta_sq / 24.0, (1.0 - cos_t) / (theta_sq + 1e-30))

    e_x = jnp.array([1.0, 0.0, 0.0])
    cross1 = jnp.cross(rod, e_x)
    cross2 = jnp.cross(rod, cross1)
    return e_x + sinc * cross1 + one_minus_cos_over_t2 * cross2


def _make_all_positions_callable(n_beads, planar, i_radius, i_gap):
    """Return a function ``(dofs, design) -> (n_beads, 3)`` of bead positions."""

    def all_positions(dofs, design):
        a_plus_g = design[i_radius] + design[i_gap]
        if planar:
            thetas = dofs[:n_beads]
            ps = jnp.stack([jnp.cos(thetas), jnp.zeros(n_beads), jnp.sin(thetas)], axis=1)
        else:
            rod = dofs[: 3 * n_beads].reshape(n_beads, 3)
            ps = jax.vmap(_rodrigues_to_tangent)(rod)
        deltas = a_plus_g * (ps[:-1] + ps[1:])  # (n_beads-1, 3)
        return jnp.concatenate([jnp.zeros((1, 3)), jnp.cumsum(deltas, axis=0)], axis=0)

    return all_positions


def _make_position_callable(i, all_positions):
    """Sphere position callable: ``(dofs, design, time) -> (3,)``."""
    return lambda dofs, design, time: all_positions(dofs, design)[i]


def _make_orientation_callable(i, planar):
    """Sphere orientation callable returning a Rodrigues 3-vector."""
    if planar:
        return lambda dofs, design, time: jnp.array([0.0, dofs[i], 0.0])
    return lambda dofs, design, time: dofs[3 * i : 3 * (i + 1)]


def _make_torque_callable(i, n_beads, planar, i_radius, i_gap, i_K):
    """Discrete biharmonic bending torque on bead ``i``.

    Linear in DOFs ⇒ exact ``C_K`` from a single ``jax.jacfwd``.
    """

    if planar:

        def torque(dofs, design, inputs):
            coef = design[i_K] / (2.0 * (design[i_radius] + design[i_gap]))
            if i == 0:
                ty = coef * (dofs[1] - dofs[0])
            elif i == n_beads - 1:
                ty = coef * (dofs[n_beads - 2] - dofs[n_beads - 1])
            else:
                ty = coef * (dofs[i - 1] - 2.0 * dofs[i] + dofs[i + 1])
            return jnp.array([0.0, ty, 0.0])

        return torque

    def torque(dofs, design, inputs):
        coef = design[i_K] / (2.0 * (design[i_radius] + design[i_gap]))
        rod_i = dofs[3 * i : 3 * (i + 1)]
        if i == 0:
            rod_p = dofs[3 : 6]
            return coef * (rod_p - rod_i)
        if i == n_beads - 1:
            rod_m = dofs[3 * (i - 1) : 3 * i]
            return coef * (rod_m - rod_i)
        rod_m = dofs[3 * (i - 1) : 3 * i]
        rod_p = dofs[3 * (i + 1) : 3 * (i + 2)]
        return coef * (rod_m - 2.0 * rod_i + rod_p)

    return torque


def _make_radius_callable(i_radius):
    """Closure that returns the shared sphere radius."""

    def radius(dofs, design):
        return design[i_radius]

    return radius


def _make_force_callable(i_mass, i_g0, i_g1, i_g2):
    """Gravity force ``[mass · g0, mass · g1, mass · g2]`` (linear in inputs)."""

    def force(dofs, design, inputs):
        m = design[i_mass]
        return jnp.array([m * inputs[i_g0], m * inputs[i_g1], m * inputs[i_g2]])

    return force

"""Flexible fiber as a chain of touching spheres (Gears Model)."""

import contextlib
import io
import warnings

import jax
import jax.numpy as jnp

from .softbody import SoftBody
from .sphere import Sphere


class FlexibleFiber(SoftBody):
    """
    Chain of identical beads representing a flexible fiber with bending elasticity.

    Implements the Gears Model of Delmotte, Climent & Plouraboué 2015
    (J. Comput. Phys. 286, 14–37, §2.3) with a **linearized** version of the
    paper's bending torque (eq. 32+34 expanded to first order in joint
    angles, §2.4). Bond length is exactly ``2a`` for any configuration
    (sphere surfaces touching); bead positions are derived from bead
    orientations via the recurrence
    ``r_{i+1} = r_i + 2a · (p_i + p_{i+1}) / |p_i + p_{i+1}|``
    with ``p_i = R(rod_i) · ê_x``, so rigid bonds are satisfied by
    construction — no Lagrange multipliers.

    The linearization keeps the bending torque linear in the orientation
    DOFs, so the framework's exact ``C_K = jacfwd(T)`` extraction continues
    to work and the per-step cost is unchanged. The approximation is
    accurate for joint angles ≲ π/4. Large-deformation regimes (e.g. the
    paper's Fig 8b buckling at ``κ_eq ≠ 0`` or Fig 10b ``B = 10000``
    horseshoe) require the full geometric ``κ`` formula and are out of
    scope.

    Parameters
    ----------
    n_beads : int
        Number of beads in the chain.
    radius : float, default 1.0
        Sphere radius ``a``. Bond length is exactly ``2a``.
    bending_rigidity : float, default 1.0
        Bending modulus ``K_b``. The linearized joint moment is
        ``m_j = (K_b / 4a) · (rod_{j+1} − rod_{j-1})`` (with the x-component
        zeroed; only y and z components carry torque) and the bead torque
        is ``γ^b_i = m_{i+1} − m_{i-1}`` with ``m_0 = m_{N-1} = 0``.
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
    Sphere 0 sits at the body origin **with its tangent identified with
    the body frame's** ``ê_x``: it has no per-bead Rodrigues DOF. The
    chain is also treated as torsionally infinitely stiff, so the
    "twist" component of each bead's Rodrigues vector (its projection
    on ``ê_x``, which leaves the bead tangent invariant under
    Rodrigues rotation) is structurally zero, not a DOF. Only the
    ``N − 1`` distal beads carry their two bending components, giving
    ``Ndof = (N − 1)`` (planar) or ``2 (N − 1)`` (3D) configurational
    degrees of freedom — the correct count for a slender, inextensible,
    torsion-free bead chain anchored at one end. DOFs are named
    ``theta_{i}`` (planar) or ``theta_{i}_y``, ``theta_{i}_z`` (3D) for
    ``i = 1 … N − 1``. All default to zero, giving a straight fiber
    along ``ê_x`` rooted at the origin.

    The gravity field is registered as a 3-D field input named ``gravity``;
    at runtime the solver supplies the body-frame components as
    ``gravity0``, ``gravity1``, ``gravity2``.

    Sign convention (planar): ``θ_i`` is the Rodrigues angle around ``+ê_y``,
    so ``p_i = R_y(θ_i) · ê_x = (cos θ_i, 0, −sin θ_i)``. The chain bends
    in the xz-plane.
    """

    def __init__(
        self,
        n_beads: int,
        radius: float = 1.0,
        bending_rigidity: float = 1.0,
        mass: float = 1.0,
        planar: bool = False,
        verbose: bool = False,
    ):
        if n_beads < 2:
            raise ValueError("n_beads must be at least 2.")
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
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
            self._build_fiber(radius, bending_rigidity, mass)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                self._build_fiber(radius, bending_rigidity, mass)

    @property
    def n_beads(self) -> int:
        """Number of beads in the chain."""
        return self._n_beads

    @property
    def planar(self) -> bool:
        """True if the fiber is restricted to xz-plane bending."""
        return self._planar

    def _build_fiber(self, radius, bending_rigidity, mass):
        n = self._n_beads
        planar = self._planar

        # ---- Design parameters (insertion order is preserved) ----
        self.add_design("radius", default=float(radius))
        self.add_design("K_b", default=float(bending_rigidity))
        self.add_design("mass", default=float(mass))
        i_radius = self.design_variables.index("radius")
        i_K = self.design_variables.index("K_b")
        i_mass = self.design_variables.index("mass")

        # ---- DOFs: bending components per *distal* bead ----
        # Sphere 0 has no per-bead orientation DOF: its tangent is
        # structurally aligned with the body frame's ê_x. This removes
        # the redundancy between the body orientation and rod_0 that
        # would otherwise leave a free uniform-rotation mode in the
        # clamped-anchor problem.
        #
        # The chain is treated as torsionally infinitely stiff: for
        # each distal bead we drop the x-component of its Rodrigues
        # vector (rotations around ê_x leave the bead tangent
        # invariant — that mode is the un-physical "twist"). The two
        # remaining components (y, z) span the two bending directions
        # perpendicular to the local bond.
        if planar:
            for i in range(1, n):
                self.add_dof(f"theta_{i}", default=0.0)
        else:
            for i in range(1, n):
                self.add_dof(f"theta_{i}_y", default=0.0)
                self.add_dof(f"theta_{i}_z", default=0.0)

        # ---- Gravity 3-D field input ----
        self.add_input("gravity", kind="field")
        i_g0 = self.input_variables.index("gravity0")
        i_g1 = self.input_variables.index("gravity1")
        i_g2 = self.input_variables.index("gravity2")

        all_positions = _make_all_positions_callable(n, planar, i_radius)
        force_callable = _make_force_callable(i_mass, i_g0, i_g1, i_g2)
        radius_callable = _make_radius_callable(i_radius)

        # ---- Add the spheres ----
        for i in range(n):
            sphere = Sphere(
                radius=radius_callable,
                position=_make_position_callable(i, all_positions),
                orientation=_make_orientation_callable(i, planar),
                force=force_callable,
                torque=_make_torque_callable(i, n, planar, i_radius, i_K),
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


def _make_all_positions_callable(n_beads, planar, i_radius):
    """Return a function ``(dofs, design) -> (n_beads, 3)`` of bead positions.

    Bond length is exactly ``2a`` for any configuration; bead k+1 is at

        r_{k+1} = r_k + 2a · (p_k + p_{k+1}) / |p_k + p_{k+1}|

    where ``p_j = R(rod_j) · ê_x``. Sphere 0's tangent is structurally
    fixed to ``p_0 = ê_x``; only spheres ``1 … N−1`` have orientation
    DOFs. The bond direction is the average of the two bead tangents,
    normalized; the normalization enforces the rigid-bond ("touching
    gears") constraint exactly regardless of joint angle. Singular only
    at δ = π (consecutive tangents antiparallel — fiber folded onto
    itself, unphysical); guarded by ``eps``.
    """
    eps = 1e-12
    e_x = jnp.array([[1.0, 0.0, 0.0]])  # sphere 0's tangent, fixed

    def all_positions(dofs, design):
        a = design[i_radius]
        if planar:
            thetas = dofs[: n_beads - 1]
            ps_distal = jnp.stack(
                [jnp.cos(thetas), jnp.zeros(n_beads - 1), jnp.sin(thetas)],
                axis=1,
            )
        else:
            # Two bending components per distal bead. Reconstruct the
            # full Rodrigues vector with x-component (twist) = 0.
            yz = dofs[: 2 * (n_beads - 1)].reshape(n_beads - 1, 2)
            rod = jnp.concatenate(
                [jnp.zeros((n_beads - 1, 1)), yz], axis=1
            )
            ps_distal = jax.vmap(_rodrigues_to_tangent)(rod)
        ps = jnp.concatenate([e_x, ps_distal], axis=0)  # (n_beads, 3)
        bond_dirs = ps[:-1] + ps[1:]  # (n_beads-1, 3)
        norms = jnp.linalg.norm(bond_dirs, axis=1, keepdims=True)
        deltas = 2.0 * a * bond_dirs / (norms + eps)
        return jnp.concatenate([jnp.zeros((1, 3)), jnp.cumsum(deltas, axis=0)], axis=0)

    return all_positions


def _make_position_callable(i, all_positions):
    """Sphere position callable: ``(dofs, design, time) -> (3,)``."""
    return lambda dofs, design, time: all_positions(dofs, design)[i]


def _make_orientation_callable(i, planar):
    """Sphere orientation callable returning a Rodrigues 3-vector.

    Sphere 0's tangent is structurally aligned with the body frame, so
    its orientation is the zero Rodrigues vector regardless of ``dofs``.
    Distal spheres (``i = 1 … N − 1``) reconstruct their Rodrigues vector
    from the bending DOFs:

    - planar: one DOF ``θ_i`` per bead, vector ``(0, θ_i, 0)``;
    - 3-D: two bending DOFs ``(θ_i^y, θ_i^z)`` per bead, vector
      ``(0, θ_i^y, θ_i^z)``. The twist component (along ``ê_x``) is
      structurally zero — see the class docstring.
    """
    if i == 0:
        return lambda dofs, design, time: jnp.zeros(3)
    if planar:
        return lambda dofs, design, time: jnp.array([0.0, dofs[i - 1], 0.0])
    return lambda dofs, design, time: jnp.concatenate(
        [jnp.zeros(1), dofs[2 * (i - 1) : 2 * i]]
    )


def _make_torque_callable(i, n_beads, planar, i_radius, i_K):
    """Linearized bending torque on bead ``i`` (Delmotte 2015, eq. 32+34).

    In the implicit-DOF parameterization where each bead carries an
    orientation DOF and bond directions are derived as ``e_{j,j+1} = (p_j +
    p_{j+1})/|·|``, the linearized bending energy is

        E_bend ≈ (K_b / (2a)) · Σ (θ_{i+1} − θ_i)²

    (i.e., a torsional-spring chain on the bead orientations — the
    small-angle limit of the joint-angle bending energy from eq. 32+34).
    The generalized force is the discrete Laplacian on the orientation
    DOFs:

        γ^b_i = (K_b / (2a)) · (θ_{i-1} − 2 θ_i + θ_{i+1})  (interior)
              = (K_b / (2a)) · (θ_{i+1} − θ_i)               (i = 0)
              = (K_b / (2a)) · (θ_{i-1} − θ_i)               (i = N-1)

    The end-bead first-difference matches Fig. 5 of the paper
    (``γ^b_1 = m(s_2)``, ``γ^b_N = −m(s_{N-1})``) in this parameterization
    where ``m(s_j) ∝ θ_j − θ_{j-1}``. Linear in DOFs ⇒ exact ``C_K`` from a
    single ``jax.jacfwd``.
    """

    if planar:

        def torque(dofs, design, inputs):
            coef = design[i_K] / (2.0 * design[i_radius])
            # Prepend theta_0 = 0 (no DOF for sphere 0) so the
            # discrete-Laplacian formulas below stay symmetric in the
            # bead index.
            theta = jnp.concatenate([jnp.zeros(1), dofs[: n_beads - 1]])
            if i == 0:
                ty = coef * (theta[1] - theta[0])
            elif i == n_beads - 1:
                ty = coef * (theta[n_beads - 2] - theta[n_beads - 1])
            else:
                ty = coef * (theta[i - 1] - 2.0 * theta[i] + theta[i + 1])
            return jnp.array([0.0, ty, 0.0])

        return torque

    def torque(dofs, design, inputs):
        coef = design[i_K] / (2.0 * design[i_radius])
        # Sphere 0 has no orientation DOF (structurally rod_0 = 0); the
        # distal beads have only their two bending components, with
        # rod_x ≡ 0 (no twist). Reconstruct the full per-bead Rodrigues
        # 3-vectors so the discrete-Laplacian formulas below stay
        # symmetric in the bead index. The torque-around-x component
        # comes out identically zero for any state, consistent with
        # treating the chain as torsionally infinitely stiff.
        yz_distal = dofs[: 2 * (n_beads - 1)].reshape(n_beads - 1, 2)
        rod_distal = jnp.concatenate(
            [jnp.zeros((n_beads - 1, 1)), yz_distal], axis=1
        )
        rod = jnp.concatenate([jnp.zeros((1, 3)), rod_distal], axis=0)
        if i == 0:
            return coef * (rod[1] - rod[0])
        if i == n_beads - 1:
            return coef * (rod[n_beads - 2] - rod[n_beads - 1])
        return coef * (rod[i - 1] - 2.0 * rod[i] + rod[i + 1])

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

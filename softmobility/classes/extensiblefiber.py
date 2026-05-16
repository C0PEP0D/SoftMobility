"""Extensible flexible fiber: Gears bonds with per-bond axial springs."""

import contextlib
import io
import warnings

import jax
import jax.numpy as jnp

from .flexiblefiber import _rodrigues_to_tangent
from .softbody import SoftBody
from .sphere import Sphere


class ExtensibleFiber(SoftBody):
    """Chain of beads with a small equilibrium gap and stiff axial springs.

    Sibling of :class:`FlexibleFiber`. Adds per-bond extension DOFs
    ``δ_i`` on top of the orientation DOFs ``θ_i``. The bond length is

    .. math::

        L_i = 2a + g_0 + \\delta_i ,

    with a fixed equilibrium gap ``g_0`` (default ``0.1 a``) and a linear
    spring potential

    .. math::

        E_s = \\tfrac{1}{2}\\, k_s\\, \\sum_i \\delta_i^2 .

    The spring is wired in by adding the force ``±k_s · δ_i · n̂_i`` on
    the two beads bordering bond ``i``, where ``n̂_i = (p_i + p_{i+1})
    / |p_i + p_{i+1}|`` is the bond unit vector. Virtual work then
    projects this onto the generalised force ``−k_s · δ_i`` conjugate
    to the extension DOF (verified analytically: the orientation-DOF
    projection vanishes because ``n̂ · ∂n̂/∂θ = 0``).

    Compared to :class:`FlexibleFiber`:

    * Position recurrence ``r_{i+1} = r_i + L_i · n̂_i`` instead of
      ``r_i + 2a · n̂_i`` (bond length is now state-dependent).
    * Two extra design parameters: ``gap`` and ``k_s``.
    * ``N − 1`` extra extension DOFs ``delta_0 … delta_{N-2}``.
    * Force callable varies per bead (closes over the bead index) to
      inject the bond-spring contribution; gravity is unchanged.
    * Torque callable is **identical** — the harmonic bending discrete
      Laplacian on orientation DOFs.

    The framework's ``C_K = jacfwd(six_component_force)`` extraction
    still works; ``C_K`` is no longer constant in the orientation DOFs
    (because ``n̂_i`` is a nonlinear function of ``θ_i``), but the
    rollout machinery already recomputes it at each step.

    Parameters
    ----------
    n_beads : int
        Number of beads in the chain. Must be ≥ 2.
    radius : float, default 1.0
        Bead radius ``a``.
    bending_rigidity : float, default 1.0
        Bending modulus ``K_b``. The bending torque is the same harmonic
        discrete Laplacian as :class:`FlexibleFiber`:
        ``γ^b_i = (K_b / 2a) · (θ_{i-1} − 2 θ_i + θ_{i+1})``.
    gap : float or None, default ``None``
        Equilibrium gap ``g_0`` between sphere surfaces. Bond
        equilibrium length is ``2a + g_0``. When ``None``, defaults to
        ``0.1 · radius``.
    spring_stiffness : float or None, default ``None``
        Axial spring constant ``k_s`` (units force/length). When
        ``None``, defaults to ``10 · K_b / radius³``. The default is a
        moderate stiffness; for "nearly inextensible" dynamics at a
        given ``dt``, the stability bound is roughly ``k_s < 1 /
        (μ_b · dt)`` with ``μ_b = 1/(6π a)`` the single-bead Stokes
        mobility.
    mass : float, default 1.0
        Per-bead mass; gravity force per bead is ``mass · g``.
    planar : bool, default False
        If True, bending is restricted to the xz-plane; one scalar
        ``theta_i`` DOF per distal bead. If False, two DOFs
        ``(theta_i_y, theta_i_z)`` per distal bead.
    verbose : bool, default False
        Pass-through to the underlying :class:`SphereAssembly`.
    allow_overlap : bool, default True
        Toggles the GRPY overlap regimes. Defaults to True because
        dynamic compression of bonds (``δ_i < 0``) can bring beads
        near or into contact even when the equilibrium has a positive
        gap.

    Notes
    -----
    **DOF layout (planar):**

    .. code-block:: text

        [theta_1, …, theta_{N-1},  delta_0, …, delta_{N-2}]

    **3-D layout:**

    .. code-block:: text

        [theta_1_y, theta_1_z, …, theta_{N-1}_z,  delta_0, …, delta_{N-2}]

    Sphere 0 is anchored at the body origin with its tangent identified
    with the body ``ê_x``; only spheres ``1 … N − 1`` carry orientation
    DOFs. The twist around each bead's tangent is structurally frozen
    (consistent with :class:`FlexibleFiber`). All DOFs default to zero,
    giving a straight fiber along ``ê_x`` with each bond at its
    equilibrium length ``2a + g_0``.

    **Rigid limit.** As ``spring_stiffness → ∞`` the extension DOFs
    are pinned at zero and the dynamics reduce to a
    :class:`FlexibleFiber` whose bonds have length ``2a + g_0`` instead
    of ``2a``. With ``gap = 0`` and ``spring_stiffness → ∞`` the
    reduction is to the touching-sphere chain.

    **The k_s — dt tradeoff.** Stiff springs are necessary to approximate
    inextensibility but incompatible with the explicit time-step.
    Empirically the stability condition for the RK4 rollout is
    ``dt ≲ 1 / (μ_b · k_s)`` where ``μ_b ≈ 1/(6π a)``. With the typical
    elastic ``dt = 0.05 · (2a)⁴ / K_b`` and ``a = 1``, this gives
    ``k_s_max ≈ 6π / (0.05 · 16 · K_b⁻¹) ≈ 24 / K_b⁻¹``. Going stiffer
    requires shrinking ``dt`` proportionally.

    Examples
    --------
    Build a planar 5-bead fiber with a 10 % equilibrium gap and a stiff
    axial spring, run a brief cantilever test under transverse gravity::

        >>> import softmobility as sm
        >>> import jax.numpy as jnp
        >>> body = sm.ExtensibleFiber(
        ...     n_beads=5, radius=1.0, bending_rigidity=30.0,
        ...     gap=0.1, spring_stiffness=300.0, mass=1.0, planar=True,
        ... )
        >>> body.Ndof  # (N-1) orient + (N-1) ext = 8
        8
        >>> rollout = sm.FlowBodyRollout(
        ...     soft_body=body, flow=sm.no_flow(),
        ...     input_map={"gravity": sm.gravity_field(g=1e-3)},
        ... )
        >>> _, _, dofs, _ = rollout.rollout_clamped_anchor(
        ...     dt=0.027, n_steps=1000,
        ...     anchor_position_fn=lambda t: jnp.zeros(3),
        ...     anchor_velocity_fn=lambda t: jnp.zeros(6),
        ... )
    """

    def __init__(
        self,
        n_beads: int,
        radius: float = 1.0,
        bending_rigidity: float = 1.0,
        gap: float | None = None,
        spring_stiffness: float | None = None,
        mass: float = 1.0,
        planar: bool = False,
        verbose: bool = False,
        allow_overlap: bool = True,
        intrinsic_curvature: float = 0.0,
    ):
        if n_beads < 2:
            raise ValueError("n_beads must be at least 2.")
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        if bending_rigidity < 0.0:
            raise ValueError("bending_rigidity must be non-negative.")

        if gap is None:
            gap = 0.1 * radius
        if gap < 0.0:
            raise ValueError("gap must be non-negative.")

        if spring_stiffness is None:
            spring_stiffness = 10.0 * bending_rigidity / radius**3
        if spring_stiffness <= 0.0:
            raise ValueError("spring_stiffness must be positive.")

        # Initialise empty SoftBody. The empty-geometry validation warning
        # is expected at this point and is suppressed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            super().__init__(verbose=False, allow_overlap=allow_overlap)

        self._n_beads = int(n_beads)
        self._planar = bool(planar)

        if verbose:
            self._build_fiber(
                radius, bending_rigidity, gap, spring_stiffness, mass,
                intrinsic_curvature,
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                self._build_fiber(
                    radius, bending_rigidity, gap, spring_stiffness, mass,
                    intrinsic_curvature,
                )

    @property
    def n_beads(self) -> int:
        """Number of beads in the chain."""
        return self._n_beads

    @property
    def planar(self) -> bool:
        """True if the fiber is restricted to xz-plane bending."""
        return self._planar

    def _build_fiber(
        self, radius, bending_rigidity, gap, spring_stiffness, mass,
        intrinsic_curvature,
    ):
        n = self._n_beads
        planar = self._planar

        # ---- Design parameters ----
        self.add_design("radius", default=float(radius))
        self.add_design("K_b", default=float(bending_rigidity))
        self.add_design("gap", default=float(gap))
        self.add_design("k_s", default=float(spring_stiffness))
        self.add_design("mass", default=float(mass))
        self.add_design("kappa_0", default=float(intrinsic_curvature))
        i_radius = self.design_variables.index("radius")
        i_K = self.design_variables.index("K_b")
        i_gap = self.design_variables.index("gap")
        i_ks = self.design_variables.index("k_s")
        i_mass = self.design_variables.index("mass")
        i_kappa0 = self.design_variables.index("kappa_0")

        # ---- Orientation DOFs (per distal bead) ----
        if planar:
            for i in range(1, n):
                self.add_dof(f"theta_{i}", default=0.0)
            n_orient_dofs = n - 1
        else:
            for i in range(1, n):
                self.add_dof(f"theta_{i}_y", default=0.0)
                self.add_dof(f"theta_{i}_z", default=0.0)
            n_orient_dofs = 2 * (n - 1)

        # ---- Per-bond extension DOFs ----
        # delta_i is the extension of bond i (between beads i and i+1).
        # Default zero ⇒ bond at equilibrium length 2a + g_0.
        for i in range(n - 1):
            self.add_dof(f"delta_{i}", default=0.0)

        # ---- Gravity 3-D field input ----
        self.add_input("gravity", kind="field")
        i_g0 = self.input_variables.index("gravity0")
        i_g1 = self.input_variables.index("gravity1")
        i_g2 = self.input_variables.index("gravity2")

        bond_quantities = _make_bond_quantities_callable(
            n, planar, i_radius, i_gap, n_orient_dofs
        )
        radius_callable = _make_radius_callable(i_radius)

        for i in range(n):
            sphere = Sphere(
                radius=radius_callable,
                position=_make_position_callable(i, bond_quantities),
                orientation=_make_orientation_callable(i, planar),
                force=_make_force_callable(
                    i, n, bond_quantities, n_orient_dofs,
                    i_mass, i_ks, i_g0, i_g1, i_g2,
                ),
                torque=_make_torque_callable(i, n, planar, i_radius, i_K, i_kappa0),
            )
            self.add_sphere(sphere)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bond_quantities_callable(n_beads, planar, i_radius, i_gap, n_orient_dofs):
    """Return a callable ``(dofs, design) -> (positions, unit_dirs)``.

    * ``positions`` has shape ``(n_beads, 3)`` — bead centres in the body frame.
    * ``unit_dirs`` has shape ``(n_beads - 1, 3)`` — unit vectors ``n̂_i``
      along each bond, computed as ``(p_i + p_{i+1}) / |p_i + p_{i+1}|``.

    Bond ``i`` (between beads ``i`` and ``i+1``) has length
    ``2a + g_0 + δ_i``; positions are accumulated as
    ``r_{i+1} = r_i + L_i · n̂_i``.
    """
    eps = 1e-12
    e_x = jnp.array([[1.0, 0.0, 0.0]])  # sphere 0's tangent, fixed

    def bond_quantities(dofs, design):
        a = design[i_radius]
        g_0 = design[i_gap]
        if planar:
            thetas = dofs[: n_beads - 1]
            ps_distal = jnp.stack(
                [jnp.cos(thetas), jnp.zeros(n_beads - 1), jnp.sin(thetas)],
                axis=1,
            )
        else:
            yz = dofs[: 2 * (n_beads - 1)].reshape(n_beads - 1, 2)
            rod = jnp.concatenate(
                [jnp.zeros((n_beads - 1, 1)), yz], axis=1
            )
            ps_distal = jax.vmap(_rodrigues_to_tangent)(rod)
        ps = jnp.concatenate([e_x, ps_distal], axis=0)  # (n_beads, 3)
        bond_dirs = ps[:-1] + ps[1:]  # (n_beads-1, 3)
        norms = jnp.linalg.norm(bond_dirs, axis=1, keepdims=True)
        unit_dirs = bond_dirs / (norms + eps)  # (n_beads-1, 3)
        deltas = dofs[n_orient_dofs : n_orient_dofs + (n_beads - 1)]
        lengths = (2.0 * a + g_0) + deltas  # (n_beads-1,)
        bond_vectors = lengths[:, None] * unit_dirs
        positions = jnp.concatenate(
            [jnp.zeros((1, 3)), jnp.cumsum(bond_vectors, axis=0)], axis=0
        )
        return positions, unit_dirs

    return bond_quantities


def _make_position_callable(i, bond_quantities):
    """Sphere ``i`` position callable: ``(dofs, design, time) -> (3,)``."""
    return lambda dofs, design, time: bond_quantities(dofs, design)[0][i]


def _make_orientation_callable(i, planar):
    """Sphere ``i`` orientation callable. Matches FlexibleFiber convention.

    Sphere 0's tangent is structurally aligned with the body frame
    (zero Rodrigues vector). Distal spheres ``i = 1 … N − 1`` carry the
    bending DOFs.
    """
    if i == 0:
        return lambda dofs, design, time: jnp.zeros(3)
    if planar:
        return lambda dofs, design, time: jnp.array([0.0, dofs[i - 1], 0.0])
    return lambda dofs, design, time: jnp.concatenate(
        [jnp.zeros(1), dofs[2 * (i - 1) : 2 * i]]
    )


def _make_force_callable(
    i, n_beads, bond_quantities, n_orient_dofs,
    i_mass, i_ks, i_g0, i_g1, i_g2,
):
    """Force on bead ``i``: gravity + axial-spring contribution.

    The spring along bond ``j`` (between beads ``j`` and ``j+1``) exerts
    ``+k_s · δ_j · n̂_j`` on bead ``j`` and ``−k_s · δ_j · n̂_j`` on bead
    ``j+1``. Summing the contributions of the (at most two) bonds adjacent
    to bead ``i`` gives the spring term below.

    Linearity: linear in inputs (gravity); nonlinear in orientation DOFs
    (through ``n̂_j``) and linear in extension DOFs (through ``δ_j``).
    The framework's ``jacfwd``-based ``C_K`` extraction handles this.
    """

    def force(dofs, design, inputs):
        m = design[i_mass]
        k_s = design[i_ks]
        gravity = jnp.array(
            [m * inputs[i_g0], m * inputs[i_g1], m * inputs[i_g2]]
        )
        _, unit_dirs = bond_quantities(dofs, design)
        deltas = dofs[n_orient_dofs : n_orient_dofs + (n_beads - 1)]

        spring = jnp.zeros(3)
        # Bond i (acts on bead i if i < N-1)
        if i < n_beads - 1:
            spring = spring + k_s * deltas[i] * unit_dirs[i]
        # Bond i-1 (acts on bead i if i > 0)
        if i > 0:
            spring = spring - k_s * deltas[i - 1] * unit_dirs[i - 1]

        return gravity + spring

    return force


def _make_torque_callable(i, n_beads, planar, i_radius, i_K, i_kappa0):
    """Linearised bending torque — identical to FlexibleFiber.

    The bending torque on bead ``i`` is the discrete Laplacian on the
    orientation DOFs, with first-difference end terms. Extension DOFs
    do not appear because the bending energy is a function of the
    orientation DOFs only. Intrinsic curvature ``κ_0`` adds a constant
    bias of ``±K_b·κ_0`` to the two boundary torques (see the matching
    docstring in :func:`softmobility.classes.flexiblefiber._make_torque_callable`).
    """

    if planar:

        def torque(dofs, design, inputs):
            coef = design[i_K] / (2.0 * design[i_radius])
            bias = design[i_K] * design[i_kappa0]
            theta = jnp.concatenate([jnp.zeros(1), dofs[: n_beads - 1]])
            if i == 0:
                ty = coef * (theta[1] - theta[0]) - bias
            elif i == n_beads - 1:
                ty = coef * (theta[n_beads - 2] - theta[n_beads - 1]) + bias
            else:
                ty = coef * (theta[i - 1] - 2.0 * theta[i] + theta[i + 1])
            return jnp.array([0.0, ty, 0.0])

        return torque

    def torque(dofs, design, inputs):
        coef = design[i_K] / (2.0 * design[i_radius])
        bias = jnp.array([0.0, design[i_K] * design[i_kappa0], 0.0])
        yz_distal = dofs[: 2 * (n_beads - 1)].reshape(n_beads - 1, 2)
        rod_distal = jnp.concatenate(
            [jnp.zeros((n_beads - 1, 1)), yz_distal], axis=1
        )
        rod = jnp.concatenate([jnp.zeros((1, 3)), rod_distal], axis=0)
        if i == 0:
            return coef * (rod[1] - rod[0]) - bias
        if i == n_beads - 1:
            return coef * (rod[n_beads - 2] - rod[n_beads - 1]) + bias
        return coef * (rod[i - 1] - 2.0 * rod[i] + rod[i + 1])

    return torque


def _make_radius_callable(i_radius):
    """Closure that returns the shared sphere radius."""

    def radius(dofs, design):
        return design[i_radius]

    return radius

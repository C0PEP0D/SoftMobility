"""SoftBody class."""

import warnings
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .sphere import Sphere
from .sphereassembly import SphereAssembly

SoftMobilityTensors = namedtuple("SoftMobilityTensors", ["M", "Mred", "M_K", "M_H", "C_E", "Pi", "p_act"])


class SoftBody(SphereAssembly):
    """
    SoftBody (simulation of deformable bodies in fluid flow).

    Extends ``SphereAssembly`` to model soft bodies with mobility tensors and
    hydrodynamic interactions.  Supports computation of mobility matrices,
    coupling tensors, and forces for soft body dynamics.  Compatible with JAX
    for automatic differentiation and just-in-time compilation.

    Parameters
    ----------
    parameters_source : str, optional
        Path to a YAML file containing assembly parameters (spheres, dofs,
        design variables, etc.).  If provided, the assembly is initialized
        from this file.
    verbose : bool, default=True
        If True, prints debug and progress information during initialization
        and computation.

    Attributes
    ----------
    SoftMobilityTensors : namedtuple
        A named tuple containing the mobility tensors:

        - M : jnp.ndarray
            Grand mobility matrix (to be multiplied by grand forces
            [F1, T1, F2, T2...]).
        - M_K : jnp.ndarray
            Elastic mobility matrix (to be multiplied by degrees of freedom).
        - M_H : jnp.ndarray
            Input mobility matrix (coupling with 3D and scalar fields).
        - C_E : jnp.ndarray
            Elastic coupling matrix.
        - Pi : jnp.ndarray
            Projection matrix :math:`\\boldsymbol{\\Pi}`.

    Examples
    --------
    Initialize a soft body from a YAML file:

    >>> soft_body = SoftBody("path/to/parameters.yaml")

    Compute mobility tensors:

    >>> tensors = soft_body.compute_tensors()

    Notes
    -----
    - The grand mobility tensor is computed using the Rotne–Prager–
      Yamakawa (GRPY) approximation across three pairwise regimes:
      far-field (``R > a_i + a_j``), partial overlap
      (``|a_i − a_j| < R ≤ a_i + a_j``), and full immersion
      (``R ≤ |a_i − a_j|``). Strict overlap is permitted but emits a
      one-shot :class:`UserWarning` per regime; call
      :meth:`reset_overlap_warnings` to re-enable, or use
      :meth:`validate_no_overlap` for an explicit Python-side check
      before simulation.
    - Inherits all functionality from ``SphereAssembly``, including sphere
      management and degree of freedom handling.
    - Compatible with JAX transformations
      (``jax.jit``, ``jax.grad``, ``jax.vmap``).
    """

    def __init__(self, *args, allow_overlap: bool = False, **kwargs):
        """Build the SoftBody.

        Parameters
        ----------
        allow_overlap : bool, default False
            When ``False`` (default), the grand mobility uses the
            single-branch far-field Rotne–Prager–Yamakawa formulas — fast,
            but unphysical (and producing NaN at exact coincidence) if any
            pair drifts into overlap. When ``True``, the three GRPY
            regimes (far-field, partial overlap, full immersion) are
            traced through ``lax.cond``; ~15–20 % slower per
            ``compute_fast_tensors`` call but correct under overlap.

            In *both* modes, a post-rollout scan
            (:meth:`scan_trajectory_for_overlap`) emits at most one
            ``UserWarning`` per regime per process if overlap is
            detected. The message text differs: ``False`` flags
            unphysical results and points at ``allow_overlap=True``;
            ``True`` just notes the regime entered.
        """
        super().__init__(*args, **kwargs)
        self.allow_overlap = bool(allow_overlap)
        self._validate_default_geometry()
        self.compute_fast_tensors = jax.jit(self.compute_tensors)

    def compute_tensors(self, dofs=None, design=None, time=None):
        """
        Compute the full mobility problem for a given system configuration.

        This method calculates the mobility matrices, coupling tensors, and
        velocity projection needed to describe the system's dynamic response
        to external forces.

        Parameters
        ----------
        dofs : list or array, optional
            Degrees of freedom. Defaults to ``dof_defaults``.
        design : list or array, optional
            Design variables. Defaults to ``design_defaults``.
        time : float or array, optional
            Time variable. Defaults to ``0.0``.

        Returns
        -------
        SoftMobilityTensors
            A named tuple containing:

            - M : grand mobility tensor ``Mred @ J.T``, shape ``(6+Ndof, 6N)``.
            - Mred : reduced mobility matrix, shape ``(6+Ndof, 6+Ndof)``.
              Maps generalized forces (force/torque at :math:`O` and internal
              forces conjugate to the DOFs) to generalized velocities
              :math:`(\\mathbf{U}, \\boldsymbol{\\Omega}, \\dot{q})`.
              For a rigid body (``Ndof = 0``) the first :math:`6 \times 6`
              block is the effective mobility; use :meth:`compute_rigid_mobility`
              to obtain it directly.
            - M_K : elastic mobility (dofs coupling).
            - M_H : input mobility (field/scalar coupling).
            - C_E : strain coupling matrix.
            - Pi : projection matrix :math:`\\boldsymbol{\\Pi}`.
            - p_act : active velocity contribution.

        Examples
        --------
        >>> M, _, V, *_ = soft_body.compute_tensors(dofs=[0, 1])

        Notes
        -----
        Use ``compute_fast_tensors`` for repeated calls — it is a
        ``jax.jit``-compiled version of this method.
        """
        dofs, design, time = self._setup_params(dofs, design, time)

        J, v_act = self.compute_Jacobian_matrix(dofs, design, time)
        Mgrand = self.compute_grand_mobility(dofs, design, time)
        Rgrand = jnp.linalg.inv(Mgrand)
        C_S = self._compute_composition_of_strain(dofs, design, time)
        R_S = self._compute_coupling_with_strain(dofs, design, time)
        C_H = self.grand_C_H(dofs, design, time)
        C_K = self.grand_C_K(dofs, design, time)

        Mred = jnp.linalg.inv(J.T @ Rgrand @ J)
        M = Mred @ J.T
        Pi = M @ Rgrand
        C_E = Pi @ C_S + M @ R_S
        M_K = M @ C_K
        M_H = M @ C_H
        p_act = -Pi @ v_act.squeeze()

        return SoftMobilityTensors(M=M, Mred=Mred, M_K=M_K, M_H=M_H, C_E=C_E, Pi=Pi, p_act=p_act)

    def compute_rigid_mobility(self, dofs=None, design=None, time=None) -> jnp.ndarray:
        """
        Compute the rigid-body equivalent mobility matrix.

        Returns the 6×6 effective mobility of the assembly treated as a
        perfectly rigid body: all spring stiffnesses are taken to infinity and
        all degrees of freedom are frozen.  Only the six rigid-body modes
        (three translations, three rotations of the reference point) are
        retained.

        This is equivalent to ``compute_tensors().Mred`` when the body has no
        degrees of freedom, and gives the rigid-body limit for soft bodies.

        Parameters
        ----------
        dofs : array-like, optional
            Degrees of freedom used to evaluate bead positions. Defaults to
            ``dof_defaults``.
        design : array-like, optional
            Design variables. Defaults to ``design_defaults``.
        time : float or array-like, optional
            Time used for time-dependent geometry.

        Returns
        -------
        jnp.ndarray, shape ``(6, 6)``
            Symmetric positive-definite mobility matrix relating a force/torque
            applied at the assembly reference point to its translational and
            rotational velocity.
        """
        dofs, design, time = self._setup_params(dofs, design, time)
        Mgrand = self.compute_grand_mobility(dofs, design, time)
        Rgrand = jnp.linalg.inv(Mgrand)
        C_U = self.compute_C_U(dofs, design, time)
        return jnp.linalg.inv(C_U.T @ Rgrand @ C_U)

    def compute_grand_mobility(self, dofs=None, design=None, time=None) -> jnp.ndarray:
        """
        Compute the grand hydrodynamic mobility tensor.

        Uses the Rotne–Prager–Yamakawa (GRPY) approximation across three
        regimes (far-field, partial overlap, full immersion). Direct calls
        are silent; overlap detection happens in
        :meth:`scan_trajectory_for_overlap`, called automatically by
        :meth:`FlowBodyRollout.rollout` after the trajectory is computed.

        The computation is vectorised with ``jax.vmap``: JAX compiles a
        single pairwise block function and maps it over all N² sphere pairs,
        giving O(1) compile time (in N) and O(N²) arithmetic cost.

        Parameters
        ----------
        dofs : array-like, optional
            Degrees of freedom. Defaults to ``dof_defaults``.
        design : array-like, optional
            Design variables. Defaults to ``design_defaults``.
        time : float or array-like, optional
            Time used for time-dependent geometry.

        Returns
        -------
        jnp.ndarray, shape ``(6*N, 6*N)``
            Grand mobility matrix mapping force/torque on each sphere to
            translational and angular velocities.
        """
        dofs, design, time = self._setup_params(dofs, design, time)
        N = self.Nspheres

        # Gather geometry as JAX arrays (Python-side, at trace time)
        positions = jnp.stack([s.position(dofs, design, time) for s in self.spheres])  # (N, 3)
        radii = jnp.stack([jnp.asarray(s.radius(dofs, design)) for s in self.spheres])  # (N,)

        off_diag_fn = _grpy_off_diag_block_all if self.allow_overlap else _grpy_off_diag_block_far

        def block_fn(pos_i, r_i, pos_j, r_j):
            """6×6 GRPY block for one pair; handles diagonal (i==j) via lax.cond."""
            return lax.cond(
                jnp.linalg.norm(pos_i - pos_j) < 1e-9,
                lambda _: _grpy_self_block(r_i),
                lambda _: off_diag_fn(pos_i, r_i, pos_j, r_j),
                None,
            )

        # Double-vmap over i (outer) and j (inner) → (N, N, 6, 6)
        compute_row = jax.vmap(block_fn, in_axes=(None, None, 0, 0))
        compute_all = jax.vmap(compute_row, in_axes=(0, 0, None, None))
        blocks = compute_all(positions, radii, positions, radii)

        # (N, N, 6, 6) → (6N, 6N): interleave sphere and component axes
        return blocks.transpose(0, 2, 1, 3).reshape(6 * N, 6 * N)

    def validate_no_overlap(self, dofs=None, design=None, time=None) -> None:
        """
        Check that no pair of spheres overlaps at the given configuration.

        Parameters
        ----------
        dofs : array-like, optional
            Degrees of freedom. Defaults to ``dof_defaults``.
        design : array-like, optional
            Design variables. Defaults to ``design_defaults``.
        time : float or array-like, optional
            Time. Defaults to ``0.0``.

        Raises
        ------
        ValueError
            If any pair (i, j) has centre-to-centre distance < r_i + r_j
            (strict overlap). Exactly-touching spheres (distance equal to
            the sum of radii within ``1e-9 · max(r_i, r_j)``) are accepted
            since they are the natural state of the Gears Model.
        """
        dofs, design, time = self._setup_params(dofs, design, time)
        positions = [np.asarray(s.position(dofs, design, time), dtype=float) for s in self.spheres]
        radii = [float(s.radius(dofs, design)) for s in self.spheres]
        for i in range(self.Nspheres):
            for j in range(i + 1, self.Nspheres):
                Rij = float(np.linalg.norm(positions[i] - positions[j]))
                ri_rj = radii[i] + radii[j]
                tol = 1e-9 * max(radii[i], radii[j])
                if Rij < ri_rj - tol:
                    raise ValueError(
                        f"Spheres {i} and {j} overlap: separation {Rij:.4f} < "
                        f"sum of radii {ri_rj:.4f}. "
                        "GRPY mobility requires non-overlapping spheres."
                    )

    @staticmethod
    def reset_overlap_warnings() -> None:
        """Clear the set of issued GRPY overlap warnings so they fire again."""
        _OVERLAP_WARN_SEEN.clear()

    @staticmethod
    def silence_overlap_warnings(silence: bool = True) -> None:
        """Disable (or re-enable) the runtime overlap warning.

        Parameters
        ----------
        silence : bool, default True
            If True, suppress all subsequent GRPY overlap warnings. Pass
            ``False`` to re-enable.
        """
        global _OVERLAP_WARN_ENABLED
        _OVERLAP_WARN_ENABLED = not silence

    def scan_trajectory_for_overlap(
        self,
        dofs_traj,
        design=None,
        times=None,
    ) -> None:
        """Scan a rollout's DOF trajectory and warn if spheres ever overlap.

        Computes pairwise body-frame distances at every step and emits at
        most one :class:`UserWarning` per regime (partial overlap or full
        immersion). Skips silently when called from inside a JAX trace
        (the inputs would be tracers, not concrete arrays).

        Parameters
        ----------
        dofs_traj : array-like, shape ``(n_steps, Ndof)``
            DOF trajectory returned by ``FlowBodyRollout.rollout``.
        design : array-like, optional
            Design parameters. Defaults to ``design_defaults``.
        times : array-like, shape ``(n_steps,)``, optional
            Time at each step. Defaults to ``arange(n_steps)`` (only matters
            for time-dependent sphere expressions).
        """
        if not _OVERLAP_WARN_ENABLED:
            return
        # Short-circuit if the relevant warning key has already fired.
        if self.allow_overlap:
            if {"partial-overlap", "full-immersion"} <= _OVERLAP_WARN_SEEN:
                return
        else:
            if "invalid-overlap" in _OVERLAP_WARN_SEEN:
                return

        try:
            dofs_np = np.asarray(dofs_traj, dtype=float)
        except Exception:
            return  # under tracing — can't materialise

        if dofs_np.ndim != 2:
            return  # nothing meaningful to scan

        n_steps = dofs_np.shape[0]
        if design is None:
            design = np.asarray(self.design_defaults, dtype=float)
        else:
            try:
                design = np.asarray(design, dtype=float)
            except Exception:
                return
        if times is None:
            times = np.arange(n_steps, dtype=float)
        else:
            try:
                times = np.asarray(times, dtype=float).reshape(-1)
            except Exception:
                return
            if times.shape[0] != n_steps:
                times = np.arange(n_steps, dtype=float)

        # Radii: assumed constant in time (the design vector is constant).
        radii = np.asarray(
            [float(s.radius(dofs_np[0], design)) for s in self.spheres],
            dtype=float,
        )
        r_sum = radii[:, None] + radii[None, :]
        r_diff = np.abs(radii[:, None] - radii[None, :])
        # Tolerance comfortably above float32 machine epsilon (~1.2e-7) so
        # exact-touching beads in the Gears Model do not register as
        # overlap after the rotation/recurrence drift.
        tol = 1e-5 * float(np.max(radii))
        eye = np.eye(self.Nspheres, dtype=bool)

        # Subsample large trajectories to keep the scan cheap.
        max_samples = 256
        if n_steps > max_samples:
            stride = max(1, n_steps // max_samples)
            sample_idx = np.arange(0, n_steps, stride)
        else:
            sample_idx = np.arange(n_steps)

        # Vectorise the position evaluation across the trajectory: one JAX
        # call covers every sampled step, avoiding per-step Python dispatch.
        try:
            dofs_jax = jnp.asarray(dofs_np[sample_idx])
            times_jax = jnp.asarray(times[sample_idx].reshape(-1, 1))
            design_jax = jnp.asarray(design)

            def _positions_at(dofs_t, t_t):
                return jnp.stack([s.position(dofs_t, design_jax, t_t) for s in self.spheres])

            pos_all = np.asarray(jax.vmap(_positions_at)(dofs_jax, times_jax))
        except Exception:
            return

        # pos_all shape: (n_samples, N, 3). Compute pairwise distances per step.
        # Skip steps that already have NaN positions (a separate overflow warning
        # would be more accurate, but the strict-overlap step *before* NaN
        # arrives is what we want to flag).
        finite_mask = np.all(np.isfinite(pos_all), axis=(1, 2))
        diff = pos_all[:, :, None, :] - pos_all[:, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)  # (n_samples, N, N)
        # Off-diagonal mask
        offdiag = ~eye[None, :, :]
        strict = (dist < r_sum[None, :, :] - tol) & offdiag
        near = (dist < r_diff[None, :, :] + tol) & offdiag
        # Only count steps where positions are finite.
        strict &= finite_mask[:, None, None]
        near &= finite_mask[:, None, None]

        saw_near = bool(near.any())
        saw_medium = bool(strict.any()) and not saw_near

        if saw_medium or saw_near:
            _emit_overlap_warning(saw_medium, saw_near, allow_overlap=self.allow_overlap)

    def _compute_composition_of_strain(self, *args):
        C_S = jnp.block(
            [[self._individual_composition_of_strain(self.spheres[i], *args)] for i in range(self.Nspheres)]
        )
        return C_S

    def _compute_coupling_with_strain(self, *args):
        svecmat = jnp.array(
            [
                [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
                [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
                [[0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [-1, 0, 0, -1, 0]],
            ]
        )

        tensor_blocks = []
        for i in range(self.Nspheres):
            tensor_block = jnp.zeros((6, 3, 3))
            for j in range(self.Nspheres):
                if i == j:
                    Gij = jnp.zeros((6, 3, 3))
                else:
                    mutd = self._compute_mu_td_ij(self.spheres[i], self.spheres[j], *args)
                    murd = self._compute_mu_rd_ij(self.spheres[i], self.spheres[j], *args)
                    Gij = jnp.concatenate(arrays=[mutd, murd], axis=0)
                tensor_block += Gij
            tensor_blocks.append(tensor_block)

        ge = jnp.concatenate(tensor_blocks, axis=0)
        rs = jnp.einsum("ijk,jkl->il", ge, svecmat)
        return rs

    # GRPY tensor helpers (three regimes; strict overlap warned at runtime) ###

    def _2spheres_grpy_quantities(self, sphere_i: Sphere, sphere_j: Sphere, dofs=None, design=None, time=None):
        dofs, design, time = self._setup_params(dofs, design, time)

        posi = jnp.asarray(sphere_i.position(dofs, design, time))
        posj = jnp.asarray(sphere_j.position(dofs, design, time))
        ai = sphere_i.radius(dofs, design)
        aj = sphere_j.radius(dofs, design)
        rvector = posi - posj
        rnorm = jnp.linalg.norm(rvector) + 1e-12
        runit = rvector / rnorm

        return rnorm, runit, ai, aj

    def _compute_mu_td_ij(self, sphere_i: Sphere, sphere_j: Sphere, *args) -> jnp.ndarray:
        """Translation–deformation mobility (3×3×3 tensor).

        Dispatches at Python (trace) time to the single-branch far-field
        helper or the three-regime helper, based on ``self.allow_overlap``.
        """
        if self.allow_overlap:
            return self._compute_mu_td_ij_all(sphere_i, sphere_j, *args)
        return self._compute_mu_td_ij_far(sphere_i, sphere_j, *args)

    def _compute_mu_td_ij_far(self, sphere_i: Sphere, sphere_j: Sphere, *args) -> jnp.ndarray:
        """Far-field-only mu_td (no ``lax.cond``)."""
        Rij, Rij_hat, ai, aj = self._2spheres_grpy_quantities(sphere_i, sphere_j, *args)
        hat1_pf = (5.0 * aj / 6.0) * (-2.0 * (5.0 * ai**2 * aj**2 + 3.0 * aj**4)) / (5.0 * Rij**4)
        hat3_pf = (5.0 * aj / 6.0) * aj**2 * (5.0 * ai**2 + 3.0 * aj**2 - 3.0 * Rij**2) / Rij**4
        matrix = hat1_pf * jnp.outer(jnp.eye(3), Rij_hat).reshape(3, 3, 3) + hat3_pf * jnp.outer(
            Rij_hat, jnp.outer(Rij_hat, Rij_hat)
        ).reshape(3, 3, 3)
        return matrix

    def _compute_mu_td_ij_all(self, sphere_i: Sphere, sphere_j: Sphere, *args) -> jnp.ndarray:
        """Three-regime mu_td (case_far / case_medium / case_near)."""
        Rij, Rij_hat, ai, aj = self._2spheres_grpy_quantities(sphere_i, sphere_j, *args)
        a_inf = jnp.minimum(ai, aj)
        a_sup = jnp.maximum(ai, aj)

        def case_far(_):
            hat1 = (5.0 * aj / 6.0) * (-2.0 * (5.0 * ai**2 * aj**2 + 3.0 * aj**4)) / (5.0 * Rij**4)
            hat3 = (5.0 * aj / 6.0) * aj**2 * (5.0 * ai**2 + 3.0 * aj**2 - 3.0 * Rij**2) / Rij**4
            return hat1, hat3

        def case_medium(_):
            calC = (
                10.0 * Rij**6
                - 24.0 * Rij**5 * ai
                - 15.0 * Rij**4 * (aj**2 - ai**2)
                + (aj - ai) ** 5 * (ai + 5.0 * aj)
            ) / (40.0 * ai * aj * Rij**4)
            calD = (((ai - aj) ** 2 - Rij**2) ** 2 * ((ai - aj) * (ai + 5.0 * aj) - Rij**2)) / (
                16.0 * ai * aj * Rij**4
            )
            return (5.0 * aj / 6.0) * calC, (5.0 * aj / 6.0) * calD

        def case_near(_):
            hat1 = -jnp.heaviside(aj - ai, 0.0) * Rij
            hat3 = jnp.zeros_like(Rij)
            return hat1, hat3

        hat1_pf, hat3_pf = lax.cond(
            Rij > ai + aj,
            case_far,
            lambda _: lax.cond(Rij > a_sup - a_inf, case_medium, case_near, None),
            None,
        )

        matrix = hat1_pf * jnp.outer(jnp.eye(3), Rij_hat).reshape(3, 3, 3) + hat3_pf * jnp.outer(
            Rij_hat, jnp.outer(Rij_hat, Rij_hat)
        ).reshape(3, 3, 3)
        return matrix

    def _compute_mu_rd_ij(self, sphere_i: Sphere, sphere_j: Sphere, *args) -> jnp.ndarray:
        """Rotation–deformation mobility (3×3×3 tensor).

        Dispatches on ``self.allow_overlap`` like :meth:`_compute_mu_td_ij`.
        """
        if self.allow_overlap:
            return self._compute_mu_rd_ij_all(sphere_i, sphere_j, *args)
        return self._compute_mu_rd_ij_far(sphere_i, sphere_j, *args)

    def _compute_mu_rd_ij_far(self, sphere_i: Sphere, sphere_j: Sphere, *args) -> jnp.ndarray:
        """Far-field-only mu_rd (no ``lax.cond``)."""
        Rij, Rij_hat, ai, aj = self._2spheres_grpy_quantities(sphere_i, sphere_j, *args)
        prefactor = -(5.0 / 2.0) * (aj / Rij) ** 3
        matrix = prefactor * jnp.outer(jnp.cross(-jnp.eye(3), Rij_hat), Rij_hat).reshape(3, 3, 3)
        return matrix

    def _compute_mu_rd_ij_all(self, sphere_i: Sphere, sphere_j: Sphere, *args) -> jnp.ndarray:
        """Three-regime mu_rd."""
        Rij, Rij_hat, ai, aj = self._2spheres_grpy_quantities(sphere_i, sphere_j, *args)
        a_inf = jnp.minimum(ai, aj)
        a_sup = jnp.maximum(ai, aj)

        def case_far(_):
            return -(5.0 / 2.0) * (aj / Rij) ** 3

        def case_medium(_):
            calB = (3.0 * ((ai - aj) ** 2 - Rij**2) ** 2 * (ai**2 + 4.0 * ai * aj + aj**2 - Rij**2)) / (
                64.0 * Rij**3
            )
            return -5.0 * calB / (3.0 * ai**3)

        def case_near(_):
            return jnp.zeros_like(Rij)

        prefactor = lax.cond(
            Rij > ai + aj,
            case_far,
            lambda _: lax.cond(Rij > a_sup - a_inf, case_medium, case_near, None),
            None,
        )

        matrix = prefactor * jnp.outer(jnp.cross(-jnp.eye(3), Rij_hat), Rij_hat).reshape(3, 3, 3)
        return matrix

    def _individual_composition_of_strain(self, sphere: Sphere, dofs=None, design=None, time=None):
        dofs, design, time = self._setup_params(dofs, design, time)

        X, Y, Z = sphere.position(dofs, design, time)
        T = jnp.array(
            [
                [X, Y, Z, 0, 0],
                [0, X, 0, Y, Z],
                [-Z, 0, X, -Z, Y],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        return T

    def _validate_default_geometry(self):
        """
        Check that the default configuration produces a finite mobility matrix.
        Warns the user early rather than letting NaN appear silently at runtime.
        """
        try:
            dofs = jnp.array(self.dof_defaults)
            design = jnp.array(self.design_defaults)
            time = jnp.array([0.0])
            tensors = self.compute_tensors(dofs, design, time)

            issues = []
            for name, tensor in [("M_H", tensors.M_H), ("M_K", tensors.M_K), ("C_E", tensors.C_E)]:
                if not jnp.all(jnp.isfinite(tensor)):
                    issues.append(name)

            if issues:
                warnings.warn(
                    f"Default configuration produces NaN/Inf in: {issues}. "
                    f"Check that default DOFs and design parameters describe a "
                    f"physically valid, non-degenerate geometry (e.g. spheres "
                    f"at zero separation).\n"
                    f"  dof_defaults    = {list(self.dof_defaults)}\n"
                    f"  design_defaults = {dict(self.design_defaults)}"
                    f"  time_default    = {[0.0]}",
                    UserWarning,
                    stacklevel=2,
                )

        except Exception as e:
            warnings.warn(
                f"Could not validate default geometry: {e}",
                UserWarning,
                stacklevel=2,
            )


# Module-level functions and state for GRPY mobility tensors ##################
# These operate on plain arrays (no Sphere objects) so they are vmappable.
#
# Three regimes per pair, selected at trace time by nested ``lax.cond``:
#   case_far    : R > a_i + a_j                  (far-field Rotne–Prager)
#   case_medium : |a_i - a_j| < R ≤ a_i + a_j    (GRPY partial-overlap, WMZS)
#   case_near   : R ≤ |a_i - a_j|                (full immersion)


# Host-side state for runtime overlap warnings. The warning is fired once
# per regime per process; call ``SoftBody.reset_overlap_warnings()`` to clear.
_OVERLAP_WARN_SEEN: set[str] = set()
_OVERLAP_WARN_ENABLED: bool = True


def _emit_overlap_warning(any_medium: bool, any_near: bool, allow_overlap: bool = True) -> None:
    """Emit at most one ``UserWarning`` per regime per process.

    Parameters
    ----------
    any_medium, any_near : bool
        Whether any sampled timestep entered the corresponding regime.
    allow_overlap : bool, default True
        Selects the warning text. ``True`` → the SoftBody was constructed
        with ``allow_overlap=True`` and the regime is handled correctly by
        the three-regime GRPY helpers; the warning is informational.
        ``False`` → the SoftBody was constructed with the default
        (far-field-only) mobility, so the overlap step produced
        unphysical / NaN values — the warning carries an "invalid
        results" message and points the user at ``allow_overlap=True``.
    """
    if not _OVERLAP_WARN_ENABLED:
        return
    if allow_overlap:
        if any_near and "full-immersion" not in _OVERLAP_WARN_SEEN:
            _OVERLAP_WARN_SEEN.add("full-immersion")
            warnings.warn(
                "GRPY mobility entered the full-immersion regime "
                "(R ≤ |a_i − a_j|, one sphere fully inside another) during a "
                "rollout. Further occurrences are suppressed; call "
                "SoftBody.reset_overlap_warnings() to re-enable.",
                UserWarning,
                stacklevel=2,
            )
        if any_medium and "partial-overlap" not in _OVERLAP_WARN_SEEN:
            _OVERLAP_WARN_SEEN.add("partial-overlap")
            warnings.warn(
                "GRPY mobility entered the partial-overlap regime "
                "(|a_i − a_j| < R < a_i + a_j, spheres strictly overlapping) "
                "during a rollout. Further occurrences are suppressed; call "
                "SoftBody.reset_overlap_warnings() to re-enable.",
                UserWarning,
                stacklevel=2,
            )
    else:
        if (any_medium or any_near) and "invalid-overlap" not in _OVERLAP_WARN_SEEN:
            _OVERLAP_WARN_SEEN.add("invalid-overlap")
            warnings.warn(
                "Spheres overlapped during a rollout (R < a_i + a_j) but "
                "this SoftBody was built with allow_overlap=False — "
                "far-field-only GRPY gives unphysical / NaN mobility "
                "there, so the trajectory is invalid past the first "
                "overlap. Pass allow_overlap=True to SoftBody(...) to "
                "enable the GRPY partial-overlap and full-immersion "
                "formulas, or shrink the design parameters that cause the "
                "overlap. Further occurrences are suppressed; call "
                "SoftBody.reset_overlap_warnings() to re-enable.",
                UserWarning,
                stacklevel=2,
            )


def _grpy_self_block(r_i: float) -> jnp.ndarray:
    """Self-mobility 6×6 block for a sphere of radius ``r_i``."""
    mu_tt = (1.0 / (6.0 * jnp.pi * r_i)) * jnp.eye(3)
    mu_rr = (1.0 / (8.0 * jnp.pi * r_i**3)) * jnp.eye(3)
    zeros = jnp.zeros((3, 3))
    return jnp.block([[mu_tt, zeros], [zeros, mu_rr]])


def _grpy_off_diag_block_far(
    pos_i: jnp.ndarray, r_i: float, pos_j: jnp.ndarray, r_j: float
) -> jnp.ndarray:
    """Far-field Rotne–Prager 6×6 block (no overlap branches).

    Selected when ``SoftBody.allow_overlap`` is ``False``. Identical to
    the pre-three-regime helper on ``main`` — single branch, no
    ``lax.cond`` overhead. Produces unphysical / NaN values if the pair
    is in strict overlap; the post-rollout
    :meth:`SoftBody.scan_trajectory_for_overlap` is responsible for
    catching that and emitting a ``UserWarning``.
    """
    rvec = pos_i - pos_j
    R = jnp.linalg.norm(rvec)
    Rhat = rvec / R

    # Translation–translation (Rotne–Prager)
    eye_pf = (1.0 + (r_i**2 + r_j**2) / (3.0 * R**2)) / (8.0 * jnp.pi * R)
    hat_pf = (1.0 - (r_i**2 + r_j**2) / R**2) / (8.0 * jnp.pi * R)
    mu_tt = eye_pf * jnp.eye(3) + hat_pf * jnp.outer(Rhat, Rhat)

    # Rotation–rotation
    mu_rr = (-1.0 / (16.0 * jnp.pi * R**3)) * jnp.eye(3) + (3.0 / (16.0 * jnp.pi * R**3)) * jnp.outer(Rhat, Rhat)

    # Rotation–translation and translation–rotation (cross-product matrix form)
    pf_rt = 1.0 / (8.0 * jnp.pi * R**2)
    mu_rt = pf_rt * jnp.cross(-jnp.eye(3), Rhat)
    # mu_tr(i,j) = mu_rt(j,i)^T  with Rhat_ji = -Rhat_ij  (radii cancel here).
    mu_tr = pf_rt * jnp.cross(-jnp.eye(3), -Rhat).T

    return jnp.block([[mu_tt, mu_tr], [mu_rt, mu_rr]])


def _grpy_off_diag_block_all(
    pos_i: jnp.ndarray, r_i: float, pos_j: jnp.ndarray, r_j: float
) -> jnp.ndarray:
    """GRPY 6×6 block over all three regimes (far / medium / near).

    Selected when ``SoftBody.allow_overlap`` is ``True``. Adds the
    Wajnryb–Mizerski–Żuk–Szymczak partial-overlap and full-immersion
    formulas. Branches via nested ``lax.cond``; ~15–20 % slower per
    ``compute_fast_tensors`` than :func:`_grpy_off_diag_block_far` even
    when no pair actually overlaps (lax.cond dispatch cost).

    Parameters
    ----------
    pos_i, pos_j : jnp.ndarray, shape (3,)
        Centre positions of spheres i and j.
    r_i, r_j : float
        Radii of spheres i and j.

    Returns
    -------
    jnp.ndarray, shape (6, 6)
        Block ``[[mu_tt, mu_tr], [mu_rt, mu_rr]]``.
    """
    rvec = pos_i - pos_j
    R = jnp.linalg.norm(rvec) + 1e-12
    Rhat = rvec / R
    a_inf = jnp.minimum(r_i, r_j)
    a_sup = jnp.maximum(r_i, r_j)

    inv8piR = 1.0 / (8.0 * jnp.pi * R)
    inv16piR3 = 1.0 / (16.0 * jnp.pi * R**3)

    def case_far(_):
        # Translation–translation (Rotne–Prager)
        eye_tt = (1.0 + (r_i**2 + r_j**2) / (3.0 * R**2)) * inv8piR
        hat_tt = (1.0 - (r_i**2 + r_j**2) / R**2) * inv8piR
        # Rotation–rotation
        eye_rr = -inv16piR3
        hat_rr = 3.0 * inv16piR3
        # Rotation–translation (symmetric in i↔j)
        pf_rt_ij = 1.0 / (8.0 * jnp.pi * R**2)
        pf_rt_ji = pf_rt_ij
        return eye_tt, hat_tt, eye_rr, hat_rr, pf_rt_ij, pf_rt_ji

    def case_medium(_):
        # Partial overlap |a_i − a_j| < R ≤ a_i + a_j (WMZS 2013)
        # mu_tt
        eye_tt = ((16.0 * R**3 * (r_i + r_j) - ((r_i - r_j) ** 2 + 3.0 * R**2) ** 2) / (32.0 * R**3)) / (
            6.0 * jnp.pi * r_i * r_j
        )
        hat_tt = (3.0 * ((r_i - r_j) ** 2 - R**2) ** 2 / (32.0 * R**3)) / (6.0 * jnp.pi * r_i * r_j)
        # mu_rr
        calA = (
            5.0 * R**6
            - 27.0 * R**4 * (r_i**2 + r_j**2)
            + 32.0 * R**3 * (r_i**3 + r_j**3)
            - 9.0 * R**2 * (r_i**2 - r_j**2) ** 2
            - (r_i - r_j) ** 4 * (r_i**2 + 4.0 * r_i * r_j + r_j**2)
        ) / (64.0 * R**3)
        calB = (
            3.0 * ((r_i - r_j) ** 2 - R**2) ** 2 * (r_i**2 + 4.0 * r_i * r_j + r_j**2 - R**2)
        ) / (64.0 * R**3)
        eye_rr = calA / (8.0 * jnp.pi * r_i**3 * r_j**3)
        hat_rr = calB / (8.0 * jnp.pi * r_i**3 * r_j**3)
        # mu_rt prefactor: asymmetric in i↔j, compute both orderings.
        pf_rt_ij = ((r_i - r_j + R) ** 2 * (r_j**2 + 2.0 * r_j * (r_i + R) - 3.0 * (r_i - R) ** 2)) / (
            8.0 * R**2
        ) / (16.0 * jnp.pi * r_i**3 * r_j)
        pf_rt_ji = ((r_j - r_i + R) ** 2 * (r_i**2 + 2.0 * r_i * (r_j + R) - 3.0 * (r_j - R) ** 2)) / (
            8.0 * R**2
        ) / (16.0 * jnp.pi * r_j**3 * r_i)
        return eye_tt, hat_tt, eye_rr, hat_rr, pf_rt_ij, pf_rt_ji

    def case_near(_):
        # Full immersion R ≤ |a_i − a_j|: the inner sphere is rigidly
        # carried by the outer one, so its mobility equals that of a
        # single sphere of radius a_sup.
        eye_tt = 1.0 / (6.0 * jnp.pi * a_sup)
        hat_tt = jnp.zeros_like(R)
        eye_rr = 1.0 / (8.0 * jnp.pi * a_sup**3)
        hat_rr = jnp.zeros_like(R)
        # Heaviside-gated direction: only the bigger sphere drags the smaller.
        pf_rt_ij = jnp.heaviside(r_i - r_j, 0.0) * R / (8.0 * jnp.pi * R**2)
        pf_rt_ji = jnp.heaviside(r_j - r_i, 0.0) * R / (8.0 * jnp.pi * R**2)
        return eye_tt, hat_tt, eye_rr, hat_rr, pf_rt_ij, pf_rt_ji

    eye_tt, hat_tt, eye_rr, hat_rr, pf_rt_ij, pf_rt_ji = lax.cond(
        R > r_i + r_j,
        case_far,
        lambda _: lax.cond(R > a_sup - a_inf, case_medium, case_near, None),
        None,
    )

    # No per-pair runtime warning here: ``jax.debug.callback`` under
    # ``lax.scan`` forces a host roundtrip per step and tanks performance
    # by ~75× on small bodies.  Overlap detection now happens once after
    # a rollout via :meth:`SoftBody.scan_trajectory_for_overlap`, called
    # automatically by :meth:`FlowBodyRollout.rollout` and the clamped
    # variant.  Direct (non-rollout) callers may invoke
    # :meth:`validate_no_overlap` or scan_trajectory_for_overlap by hand.

    mu_tt = eye_tt * jnp.eye(3) + hat_tt * jnp.outer(Rhat, Rhat)
    mu_rr = eye_rr * jnp.eye(3) + hat_rr * jnp.outer(Rhat, Rhat)
    mu_rt = pf_rt_ij * jnp.cross(-jnp.eye(3), Rhat)
    # mu_tr(i,j) = mu_rt(j,i)^T  with Rhat_ji = -Rhat_ij  (and swapped radii).
    mu_tr = pf_rt_ji * jnp.cross(-jnp.eye(3), -Rhat).T

    return jnp.block([[mu_tt, mu_tr], [mu_rt, mu_rr]])

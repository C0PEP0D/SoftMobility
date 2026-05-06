"""SoftBody class."""

import warnings
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .sphere import Sphere
from .sphereassembly import SphereAssembly

SoftMobilityTensors = namedtuple("SoftMobilityTensors", ["M", "M_K", "M_H", "C_E", "P", "p_act"])


# Pure module-level functions for far-field GRPY mobility tensors ##############
# These operate on plain arrays (no Sphere objects) so they are vmappable.

def _grpy_self_block(r_i: float) -> jnp.ndarray:
    """Self-mobility 6×6 block for a sphere of radius ``r_i``."""
    mu_tt = (1.0 / (6.0 * jnp.pi * r_i)) * jnp.eye(3)
    mu_rr = (1.0 / (8.0 * jnp.pi * r_i**3)) * jnp.eye(3)
    zeros = jnp.zeros((3, 3))
    return jnp.block([[mu_tt, zeros], [zeros, mu_rr]])


def _grpy_off_diag_block(
    pos_i: jnp.ndarray, r_i: float, pos_j: jnp.ndarray, r_j: float
) -> jnp.ndarray:
    """Far-field GRPY 6×6 block for a non-overlapping sphere pair (i ≠ j).

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
    R = jnp.linalg.norm(rvec)
    Rhat = rvec / R

    # Translation–translation (Rotne–Prager)
    eye_pf = (1.0 + (r_i**2 + r_j**2) / (3.0 * R**2)) / (8.0 * jnp.pi * R)
    hat_pf = (1.0 - (r_i**2 + r_j**2) / R**2) / (8.0 * jnp.pi * R)
    mu_tt = eye_pf * jnp.eye(3) + hat_pf * jnp.outer(Rhat, Rhat)

    # Rotation–rotation
    mu_rr = (
        (-1.0 / (16.0 * jnp.pi * R**3)) * jnp.eye(3)
        + (3.0 / (16.0 * jnp.pi * R**3)) * jnp.outer(Rhat, Rhat)
    )

    # Rotation–translation and translation–rotation (cross-product matrix form)
    pf_rt = 1.0 / (8.0 * jnp.pi * R**2)
    mu_rt = pf_rt * jnp.cross(-jnp.eye(3), Rhat)
    # mu_tr(i,j) = mu_rt(j,i)^T  where Rhat_ji = -Rhat_ij
    mu_tr = pf_rt * jnp.cross(-jnp.eye(3), -Rhat).T

    return jnp.block([[mu_tt, mu_tr], [mu_rt, mu_rr]])


###############################################################################


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
        - P : jnp.ndarray
            Projection matrix.

    Examples
    --------
    Initialize a soft body from a YAML file:

    >>> soft_body = SoftBody("path/to/parameters.yaml")

    Compute mobility tensors:

    >>> tensors = soft_body.compute_tensors()

    Notes
    -----
    - The grand mobility tensor is computed using the far-field Rotne–
      Prager–Yamakawa (GRPY) approximation. Spheres must **not** overlap;
      call :meth:`validate_no_overlap` to verify the geometry before
      simulation.
    - Inherits all functionality from ``SphereAssembly``, including sphere
      management and degree of freedom handling.
    - Compatible with JAX transformations
      (``jax.jit``, ``jax.grad``, ``jax.vmap``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

            - M : grand mobility matrix.
            - M_K : elastic mobility (dofs coupling).
            - M_H : input mobility (field/scalar coupling).
            - C_E : strain coupling matrix.
            - P : projection matrix.
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
        Mgrand = self.compute_mobility_tensor(dofs, design, time)
        Rgrand = jnp.linalg.inv(Mgrand)
        C_S = self._compute_composition_of_strain(dofs, design, time)
        R_S = self._compute_coupling_with_strain(dofs, design, time)
        C_H = self.grand_C_H(dofs, design, time)
        C_K = self.grand_C_K(dofs, design, time)

        Mred = jnp.linalg.inv(J.T @ Rgrand @ J)
        M = Mred @ J.T
        P = M @ Rgrand
        C_E = P @ C_S + M @ R_S
        M_K = M @ C_K
        M_H = M @ C_H
        p_act = -P @ v_act.squeeze()

        return SoftMobilityTensors(M=M, M_K=M_K, M_H=M_H, C_E=C_E, P=P, p_act=p_act)

    def compute_mobility_tensor(self, dofs=None, design=None, time=None) -> jnp.ndarray:
        """
        Compute the grand hydrodynamic mobility tensor.

        Uses the far-field Rotne–Prager–Yamakawa (GRPY) approximation.
        Spheres must not overlap; see :meth:`validate_no_overlap`.

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

        def block_fn(pos_i, r_i, pos_j, r_j):
            """6×6 GRPY block for one pair; handles diagonal (i==j) via lax.cond."""
            return lax.cond(
                jnp.linalg.norm(pos_i - pos_j) < 1e-9,
                lambda _: _grpy_self_block(r_i),
                lambda _: _grpy_off_diag_block(pos_i, r_i, pos_j, r_j),
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
            If any pair (i, j) has centre-to-centre distance ≤ r_i + r_j.
        """
        dofs, design, time = self._setup_params(dofs, design, time)
        positions = [np.asarray(s.position(dofs, design, time), dtype=float) for s in self.spheres]
        radii = [float(s.radius(dofs, design)) for s in self.spheres]
        for i in range(self.Nspheres):
            for j in range(i + 1, self.Nspheres):
                Rij = float(np.linalg.norm(positions[i] - positions[j]))
                ri_rj = radii[i] + radii[j]
                if Rij <= ri_rj:
                    raise ValueError(
                        f"Spheres {i} and {j} overlap: separation {Rij:.4f} ≤ "
                        f"sum of radii {ri_rj:.4f}. "
                        "GRPY mobility requires non-overlapping spheres."
                    )

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

    # GRPY tensor helpers (far-field only) ####################################

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
        """Far-field translation–deformation mobility (3×3×3 tensor)."""
        Rij, Rij_hat, ai, aj = self._2spheres_grpy_quantities(sphere_i, sphere_j, *args)

        hat1_pf = (5.0 * aj / 6.0) * (-2.0 * (5.0 * ai**2 * aj**2 + 3.0 * aj**4)) / (5.0 * Rij**4)
        hat3_pf = (5.0 * aj / 6.0) * aj**2 * (5.0 * ai**2 + 3.0 * aj**2 - 3.0 * Rij**2) / Rij**4

        matrix = (
            hat1_pf * jnp.outer(jnp.eye(3), Rij_hat).reshape(3, 3, 3)
            + hat3_pf * jnp.outer(Rij_hat, jnp.outer(Rij_hat, Rij_hat)).reshape(3, 3, 3)
        )
        return matrix

    def _compute_mu_rd_ij(self, sphere_i: Sphere, sphere_j: Sphere, *args) -> jnp.ndarray:
        """Far-field rotation–deformation mobility (3×3×3 tensor)."""
        Rij, Rij_hat, ai, aj = self._2spheres_grpy_quantities(sphere_i, sphere_j, *args)

        prefactor = -(5.0 / 2.0) * (aj / Rij) ** 3
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
                    f"physically valid, non-degenerate geometry (e.g. spheres not "
                    f"overlapping or at zero separation).\n"
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

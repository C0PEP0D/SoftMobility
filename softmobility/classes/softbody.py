"""SoftBody class."""

from collections import namedtuple
import warnings
import jax.numpy as jnp
import jax
from jax import lax
from .sphere import Sphere
from .sphereassembly import SphereAssembly

SoftMobilityTensors = namedtuple("SoftMobilityTensors", ["M", "M_K", "M_H", "C_E", "P"])


class SoftBody(SphereAssembly):
    """
    SoftBody (simulation of deformable bodies in fluid flow).

    Extends ``SphereAssembly`` to model soft bodies with mobility tensors and hydrodynamic interactions.
    Supports computation of mobility matrices, coupling tensors, and forces for soft body dynamics.
    Compatible with JAX for automatic differentiation and just-in-time compilation.

    Parameters
    ----------
    parameters_source : str, optional
        Path to a YAML file containing assembly parameters (spheres, dofs, design variables, etc.).
        If provided, the assembly is initialized from this file.
    verbose : bool, default=True
        If True, prints debug and progress information during initialization and computation.

    Attributes
    ----------
    SoftMobilityTensors : namedtuple
        A named tuple containing the mobility tensors:

        - M : jnp.ndarray
            Grand mobility matrix (to be multiplied by grand forces [F1, T1, F2, T2...]).
        - M_K : jnp.ndarray
            Elastic mobility matrix (to be multiplied by degrees of freedom dofs).
        - M_H : jnp.ndarray
            Input mobility matrix (coupling with 3D and scalar fields).
        - C_E : jnp.ndarray
            Elastic coupling matrix.
        - P : jnp.ndarray
            Projection matrix.

    Examples
    --------
    Initialize a soft body from a YAML file:

    >>> soft_body = SoftBody.from_file("path/to/parameters.yaml")

    Compute mobility tensors:

    >>> tensors = soft_body.compute_tensors(dofs, design)

    Notes
    -----
    - Inherits all functionality from ``SphereAssembly``, including sphere management and degree of freedom handling.
    - Mobility tensors are computed using symbolic expressions for efficiency and accuracy.
    - Compatible with JAX transformations (``jax.jit``, ``jax.grad``, ``jax.vmap``).
    """

    def __init__(self, *args, **kwargs):
        # Call the __init__ method of the parent class (SphereAssembly)
        super().__init__(*args, **kwargs)
        self._validate_default_geometry()
        self.compute_fast_tensors = jax.jit(self.compute_tensors)

    def compute_tensors(self, dofs=None, design=None):
        """
        Compute the full mobility problem for a given system configuration.

        This method calculates the mobility matrices, coupling tensors, and velocity projection
        needed to describe the system's dynamic response to external forces.

        Parameters
        ----------
        dofs : list or array, optional
            Degrees of freedom of the soft plankton.
        params : list or array, optional
            Parameters defining the soft plankton characteristics.

        Returns
        -------
        SoftMobilityTensors
            A named tuple containing:
            - M (jax.numpy.ndarray): Mobility matrix for forces/torques expressed at the center of spheres.
            - M_K (jax.numpy.ndarray): Mobility matrix with degrees.
            - M_H (jax.numpy.ndarray): Mobility matrix with inputs (3D and scalar fields).
            - C_E (jax.numpy.ndarray): Coupling matrix with strain.
            - P (jax.numpy.ndarray): Projection matrix.

        Examples
        --------
        Unpacking using a tuple:

        >>> M, _, V, *_ = soft_plankton.compute_mobility_problem(dofs=[0, 1])
        >>> print(M, V)

        Unpacking using the named tuple (preferred method):

        >>> matrices = soft_plankton.compute_mobility_problem()
        >>> print(matrices.M, matrices.V)

        Notes
        -----
        - This function leverages JAX for automatic differentiation and efficient computation.
        """
        # Handling dofs and params, passing default value if None are given
        dofs, design = self._setup_params(dofs, design)

        J = self.compute_Jacobian_matrix(dofs, design)
        Mgrand = self.compute_mobility_tensor(dofs, design)
        Rgrand = jnp.linalg.inv(Mgrand)
        C_S = self._compute_composition_of_strain(dofs, design)
        R_S = self._compute_coupling_with_strain(dofs, design)
        C_H = self.grand_c_field(dofs, design)
        C_K = self.grand_c_stiff(dofs, design)

        # Compute soft mobility tensors
        Mred = jnp.linalg.inv(J.T @ Rgrand @ J)
        M = Mred @ J.T
        P = M @ Rgrand
        C_E = P @ C_S + M @ R_S
        M_K = M @ C_K
        M_H = M @ C_H

        return SoftMobilityTensors(M, M_K, M_H, C_E, P)

    def compute_mobility_tensor(self, dofs=None, design=None):
        dofs, design = self._setup_params(dofs, design)

        # Function to compute diagonal blocks (i == j)
        def compute_diag_block(i):
            mu_tt = self._compute_mu_tt_ii(self.spheres[i], dofs, design)
            mu_rr = self._compute_mu_rr_ii(self.spheres[i], dofs, design)
            mu_rt = jnp.zeros((3, 3))
            mu_tr = jnp.zeros((3, 3))
            return jnp.block([[mu_tt, mu_tr], [mu_rt, mu_rr]])

        # Function to compute off-diagonal blocks (i ≠ j)
        def compute_off_diag_block(i, j):
            mu_tt = self._compute_mu_tt_ij(self.spheres[i], self.spheres[j], dofs, design)
            mu_rr = self._compute_mu_rr_ij(self.spheres[i], self.spheres[j], dofs, design)
            mu_rt = self._compute_mu_rt_ij(self.spheres[i], self.spheres[j], dofs, design)
            mu_tr = self._compute_mu_rt_ij(self.spheres[j], self.spheres[i], dofs, design).T
            return jnp.block([[mu_tt, mu_tr], [mu_rt, mu_rr]])

        # Function to select correct block using lax.cond
        def compute_block(i, j):
            return lax.cond(
                i == j, lambda _: compute_diag_block(i), lambda _: compute_off_diag_block(i, j), operand=None
            )

        # Assemble full mobility matrix
        M = jnp.block([[compute_block(i, j) for j in range(self.Nspheres)] for i in range(self.Nspheres)])

        return M

    def _compute_mobility_tensor_alt(self, dofs=None, design=None, inputs=None):
        dofs, design = self._setup_params(dofs, design)
        # Define the blocks functions
        mu_tt = jnp.block(
            [
                [
                    self._compute_mu_tt_ij(self.spheres[i], self.spheres[j], dofs, design)
                    for j in range(self.Nspheres)
                ]
                for i in range(self.Nspheres)
            ]
        )
        mu_rr = jnp.block(
            [
                [
                    self._compute_mu_rr_ij(self.spheres[i], self.spheres[j], dofs, design)
                    for j in range(self.Nspheres)
                ]
                for i in range(self.Nspheres)
            ]
        )
        mu_rt = jnp.block(
            [
                [
                    self._compute_mu_rt_ij(self.spheres[i], self.spheres[j], dofs, design)
                    for j in range(self.Nspheres)
                ]
                for i in range(self.Nspheres)
            ]
        )
        mu_tr = jnp.block(
            [
                [
                    self._compute_mu_rt_ij(self.spheres[j], self.spheres[i], dofs, design).T
                    for j in range(self.Nspheres)
                ]
                for i in range(self.Nspheres)
            ]
        )
        # Assemble the blocks into a full matrix
        M = jnp.block(jnp.block([[mu_tt, mu_tr], [mu_rt, mu_rr]]))

        return M

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

        # Initialize tensor for each sphere
        tensor_blocks = []

        # Compute the sum of mutd and murd blocks for each sphere
        for i in range(self.Nspheres):
            # Initialize a 6x3x3 tensor with zeros
            tensor_block = jnp.zeros((6, 3, 3))

            # Iterate over all spheres, including i itself
            for j in range(self.Nspheres):
                if i == j:
                    # For the same sphere, no interaction terms
                    Gij = jnp.zeros((6, 3, 3))
                else:
                    # For distinct spheres, compute mutd and murd
                    mutd = self._compute_mu_td_ij(self.spheres[i], self.spheres[j], *args)
                    murd = self._compute_mu_rd_ij(self.spheres[i], self.spheres[j], *args)
                    Gij = jnp.concatenate(arrays=[mutd, murd], axis=0)

                # Sum over j
                tensor_block += Gij

            # Append the computed block for the current sphere
            tensor_blocks.append(tensor_block)

        # Stack all blocks into a final 6 * Nspheres x 3 x 3 tensor
        ge = jnp.concatenate(tensor_blocks, axis=0)

        # Perform tensor contraction with Svecmat (if needed)
        rs = jnp.einsum("ijk,jkl->il", ge, svecmat)  # Adjust contraction as needed

        return rs

    # Functions to compute the GRPY tensors #######################################
    def _1sphere_grpy_quantities(self, sphere: Sphere, dofs=None, design=None, inputs=None):
        dofs, design = self._setup_params(dofs, design)

        return sphere.radius(dofs, design)

    def _2spheres_grpy_quantities(self, sphere_i: Sphere, sphere_j: Sphere, dofs=None, design=None, inputs=None):
        dofs, design = self._setup_params(dofs, design)

        position_i = sphere_i.position(dofs, design)
        position_j = sphere_j.position(dofs, design)
        radius_i = sphere_i.radius(dofs, design)
        radius_j = sphere_j.radius(dofs, design)
        posi = jnp.array(position_i)
        posj = jnp.array(position_j)
        rvector = posi - posj
        rnorm = jnp.linalg.norm(rvector) + 1e-12  # avoid division by zero
        runit = rvector / rnorm

        return rnorm, runit, radius_i, radius_j

    def _compute_mu_tt_ii(self, sphere, *args):
        a_i = self._1sphere_grpy_quantities(sphere, *args)
        matrix = 1 / (6 * jnp.pi * a_i) * jnp.eye(3)

        return matrix

    def _compute_mu_rr_ii(self, sphere, *args):
        a_i = self._1sphere_grpy_quantities(sphere, *args)
        matrix = 1 / (8 * jnp.pi * a_i**3) * jnp.eye(3)

        return matrix

    def _compute_mu_tt_ij(self, sphere_i, sphere_j, *args):
        Rij, Rij_hat, ai, aj = self._2spheres_grpy_quantities(sphere_i, sphere_j, *args)

        a_inf = jnp.minimum(ai, aj)
        a_sup = jnp.maximum(ai, aj)

        # Use JAX `lax.cond()` for branching logic
        def case_far(_):
            """Case when Rij > ai + aj"""
            eye_prefactor = (1 + (ai**2 + aj**2) / (3 * Rij**2)) / (8 * jnp.pi * Rij)
            hat_prefactor = (1 - (ai**2 + aj**2) / Rij**2) / (8 * jnp.pi * Rij)
            return eye_prefactor, hat_prefactor

        def case_medium(_):
            """Case when a_sup - a_inf < Rij <= ai + aj"""
            eye_prefactor = (
                (16 * Rij**3 * (ai + aj) - ((ai - aj) ** 2 + 3 * Rij**2) ** 2) / (32 * Rij**3)
            ) / (6 * jnp.pi * ai * aj)
            hat_prefactor = (3 * ((ai - aj) ** 2 - Rij**2) ** 2 / (32 * Rij**3)) / (6 * jnp.pi * ai * aj)
            return eye_prefactor, hat_prefactor

        def case_near(_):
            """Case when Rij <= a_sup - a_inf"""
            return 1 / (6 * jnp.pi * a_sup), 0.0

        # Nested conditions
        eye_prefactor, hat_prefactor = lax.cond(
            Rij > ai + aj, case_far, lambda _: lax.cond(Rij > a_sup - a_inf, case_medium, case_near, None), None
        )

        # Compute the 3x3 mobility tensor
        matrix = eye_prefactor * jnp.eye(3) + hat_prefactor * jnp.outer(Rij_hat, Rij_hat)

        return matrix

    def _compute_mu_rr_ij(self, sphere_i: Sphere, sphere_j: Sphere, *args):
        Rij, Rij_hat, ai, aj = self._2spheres_grpy_quantities(sphere_i, sphere_j, *args)

        a_inf = jnp.minimum(ai, aj)
        a_sup = jnp.maximum(ai, aj)

        # Use JAX `lax.cond()` for branching logic
        def case_far(_):
            """Case when Rij > ai + aj"""
            eye_prefactor = -1.0 / (16 * jnp.pi * Rij**3)
            hat_prefactor = 3.0 / (16 * jnp.pi * Rij**3)
            return eye_prefactor, hat_prefactor

        def case_medium(_):
            """Case when a_sup - a_inf < Rij <= ai + aj"""
            calA = (
                +5.0 * Rij**6
                - 27 * Rij**4 * (ai**2 + aj**2)
                + 32 * Rij**3 * (ai**3 + aj**3)
                - 9.0 * Rij**2 * (ai**2 - aj**2) ** 2
                - (ai - aj) ** 4 * (ai**2 + 4 * aj * ai + aj**2)
            ) / (64 * Rij**3)
            calB = (3.0 * ((ai - aj) ** 2 - Rij**2) ** 2 * (ai**2 + 4 * aj * ai + aj**2 - Rij**2)) / (64 * Rij**3)
            eye_prefactor = calA / (8 * jnp.pi * ai**3 * aj**3)
            hat_prefactor = calB / (8 * jnp.pi * ai**3 * aj**3)
            return eye_prefactor, hat_prefactor

        def case_near(_):
            """Case when Rij <= a_sup - a_inf"""
            eye_prefactor = 1.0 / (8 * jnp.pi * a_sup**3)
            hat_prefactor = 0.0
            return eye_prefactor, hat_prefactor

        # Nested conditions
        eye_prefactor, hat_prefactor = lax.cond(
            Rij > ai + aj, case_far, lambda _: lax.cond(Rij > a_sup - a_inf, case_medium, case_near, None), None
        )

        # Compute the 3x3 matrix
        matrix = eye_prefactor * jnp.eye(3) + hat_prefactor * jnp.outer(Rij_hat, Rij_hat)

        return matrix

    def _compute_mu_rt_ij(self, sphere_i: Sphere, sphere_j: Sphere, *args):
        Rij, Rij_hat, ai, aj = self._2spheres_grpy_quantities(sphere_i, sphere_j, *args)

        a_inf = jnp.minimum(ai, aj)
        a_sup = jnp.maximum(ai, aj)

        # Use JAX `lax.cond()` for branching logic
        def case_far(_):
            """Case when Rij > ai + aj"""
            prefactor = 1 / (8 * jnp.pi * Rij**2)
            return prefactor

        def case_medium(_):
            """Case when a_sup - a_inf < Rij <= ai + aj"""
            prefactor = (
                ((ai - aj + Rij) ** 2 * (aj**2 + 2 * aj * (ai + Rij) - 3 * (ai - Rij) ** 2))
                / (8 * Rij**2)
                / (16 * jnp.pi * ai**3 * aj)
            )
            return prefactor

        def case_near(_):
            """Case when Rij <= a_sup - a_inf"""
            prefactor = jnp.heaviside(ai - aj, 0) * Rij / (8 * jnp.pi * Rij**2)
            return prefactor

        # Nested conditions
        prefactor = lax.cond(
            Rij > ai + aj, case_far, lambda _: lax.cond(Rij > a_sup - a_inf, case_medium, case_near, None), None
        )

        # Compute the 3x3 matrix
        matrix = prefactor * jnp.cross(-jnp.eye(3), Rij_hat)

        return matrix

    def _compute_mu_td_ij(self, sphere_i: Sphere, sphere_j: Sphere, *args):
        Rij, Rij_hat, ai, aj = self._2spheres_grpy_quantities(sphere_i, sphere_j, *args)

        a_inf = jnp.minimum(ai, aj)
        a_sup = jnp.maximum(ai, aj)

        # Use JAX `lax.cond()` for branching logic
        def case_far(_):
            """Case when Rij > ai + aj"""
            hat1_prefactor = (5.0 * aj / 6) * (-2 * (5 * ai**2 * aj**2 + 3 * aj**4)) / (5 * Rij**4)
            hat3_prefactor = (5.0 * aj / 6) * aj**2 * (5 * ai**2 + 3 * aj**2 - 3 * Rij**2) / (Rij**4)
            return hat1_prefactor, hat3_prefactor

        def case_medium(_):
            """Case when a_sup - a_inf < Rij <= ai + aj"""
            calC = (
                +10.0 * Rij**6 - 24 * Rij**5 * ai - 15 * Rij**4 * (aj**2 - ai**2) + (aj - ai) ** 5 * (ai + 5 * aj)
            ) / (40 * ai * aj * Rij**4)
            calD = (((ai - aj) ** 2 - Rij**2) ** 2 * ((ai - aj) * (ai + 5 * aj) - Rij**2)) / (
                16.0 * ai * aj * Rij**4
            )
            hat1_prefactor = (5.0 * aj / 6) * calC
            hat3_prefactor = (5.0 * aj / 6) * calD
            return hat1_prefactor, hat3_prefactor

        def case_near(_):
            """Case when Rij <= a_sup - a_inf"""
            hat1_prefactor = -jnp.heaviside(aj - ai, 0.0) * Rij
            hat3_prefactor = 0.0
            return hat1_prefactor, hat3_prefactor

        # Nested conditions
        hat1_prefactor, hat3_prefactor = lax.cond(
            Rij > ai + aj, case_far, lambda _: lax.cond(Rij > a_sup - a_inf, case_medium, case_near, None), None
        )

        # Compute the 3x3x3 tensor
        matrix = hat1_prefactor * jnp.outer(jnp.eye(3), Rij_hat).reshape(3, 3, 3) + hat3_prefactor * jnp.outer(
            Rij_hat, jnp.outer(Rij_hat, Rij_hat)
        ).reshape(3, 3, 3)

        return matrix

    def _compute_mu_rd_ij(self, sphere_i: Sphere, sphere_j: Sphere, *args):
        Rij, Rij_hat, ai, aj = self._2spheres_grpy_quantities(sphere_i, sphere_j, *args)

        a_inf = jnp.minimum(ai, aj)
        a_sup = jnp.maximum(ai, aj)

        # Use JAX `lax.cond()` for branching logic
        def case_far(_):
            """Case when Rij > ai + aj"""
            prefactor = -(5.0 / 2) * (aj / Rij) ** 3
            return prefactor

        def case_medium(_):
            calB = (3.0 * ((ai - aj) ** 2 - Rij**2) ** 2 * (ai**2 + 4 * aj * ai + aj**2 - Rij**2)) / (64 * Rij**3)
            prefactor = -5 * calB / (3 * ai**3)
            return prefactor

        def case_near(_):
            """Case when Rij <= a_sup - a_inf"""
            prefactor = 0.0
            return prefactor

        # Nested conditions
        prefactor = lax.cond(
            Rij > ai + aj, case_far, lambda _: lax.cond(Rij > a_sup - a_inf, case_medium, case_near, None), None
        )

        # Compute the 3x3 matrix
        matrix = prefactor * jnp.outer(jnp.cross(-jnp.eye(3), Rij_hat), Rij_hat).reshape(3, 3, 3)

        return matrix

    def _individual_composition_of_strain(self, sphere: Sphere, dofs=None, design=None, inputs=None):
        dofs, design = self._setup_params(dofs, design)

        X, Y, Z = sphere.position(dofs, design)
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
            tensors = self.compute_tensors(dofs, design)

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
                    f"  design_defaults = {dict(self.design_defaults)}",
                    UserWarning,
                    stacklevel=2,
                )

        except Exception as e:
            warnings.warn(
                f"Could not validate default geometry: {e}",
                UserWarning,
                stacklevel=2,
            )

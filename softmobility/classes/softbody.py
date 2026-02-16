import jax.numpy as jnp
import jax
from jax import lax
from collections import namedtuple
from .sphere import Sphere
from .sphereassembly import SphereAssembly


class SoftBody(SphereAssembly):

    MobilityResult = namedtuple("FullMobilityResult", ["M", "Mtilde", "G", "V", "P"])
    # FastMobilityResult = namedtuple(
    #     "FastMobilityResult", ["Mk", "Mkmean", "Mkm", "Gk", "Mdof", "Mdofm", "Gdof", "x_cr", "x_cm"]
    # )

    def __init__(self, *args, **kwargs):
        # Call the __init__ method of the parent class (SphereAssembly)
        super().__init__(*args, **kwargs)

    def compute_mobility_problem(self, dofs=None, params=None):
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
        MobilityResult
            A named tuple containing:
            - M (jax.numpy.ndarray): Mobility matrix for forces expressed in the coordinate of assembly.
            - G (jax.numpy.ndarray): Coupling matrix with strain.
            - V (jax.numpy.ndarray): Velocity projection matrix.
            - P (jax.numpy.ndarray): Active velocity projection matrix.

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
        dofs, params = self._setup_params(dofs, params)

        # Compute Jacobian matrix J mapping velocities to dofs and assembly velocities
        J = self.compute_Jacobian_matrix(dofs, params)

        # Compute the resistance tensor R linking forces to velocities
        R = self.compute_resistance_tensor(dofs, params)

        # Compute coupling tensor Gfull linking strain rate to velocities
        Gfull = self._compute_coupling_with_strain(dofs, params)

        # Compute velocity, force, and gravity composition operators from spheres to assembly
        # Tu = self.compute_composition_of_velocity(dofs, params)
        Tf = self.compute_composition_of_forces(dofs, params)
        # Tm = self.compute_composition_of_gravity(dofs, params)

        # Compute reduced mobility matrix Mred = (J R J^T)^-1
        Mred = self._compute_Mred_jitted(jnp.einsum("ji, jk, kl->il", J, R, J))

        # Projection matrix Proj = Mred J R
        P = jnp.einsum("ij, kj, kl->il", Mred, J, R)

        # Compute velocity projection matrix V
        # Vverif = jnp.einsum("ij, jk->ik", Proj, Tu)
        V = jnp.vstack([jnp.zeros((J.shape[1] - 6, 6)), jnp.eye(6)])

        # Sanity check to ensure V is correct
        # assert jnp.allclose(V, Vverif, atol=1e-5), "Matrix V is not correct!"

        # Compute mobility matrices
        M = jnp.einsum("ij, kj, kl->il", Mred, J, Tf)  # Mobility matrix for forces
        Mtilde = jnp.einsum("ij, kj->ik", Mred, J)  # Mobility matrix in the frame of spheres
        # Mm = jnp.einsum("ij, kj, kl->il", Mred, J, Tm)  # Reduced Mobility matrix for gravity
        G = jnp.einsum("ij, jk->ik", P, Gfull)  # Coupling with strain

        return self.MobilityResult(M, Mtilde, G, V, P)

    # def compute_fast_mobility_problem(self, dofs=None, params=None):
    #     """
    #     Compute the fast mobility problem for a given system configuration.

    #     This method derives an approximation of the mobility problem by leveraging the fast
    #     dynamics of the degrees of freedom relative to the backgound flow unsteadiness. It
    #     projects mobility quantities onto a lower-dimensional space to accelerate computations.

    #     Parameters
    #     ----------
    #     dofs : list or array, optional
    #         Degrees of freedom of the system.
    #     params : list or array, optional
    #         Parameters defining system properties.

    #     Returns
    #     -------
    #     FastMobilityResult
    #         A named tuple containing:
    #         - Mk (jax.numpy.ndarray): Projected mobility matrix.
    #         - Mkmean (jax.numpy.ndarray): Mean projected mobility matrix.
    #         - Mkm (jax.numpy.ndarray): Projected mobility matrix for gravity-like forces.
    #         - Gk (jax.numpy.ndarray): Projected coupling matrix with strain.
    #         - Mdof (jax.numpy.ndarray): Mobility matrix projected onto degrees of freedom.
    #         - Mdofm (jax.numpy.ndarray): Mobility matrix projected onto gravity-like forces.
    #         - Gdof (jax.numpy.ndarray): Coupling matrix projected onto degrees of freedom.
    #         - x_cr (jax.numpy.ndarray): Coordinate of the center of resistance.

    #     Examples
    #     --------
    #     Unpacking using a tuple:

    #     >>> Mk, _, Mkm, *_ = soft_plankton.compute_fast_mobility_problem(dofs=[0, 1])
    #     >>> print(Mk, Mkm)

    #     Unpacking using the named tuple (preferred method):

    #     >>> matrices = soft_plankton.compute_fast_mobility_problem()
    #     >>> print(matrices.Mk, matrices.Mkm)

    #     Notes
    #     -----
    #     - This function builds upon `compute_mobility_problem` to derive an optimized mobility problem.
    #     - Ensure that all input lists/arrays are compatible with JAX operations.
    #     - The method assumes a resistance-mobility-stiffness formulation.
    #     """
    #     # Handling dofs and params, passing default value if None are given
    #     dofs, params = self._setup_params(dofs, params)

    #     # Compute full mobility matrices
    #     M, _, Mm, G, V, *_ = self.compute_mobility_problem(dofs, params)

    #     # Compute stiffness matrix
    #     K = self.compute_stiffness_matrix(dofs, params)

    #     # Projection matrices mapping to degrees of freedom (I1) and rigid-body motion (I2)
    #     I1 = jnp.hstack([jnp.eye(self.Ndof), jnp.zeros((self.Ndof, 6))])
    #     I2 = jnp.hstack([jnp.zeros((6, self.Ndof)), jnp.eye(6)])

    #     # Compute inverse projected stiffness-weighted mobility matrix: Rk = (I1 M K)^-1
    #     Rk = jnp.linalg.inv(jnp.einsum("ij, jk, kl->il", I1, M, K))

    #     # Compute matrix D used for projection: D= M K Rk I1
    #     D = jnp.einsum("ij, jk, kl, lm->im", M, K, Rk, I1)

    #     # Projection operator P = I2 (I - D)
    #     P = jnp.einsum("ij, jk->ik", I2, jnp.eye(self.Ndof + 6) - D)

    #     # Compute projected velocity matrix Vk
    #     Vk = jnp.einsum("ij, jk->ik", P, V)

    #     # Sanity check: Vk should be an 6x6 identity matrix
    #     assert jnp.allclose(Vk, jnp.eye(6), atol=1e-5), "Matrix Vk is not correct!"

    #     # Compute projected mobility matrices
    #     Mk = jnp.einsum("ij, jk->ik", P, M)  # Projected mobility matrix
    #     Mkmean = self._compute_mean(Mk)  # Mean projected mobility matrix
    #     # Mkm = jnp.einsum("ij, jk->ik", P, Mm)  # Projected gravity mobility matrix
    #     Gk = jnp.einsum("ij, jk->ik", P, G)  # Projected coupling matrix

    #     # Compute mobility matrices in degrees of freedom space
    #     Mdof = jnp.einsum("ij, jk, kl->il", -Rk, I1, M)
    #     # Mdofm = jnp.einsum("ij, jk, kl->il", -Rk, I1, Mm)
    #     Gdof = jnp.einsum("ij, jk, kl->il", -Rk, I1, G)

    #     x_cr, x_cm = self._compute_hydrodynamic_centers(Mkmean)

    #     return self.FastMobilityResult(Mk, Mkmean, 0, Gk, Mdof, 0, Gdof, x_cr, x_cm)

    def compute_mobility_tensor(self, dofs=None, params=None):
        dofs, params = self._setup_params(dofs, params)

        # Function to compute diagonal blocks (i == j)
        def compute_diag_block(i):
            mu_tt = self._compute_mu_tt_ii(self.spheres[i], dofs, params)
            mu_rr = self._compute_mu_rr_ii(self.spheres[i], dofs, params)
            mu_rt = jnp.zeros((3, 3))
            mu_tr = jnp.zeros((3, 3))
            return jnp.block([[mu_tt, mu_tr], [mu_rt, mu_rr]])

        # Function to compute off-diagonal blocks (i ≠ j)
        def compute_off_diag_block(i, j):
            mu_tt = self._compute_mu_tt_ij(self.spheres[i], self.spheres[j], dofs, params)
            mu_rr = self._compute_mu_rr_ij(self.spheres[i], self.spheres[j], dofs, params)
            mu_rt = self._compute_mu_rt_ij(self.spheres[i], self.spheres[j], dofs, params)
            mu_tr = self._compute_mu_rt_ij(self.spheres[j], self.spheres[i], dofs, params).T
            return jnp.block([[mu_tt, mu_tr], [mu_rt, mu_rr]])

        # Function to select correct block using lax.cond
        def compute_block(i, j):
            return lax.cond(
                i == j, lambda _: compute_diag_block(i), lambda _: compute_off_diag_block(i, j), operand=None
            )

        # Assemble full mobility matrix
        M = jnp.block([[compute_block(i, j) for j in range(self.Nspheres)] for i in range(self.Nspheres)])

        return M

    def compute_resistance_tensor(self, dofs, params):
        M = self.compute_mobility_tensor(dofs, params)
        return self._compute_resistance_tensor_jitted(M)

    @staticmethod
    @jax.jit
    def _compute_resistance_tensor_jitted(M):
        return jnp.linalg.inv(M)

    @staticmethod
    @jax.jit
    def _compute_Mred_jitted(M):
        return jnp.linalg.inv(M)

    def compute_mobility_tensor_alt(self, dofs=None, params=None):
        dofs, params = self._setup_params(dofs, params)
        # Define the blocks functions
        mu_tt = jnp.block(
            [
                [
                    self._compute_mu_tt_ij(self.spheres[i], self.spheres[j], dofs, params)
                    for j in range(self.Nspheres)
                ]
                for i in range(self.Nspheres)
            ]
        )
        mu_rr = jnp.block(
            [
                [
                    self._compute_mu_rr_ij(self.spheres[i], self.spheres[j], dofs, params)
                    for j in range(self.Nspheres)
                ]
                for i in range(self.Nspheres)
            ]
        )
        mu_rt = jnp.block(
            [
                [
                    self._compute_mu_rt_ij(self.spheres[i], self.spheres[j], dofs, params)
                    for j in range(self.Nspheres)
                ]
                for i in range(self.Nspheres)
            ]
        )
        mu_tr = jnp.block(
            [
                [
                    self._compute_mu_rt_ij(self.spheres[j], self.spheres[i], dofs, params).T
                    for j in range(self.Nspheres)
                ]
                for i in range(self.Nspheres)
            ]
        )
        # Assemble the blocks into a full matrix
        M = jnp.block(jnp.block([[mu_tt, mu_tr], [mu_rt, mu_rr]]))

        return M

    # def _compute_mean(self, M: jnp.ndarray) -> jnp.ndarray:
    #     """
    #     Compute the mean of Nspheres blocks extracted from the input array M.

    #     Args:
    #         M (jnp.ndarray): A 2D Jax numpy array representing concatenated data for multiple spheres. Each sphere's data is a 6x6 matrix, so M has shape (36 * Nspheres,).

    #     Returns:
    #         jnp.ndarray: A 2D Jax numpy array of shape (6, 6) which is the element-wise mean of all blocks.

    #     Note:
    #         - The function reshapes M into a 3D array with dimensions (Nspheres, 6, 6), where each slice along the first dimension represents a sphere's data matrix.
    #         - It checks if all spheres' data matrices are nearly identical within an absolute tolerance of 1e-5. If not, it prints a warning message.
    #     """
    #     N = self.Nspheres
    #     # Reshape M into a 3D array where each element is a 6x6 matrix representing one sphere's data
    #     M_blocks = M.reshape(6, 6 * N).T.reshape(N, 6, 6)

    #     # Check if all blocks are close to the first block with tolerance 1e-5
    #     warning_printed = False
    #     for i in range(1, N):
    #         if not jnp.isclose(M_blocks[i], M_blocks[0], atol=1e-5).all():
    #             warning_printed = True
    #             break

    #     # Print a warning if any block differs significantly from the first one
    #     if warning_printed:
    #         print("Warning: Matrix blocks of Mk are different (absolute tol 1e-5)")

    #     # Compute the mean matrix by summing all blocks and dividing by Nspheres (implicitly through summation along axis=0)
    #     M_mean = jnp.mean(M_blocks, axis=0)

    #     return M_mean

    def _compute_composition_of_strain(self, *args):
        T = jnp.block(
            [[self._individual_composition_of_strain(self.spheres[i], *args)] for i in range(self.Nspheres)]
        )

        return T

    def _compute_coupling_with_strain(self, *args):
        Svecmat = jnp.array(
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
        GE = jnp.concatenate(tensor_blocks, axis=0)

        # Perform tensor contraction with Svecmat (if needed)
        G = self._compute_composition_of_strain(*args) + jnp.einsum(
            "ijk,jkl->il", GE, Svecmat
        )  # Adjust contraction as needed

        return G

    # def _compute_hydrodynamic_centers(self, mobility_matrix: jnp.ndarray) -> jnp.ndarray:
    #     if mobility_matrix.shape != (6, 6):
    #         raise ValueError(f"Mobility matrix must have shape (6, 6), but got {mobility_matrix.shape}.")

    #     # Tensors to compute the mobility center
    #     b = mobility_matrix[3:, :3]
    #     c = mobility_matrix[3:, 3:]

    #     # Tensors to compute the resistance center
    #     resistance_matrix = jnp.linalg.inv(mobility_matrix)
    #     A = resistance_matrix[:3, :3]
    #     B = resistance_matrix[3:, :3]

    #     levicivita = jnp.array(
    #         [
    #             [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    #             [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    #             [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #         ]
    #     )

    #     # From Kim & Karila 5.2.2 and 5.3.2
    #     traAmAinv = jnp.linalg.inv(jnp.trace(A) * jnp.eye(3) - A)
    #     x_cr = 0.5 * jnp.einsum("ij,jkl,kl->i", traAmAinv, levicivita, B - B.transpose())
    #     trcmcinv = jnp.linalg.inv(c - jnp.trace(c) * jnp.eye(3))
    #     x_cm = 0.5 * jnp.einsum("ij,jkl,kl->i", trcmcinv, levicivita, b - b.transpose())

    #     return x_cr, x_cm

    # Functions to compute the GRPY tensors #######################################
    def _1sphere_GRPY_quantities(self, sphere: Sphere, dofs=None, params=None):
        dofs, params = self._setup_params(dofs, params)

        return sphere.radius(dofs, params)

    def _2spheres_GRPY_quantities(self, sphere_i: Sphere, sphere_j: Sphere, dofs=None, params=None):
        dofs, params = self._setup_params(dofs, params)

        position_i = sphere_i.position(dofs, params)
        position_j = sphere_j.position(dofs, params)
        radius_i = sphere_i.radius(dofs, params)
        radius_j = sphere_j.radius(dofs, params)
        posi = jnp.array(position_i)
        posj = jnp.array(position_j)
        rvector = posi - posj
        rnorm = jnp.linalg.norm(rvector) + 1e-12  # avoid division by zero
        runit = rvector / rnorm

        return rnorm, runit, radius_i, radius_j

    def _compute_mu_tt_ii(self, sphere, *args):
        a_i = self._1sphere_GRPY_quantities(sphere, *args)
        matrix = 1 / (6 * jnp.pi * a_i) * jnp.eye(3)

        return matrix

    def _compute_mu_rr_ii(self, sphere, *args):
        a_i = self._1sphere_GRPY_quantities(sphere, *args)
        matrix = 1 / (8 * jnp.pi * a_i**3) * jnp.eye(3)

        return matrix

    def _compute_mu_tt_ij(self, sphere_i, sphere_j, *args):
        Rij, Rij_hat, ai, aj = self._2spheres_GRPY_quantities(sphere_i, sphere_j, *args)

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
        Rij, Rij_hat, ai, aj = self._2spheres_GRPY_quantities(sphere_i, sphere_j, *args)

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
        Rij, Rij_hat, ai, aj = self._2spheres_GRPY_quantities(sphere_i, sphere_j, *args)

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
        Rij, Rij_hat, ai, aj = self._2spheres_GRPY_quantities(sphere_i, sphere_j, *args)

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
        Rij, Rij_hat, ai, aj = self._2spheres_GRPY_quantities(sphere_i, sphere_j, *args)

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

    def _individual_composition_of_strain(self, sphere: Sphere, dofs=None, params=None):
        dofs, params = self._setup_params(dofs, params)

        X, Y, Z = sphere.position(dofs, params)
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

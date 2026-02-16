import jax.numpy as jnp
import jax
from jax import lax

from softmobility import SoftBody, Flow


class FlowBody:
    def __init__(
        self,
        soft_body: SoftBody,
        flow: Flow,
        init_position=[0, 0, 0],
        init_orientation=[0, 0, 0],
        dt=0.01,
        integrator="RK2",
    ):
        """
        Solver for fluid-structure interaction.

        Parameters:
        - soft_plankton: SoftBody object
        - flow: A Flow object
        - init_position: a 3D list or array (default [0, 0, 0])
        - init_orientation: a 3D list or array (default [0, 0, 0])
        - dt: Time step for integration (default 0.01)
        - integrator: str, one of: "Euler", "RK2", or "RK4" (default "RK2")
        """
        self.soft_plankton = soft_body
        self.flow = flow
        self.dof = self.soft_plankton.dof_defaults  # Degrees of freedom
        self.position = jnp.array(init_position)
        self.orientation = jnp.array(init_orientation)
        self.dt = dt
        self.integrator = integrator
        self.trajectory = [[self.dof, self.position, self.orientation]]
        self.time = 0.0
        self.compute_fast_mobility = jax.jit(self._compute_fast_mobility)
        self.compute_grand_velocity = jax.jit(self._compute_grand_velocity)

    def _compute_fast_mobility(self, dofs):
        """JIT-compiled function for computing mobility."""
        params = self.soft_plankton.param_defaults
        return self.soft_plankton.compute_mobility_problem(dofs, params)

    def grand_velocity(self):
        """Compute grand velocity of the plankton."""
        # Extract state variables
        position = self.position
        orientation = self.orientation
        dof = self.dof

        # Compute the flow at plankton position (in the lab framework)
        v_lab = self.flow.velocity(position)
        omega_lab, s_lab = self.flow.omega_s(position)

        return self.compute_grand_velocity(orientation, dof, v_lab, omega_lab, s_lab)

    def _compute_grand_velocity(self, orientation, dof, v_lab, omega_lab, s_lab):
        # Compute rotation matrices
        R, grand_R = rotation_matrix_from_Rodrigues(orientation, Ndof=self.soft_plankton.Ndof)

        # Compute flow velocities in the particle framework
        v_inf = R.T @ v_lab
        omega_inf = R.T @ omega_lab
        u_inf = jnp.block([v_inf, omega_inf])

        # Compute rate-of-strain in the particle framework
        s_part = R.T @ s_lab @ R
        s_inf = jnp.array([s_part[0, 0], s_part[0, 1], s_part[0, 2], s_part[1, 1], s_part[1, 2]])

        matrices = self.compute_fast_mobility(dof)
        Mtilde = matrices.Mtilde
        Mm = matrices.Mm
        G = matrices.G
        V = matrices.V

        # Extract elastic forces/torques
        F_elastic = self.soft_plankton.grand_forces_func(dofs=dof)

        # Compute soft mobility equation
        gravity = jnp.array([0, 0, -1])
        grand_velocity = Mtilde @ F_elastic + Mm @ gravity + V @ u_inf + G @ s_inf

        # Compute velocities in lab framework
        grand_velocity_lab = grand_R @ grand_velocity

        return grand_velocity_lab

    def integrate_euler(self):
        """Euler first-order integration."""

        Qdot = self.grand_velocity()

        self.dof += self.dt * Qdot[:-6]
        self.position += self.dt * Qdot[-6:-3]
        self.orientation += self.dt * dot_orientation(self.orientation, Qdot[-3:])
        self.orientation = rescale_orientation(self.orientation)

        self.time += self.dt
        self.trajectory.append([self.dof, self.position, self.orientation])

    def integrate_rk2(self):
        """Runge-Kutta 2nd order (Midpoint) integration."""

        # Step 1: Store current state
        current_dof = self.dof.copy()
        current_position = self.position.copy()
        current_orientation = self.orientation.copy()

        # First step (calculate k1)
        Qdot = self.grand_velocity()  # Compute the velocity

        k1_dof = Qdot[:-6]
        k1_position = Qdot[-6:-3]
        k1_orientation = Qdot[-3:]

        # Second step (calculate k2), calculate the mid-point velocity (using the current values)
        self.dof = current_dof + self.dt * k1_dof / 2
        self.position = current_position + self.dt * k1_position / 2
        self.orientation = current_orientation + self.dt * k1_orientation / 2

        # Compute Qdot at the midpoint
        Qdot_mid = self.grand_velocity()  # Compute midpoint velocity
        k2_dof = Qdot_mid[:-6]
        k2_position = Qdot_mid[-6:-3]
        k2_orientation = Qdot_mid[-3:]

        # Update the state with weighted sum of k1 and k2
        self.dof = current_dof + self.dt * (k1_dof + k2_dof) / 2
        self.position = current_position + self.dt * (k1_position + k2_position) / 2
        self.orientation = current_orientation + self.dt * (k1_orientation + k2_orientation) / 2
        self.orientation = rescale_orientation(self.orientation)  # Ensure orientation is valid

        # Update time and store trajectory
        self.time += self.dt
        self.trajectory.append([self.dof, self.position, self.orientation])

    def integrate_rk4(self):
        """Runge-Kutta 4th order integration."""

        # Step 1: Store current state
        current_dof = self.dof.copy()
        current_position = self.position.copy()
        current_orientation = self.orientation.copy()

        # First step (calculate k1)
        Qdot = self.grand_velocity()  # Compute the velocity

        k1_dof = Qdot[:-6]
        k1_position = Qdot[-6:-3]
        k1_orientation = Qdot[-3:]

        # Second step (calculate k2)
        self.dof = current_dof + self.dt * k1_dof / 2
        self.position = current_position + self.dt * k1_position / 2
        self.orientation = current_orientation + self.dt * k1_orientation / 2
        Qdot_k2 = self.grand_velocity()
        k2_dof = Qdot_k2[:-6]
        k2_position = Qdot_k2[-6:-3]
        k2_orientation = Qdot_k2[-3:]

        # Third step (calculate k3)
        self.dof = current_dof + self.dt * k2_dof / 2
        self.position = current_position + self.dt * k2_position / 2
        self.orientation = current_orientation + self.dt * k2_orientation / 2
        Qdot_k3 = self.grand_velocity()
        k3_dof = Qdot_k3[:-6]
        k3_position = Qdot_k3[-6:-3]
        k3_orientation = Qdot_k3[-3:]

        # Fourth step (calculate k4)
        self.dof = current_dof + self.dt * k3_dof
        self.position = current_position + self.dt * k3_position
        self.orientation = current_orientation + self.dt * k3_orientation
        Qdot_k4 = self.grand_velocity()
        k4_dof = Qdot_k4[:-6]
        k4_position = Qdot_k4[-6:-3]
        k4_orientation = Qdot_k4[-3:]

        # Update the state with weighted sum of k1, k2, k3, k4
        self.dof = current_dof + self.dt * (k1_dof + 2 * k2_dof + 2 * k3_dof + k4_dof) / 6
        self.position = (
            current_position + self.dt * (k1_position + 2 * k2_position + 2 * k3_position + k4_position) / 6
        )
        self.orientation = (
            current_orientation
            + self.dt * (k1_orientation + 2 * k2_orientation + 2 * k3_orientation + k4_orientation) / 6
        )
        self.orientation = rescale_orientation(self.orientation)  # Ensure orientation is valid

        # Update time and store trajectory
        self.time += self.dt
        self.trajectory.append([self.dof, self.position, self.orientation])

    def simulate(self, T):
        """Run the simulation for time T."""
        num_steps = int(T / self.dt)
        for t in range(num_steps):
            if t % 100 == 0:
                print(f"Time: {self.time:.3f} / {T:.3f}  Integrator {self.integrator}")
            if self.integrator == "Euler":
                self.integrate_euler()
            elif self.integrator == "RK2":
                self.integrate_rk2()
            else:
                self.integrate_rk4()

        return self.trajectory


# Useful functions for rotation with rotation vector ########################
@jax.jit
def rescale_orientation(rvec):
    """
    Rescale the orientation vector to avoid singularities.
    """
    rvec = jnp.array(rvec)
    r = jnp.linalg.norm(rvec)

    def rescale(_):
        return rvec - 2 * jnp.pi * rvec / r

    return lax.cond(r >= jnp.pi, rescale, lambda _: rvec, None)


@jax.jit
def dot_orientation(rvec, omega):
    """
    Compute the time derivative of the orientation vector using the Bortz formula.
    """
    rvec = jnp.array(rvec)
    r = jnp.linalg.norm(rvec)

    def small_r_case(_):
        """Return omega directly if r is small."""
        return omega

    def normal_case(_):
        """Compute Bortz derivative normally."""
        runit = rvec / r
        term1 = omega
        term2 = 0.5 * jnp.cross(runit, omega)
        correction_factor = (1 / r**2) * (1 - (r / 2) / jnp.tan(r / 2))
        term3 = correction_factor * jnp.cross(runit, jnp.cross(runit, omega))
        return term1 + term2 + term3

    return lax.cond(r < 1e-6, small_r_case, normal_case, None)


rotation_matrix_from_Rodrigues = jax.jit(
    lambda rvec, Ndof: _rotation_matrix_from_Rodrigues_impl(rvec, Ndof), static_argnums=(1,)
)


def _rotation_matrix_from_Rodrigues_impl(rvec, Ndof):
    """
    Rotation matrix from rotation vector r using Rodrigues' rotation formula.
    """
    rvec = jnp.array(rvec)
    theta = jnp.linalg.norm(rvec)

    def no_rotation(_):
        """Return identity matrix when theta is very small."""
        R = jnp.eye(3)
        grand_R = jnp.eye(Ndof + 6)
        return R, grand_R

    def compute_rotation(_):
        """Compute Rodrigues' rotation matrix."""
        k = rvec / theta
        kx, ky, kz = k
        K = jnp.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        R = jnp.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * jnp.dot(K, K)

        grand_R = jnp.block(
            [
                [jnp.eye(Ndof), jnp.zeros((Ndof, 6))],
                [jnp.zeros((3, Ndof)), R, jnp.zeros((3, 3))],
                [jnp.zeros((3, Ndof)), jnp.zeros((3, 3)), R],
            ]
        )
        return R, grand_R

    return lax.cond(theta < 1e-6, no_rotation, compute_rotation, None)

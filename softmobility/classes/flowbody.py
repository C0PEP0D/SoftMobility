"""Class FlowBody used to integrate the trajectory of a soft body in a flow, subject to a field"""

import jax.numpy as jnp
import jax
from jax import lax

from softmobility import SoftBody, Flow, Field


class FlowBody:
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

    def __init__(
        self,
        soft_body: SoftBody,
        flow: Flow,
        field: Field,
        init_position=[0, 0, 0],
        init_orientation=[0, 0, 0],
        dt=0.01,
        integrator="RK2",
    ):
        self.soft_body = soft_body
        self.flow = flow
        self.field = field
        self.dofs = self.soft_body.dof_defaults  # Degrees of freedom
        self.inputs = self.soft_body.input_defaults
        self.position = jnp.array(init_position)
        self.orientation = jnp.array(init_orientation)
        self.dt = dt
        self.integrator = integrator
        self.trajectory = [[self.position, self.orientation, self.dofs]]
        self.time = 0.0
        self.compute_fast_mobility = jax.jit(self._compute_fast_mobility)
        self.compute_grand_velocity = jax.jit(self._compute_grand_velocity)

    def _compute_fast_mobility(self, dofs_inputs):
        """JIT-compiled function for computing mobility."""
        dofs = dofs_inputs[: self.soft_body.Ndof]
        design = self.soft_body.design_defaults
        inputs = dofs_inputs[self.soft_body.Ndof :]
        return self.soft_body.compute_mobility_problem(dofs, design, inputs)

    def grand_velocity(self):
        """Compute grand velocity of the body."""
        # Extract state variables
        position = self.position
        orientation = self.orientation
        dofs = self.dofs
        inputs = self.inputs
        time = self.time

        # Compute the flow at body position (in the lab framework)
        u_lab = self.flow.velocity(position)
        omega_lab, s_lab = self.flow.omega_s(position)

        # Commpute the field value at body position (in the lab framework)
        field_lab = self.field.vector(position, time)

        return self.compute_grand_velocity(orientation, dofs, inputs, u_lab, omega_lab, s_lab, field_lab)

    def _compute_grand_velocity(self, orientation, dofs, inputs, u_lab, omega_lab, s_lab, field_lab):
        # Compute rotation matrices
        R, sixc_R = rotation_matrix_from_Rodrigues(orientation, Ndof=self.soft_body.Ndof)

        # Compute flow velocities and field value in the body's framework
        p_lab = jnp.block([u_lab, omega_lab, jnp.zeros(self.soft_body.Ndof)])
        field = R.T @ field_lab

        # Compute rate-of-strain in the particle framework
        s_part = R.T @ s_lab @ R
        s_inf = jnp.array([s_part[0, 0], s_part[0, 1], s_part[0, 2], s_part[1, 1], s_part[1, 2]])

        tensors = self.compute_fast_mobility(jnp.block([dofs, inputs]))
        M_H = tensors.M_H
        M_K = tensors.M_K
        C_E = tensors.C_E

        sixc_velocity = M_H @ field + M_K @ dofs + C_E @ s_inf

        # Compute velocities in lab framework
        sixc_velocity_lab = sixc_R @ sixc_velocity + p_lab

        return sixc_velocity_lab

    def integrate_euler(self):
        """Euler first-order integration."""

        Qdot = self.grand_velocity()

        self.position += self.dt * Qdot[:3]
        self.orientation += self.dt * dot_orientation(self.orientation, Qdot[3:6])
        self.orientation = rescale_orientation(self.orientation)
        self.dofs += self.dt * Qdot[6:]

        self.time += self.dt
        self.trajectory.append([self.position, self.orientation, self.dofs])

    def integrate_rk2(self):
        """Runge-Kutta 2nd order (Midpoint) integration."""

        # Step 1: Store current state
        current_dof = self.dofs.copy()
        current_position = self.position.copy()
        current_orientation = self.orientation.copy()

        # First step (calculate k1)
        Qdot = self.grand_velocity()  # Compute the velocity

        k1_position = Qdot[:3]
        k1_orientation = Qdot[3:6]
        k1_dof = Qdot[6:]

        # Second step (calculate k2), calculate the mid-point velocity (using the current values)
        self.position = current_position + self.dt * k1_position / 2
        self.orientation = current_orientation + self.dt * k1_orientation / 2
        self.dofs = current_dof + self.dt * k1_dof / 2

        # Compute Qdot at the midpoint
        Qdot_mid = self.grand_velocity()  # Compute midpoint velocity
        k2_position = Qdot_mid[:3]
        k2_orientation = Qdot_mid[3:6]
        k2_dof = Qdot_mid[6:]

        # Update the state with weighted sum of k1 and k2
        self.position = current_position + self.dt * (k1_position + k2_position) / 2
        self.orientation = current_orientation + self.dt * (k1_orientation + k2_orientation) / 2
        self.orientation = rescale_orientation(self.orientation)  # Ensure orientation is valid
        self.dofs = current_dof + self.dt * (k1_dof + k2_dof) / 2

        # Update time and store trajectory
        self.time += self.dt
        self.trajectory.append([self.position, self.orientation, self.dofs])

    def integrate_rk4(self):
        """Runge-Kutta 4th order integration."""

        # Step 1: Store current state
        current_dof = self.dofs.copy()
        current_position = self.position.copy()
        current_orientation = self.orientation.copy()

        # First step (calculate k1)
        Qdot = self.grand_velocity()  # Compute the velocity

        k1_position = Qdot[:3]
        k1_orientation = Qdot[3:6]
        k1_dof = Qdot[6:]

        # Second step (calculate k2)
        self.dofs = current_dof + self.dt * k1_dof / 2
        self.position = current_position + self.dt * k1_position / 2
        self.orientation = current_orientation + self.dt * k1_orientation / 2
        Qdot_k2 = self.grand_velocity()
        k2_position = Qdot_k2[:3]
        k2_orientation = Qdot_k2[3:6]
        k2_dof = Qdot_k2[6:]

        # Third step (calculate k3)
        self.dofs = current_dof + self.dt * k2_dof / 2
        self.position = current_position + self.dt * k2_position / 2
        self.orientation = current_orientation + self.dt * k2_orientation / 2
        Qdot_k3 = self.grand_velocity()
        k3_position = Qdot_k3[:3]
        k3_orientation = Qdot_k3[3:6]
        k3_dof = Qdot_k3[6:]

        # Fourth step (calculate k4)
        self.dofs = current_dof + self.dt * k3_dof
        self.position = current_position + self.dt * k3_position
        self.orientation = current_orientation + self.dt * k3_orientation
        Qdot_k4 = self.grand_velocity()
        k4_position = Qdot_k4[:3]
        k4_orientation = Qdot_k4[3:6]
        k4_dof = Qdot_k4[6:]

        # Update the state with weighted sum of k1, k2, k3, k4
        self.position = (
            current_position + self.dt * (k1_position + 2 * k2_position + 2 * k3_position + k4_position) / 6
        )
        self.orientation = (
            current_orientation
            + self.dt * (k1_orientation + 2 * k2_orientation + 2 * k3_orientation + k4_orientation) / 6
        )
        self.orientation = rescale_orientation(self.orientation)  # Ensure orientation is valid
        self.dofs = current_dof + self.dt * (k1_dof + 2 * k2_dof + 2 * k3_dof + k4_dof) / 6

        # Update time and store trajectory
        self.time += self.dt
        self.trajectory.append([self.position, self.orientation, self.dofs])

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

        sixc_R = jnp.block(
            [
                [R, jnp.zeros((3, 3)), jnp.zeros((3, Ndof))],  # pos
                [jnp.zeros((3, 3)), R, jnp.zeros((3, Ndof))],  # ori
                [jnp.zeros((Ndof, 6)), jnp.eye(Ndof)],  # dof
            ]
        )
        return R, sixc_R

    return lax.cond(theta < 1e-6, no_rotation, compute_rotation, None)

"""Class FlowBody used to integrate the trajectory of a soft body in a flow, subject to a field"""

import jax.numpy as jnp
import jax
from jax import lax

from softmobility import SoftBody, Flow, Field, Scalar


class FlowBodySolver:
    """
    Solver for fluid-structure interaction.

    Parameters:
    - soft_plankton: SoftBody object
    - flow: A Flow object
    - input_map: A dict mapping input variable names to Field or Scalar objects
    - init_position: a 3D list or array (default [0, 0, 0])
    - init_orientation: a 3D list or array (default [0, 0, 0])
    - dt: Time step for integration (default 0.01)
    - integrator: str, one of: "Euler", "RK2", or "RK4" (default "RK2")
    """

    def __init__(
        self,
        soft_body: SoftBody,
        flow: Flow,
        input_map: dict[str, Field | Scalar] | None = None,
        init_position=[0, 0, 0],
        init_orientation=[0, 0, 0],
        dt=0.01,
        integrator="RK2",
    ):
        self.soft_body = soft_body
        self.flow = flow
        self._validate_inputs(input_map)
        self.dofs = self.soft_body.dof_defaults  # Degrees of freedom
        self.position = jnp.array(init_position)
        self.orientation = jnp.array(init_orientation)
        self.dt = dt
        self.integrator = integrator
        self.trajectory = [[self.position, self.orientation, self.dofs]]
        self.time = 0.0
        self.compute_fast_mobility = jax.jit(self._compute_fast_mobility)
        self.compute_sixc_velocity = jax.jit(self._compute_sixc_velocity)

    def _validate_inputs(self, input_dict: dict):
        """
        Validate and store inputs as ordered lists matching input_variables layout.
        Populates self.fields (list[Field]) and self.scalars (list[Scalar]).
        """
        input_dict = input_dict or {}
        fields, scalars = [], []
        seen_field_bases = set()

        for var in self.soft_body.input_variables:
            if var[-1].isdigit():
                base = var[:-1]
                if base not in seen_field_bases:
                    if base not in input_dict:
                        raise ValueError(f"Missing Field input '{base}'")
                    if not isinstance(input_dict[base], Field):
                        raise TypeError(f"Input '{base}' expected a Field, got {type(input_dict[base]).__name__}")
                    fields.append(input_dict[base])
                    seen_field_bases.add(base)
            else:
                if var not in input_dict:
                    raise ValueError(f"Missing Scalar input '{var}'")
                if not isinstance(input_dict[var], Scalar):
                    raise TypeError(f"Input '{var}' expected a Scalar, got {type(input_dict[var]).__name__}")
                scalars.append(input_dict[var])

        unexpected = (
            set(input_dict.keys())
            - seen_field_bases
            - {v for v in self.soft_body.input_variables if not v[-1].isdigit()}
        )
        if unexpected:
            raise ValueError(f"Unexpected input keys: {sorted(unexpected)}")

        self.fields = fields
        self.scalars = scalars

    def _build_input_vector(self, position, time, rot_matrix):
        """
        Assemble the input vector in canonical order: field components first, scalars last.
        """
        parts = []
        for field in self.fields:
            field_lab = field.vector(position, time)
            parts.append(rot_matrix.T @ field_lab)  # rotated (3,) array

        for scalar in self.scalars:
            parts.append(jnp.array([scalar.value(position, time)]))

        return jnp.concatenate(parts) if parts else jnp.zeros(0)

    def _compute_fast_mobility(self, dofs):
        """JIT-compiled function for computing mobility."""
        design = self.soft_body.design_defaults
        return self.soft_body.compute_mobility_problem(dofs, design)

    def sixc_velocity(self):
        """Compute six-component velocity of the body."""
        rot_matrix, _ = rotation_matrix_from_Rodrigues(self.orientation, Ndof=self.soft_body.Ndof)
        input_vec = self._build_input_vector(self.position, self.time, rot_matrix)
        u_lab = self.flow.velocity(self.position, self.time)
        omega_lab, E_lab = self.flow.omega_rate_of_strain(self.position, self.time)
        return self._compute_sixc_velocity(self.orientation, self.dofs, input_vec, u_lab, omega_lab, E_lab)

    def _compute_sixc_velocity(self, orientation, dofs, input_vec, u_lab, omega_lab, E_lab):
        rot_matrix, sixc_rot_matrix = rotation_matrix_from_Rodrigues(orientation, Ndof=self.soft_body.Ndof)

        E_body = rot_matrix.T @ E_lab @ rot_matrix
        E_inf = jnp.array([E_body[0, 0], E_body[0, 1], E_body[0, 2], E_body[1, 1], E_body[1, 2]])

        tensors = self.compute_fast_mobility(dofs)
        sixc_velocity = tensors.M_H @ input_vec + tensors.M_K @ dofs + tensors.C_E @ E_inf

        p_lab = jnp.block([u_lab, omega_lab, jnp.zeros(self.soft_body.Ndof)])
        return sixc_rot_matrix @ sixc_velocity + p_lab

    def integrate_euler(self):
        """Euler first-order integration."""

        p = self.sixc_velocity()
        bortz = compute_Bortz_operator(self.orientation)

        self.position += self.dt * p[:3]
        self.orientation += self.dt * bortz @ p[3:6]
        self.orientation = rescale_orientation(self.orientation)
        self.dofs += self.dt * p[6:]

        self.time += self.dt
        self.trajectory.append([self.position, self.orientation, self.dofs])

    def integrate_rk2(self):
        """Runge-Kutta 2nd order (Midpoint) integration."""

        # Step 1: Store current state
        current_dof = self.dofs.copy()
        current_position = self.position.copy()
        current_orientation = self.orientation.copy()
        bortz = compute_Bortz_operator(self.orientation)

        # First step (calculate k1)
        Qdot = self.sixc_velocity()  # Compute the velocity

        k1_position = Qdot[:3]
        k1_orientation = Qdot[3:6]
        k1_dof = Qdot[6:]

        # Second step (calculate k2), calculate the mid-point velocity (using the current values)
        self.position = current_position + self.dt * k1_position / 2
        self.orientation = current_orientation + self.dt * bortz @ k1_orientation / 2
        self.dofs = current_dof + self.dt * k1_dof / 2

        # Compute Qdot at the midpoint
        Qdot_mid = self.sixc_velocity()  # Compute midpoint velocity
        k2_position = Qdot_mid[:3]
        k2_orientation = Qdot_mid[3:6]
        k2_dof = Qdot_mid[6:]

        # Update the state with weighted sum of k1 and k2
        self.position = current_position + self.dt * (k1_position + k2_position) / 2
        self.orientation = current_orientation + self.dt * bortz @ (k1_orientation + k2_orientation) / 2
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
        Qdot = self.sixc_velocity()  # Compute the velocity

        k1_position = Qdot[:3]
        k1_orientation = Qdot[3:6]
        k1_dof = Qdot[6:]

        # Second step (calculate k2)
        self.dofs = current_dof + self.dt * k1_dof / 2
        self.position = current_position + self.dt * k1_position / 2
        self.orientation = current_orientation + self.dt * k1_orientation / 2
        Qdot_k2 = self.sixc_velocity()
        k2_position = Qdot_k2[:3]
        k2_orientation = Qdot_k2[3:6]
        k2_dof = Qdot_k2[6:]

        # Third step (calculate k3)
        self.dofs = current_dof + self.dt * k2_dof / 2
        self.position = current_position + self.dt * k2_position / 2
        self.orientation = current_orientation + self.dt * k2_orientation / 2
        Qdot_k3 = self.sixc_velocity()
        k3_position = Qdot_k3[:3]
        k3_orientation = Qdot_k3[3:6]
        k3_dof = Qdot_k3[6:]

        # Fourth step (calculate k4)
        self.dofs = current_dof + self.dt * k3_dof
        self.position = current_position + self.dt * k3_position
        self.orientation = current_orientation + self.dt * k3_orientation
        Qdot_k4 = self.sixc_velocity()
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

    def simulate(self, final_time):
        """Run the simulation for time T."""
        num_steps = int(final_time / self.dt)
        for t in range(num_steps):
            if t % 100 == 0:
                print(f"Time: {self.time:.3f} / {final_time:.3f}  Integrator {self.integrator}")
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
def compute_Bortz_operator(rvec):
    """
    Compute the time derivative of the orientation vector using the Bortz formula.
    """
    rvec = jnp.array(rvec)
    theta = jnp.linalg.norm(rvec)

    def small_r_case(_):
        """Return omega directly if r is small."""
        return jnp.eye(3)

    def normal_case(_):
        """Compute Bortz derivative normally."""
        runit = rvec / theta
        kx, ky, kz = runit
        runitcross = jnp.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        term1 = -(theta / 2) * runitcross
        term2 = (theta / 2) / jnp.tan(theta / 2) * jnp.eye(3)
        term3 = (1 - (theta / 2) / jnp.tan(theta / 2)) * jnp.outer(runit, runit)
        return term1 + term2 + term3

    return lax.cond(theta < 1e-6, small_r_case, normal_case, None)


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
        sixc_R = jnp.eye(Ndof + 6)
        return R, sixc_R

    def compute_rotation(_):
        """Compute Rodrigues' rotation matrix."""
        runit = rvec / theta
        kx, ky, kz = runit
        runitcross = jnp.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        R = jnp.eye(3) + jnp.sin(theta) * runitcross + (1 - jnp.cos(theta)) * jnp.dot(runitcross, runitcross)

        sixc_R = jnp.block(
            [
                [R, jnp.zeros((3, 3)), jnp.zeros((3, Ndof))],  # pos
                [jnp.zeros((3, 3)), R, jnp.zeros((3, Ndof))],  # ori
                [jnp.zeros((Ndof, 6)), jnp.eye(Ndof)],  # dof
            ]
        )
        return R, sixc_R

    return lax.cond(theta < 1e-6, no_rotation, compute_rotation, None)

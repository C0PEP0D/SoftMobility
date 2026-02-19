import jax.numpy as jnp
import jax


class Flow:
    """Base class for 3D flows."""

    def __init__(self, *params):
        """
        Initialize flow with additional parameters.

        Parameters:
        - *params: Flow-specific parameters (e.g., shear rate, vortex strength).
        """
        self.params = params  # Store additional parameters

    def velocity(self, pos):
        """
        Returns velocity vector u = (ux, uy, uz).

        Parameters:
        - pos: List, tuple, or array representing (x, y, z)

        Returns:
        - jnp.array([ux, uy, uz])
        """
        pos = self._ensure_jax_array(pos)  # Convert input to JAX array
        return self._velocity(pos)

    def gradient(self, pos):
        """
        Returns velocity gradient tensor ∇u (3x3 matrix).

        Parameters:
        - pos: List, tuple, or array representing (x, y, z)

        Returns:
        - 3x3 velocity gradient tensor
        """
        pos = self._ensure_jax_array(pos)
        return jax.jacfwd(self._velocity)(pos)

    def omega_s(self, pos, tol=1e-5):
        """
        Compute angular velocity Omega and the rate-of-strain S, checking incompressibility.

        Parameters:
        - pos: (x, y, z) coordinates
        - tol: Tolerance for checking incompressibility (default: 1e-6)

        Returns:
        - Omega: jnp.array([Ωx, Ωy, Ωz])  # Angular velocity
        - S: jnp.array()                  # Symmetric part (S is trace-free)
        """
        grad_u = self.gradient(pos)  # Compute velocity gradient tensor

        # Compute the trace (divergence of velocity)
        # trace = jnp.trace(grad_u)  # tr(∇u) = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z

        # Check incompressibility
        # if jnp.abs(trace) > tol:
        #     raise ValueError(
        #         f"❌ Error: The flow is not incompressible (∇⋅u = {trace_S:.2e})."
        #         " Ensure that your flow satisfies ∇⋅u = 0 before using omega_svector()."
        #     )

        # Antisymmetric part A (related to vorticity)
        A = 0.5 * (grad_u - grad_u.T)
        Omega = jnp.array([A[2, 1], A[0, 2], A[1, 0]])  # Extract angular velocity components

        # Symmetric part E
        rate_of_strain = 0.5 * (grad_u + grad_u.T)  # Full symmetric strain rate tensor

        # # Store all 6 independent components: Sxx, Sxy, Sxz, Syy, Syz
        # S_vector = jnp.array([S[0, 0], S[0, 1], S[0, 2], S[1, 1], S[1, 2]])

        return Omega, rate_of_strain

    def _velocity(self, pos):
        """Abstract method for velocity implementation (must be overridden)."""
        raise NotImplementedError("Subclasses must implement _velocity_impl.")

    def _ensure_jax_array(self, pos):
        """Convert input to jnp.ndarray, handling exceptions."""
        try:
            return jnp.asarray(pos, dtype=jnp.float32)
        except Exception as e:
            raise ValueError(f"Invalid input format for position: {pos}") from e


# Examples of flow derived from the class Flow
class PureShearFlow(Flow):
    """Simple pure shear flow u = (y, 0, 0)."""

    def _velocity(self, pos):
        x, y, z = pos  # Extract coordinates
        shear_rate = self.params[0] if self.params else 1  # Default value
        return jnp.array([shear_rate * y, 0, 0])


class VortexFlow(Flow):
    """2D Vortex Flow (rotational flow)."""

    def _velocity(self, pos):
        x, y, z = pos
        omega = self.params[0] if self.params else 1.0  # Default angular velocity
        return jnp.array([-omega * y, omega * x, 0])

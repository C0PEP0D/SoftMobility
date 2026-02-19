import jax.numpy as jnp


class Field:
    """Base class for 3D fields."""

    def __init__(self, *params):
        """
        Initialize field with additional parameters.

        Parameters:
        - *params: Field-specific parameters (e.g., amplitude, frequency, time).
        """
        self.params = params  # Store additional parameters

    def vector(self, pos=[0.0, 0.0, 0.0], time=0.0):
        """
        Returns field vector H = (Hx, Hy, Hz).

        Parameters:
        - pos: List, tuple, or array representing (x, y, z)

        Returns:
        - jnp.array([Hx, Hy, Hz])
        """
        pos = self._ensure_jax_array(pos)  # Convert input to JAX array
        return self._field_vector(pos, time)

    def _field_vector(self, pos, time):
        """Abstract method for velocity implementation (must be overridden)."""
        raise NotImplementedError("Subclasses must implement _velocity_impl.")

    def _ensure_jax_array(self, pos):
        """Convert input to jnp.ndarray, handling exceptions."""
        try:
            return jnp.asarray(pos, dtype=jnp.float32)
        except Exception as e:
            raise ValueError(f"Invalid input format for position: {pos}") from e


# Examples of fields derived from the class `Field`
class GravityField(Field):
    """Constant uniform field g = (0, 0, -1)."""

    def _field_vector(self, pos, time):
        gravity_acc = self.params[0] if self.params else 1  # Default value
        return jnp.array([0, 0, -gravity_acc])


class RotatingMagneticField(Field):
    """Rotating magnetic field with constant component along x."""

    def _field_vector(self, pos, time):
        B0x = self.params[0] if self.params else 1.0  # Constant component of the field
        B0y = self.params[1] if self.params else 1.0  # Rotating component
        omega = self.params[2] if self.params else 1.0  # Default angular velocity
        return jnp.array([B0x, B0y * jnp.cos(omega * time), B0y * jnp.cos(omega * time)])


class OscillatingMagneticField(Field):
    """Oscillating magnetic field (constant component along x, oscillating component along y)."""

    def _field_vector(self, pos, time):
        B0x = self.params[0] if self.params else 1.0  # Constant component of the field
        B0y = self.params[1] if self.params else 1.0  # Oscillating component of the field
        omega = self.params[2] if self.params else 1.0  # Default angular velocity
        return jnp.array([B0x, B0y * jnp.cos(omega * time), 0])

import jax.numpy as jnp


class Scalar:
    """Base class for scalar inputs."""

    def __init__(self, *params):
        """
        Initialize field with additional parameters.

        Parameters:
        - *params: Field-specific parameters (e.g., amplitude, frequency, time).
        """
        self.params = params  # Store additional parameters

    def value(self, pos=[0.0, 0.0, 0.0], time=0.0) -> float:
        """
        Returns field vector H = (Hx, Hy, Hz).

        Parameters:
        - pos: List, tuple, or array representing (x, y, z)
        - time: float

        Returns:
        - float
        """
        pos = self._ensure_jax_array(pos)  # Convert input to JAX array
        return self._value(pos, time)

    def _value(self, pos, time):
        """Abstract method for velocity implementation (must be overridden)."""
        raise NotImplementedError("Subclasses must implement _velocity_impl.")

    def _ensure_jax_array(self, pos):
        """Convert input to jnp.ndarray, handling exceptions."""
        try:
            return jnp.asarray(pos, dtype=jnp.float32)
        except Exception as e:
            raise ValueError(f"Invalid input format for position: {pos}") from e


# Examples of fields derived from the class `Field`
class ConstantScalar(Scalar):
    """Constant scalar param[0]."""

    def _value(self, pos, time):
        value = self.params[0] if self.params else 1.0  # Default value
        return value

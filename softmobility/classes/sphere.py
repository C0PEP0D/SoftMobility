import jax.numpy as jnp
from jax import lax


class Sphere:
    def __init__(
        self,
        radius=None,
        position=None,
        orientation=None,
        force=None,
        torque=None,
    ):
        """Initialize a new Sphere instance with flexible input handling."""
        self._radius_func = _convert_to_scalar_callable(radius, "radius", 1.0)
        self._position_func = _convert_to_vector_callable(position, "position")
        self._orientation_func = _convert_to_vector_callable(orientation, "orientation")
        self._force_func = _convert_to_vector_callable(force, "force")
        self._torque_func = _convert_to_vector_callable(torque, "torque")

    def radius(self, dofs: jnp.ndarray, design: jnp.ndarray, inputs: jnp.ndarray) -> float:
        return self._radius_func(dofs, design, inputs)

    def position(self, dofs: jnp.ndarray, design: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        return self._position_func(dofs, design, inputs)

    def orientation(self, dofs: jnp.ndarray, design: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        return self._orientation_func(dofs, design, inputs)

    def force(self, dofs: jnp.ndarray, design: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        return self._force_func(dofs, design, inputs)

    def torque(self, dofs: jnp.ndarray, design: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        return self._torque_func(dofs, design, inputs)

    def __str__(self):
        return f"sphere object"

    def _bortz_equation(self, *args):
        """
        Compute the Bortz Jacobian for a 3D rotation vector.

        Args:
            rotation_vector (jnp.ndarray): 3D rotation vector (r_x, r_y, r_z).

        Returns:
            jnp.ndarray: 3x3 Jacobian matrix.
        """
        rotation_vector = self.orientation(*args)
        norm_r = jnp.linalg.norm(rotation_vector)

        def zero_case(_):
            """Case when norm_r is zero: Return identity matrix."""
            return jnp.eye(3)

        def nonzero_case(norm_r):
            """Compute Bortz Jacobian for nonzero rotation."""
            unit_r = rotation_vector / norm_r
            sin_norm_r = jnp.sin(norm_r)
            cos_norm_r = jnp.cos(norm_r)

            # Skew-symmetric matrix of the vectorial product with unit_r
            skew_unit_r = jnp.array(
                [
                    [0, -unit_r[2], unit_r[1]],
                    [unit_r[2], 0, -unit_r[0]],
                    [-unit_r[1], unit_r[0], 0],
                ]
            )

            return (
                (sin_norm_r / norm_r) * jnp.eye(3)
                + (1 - sin_norm_r / norm_r) * jnp.outer(unit_r, unit_r)
                + ((1 - cos_norm_r) / norm_r) * skew_unit_r
            )

        # Use JAX's lax.cond to switch between cases without breaking JIT
        return lax.cond(norm_r < 1e-6, zero_case, nonzero_case, norm_r)

    def bortz_jacobian(self, *args):

        # Compute the Bortz Jacobian for the rotation vector
        B = self._bortz_equation(*args)

        # Construct the 6x6 block diagonal Jacobian
        jacobian_matrix = jnp.block(
            [
                [jnp.eye(3), jnp.zeros((3, 3))],  # Identity for position
                [jnp.zeros((3, 3)), B],  # Bortz Jacobian for rotation
            ]
        )

        return jacobian_matrix

    def composition_of_velocity(self, *args):
        position = self.position(*args)
        T = jnp.array(
            [
                [1, 0, 0, 0, position[2], -position[1]],
                [0, 1, 0, -position[2], 0, position[0]],
                [0, 0, 1, position[1], -position[0], 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        return T

    def composition_of_force(self, *args):
        position = self.position(*args)
        T = jnp.array(
            [
                [1, 0, 0, 0, -position[2], position[1]],
                [0, 1, 0, position[2], 0, -position[0]],
                [0, 0, 1, -position[1], position[0], 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        return T.transpose()


# Functions to build and check callables
def _validate_callable(func, name):
    """Ensure the function is callable and takes exactly two arguments."""
    if not callable(func):
        raise TypeError(f"{name} must be a callable function.")
    if func.__code__.co_argcount != 3:
        raise ValueError(f"{name} must accept exactly three arguments: 'dofs', 'design', and 'inputs'.")


def _convert_to_scalar_callable(value, name, default=1.0):
    """Convert a scalar value or callable to a callable function returning a constant float."""
    if value is None:
        return lambda dofs, design, inputs: default
    try:
        float_value = float(value)
        return lambda dofs, design, inputs: float_value
    except (TypeError, ValueError):
        pass

    if callable(value):
        _validate_callable(value, name)
        return value

    raise TypeError(f"{name} must be a callable or a scalar.")


def _convert_to_vector_callable(value, name, default=jnp.array([0, 0, 0])):
    """Convert scalars, lists, or arrays to a callable function returning a constant value."""
    if value is None:
        return lambda dofs, design, inputs: default
    try:
        vector_value = jnp.array(value)
        if vector_value.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), but got {vector_value.shape}.")
        return lambda dofs, design, inputs: vector_value
    except TypeError:
        pass

    if callable(value):
        _validate_callable(value, name)
        return value

    raise TypeError(f"{name} must be a callable, an array, or a list.")

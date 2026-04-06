import jax.numpy as jnp
from jax import lax, Array


class Sphere:
    def __init__(
        self,
        radius=None,
        position=None,
        orientation=None,
        c_field=None,
        c_stiff=None,
    ):
        """Initialize a new Sphere instance with flexible input handling."""
        self._radius_func = _convert_to_scalar_callable(radius, "radius", 1.0)
        self._position_func = _convert_to_vector_callable_time(position, "position")
        self._orientation_func = _convert_to_vector_callable_time(orientation, "orientation")
        self._c_field_func = _convert_to_array_callable(c_field, "c_field")
        self._c_stiff_func = _convert_to_array_callable(c_stiff, "c_stiff")

    def radius(self, dofs: Array, design: Array) -> float:
        """radius of the sphere"""
        return self._radius_func(dofs, design)

    def position(self, dofs: Array, design: Array, time: Array) -> jnp.ndarray:
        """position relative to the reference point"""
        return self._position_func(dofs, design, time)

    def orientation(self, dofs: Array, design: Array, time: Array) -> jnp.ndarray:
        """orientation vector"""
        return self._orientation_func(dofs, design, time)

    def c_field(self, dofs: Array, design: Array) -> jnp.ndarray:
        """forces applied to the center"""
        return self._c_field_func(dofs, design)

    def c_stiff(self, dofs: Array, design: Array) -> jnp.ndarray:
        """forces applied to the center"""
        return self._c_stiff_func(dofs, design)

    def _bortz_equation(self, *args):
        rotation_vector = self.orientation(*args)
        r_sq = jnp.dot(rotation_vector, rotation_vector)
        safe_r = jnp.sqrt(jnp.maximum(r_sq, 1e-12))
        unit_r = rotation_vector / safe_r

        sin_r = jnp.sin(safe_r)
        cos_r = jnp.cos(safe_r)

        skew_unit_r = jnp.array(
            [
                [0, -unit_r[2], unit_r[1]],
                [unit_r[2], 0, -unit_r[0]],
                [-unit_r[1], unit_r[0], 0],
            ]
        )

        full = (
            (sin_r / safe_r) * jnp.eye(3)
            + (1 - sin_r / safe_r) * jnp.outer(unit_r, unit_r)
            + (1 - cos_r) / safe_r * skew_unit_r
        )

        return jnp.where(r_sq < 1e-12, jnp.eye(3), full)

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
    if func.__code__.co_argcount != 2:
        raise ValueError(f"{name} must accept exactly two arguments: 'dofs', 'design'.")


def _validate_callable_time(func, name):
    """Ensure the function is callable and takes exactly three arguments."""
    if not callable(func):
        raise TypeError(f"{name} must be a callable function.")
    if func.__code__.co_argcount != 3:
        raise ValueError(f"{name} must accept exactly three arguments: 'dofs', 'design', 'time'.")


def _convert_to_scalar_callable(value, name, default=1.0):
    """Convert a scalar value or callable to a callable function returning a constant float."""
    if value is None:
        return lambda dofs, design: default
    try:
        float_value = float(value)
        return lambda dofs, design: float_value
    except (TypeError, ValueError):
        pass

    if callable(value):
        _validate_callable(value, name)
        return value

    raise TypeError(f"{name} must be a callable or a scalar.")


def _convert_to_vector_callable(value, name, default=jnp.array([0, 0, 0])):
    """Convert scalars, lists, or arrays to a callable function returning a constant value."""
    if value is None:
        return lambda dofs, design, time: default
    try:
        vector_value = jnp.array(value)
        if vector_value.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), but got {vector_value.shape}.")
        return lambda dofs, design, time: vector_value
    except TypeError:
        pass

    if callable(value):
        _validate_callable(value, name)
        return value

    raise TypeError(f"{name} must be a callable, an array, or a list.")


def _convert_to_vector_callable_time(value, name, default=jnp.array([0, 0, 0])):
    """Convert scalars, lists, or arrays to a callable function returning a constant value."""
    if value is None:
        return lambda dofs, design, time: default
    try:
        vector_value = jnp.array(value)
        if vector_value.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), but got {vector_value.shape}.")
        return lambda dofs, design, time: vector_value
    except TypeError:
        pass

    if callable(value):
        _validate_callable_time(value, name)
        return value

    raise TypeError(f"{name} must be a callable, an array, or a list.")


def _convert_to_array_callable(value, name, default=jnp.zeros((6, 0))):
    """Validate that value is a callable returning a (6, Ninput) array."""
    if value is None:
        return lambda dofs, design: default
    try:
        array_value = jnp.array(value)
        if array_value.shape[0] != 6:
            raise ValueError(f"{name} must have shape (6,*), but got {array_value.shape}.")
        return lambda dofs, design: array_value

    except TypeError:
        pass

    if callable(value):
        _validate_callable(value, name)
        return value

    raise TypeError(f"{name} must be a callable or an array.")

import jax.numpy as jnp
from jax import Array


class Sphere:
    """
    Spherical element used to build a deformable assembly.

    A ``Sphere`` stores geometry and force-coupling callables as functions of
    the assembly degrees of freedom and design variables. Constants and arrays
    are converted to constant callables for convenience.

    Parameters
    ----------
    radius : float or callable, optional
        Sphere radius, or callable ``radius(dofs, design)``.
    position : array-like or callable, optional
        Body-frame center position with shape ``(3,)``, or callable
        ``position(dofs, design, time)``.
    orientation : array-like or callable, optional
        Rodrigues orientation vector with shape ``(3,)``, or callable
        ``orientation(dofs, design, time)``.
    force : array-like or callable, optional
        Force on the sphere with shape ``(3,)``. It may be constant, a callable
        ``force(dofs, design, inputs)``, or a three-component sequence whose
        entries are constants or scalar callables.
    torque : array-like or callable, optional
        Torque on the sphere with shape ``(3,)``. It accepts the same forms as
        ``force``.
    C_H : array-like or callable, optional
        Matrix with first dimension 6 mapping external inputs to force and
        torque on the sphere.
    C_K : array-like or callable, optional
        Matrix with first dimension 6 mapping degrees of freedom to elastic
        force and torque.

    Notes
    -----
    The six force/velocity components are ordered as translation followed by
    rotation: ``[x, y, z, rx, ry, rz]``.
    """

    def __init__(
        self,
        radius=None,
        position=None,
        orientation=None,
        force=None,
        torque=None,
        C_H=None,
        C_K=None,
    ):
        """Initialize a new Sphere instance with flexible input handling."""
        if (force is not None or torque is not None) and (C_H is not None or C_K is not None):
            raise ValueError("Specify either force/torque or C_H/C_K, not both.")

        self._radius_func = _convert_to_scalar_callable(radius, "radius", 1.0)
        self._position_func = _convert_to_vector_callable_time(position, "position")
        self._orientation_func = _convert_to_vector_callable_time(orientation, "orientation")
        self._force_func = _convert_to_force_callable(force, "force")
        self._torque_func = _convert_to_force_callable(torque, "torque")
        self._has_explicit_couplings = C_H is not None or C_K is not None
        self._C_H_func = _convert_to_array_callable(C_H, "C_H")
        self._C_K_func = _convert_to_array_callable(C_K, "C_K")

    def radius(self, dofs: Array, design: Array) -> float:
        """Evaluate the sphere radius."""
        return self._radius_func(dofs, design)

    def position(self, dofs: Array, design: Array, time: Array) -> jnp.ndarray:
        """Evaluate the body-frame sphere-center position."""
        return self._position_func(dofs, design, time)

    def orientation(self, dofs: Array, design: Array, time: Array) -> jnp.ndarray:
        """Evaluate the sphere orientation as a Rodrigues vector."""
        return self._orientation_func(dofs, design, time)

    def force(self, dofs: Array, design: Array, inputs: Array) -> jnp.ndarray:
        """Evaluate the three-component force on the sphere."""
        return self._force_func(dofs, design, inputs)

    def torque(self, dofs: Array, design: Array, inputs: Array) -> jnp.ndarray:
        """Evaluate the three-component torque on the sphere."""
        return self._torque_func(dofs, design, inputs)

    def six_component_force(self, dofs: Array, design: Array, inputs: Array) -> jnp.ndarray:
        """Evaluate ``[force_x, force_y, force_z, torque_x, torque_y, torque_z]``."""
        return jnp.concatenate([self.force(dofs, design, inputs), self.torque(dofs, design, inputs)])

    def C_H(self, dofs: Array, design: Array) -> jnp.ndarray:
        """Evaluate the matrix coupling external inputs to force and torque."""
        return self._C_H_func(dofs, design)

    def C_K(self, dofs: Array, design: Array) -> jnp.ndarray:
        """Evaluate the matrix coupling degrees of freedom to force and torque."""
        return self._C_K_func(dofs, design)

    def _set_coupling_functions(self, C_H, C_K):
        """Set coupling functions derived by ``SphereAssembly``."""
        self._C_H_func = _convert_to_array_callable(C_H, "C_H")
        self._C_K_func = _convert_to_array_callable(C_K, "C_K")

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
        """
        Compute the six-component Jacobian for position and orientation.

        Returns
        -------
        jnp.ndarray
            Block-diagonal matrix of shape ``(6, 6)``. The translational block
            is the identity and the rotational block is the Bortz Jacobian for
            the sphere orientation.
        """
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
        """
        Map body-reference velocity to this sphere velocity.

        Returns
        -------
        jnp.ndarray
            Matrix of shape ``(6, 6)`` that composes the translational and
            angular velocity of the body reference into translational and
            angular velocity at the sphere center.
        """
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
        """
        Map this sphere force and torque to the body reference.

        Returns
        -------
        jnp.ndarray
            Matrix of shape ``(6, 6)`` used to compose force and torque about
            the sphere center into force and torque about the body reference.
        """
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


def _validate_force_callable(func, name):
    """Ensure a force/torque callable takes dofs, design, and inputs."""
    if not callable(func):
        raise TypeError(f"{name} must be a callable function.")
    if func.__code__.co_argcount != 3:
        raise ValueError(f"{name} must accept exactly three arguments: 'dofs', 'design', 'inputs'.")


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


def _convert_to_vector_callable(value, name, default=None):
    """Convert scalars, lists, or arrays to a callable function returning a constant value."""
    if default is None:
        default = jnp.array([0, 0, 0])
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


def _convert_to_vector_callable_time(value, name, default=None):
    """Convert scalars, lists, or arrays to a callable function returning a constant value."""
    if default is None:
        default = jnp.array([0, 0, 0])
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
        _fn = value
        return lambda dofs, design, time: jnp.asarray(_fn(dofs, design, time), dtype=float)

    raise TypeError(f"{name} must be a callable, an array, or a list.")


def _contains_callable(value):
    return isinstance(value, (list, tuple)) and any(callable(component) for component in value)


def _convert_to_force_component_callable(value, name):
    try:
        float_value = float(value)
        return lambda dofs, design, inputs: float_value
    except (TypeError, ValueError):
        pass

    if callable(value):
        _validate_force_callable(value, name)
        return value

    raise TypeError(f"{name} must be a scalar or a callable.")


def _convert_to_force_callable(value, name, default=None):
    """Convert force/torque entries into a callable returning a three-vector."""
    if default is None:
        default = jnp.zeros(3)
    if value is None:
        return lambda dofs, design, inputs: default

    if callable(value):
        _validate_force_callable(value, name)
        _fn = value

        def wrapper(dofs, design, inputs):
            vector = jnp.asarray(_fn(dofs, design, inputs), dtype=float)
            if vector.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), but got {vector.shape}.")
            return vector

        wrapper._raw = _fn
        return wrapper

    if _contains_callable(value):
        if len(value) != 3:
            raise ValueError(f"{name} must have shape (3,), but got ({len(value)},).")
        component_funcs = [
            _convert_to_force_component_callable(component, f"{name}[{i}]") for i, component in enumerate(value)
        ]

        def wrapper(dofs, design, inputs):
            return jnp.stack(
                [
                    jnp.asarray(component_func(dofs, design, inputs), dtype=float).reshape(())
                    for component_func in component_funcs
                ]
            )

        return wrapper

    try:
        vector_value = jnp.asarray(value, dtype=float)
        if vector_value.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), but got {vector_value.shape}.")
        return lambda dofs, design, inputs: vector_value
    except TypeError:
        pass

    raise TypeError(f"{name} must be a callable, an array, or a list.")


def _convert_to_array_callable(value, name, default=None):
    """Validate that value is a callable returning a (6, Ninput) array."""
    if default is None:
        default = jnp.zeros((6, 0))
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

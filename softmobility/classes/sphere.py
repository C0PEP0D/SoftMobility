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

        self._radius_func = _convert_to_callable(
            radius, "radius", ("dofs", "design"), output_shape=(), default=1.0
        )
        self._position_func = _convert_to_callable(
            position, "position", ("dofs", "design", "time"), output_shape=(3,)
        )
        self._orientation_func = _convert_to_callable(
            orientation, "orientation", ("dofs", "design", "time"), output_shape=(3,)
        )
        self._force_func = _convert_to_callable(
            force, "force", ("dofs", "design", "inputs"), output_shape=(3,)
        )
        self._torque_func = _convert_to_callable(
            torque, "torque", ("dofs", "design", "inputs"), output_shape=(3,)
        )
        self._has_explicit_couplings = C_H is not None or C_K is not None
        self._C_H_func = _convert_to_callable(C_H, "C_H", ("dofs", "design"), output_shape=(6, None))
        self._C_K_func = _convert_to_callable(C_K, "C_K", ("dofs", "design"), output_shape=(6, None))

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
        self._C_H_func = _convert_to_callable(C_H, "C_H", ("dofs", "design"), output_shape=(6, None))
        self._C_K_func = _convert_to_callable(C_K, "C_K", ("dofs", "design"), output_shape=(6, None))

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


# Functions to build and check callables ###################################


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


def _validate_callable(func, name, arg_names, output_shape=None):
    """Validate a user-supplied callable.

    Parameters
    ----------
    func : callable
        Function to validate.
    name : str
        Field name (e.g. ``"radius"``, ``"position"``) used in error messages.
    arg_names : tuple of str
        Names of the positional arguments the callable must accept. The
        expected argument count is ``len(arg_names)`` and the names are
        quoted in the error message.
    output_shape : tuple of int or None, optional
        Documented output shape; reserved for future use. Runtime shape
        checks are performed by the wrapper produced in
        :func:`_convert_to_callable`.
    """
    del output_shape  # documented but not used today
    if not callable(func):
        raise TypeError(f"{name} must be a callable function.")
    n = len(arg_names)
    if func.__code__.co_argcount != n:
        word = _NUMBER_WORDS.get(n, str(n))
        names_str = ", ".join(f"'{a}'" for a in arg_names)
        raise ValueError(f"{name} must accept exactly {word} arguments: {names_str}.")


def _contains_callable(value):
    return isinstance(value, (list, tuple)) and any(callable(component) for component in value)


def _shape_matches(actual, expected):
    """Return True if ``actual`` matches ``expected`` (``None`` entries are wildcards)."""
    if len(actual) != len(expected):
        return False
    return all(e is None or a == e for a, e in zip(actual, expected, strict=True))


def _format_shape(shape):
    """Render a shape tuple, with ``None`` entries shown as ``*``."""
    parts = ["*" if e is None else str(e) for e in shape]
    if len(parts) == 1:
        return f"({parts[0]},)"
    return "(" + ",".join(parts) + ")"


def _shape_error(name, expected, actual):
    return ValueError(
        f"{name} must have shape {_format_shape(expected)}, but got {tuple(actual)}."
    )


def _make_constant(const, n_args):
    """Return a callable of arity ``n_args`` that always returns ``const``."""
    if n_args == 2:
        return lambda dofs, design: const
    if n_args == 3:
        return lambda dofs, design, third: const
    raise ValueError(f"Unsupported arity: {n_args}.")


def _convert_to_callable(value, name, arg_names, output_shape, default=None):
    """Normalise a Sphere field into a callable matching ``arg_names`` and ``output_shape``.

    Handles four input categories in order: ``None`` (constant default),
    a user callable, a sequence of per-component callables (only when
    ``output_shape`` is 1-D), and a scalar/array constant. Anything else
    raises :class:`TypeError`.

    Parameters
    ----------
    value : None, scalar, sequence, or callable
        User-supplied input.
    name : str
        Field name for error messages.
    arg_names : tuple of str
        Argument names the produced callable will accept. The expected
        arity is ``len(arg_names)``.
    output_shape : tuple of int or None
        Expected output shape. ``()`` means a 0-d scalar array; ``None``
        entries match any size on that axis (e.g. ``(6, None)``).
    default : optional
        Constant value returned when ``value is None``. Defaults to
        ``jnp.zeros(output_shape)``, with ``None`` axes collapsed to
        size ``0``.
    """
    n_args = len(arg_names)
    accept_components = len(output_shape) == 1

    # 1. None → constant default (always a JAX array, including 0-d for scalar shapes)
    if value is None:
        if default is None:
            default = jnp.zeros(tuple(0 if e is None else e for e in output_shape))
        else:
            default = jnp.asarray(default, dtype=float)
        return _make_constant(default, n_args)

    # 2. User callable
    if callable(value):
        _validate_callable(value, name, arg_names, output_shape)
        _fn = value

        def wrapper(*args):
            out = jnp.asarray(_fn(*args), dtype=float)
            if not _shape_matches(out.shape, output_shape):
                raise _shape_error(name, output_shape, out.shape)
            return out

        wrapper._raw = _fn
        return wrapper

    # 3. Per-component callables (1-D shape with at least one callable element)
    if accept_components and _contains_callable(value):
        if output_shape[0] is not None and len(value) != output_shape[0]:
            raise _shape_error(name, output_shape, (len(value),))
        component_funcs = [
            _convert_to_callable(component, f"{name}[{i}]", arg_names, output_shape=())
            for i, component in enumerate(value)
        ]

        def wrapper(*args):
            return jnp.stack(
                [jnp.asarray(f(*args), dtype=float).reshape(()) for f in component_funcs]
            )

        return wrapper

    # 4. Constant scalar / array
    try:
        const = jnp.asarray(value, dtype=float)
    except (TypeError, ValueError):
        # Not coercible — fall through to rejection.
        const = None
    if const is not None:
        if not _shape_matches(const.shape, output_shape):
            raise _shape_error(name, output_shape, const.shape)
        return _make_constant(const, n_args)

    # 5. Reject
    if output_shape == ():
        raise TypeError(f"{name} must be a callable or a scalar.")
    raise TypeError(f"{name} must be a callable, an array, or a list.")

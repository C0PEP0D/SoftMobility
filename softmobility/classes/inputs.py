# inputs.py
"""Field and Scalar input classes for position- and time-dependent inputs."""

import jax.numpy as jnp
import jax

# =============================================================================
# Scalar
# =============================================================================


class Scalar:
    """
    Scalar input (active force, control signal, ...).

    Wraps a callable with signature ``(pos, time)`` or ``(pos, time, params)``
    into a JAX-compatible scalar input. Supports optional named parameters that
    can be updated at runtime and differentiated through with ``jax.grad``.

    Parameters
    ----------
    func : callable
        Signature ``(pos, time)`` or ``(pos, time, params)`` where:

        - ``pos``    : jnp.ndarray of shape (3,)
        - ``time``   : float
        - ``params`` : list of jnp.ndarray, one per parameter (only if params are defined)

        Must return a scalar (float or 0-d array).
    params : float, array-like, or list thereof, optional
        Initial parameter values. A single value is treated as a one-element list.
        Each element is converted to a 1D JAX array via ``jnp.atleast_1d``.
        If None, ``func`` is called without ``params``.
    param_names : str or list of str, optional
        Names for each parameter, enabling named access and updates.
        If provided, must have the same length as ``params``.

    Attributes
    ----------
    param_names : list of str or None
        Names of the parameters, if provided.

    Examples
    --------
    Simple scalar with no parameters:

    >>> torque = Scalar(lambda pos, t: 0.5 * jnp.sin(2 * t))

    Single named parameter, updatable at runtime:

    >>> force = Scalar(
    ...     lambda pos, t, p: p[0],
    ...     params      = 1.0,
    ...     param_names = "magnitude",
    ... )
    >>> force.update_params(magnitude=2.0)

    Multiple parameters with different shapes:

    >>> signal = Scalar(
    ...     lambda pos, t, p: p[0] * jnp.sin(p[1] @ jnp.array([t, t**2])),
    ...     params      = [1.0,         jnp.array([2.0, 0.1])],
    ...     param_names = ["amplitude", "freq_coeffs"],
    ... )

    Differentiating through a parameter with ``jax.grad``:

    >>> grad = jax.grad(lambda p: signal.value(jnp.zeros(3), 1.0))(signal._params)

    Notes
    -----
    Internally, params are stored as a list of JAX arrays, which is a valid
    JAX pytree. This means ``jax.jit``, ``jax.grad``, and ``jax.vmap`` trace
    through params transparently without any special handling. Parameter shapes
    must remain fixed after construction, as JAX requires static shapes at
    trace time.
    """

    def __init__(self, func, params=None, param_names=None):
        if not callable(func):
            raise TypeError(f"Scalar expects a callable, got {type(func).__name__}.")
        self._func = func

        # Normalize params to a list of JAX arrays
        if params is None:
            self._params = None
        else:
            self._params = (
                [_to_jax_param(p) for p in params]
                if isinstance(params, (list, tuple))
                else [_to_jax_param(params)]
            )

        if param_names is not None:
            param_names = [param_names] if isinstance(param_names, str) else list(param_names)
            if self._params is not None and len(param_names) != len(self._params):
                raise ValueError(
                    f"param_names length {len(param_names)} does not match "
                    f"number of params {len(self._params)}"
                )
        self.param_names = param_names

    def update_params(self, params=None, **kwargs):
        """
        Update parameter values by full replacement or by name.

        Parameters
        ----------
        params : float, array-like, or list thereof, optional
            Full replacement of all parameters. A single value is treated as
            a one-element list. Ignored if keyword arguments are provided.
        **kwargs : float or array-like
            Named parameter updates. Keys must match names provided at
            construction. Can update one or several parameters at a time.
            Each value is converted to a 1D JAX array via ``jnp.atleast_1d``.

        Raises
        ------
        ValueError
            If this Scalar has no params, if named updates are attempted
            without ``param_names``, or if an unknown name is provided.

        Examples
        --------
        Full update:

        >>> s.update_params([1.0, jnp.array([0.5, 0.3])])

        Single unnamed param:

        >>> s.update_params(1.0)

        Named updates:

        >>> s.update_params(amplitude=0.5)
        >>> s.update_params(amplitude=0.5, phase=jnp.array([0.1, 0.2]))
        """
        if self._params is None:
            raise ValueError("This Scalar has no params to update.")

        if kwargs:
            if self.param_names is None:
                raise ValueError("Cannot use named update — no param_names defined.")
            new_params = list(self._params)
            for name, value in kwargs.items():
                if name not in self.param_names:
                    raise ValueError(f"Unknown param '{name}'. Known: {self.param_names}")
                idx = self.param_names.index(name)
                new_params[idx] = jnp.atleast_1d(jnp.asarray(value, dtype=float))
            self._params = new_params
        elif params is not None:
            if isinstance(params, (list, tuple)):
                self._params = [jnp.atleast_1d(jnp.asarray(p, dtype=float)) for p in params]
            else:
                self._params = [jnp.atleast_1d(jnp.asarray(params, dtype=float))]

    def get_param(self, name):
        """
        Retrieve a parameter array by name.

        Parameters
        ----------
        name : str
            Name of the parameter to retrieve. Must match one of the names
            provided at construction.

        Returns
        -------
        jnp.ndarray
            The parameter array of shape (n,).

        Raises
        ------
        ValueError
            If no ``param_names`` were defined at construction, or if
            ``name`` is not among them.

        Examples
        --------
        >>> amp = signal.get_param("amplitude")
        """
        if self.param_names is None:
            raise ValueError("No param_names defined.")
        return self._params[self.param_names.index(name)]

    def value(self, pos=jnp.zeros(3), time=0.0):
        """
        Evaluate the scalar at a given position and time.

        JAX-safe: compatible with ``jax.jit``, ``jax.grad``, and ``jax.vmap``.
        Gradients can be taken with respect to ``pos``, ``time``, or any element
        of ``_params``.

        Parameters
        ----------
        pos : jnp.ndarray of shape (3,), optional
            Position in lab frame. Defaults to the origin.
        time : float, optional
            Current time. Defaults to 0.

        Returns
        -------
        jnp.ndarray
            Scalar value as a 0-d JAX array.

        Examples
        --------
        >>> v = force.value(jnp.array([0., 0., 0.]), time=1.0)
        """
        pos = jnp.asarray(pos, dtype=float)
        if self._params is not None:
            result = self._func(pos, time, self._params)
        else:
            result = self._func(pos, time)
        return jnp.asarray(result, dtype=float).squeeze()  # always 0-d scalar

    def __repr__(self):
        if self._params is None:
            return "Scalar(no params)"
        if self.param_names is not None:
            pairs = ", ".join(
                f"{k}={v.tolist() if v.size > 1 else float(v):.4g}" for k, v in zip(self.param_names, self._params)
            )
            return f"Scalar({pairs})"
        return f"Scalar(params={[p.tolist() for p in self._params]})"


# class Scalar:
#     """
#     Scalar input (active force, control signal, ...).

#     Parameters
#     ----------
#     func : callable
#         Signature: (pos: jnp.ndarray, time: float) -> float

#     Examples
#     --------
#     torque  = Scalar(lambda pos, t: 0.5 * jnp.sin(2 * t))
#     control = Scalar(lambda pos, t: pid_controller(pos, t))  # any callable works
#     """

#     def __init__(self, func, param_shape=None):
#         if not callable(func):
#             raise TypeError(f"Scalar expects a callable, got {type(func).__name__}.")
#         self._func = func
#         self.param_shape = param_shape  # e.g., (2,) for a 2D parameter
#         self._params = jnp.zeros(param_shape) if param_shape else None

#     def update_from_parameter(self, param):
#         if self._params is not None:
#             self._params = jnp.atleast_1d(jnp.array(param, dtype=float))

#     def value(self, pos=jnp.zeros(3), time=0.0) -> float:
#         """Returns scalar value at position pos and time."""
#         pos = jnp.asarray(pos, dtype=float)
#         return jnp.asarray(self._func(pos, time, self._params), dtype=float)


# =============================================================================
# Field
# =============================================================================


class Field:
    """
    3D vector field (gravity, magnetic field, ...).

    Can be instantiated directly with a callable, or subclassed for complex cases.

    Parameters
    ----------
    func : callable
        Function with signature (pos: jnp.ndarray, time: float) -> jnp.ndarray of shape (3,).

    Examples
    --------
    # Simple: one line
    gravity = GravityField(g=9.81)

    # Intermediate: inline lambda
    B = Field(lambda pos, t: jnp.array([1.0, jnp.cos(2*t), jnp.sin(2*t)]))

    # Advanced: link to external solver or database
    B = Field(my_fem_solver.interpolate)   # anything with (pos, t) -> (3,) signature

    # Advanced: stateful object with __call__
    class MeasuredField:
        def __init__(self, data):
            self.data = data        # e.g. loaded from HDF5
        def __call__(self, pos, t):
            return self.data.interpolate(pos, t)

    B = Field(MeasuredField(hdf5_data))
    """

    def __init__(self, func, param_shape=None):
        if not callable(func):
            raise TypeError(f"Field expects a callable, got {type(func).__name__}.")
        self._func = func
        self.param_shape = param_shape  # e.g., (2,) for a 2D parameter
        self._params = jnp.zeros(param_shape) if param_shape else None

    def update_from_parameter(self, param):
        if self._params is not None:
            self._params = jnp.atleast_1d(jnp.array(param, dtype=float))

    def vector(self, pos=jnp.zeros(3), time=0.0):
        """Returns field vector (3,) at position pos and time."""
        pos = jnp.asarray(pos, dtype=float)
        result = jnp.asarray(self._func(pos, time, self._params), dtype=float)
        if result.shape != (3,):
            raise ValueError(f"Field must return a (3,) array, got shape {result.shape}.")
        return result


# =============================================================================
# Flow
# =============================================================================


class Flow:
    """
    3D flow field, possibly unsteady.

    Parameters
    ----------
    func : callable
        Signature: (pos: jnp.ndarray, time: float) -> jnp.ndarray of shape (3,)

    Examples
    --------
    shear      = Flow(lambda pos, t: jnp.array([pos[1], 0., 0.]))
    oscillating = Flow(lambda pos, t: jnp.array([jnp.sin(t) * pos[1], 0., 0.]))
    flow       = Flow(my_cfd_solver.interpolate)   # any (pos, t) -> (3,) callable
    """

    def __init__(self, func, param_shape=None):
        if not callable(func):
            raise TypeError(f"Flow expects a callable, got {type(func).__name__}.")
        self._func = func
        self.param_shape = param_shape  # e.g., (2,) for a 2D parameter
        self._params = jnp.zeros(param_shape) if param_shape else None

    def update_from_parameter(self, param):
        if self._params is not None:
            self._params = jnp.atleast_1d(jnp.array(param, dtype=float))

    def velocity(self, pos=jnp.zeros(3), time=0.0):
        """Returns velocity vector of shape (3,) at position pos and time."""
        pos = jnp.asarray(pos, dtype=float)
        result = jnp.asarray(self._func(pos, time, self._params), dtype=float)
        if result.shape != (3,):
            raise ValueError(f"Flow must return a (3,) array, got shape {result.shape}.")
        return result

    def gradient(self, pos=jnp.zeros(3), time=0.0):
        """Returns spatial velocity gradient ∇u of shape (3, 3) at position pos and time."""
        pos = jnp.asarray(pos, dtype=float)
        # Freeze time so jacfwd differentiates w.r.t. pos only
        return jax.jacfwd(lambda p: self._func(p, time, self._params))(pos)

    def omega_rate_of_strain(self, pos=jnp.zeros(3), time=0.0):
        """
        Returns vorticity vector Omega (3,) and rate-of-strain tensor E (3, 3).

        Decomposition: ∇u = A + E
            A = 0.5 * (∇u - ∇uᵀ)  antisymmetric → Omega
            E = 0.5 * (∇u + ∇uᵀ)  symmetric     → rate-of-strain
        """
        grad_u = self.gradient(pos, time)
        A = 0.5 * (grad_u - grad_u.T)
        E = 0.5 * (grad_u + grad_u.T)
        Omega = jnp.array([A[2, 1], A[0, 2], A[1, 0]])
        return Omega, E


# ---------------------------------------------------------------------------
# Useful functions
# ---------------------------------------------------------------------------


def _to_jax_param(p):
    """Convert a param to JAX array, preserving shape for arrays, scalar for single values."""
    p = jnp.asarray(p, dtype=float)
    return p.squeeze() if p.size == 1 else p  # 0-d for scalars, (n,) for vectors


# ---------------------------------------------------------------------------
# Named constructors for common scalars
# ---------------------------------------------------------------------------


def constant_scalar(value=1.0):
    """Constant scalar input with updatable value."""
    return Scalar(
        lambda pos, t, p: p[0],  # p[0] is a 0-d scalar directly
        params=float(value),
        param_names="value",
    )


def oscillating_scalar(amplitude=1.0, omega=1.0, phase=0.0):
    """Sinusoidally oscillating scalar with updatable parameters."""
    return Scalar(
        lambda pos, t, p: p[0] * jnp.sin(p[1] * t + p[2]),  # clean, no double indexing
        params=[float(amplitude), float(omega), float(phase)],
        param_names=["amplitude", "omega", "phase"],
    )


# def constant_scalar(value=1):
#     """Constant scalar input."""
#     return Scalar(lambda pos, t, params: value)


# def oscillating_scalar(amplitude=1, omega=1, phase=0.0):
#     """Sinusoidally oscillating scalar input."""
#     return Scalar(lambda pos, t, params: amplitude * jnp.sin(omega * t + phase))


# ---------------------------------------------------------------------------
# Named constructors for common fields
# ---------------------------------------------------------------------------


def gravity_field(g=9.81):
    """Uniform gravity along -z."""
    return Field(lambda pos, t, params: jnp.array([0.0, 0.0, -g]))


def rotating_magnetic_field(amp_x=1, amp_y=1, omega=1):
    """Constant component along x, rotating in y-z plane."""
    return Field(lambda pos, t, params: jnp.array([amp_x, amp_y * jnp.cos(omega * t), amp_y * jnp.sin(omega * t)]))


def oscillating_magnetic_field(amp_x=1, amp_y=1, omega=1):
    """Constant component along x, oscillating along y."""
    return Field(lambda pos, t, params: jnp.array([amp_x, amp_y * jnp.sin(omega * t), 0.0]))


# ---------------------------------------------------------------------------
# Named constructors for common flows
# ---------------------------------------------------------------------------


def no_flow():
    """Quiet fluid."""
    return Flow(lambda pos, t, params: jnp.zeros(3))


def shear_flow(shear_rate=1.0):
    """Simple shear flow u = (shear_rate * y, 0, 0)."""
    return Flow(lambda pos, t, params: jnp.array([shear_rate * pos[1], 0.0, 0.0]))


def rotating_flow(omega=1.0):
    """Solid-body rotation u = (-omega*y, omega*x, 0)."""
    return Flow(lambda pos, t, params: jnp.array([-omega * pos[1], omega * pos[0], 0.0]))


def extensional_flow(rate=1.0):
    """Uniaxial extensional flow u = (rate*x, -rate/2*y, -rate/2*z)."""
    return Flow(lambda pos, t, params: jnp.array([rate * pos[0], -rate / 2 * pos[1], -rate / 2 * pos[2]]))


def Taylor_Green_flow(omega=1.0):
    """Taylor-Green vortex flow u = 0.5 * omega * (sin(x)cos(y), -cos(x)sin(y), 0)."""
    return Flow(
        lambda pos, t, params: 0.5
        * omega
        * jnp.array([0.0, jnp.sin(pos[1]) * jnp.cos(pos[2]), -jnp.cos(pos[1]) * jnp.sin(pos[2])])
    )

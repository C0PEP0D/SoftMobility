"""Parametric inputs (3D and scalar fields) for softmobility."""

import jax
import jax.numpy as jnp

# =============================================================================
# Parent class
# =============================================================================


class _ParametricBase:
    """Shared param logic for Field, Scalar, Flow."""

    def __init__(self, params, param_names):
        self._params = (
            ([_to_jax_param(p) for p in params] if isinstance(params, (list, tuple)) else [_to_jax_param(params)])
            if params is not None
            else None
        )
        if param_names is not None:
            param_names = [param_names] if isinstance(param_names, str) else list(param_names)
            if self._params is not None and len(param_names) != len(self._params):
                raise ValueError(f"param_names length {len(param_names)} != {len(self._params)}")
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
            Each value is converted to a JAX array using the same scalar-
            preserving conversion as construction.

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
                new_params[idx] = _to_jax_param(value)
            self._params = new_params
        elif params is not None:
            if isinstance(params, (list, tuple)):
                self._params = [_to_jax_param(p) for p in params]
            else:
                self._params = [_to_jax_param(params)]

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


# =============================================================================
# Scalar
# =============================================================================


class Scalar(_ParametricBase):
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
        super().__init__(params, param_names)
        if not callable(func):
            raise TypeError(f"Scalar expects a callable, got {type(func).__name__}.")
        self._func = func

    def value(self, pos=None, time=0.0):
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
        if pos is None:
            pos = jnp.zeros(3)
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
                f"{k}={v.tolist() if v.size > 1 else float(v):.4g}"
                for k, v in zip(self.param_names, self._params, strict=False)
            )
            return f"Scalar({pairs})"
        return f"Scalar(params={[p.tolist() for p in self._params]})"


# =============================================================================
# Field
# =============================================================================


class Field(_ParametricBase):
    """
    Field input (force field, magnetic field, ...).

    Wraps a callable with signature ``(pos, time)`` or ``(pos, time, params)``
    into a JAX-compatible field input. Supports optional named parameters that
    can be updated at runtime and differentiated through with ``jax.grad``.

    Parameters
    ----------
    func : callable
        Signature ``(pos, time)`` or ``(pos, time, params)`` where:

        - ``pos``    : jnp.ndarray of shape (3,)
        - ``time``   : float
        - ``params`` : list of jnp.ndarray, one per parameter (only if params are defined)

        Must return a 3D vector (jnp.ndarray of shape (3,)).
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
    Uniform gravity field along -z:

    >>> gravity = Field(lambda pos, t: jnp.array([0.0, 0.0, -9.81]))

    Rotating magnetic field in y-z plane:

    >>> def rotating_magnetic_field(pos, t, params):
    ...     return jnp.array([
    ...         params[0],
    ...         params[1] * jnp.cos(params[2] * t),
    ...         params[1] * jnp.sin(params[2] * t)])
    >>> mag_field = Field(
    ...     rotating_magnetic_field,
    ...     params=[1.0, 2.0, 3.0],
    ...     param_names=['amp_x', 'amp_y', 'omega']
    ... )

    Differentiating through a parameter with ``jax.grad``:

    >>> grad = jax.grad(lambda p: mag_field.vector(jnp.zeros(3), 1.0)[0])(mag_field._params)

    Notes
    -----
    Internally, params are stored as a list of JAX arrays, which is a valid
    JAX pytree. This means ``jax.jit``, ``jax.grad``, and ``jax.vmap`` trace
    through params transparently without any special handling. Parameter shapes
    must remain fixed after construction, as JAX requires static shapes at
    trace time.
    """

    def __init__(self, func, params=None, param_names=None):
        super().__init__(params, param_names)
        if not callable(func):
            raise TypeError(f"Field expects a callable, got {type(func).__name__}.")
        self._func = func

    def vector(self, pos=None, time=0.0):
        """
        Evaluate the field at a given position and time.

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
            return a 3D vector (jnp.ndarray of shape (3,)).

        Examples
        --------
        >>> v = myfiel.vector(jnp.array([0., 0., 0.]), time=1.0)
        """
        if pos is None:
            pos = jnp.zeros(3)
        pos = jnp.asarray(pos, dtype=float)
        result = (
            jnp.asarray(self._func(pos, time, self._params), dtype=float)
            if self._params is not None
            else jnp.asarray(self._func(pos, time), dtype=float)
        )
        if result.shape != (3,):
            raise ValueError(f"Field must return a (3,) array, got shape {result.shape}.")
        return result

    def __repr__(self):
        if self._params is None:
            return "Field(no params)"
        if self.param_names is not None:
            pairs = ", ".join(
                f"{k}={v.tolist() if v.size > 1 else float(v):.4g}"
                for k, v in zip(self.param_names, self._params, strict=False)
            )
            return f"Field({pairs})"
        return f"Field(params={[p.tolist() for p in self._params]})"


# =============================================================================
# Flow
# =============================================================================


class Flow(_ParametricBase):
    """
    Flow input (fluid flow, velocity field, ...).

    Wraps a callable with signature ``(pos, time)`` or ``(pos, time, params)``
    into a JAX-compatible flow input. Supports optional named parameters that
    can be updated at runtime and differentiated through with ``jax.grad`` and ``jax.jacfwd``.

    Parameters
    ----------
    func : callable
        Signature ``(pos, time)`` or ``(pos, time, params)`` where:

        - ``pos``    : jnp.ndarray of shape (3,)
        - ``time``   : float
        - ``params`` : list of jnp.ndarray, one per parameter (only if params are defined)

        Must return a 3D vector (jnp.ndarray of shape (3,)), representing the flow velocity.
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
    Simple shear flow:

    >>> shear = Flow(lambda pos, t: jnp.array([1.0 * pos[1], 0.0, 0.0]))

    Flow with parameters:

    >>> def custom_flow(pos, t, params):
    ...     return jnp.array([params[0] * pos[1], params[1] * pos[0], 0.0])
    >>> flow = Flow(
    ...     custom_flow,
    ...     params=[2.0, 3.0],
    ...     param_names=['shear_rate', 'rotation_rate']
    ... )

    Notes
    -----
    Internally, params are stored as a list of JAX arrays, which is a valid
    JAX pytree. This means ``jax.jit``, ``jax.grad``, and ``jax.vmap`` trace
    through params transparently without any special handling. Parameter shapes
    must remain fixed after construction, as JAX requires static shapes at
    trace time.
    """

    def __init__(self, func, params=None, param_names=None):
        super().__init__(params, param_names)
        if not callable(func):
            raise TypeError(f"Flow expects a callable, got {type(func).__name__}.")
        self._func = func

    def velocity(self, pos=None, time=0.0):
        """
        Evaluate the flow velocity at a given position and time.

        JAX-safe: compatible with ``jax.jit``, ``jax.grad``, and ``jax.vmap``.

        Parameters
        ----------
        pos : jnp.ndarray of shape (3,), optional
            Position in lab frame. Defaults to the origin.
        time : float, optional
            Current time. Defaults to 0.

        Returns
        -------
        jnp.ndarray
            Flow velocity as a 3D vector (array of shape (3,)).

        Examples
        --------
        >>> v = shear.velocity(jnp.array([0., 0., 0.]), time=1.0)
        """
        if pos is None:
            pos = jnp.zeros(3)
        pos = jnp.asarray(pos, dtype=float)
        result = (
            jnp.asarray(self._func(pos, time, self._params), dtype=float)
            if self._params is not None
            else jnp.asarray(self._func(pos, time), dtype=float)
        )
        if result.shape != (3,):
            raise ValueError(f"Flow must return a (3,) array, got shape {result.shape}.")
        return result

    def gradient(self, pos=None, time=0.0):
        """
        Evaluate the Jacobian (gradient) of the flow velocity field at a given position and time.

        JAX-safe: compatible with ``jax.jit`` and ``jax.jacfwd``.

        Parameters
        ----------
        pos : jnp.ndarray of shape (3,), optional
            Position in lab frame. Defaults to the origin.
        time : float, optional
            Current time. Defaults to 0.

        Returns
        -------
        jnp.ndarray
            Jacobian matrix of shape (3, 3).

        Examples
        --------
        >>> grad_u = shear.gradient(jnp.array([1., 2., 3.]), time=1.0)
        """
        if pos is None:
            pos = jnp.zeros(3)
        pos = jnp.asarray(pos, dtype=float)
        fn = (
            (lambda p: self._func(p, time, self._params))
            if self._params is not None
            else (lambda p: self._func(p, time))
        )
        return jax.jacfwd(fn)(pos)

    def omega_rate_of_strain(self, pos=None, time=0.0):
        """
        Evaluate the vorticity vector and rate-of-strain tensor.

        The vorticity vector comes from the skew-symmetric part of the velocity gradient,
        and the rate-of-strain tensor is the symmetric part.

        Parameters
        ----------
        pos : jnp.ndarray of shape (3,), optional
            Position in lab frame. Defaults to the origin.
        time : float, optional
            Current time. Defaults to 0.

        Returns
        -------
        tuple
            A tuple `(omega, E)` where:
            - `omega` : jnp.ndarray of shape (3,), the vorticity vector
            - `E` : jnp.ndarray of shape (3, 3), the rate-of-strain tensor

        Examples
        --------
        >>> omega, E = shear.omega_rate_of_strain(jnp.array([1., 2., 3.]), time=1.0)
        """
        grad_u = self.gradient(pos, time)
        A = 0.5 * (grad_u - grad_u.T)
        E = 0.5 * (grad_u + grad_u.T)
        return jnp.array([A[2, 1], A[0, 2], A[1, 0]]), E

    def __repr__(self):
        if self._params is None:
            return "Flow(no params)"
        if self.param_names is not None:
            pairs = ", ".join(
                f"{k}={v.tolist() if v.size > 1 else float(v):.4g}"
                for k, v in zip(self.param_names, self._params, strict=False)
            )
            return f"Flow({pairs})"
        return f"Flow(params={[p.tolist() for p in self._params]})"


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
    """
    Create a scalar input with a constant value.

    Parameters
    ----------
    value : float, default=1.0
        Constant value returned by the scalar.

    Returns
    -------
    Scalar
        Parametric scalar with one named parameter, ``"value"``.

    Examples
    --------
    >>> signal = constant_scalar(2.0)
    >>> signal.value()
    Array(2., dtype=float32)
    >>> signal.update_params(value=3.0)
    """
    return Scalar(
        lambda pos, t, p: p[0],  # p[0] is a 0-d scalar directly
        params=float(value),
        param_names="value",
    )


def oscillating_scalar(amplitude=1.0, omega=1.0, phase=0.0):
    """
    Create a sinusoidal scalar input.

    The returned scalar evaluates
    ``amplitude * sin(omega * time + phase)``.

    Parameters
    ----------
    amplitude : float, default=1.0
        Oscillation amplitude.
    omega : float, default=1.0
        Angular frequency.
    phase : float, default=0.0
        Phase shift in radians.

    Returns
    -------
    Scalar
        Parametric scalar with named parameters ``"amplitude"``, ``"omega"``,
        and ``"phase"``.
    """
    return Scalar(
        lambda pos, t, p: p[0] * jnp.sin(p[1] * t + p[2]),
        params=[float(amplitude), float(omega), float(phase)],
        param_names=["amplitude", "omega", "phase"],
    )


# ---------------------------------------------------------------------------
# Named constructors for common fields
# ---------------------------------------------------------------------------


def gravity_field(g=9.81):
    """
    Create a uniform gravity field directed along negative z.

    Parameters
    ----------
    g : float, default=9.81
        Magnitude of gravitational acceleration.

    Returns
    -------
    Field
        Field whose value is ``[0, 0, -g]`` and whose named parameter is
        ``"g"``.
    """
    return Field(
        lambda pos, t, param: jnp.array([0.0, 0.0, -param[0]]),
        params=float(g),
        param_names="g",
    )


def rotating_magnetic_field(amp_x=1, amp_y=1, omega=1):
    """
    Create a magnetic-field-like vector rotating in the y-z plane.

    The field is ``[amp_x, amp_y*cos(omega*time),
    amp_y*sin(omega*time)]``.

    Parameters
    ----------
    amp_x : float, default=1
        Constant x component.
    amp_y : float, default=1
        Amplitude of the rotating y-z component.
    omega : float, default=1
        Angular frequency.

    Returns
    -------
    Field
        Parametric field with named parameters ``"amp_x"``, ``"amp_y"``, and
        ``"omega"``.
    """
    return Field(
        lambda pos, t, p: jnp.array([p[0], p[1] * jnp.cos(p[2] * t), p[1] * jnp.sin(p[2] * t)]),
        params=[float(amp_x), float(amp_y), float(omega)],
        param_names=["amp_x", "amp_y", "omega"],
    )


def oscillating_magnetic_field(amp_x=1, amp_y=1, omega=1):
    """
    Create a magnetic-field-like vector oscillating along y.

    The field is ``[amp_x, amp_y*sin(omega*time), 0]``.

    Parameters
    ----------
    amp_x : float, default=1
        Constant x component.
    amp_y : float, default=1
        Amplitude of the oscillating y component.
    omega : float, default=1
        Angular frequency.

    Returns
    -------
    Field
        Parametric field with named parameters ``"amp_x"``, ``"amp_y"``, and
        ``"omega"``.
    """
    return Field(
        lambda pos, t, p: jnp.array([p[0], p[1] * jnp.sin(p[2] * t), 0.0]),
        params=[float(amp_x), float(amp_y), float(omega)],
        param_names=["amp_x", "amp_y", "omega"],
    )


# ---------------------------------------------------------------------------
# Named constructors for common flows
# ---------------------------------------------------------------------------


def no_flow():
    """
    Create a quiescent background flow.

    Returns
    -------
    Flow
        Flow whose velocity is zero everywhere.
    """
    return Flow(lambda pos, t: jnp.zeros(3))


def shear_flow(shear_rate=1.0):
    """
    Create a simple shear flow.

    The velocity is ``u = (shear_rate * y, 0, 0)``.

    Parameters
    ----------
    shear_rate : float, default=1.0
        Shear rate multiplying the y coordinate.

    Returns
    -------
    Flow
        Parametric flow with named parameter ``"shear_rate"``.
    """
    return Flow(
        lambda pos, t, shear_rate: jnp.array([shear_rate[0] * pos[1], 0.0, 0.0]),
        params=float(shear_rate),
        param_names="shear_rate",
    )


def rotating_flow(omega=1.0):
    """
    Create a solid-body rotation flow.

    The velocity is ``u = (-omega*y, omega*x, 0)``.

    Parameters
    ----------
    omega : float, default=1.0
        Angular speed of the background rotation.

    Returns
    -------
    Flow
        Parametric flow with named parameter ``"omega"``.
    """
    return Flow(
        lambda pos, t, omega: jnp.array([-omega[0] * pos[1], omega[0] * pos[0], 0.0]),
        params=float(omega),
        param_names="omega",
    )


def extensional_flow(rate=1.0):
    """
    Create a uniaxial extensional flow.

    The velocity is ``u = (rate*x, -rate*y/2, -rate*z/2)``.

    Parameters
    ----------
    rate : float, default=1.0
        Extension rate along x.

    Returns
    -------
    Flow
        Parametric flow with named parameter ``"rate"``.
    """
    return Flow(
        lambda pos, t, rate: jnp.array([rate[0] * pos[0], -rate[0] / 2 * pos[1], -rate[0] / 2 * pos[2]]),
        params=float(rate),
        param_names="rate",
    )


def taylor_green_flow(omega=1.0):
    """
    Create a Taylor-Green-style vortex flow.

    The implemented velocity is
    ``0.5 * omega * [0, sin(y)*cos(z), -cos(y)*sin(z)]``.

    Parameters
    ----------
    omega : float, default=1.0
        Velocity scale.

    Returns
    -------
    Flow
        Parametric flow with named parameter ``"omega"``.
    """
    return Flow(
        lambda pos, t, omega: 0.5
        * omega[0]
        * jnp.array([0.0, jnp.sin(pos[1]) * jnp.cos(pos[2]), -jnp.cos(pos[1]) * jnp.sin(pos[2])]),
        params=float(omega),
        param_names="omega",
    )

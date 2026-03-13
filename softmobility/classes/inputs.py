# inputs.py
"""Field and Scalar input classes for position- and time-dependent inputs."""

import jax.numpy as jnp
import jax


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

    def __init__(self, func):
        if not callable(func):
            raise TypeError(f"Field expects a callable, got {type(func).__name__}.")
        self._func = func

    def vector(self, pos=jnp.zeros(3), time=0.0):
        """Returns field vector (3,) at position pos and time."""
        pos = jnp.asarray(pos, dtype=float)
        result = jnp.asarray(self._func(pos, time), dtype=float)
        if result.shape != (3,):
            raise ValueError(f"Field must return a (3,) array, got shape {result.shape}.")
        return result


class Scalar:
    """
    Scalar input (active force, control signal, ...).

    Parameters
    ----------
    func : callable
        Signature: (pos: jnp.ndarray, time: float) -> float

    Examples
    --------
    torque  = Scalar(lambda pos, t: 0.5 * jnp.sin(2 * t))
    control = Scalar(lambda pos, t: pid_controller(pos, t))  # any callable works
    """

    def __init__(self, func):
        if not callable(func):
            raise TypeError(f"Scalar expects a callable, got {type(func).__name__}.")
        self._func = func

    def value(self, pos=jnp.zeros(3), time=0.0) -> float:
        """Returns scalar value at position pos and time."""
        pos = jnp.asarray(pos, dtype=float)
        result = float(self._func(pos, time))
        return result


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

    def __init__(self, func):
        if not callable(func):
            raise TypeError(f"Flow expects a callable, got {type(func).__name__}.")
        self._func = func

    def velocity(self, pos=jnp.zeros(3), time=0.0):
        """Returns velocity vector of shape (3,) at position pos and time."""
        pos = jnp.asarray(pos, dtype=float)
        result = jnp.asarray(self._func(pos, time), dtype=float)
        if result.shape != (3,):
            raise ValueError(f"Flow must return a (3,) array, got shape {result.shape}.")
        return result

    def gradient(self, pos=jnp.zeros(3), time=0.0):
        """Returns spatial velocity gradient ∇u of shape (3, 3) at position pos and time."""
        pos = jnp.asarray(pos, dtype=float)
        # Freeze time so jacfwd differentiates w.r.t. pos only
        return jax.jacfwd(lambda p: self._func(p, time))(pos)

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
# Named constructors for common fields
# ---------------------------------------------------------------------------


def gravity_field(g=9.81):
    """Uniform gravity along -z."""
    return Field(lambda pos, t: jnp.array([0.0, 0.0, -g]))


def rotating_magnetic_field(amp_x=1, amp_y=1, omega=1):
    """Constant component along x, rotating in y-z plane."""
    return Field(lambda pos, t: jnp.array([amp_x, amp_y * jnp.cos(omega * t), amp_y * jnp.sin(omega * t)]))


def oscillating_magnetic_field(amp_x=1, amp_y=1, omega=1):
    """Constant component along x, oscillating along y."""
    return Field(lambda pos, t: jnp.array([amp_x, amp_y * jnp.sin(omega * t), 0.0]))


# ---------------------------------------------------------------------------
# Named constructors for common scalars
# ---------------------------------------------------------------------------


def constant_scalar(value):
    """Constant scalar input."""
    return Scalar(lambda pos, t: value)


def oscillating_scalar(amplitude=1, omega=1, phase=0.0):
    """Sinusoidally oscillating scalar input."""
    return Scalar(lambda pos, t: amplitude * jnp.sin(omega * t + phase))


# ---------------------------------------------------------------------------
# Named constructors for common flows
# ---------------------------------------------------------------------------


def no_flow():
    """Quiet fluid."""
    return Flow(lambda pos, t: jnp.zeros_like(3))


def shear_flow(shear_rate=1.0):
    """Simple shear flow u = (shear_rate * y, 0, 0)."""
    return Flow(lambda pos, t: jnp.array([shear_rate * pos[1], 0.0, 0.0]))


def rotating_flow(omega=1.0):
    """Solid-body rotation u = (-omega*y, omega*x, 0)."""
    return Flow(lambda pos, t: jnp.array([-omega * pos[1], omega * pos[0], 0.0]))


def extensional_flow(rate=1.0):
    """Uniaxial extensional flow u = (rate*x, -rate/2*y, -rate/2*z)."""
    return Flow(lambda pos, t: jnp.array([rate * pos[0], -rate / 2 * pos[1], -rate / 2 * pos[2]]))


def Taylor_Green_flow(omega=1.0):
    """Taylor-Green vortex flow u = 0.5 * omega * (sin(x)cos(y), -cos(x)sin(y), 0)."""
    return Flow(
        lambda pos, t: 0.5
        * omega
        * jnp.array([jnp.sin(pos[0]) * jnp.cos(pos[1]), -jnp.cos(pos[0]) * jnp.sin(pos[1])])
    )

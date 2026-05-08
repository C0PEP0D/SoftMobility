"""
JAX-traceable simulation and optimization for soft bodies in flows.

Classes
-------
- FlowBodyRollout  : pure functional simulation (jit/grad/vmap compatible)
- FlowBodyOptimizer: gradient-based design optimization
- FlowBodyRL       : actor-critic RL loop (design = actor)
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax

from softmobility import Field, Flow, Scalar, SoftBody

# =============================================================================
# FlowBodyRollout
# =============================================================================


class FlowBodyRollout:
    """
    FlowBodyRollout (soft body simulation in fluid flow).

    Wraps a soft body, flow, and input map into a JAX-compatible rollout for simulating
    advection and deformation of a soft body in a flow. Supports automatic differentiation
    with ``jax.grad`` and just-in-time compilation with ``jax.jit``.

    Parameters
    ----------
    soft_body : SoftBody
        The soft body to be simulated. Must implement methods for position, orientation,
        and degrees of freedom.
    flow : Flow
        The flow field in which the soft body is advected. Must provide velocity and
        gradient methods.
    input_map : dict[str, Field | Scalar], optional
        Mapping of input variable names to their corresponding field or scalar objects.
        Used to pass external inputs to the soft body during simulation.

    Attributes
    ----------
    fields : list of Field
        List of field inputs extracted from the input_map.
    scalars : list of Scalar
        List of scalar inputs extracted from the input_map.

    Examples
    --------
    Basic rollout simulation:

    >>> mysoftbody = SoftBody(...)
    >>> myflow = Flow(...)
    >>> myfield = Field(...)
    >>> myscalar = Scalar(...)
    >>> mymap = {"field": field, "scalar": scalar}
    >>> rollout = FlowBodyRollout(soft_body=mysoftbody, flow=myflow, input_map=mymap)
    >>> positions, orientations, dofs = rollout.rollout(dt, n_steps)

    Using JAX for optimization:

    >>> grad_fn = jax.jit(jax.grad(lambda d: rollout.rollout(dt, n_steps, init_pos, init_ori, init_dofs, d)[0][2]))
    >>> gradient = grad_fn(design)

    Notes
    -----
    - The rollout is pure-functional and stateless, making it compatible with JAX transformations.
    - Inputs (fields and scalars) are validated during initialization to ensure compatibility with
      the soft body and flow.
    - The velocity method computes the soft body's linear velocity, angular velocity, and dof
      derivatives in the lab frame.
    """

    def __init__(
        self,
        soft_body: SoftBody,
        flow: Flow,
        input_map: dict[str, Field | Scalar] | None = None,
    ):
        self.soft_body = soft_body
        self.flow = flow
        self.fields, self.scalars = self._validate_inputs(input_map)

    def _velocity(self, design, position, orientation, dofs, time):
        """Returns (v_lab, omega_lab, dot_dofs), the soft body's linear velocity,
        angular velocity and derivative of degrees-of-freedom, all in lab frame."""
        rot, sixc_rot = rotation_matrix_from_Rodrigues(orientation, Ndof=self.soft_body.Ndof)
        inputs = self._build_inputs(design, position, time, rot)
        u_lab = self.flow.velocity(position, time)
        omega_lab, E_lab = self.flow.omega_rate_of_strain(position, time)
        E_body = rot.T @ E_lab @ rot
        E_inf = jnp.array([E_body[0, 0], E_body[0, 1], E_body[0, 2], E_body[1, 1], E_body[1, 2]])
        tensors = self.soft_body.compute_tensors(dofs, design, time)

        # Soft mobility equation in the body frame
        p_body = tensors.M_H @ inputs + tensors.M_K @ dofs + tensors.C_E @ E_inf + tensors.p_act

        # Rotate body-frame result back to lab frame
        p_lab = sixc_rot @ p_body

        v_lab = p_lab[:3] + u_lab
        omega = p_lab[3:6] + omega_lab
        dot_dofs = p_lab[6:]

        return v_lab, omega, dot_dofs

    def _step_rk2(self, carry, t, design, dt):
        """Classical midpoint (RK2) step, ``lax.scan`` compatible.

        The Bortz operator is recomputed at the Euler-predicted mid-step
        orientation, which is required for the orientation update to be
        truly second-order accurate.
        """
        position, orientation, dofs = carry
        time = t * dt

        def vel(pos, ori, dof, time):
            return self._velocity(design, pos, ori, dof, time)

        bortz0 = compute_bortz_operator(orientation)
        v1, w1, d1 = vel(position, orientation, dofs, time)

        ori_mid = orientation + 0.5 * dt * bortz0 @ w1
        bortz_mid = compute_bortz_operator(ori_mid)

        v2, w2, d2 = vel(
            position + 0.5 * dt * v1,
            ori_mid,
            dofs + 0.5 * dt * d1,
            time + 0.5 * dt,
        )

        pos_new = position + dt * v2
        ori_new = orientation + dt * bortz_mid @ w2
        ori_new = rescale_orientation(ori_new)
        dof_new = dofs + dt * d2

        return (pos_new, ori_new, dof_new), (pos_new, ori_new, dof_new)

    # Backward-compatible alias for the default scheme.
    _step = _step_rk2

    def _step_rk4(self, carry, t, design, dt):
        """Classical 4-stage Runge–Kutta step, ``lax.scan`` compatible.

        Recomputes the Bortz operator at every stage so the orientation update
        is truly fourth-order accurate.
        """
        position, orientation, dofs = carry
        time = t * dt

        def deriv(pos, ori, dof, t):
            v, w, dd = self._velocity(design, pos, ori, dof, t)
            return v, compute_bortz_operator(ori) @ w, dd

        v1, do1, dd1 = deriv(position, orientation, dofs, time)
        v2, do2, dd2 = deriv(
            position + 0.5 * dt * v1,
            orientation + 0.5 * dt * do1,
            dofs + 0.5 * dt * dd1,
            time + 0.5 * dt,
        )
        v3, do3, dd3 = deriv(
            position + 0.5 * dt * v2,
            orientation + 0.5 * dt * do2,
            dofs + 0.5 * dt * dd2,
            time + 0.5 * dt,
        )
        v4, do4, dd4 = deriv(
            position + dt * v3,
            orientation + dt * do3,
            dofs + dt * dd3,
            time + dt,
        )

        pos_new = position + dt / 6.0 * (v1 + 2 * v2 + 2 * v3 + v4)
        ori_new = orientation + dt / 6.0 * (do1 + 2 * do2 + 2 * do3 + do4)
        ori_new = rescale_orientation(ori_new)
        dof_new = dofs + dt / 6.0 * (dd1 + 2 * dd2 + 2 * dd3 + dd4)

        return (pos_new, ori_new, dof_new), (pos_new, ori_new, dof_new)

    _SCHEMES = {"rk2": _step_rk2, "rk4": _step_rk4}

    def rollout(
        self,
        dt,
        n_steps,
        init_position=None,
        init_orientation=None,
        init_dofs=None,
        design=None,
        scheme="rk4",
    ):
        """
        Simulate the advection and deformation of the soft body over n_steps.

        Core function for JAX transformations (``jax.jit``, ``jax.grad``, ``jax.vmap``).
        Computes the trajectory of the soft body's position, orientation, and degrees of freedom
        from an initial state, given a time step and number of steps.

        Parameters
        ----------
        dt : float
            Time step for each simulation step.
        n_steps : int
            Number of simulation steps to perform.
        init_position : jnp.ndarray or list of shape (3,), optional
            Initial position of the soft body. Defaults to the origin.
        init_orientation : jnp.ndarray or list of shape (3,), optional
            Initial orientation of the soft body. Defaults to zero orientation.
        init_dofs : jnp.ndarray or list, optional
            Initial degrees of freedom. If None, uses the soft body's default values.
        design : jnp.ndarray or list, optional
            Design parameters. If None, uses the soft body's default values.
        scheme : {"rk4", "rk2"}, default "rk4"
            Time-integration scheme. ``"rk4"`` (default) is a four-stage
            classical Runge–Kutta with the Bortz operator recomputed at every
            stage and converges as ``O(dt^4)``. ``"rk2"`` is the explicit
            midpoint method (with Bortz recomputed at the predicted mid-step)
            and converges as ``O(dt^2)``. RK4 costs roughly 2× per step but
            is typically orders of magnitude more accurate at any
            non-trivial tolerance, so it is the recommended default.

        Returns
        -------
        tuple of jnp.ndarray
            - positions : jnp.ndarray of shape (n_steps, 3)
                Positions of the soft body at each step.
            - orientations : jnp.ndarray of shape (n_steps, 3)
                Orientations of the soft body at each step.
            - dofs : jnp.ndarray of shape (n_steps, Ndof)
                Degrees of freedom of the soft body at each step.

        Notes
        -----
        - This method is stateless and pure-functional, making it fully compatible with JAX transformations.
        - The simulation is performed using ``jax.lax.scan`` for efficient, JAX-compatible iteration.
        """
        init_position = jnp.zeros(3) if init_position is None else jnp.asarray(init_position, dtype=float)
        init_orientation = jnp.zeros(3) if init_orientation is None else jnp.asarray(init_orientation, dtype=float)
        init_dofs = (
            jnp.asarray(self.soft_body.dof_defaults, dtype=float)
            if init_dofs is None
            else jnp.asarray(init_dofs, dtype=float)
        )
        design = (
            jnp.asarray(self.soft_body.design_defaults, dtype=float)
            if design is None
            else jnp.asarray(design, dtype=float)
        )
        dt = jnp.asarray(dt, dtype=float)

        try:
            step_fn = self._SCHEMES[scheme]
        except KeyError as exc:
            raise ValueError(
                f"Unknown integration scheme {scheme!r}; choose from {sorted(self._SCHEMES)}"
            ) from exc

        carry = (init_position, init_orientation, init_dofs)
        _, (positions, orientations, dofs) = jax.lax.scan(
            partial(step_fn, self, design=design, dt=dt), carry, jnp.arange(n_steps)
        )
        return positions, orientations, dofs

    def _validate_inputs(self, input_dict: dict):
        """
        Validate and store inputs as ordered lists matching input_variables layout.
        Populates self.fields (list[Field]) and self.scalars (list[Scalar]).
        """
        input_dict = input_dict or {}
        fields, scalars = [], []
        seen_field_bases = set()

        for var in self.soft_body.input_variables:
            if var[-1].isdigit():
                base = var[:-1]
                if base not in seen_field_bases:
                    if base not in input_dict:
                        raise ValueError(f"Missing Field input '{base}'")
                    if not isinstance(input_dict[base], Field):
                        raise TypeError(f"Input '{base}' expected a Field, got {type(input_dict[base]).__name__}")
                    fields.append(input_dict[base])
                    seen_field_bases.add(base)
            else:
                if var not in input_dict:
                    raise ValueError(f"Missing Scalar input '{var}'")
                if not isinstance(input_dict[var], Scalar):
                    raise TypeError(f"Input '{var}' expected a Scalar, got {type(input_dict[var]).__name__}")
                scalars.append(input_dict[var])

        unexpected = (
            set(input_dict.keys())
            - seen_field_bases
            - {v for v in self.soft_body.input_variables if not v[-1].isdigit()}
        )
        if unexpected:
            raise ValueError(f"Unexpected input keys: {sorted(unexpected)}")

        return fields, scalars

    def _build_inputs(self, design, position, time, rot_matrix):
        """
        Assemble the input vector in canonical order: field components first, scalars last.
        """
        parts = [rot_matrix.T @ f.vector(position, time) for f in self.fields]
        parts += [jnp.array([s.value(position, time)]) for s in self.scalars]
        return jnp.concatenate(parts) if parts else jnp.zeros(0)


# =============================================================================
# FlowBodyOptimizer
# =============================================================================


class FlowBodyOptimizer:
    """
    FlowBodyOptimizer (gradient-based optimization for design parameters).

    Optimizes design parameters for a soft body rollout using gradient-based methods.
    Supports JAX-compatible optimizers (e.g., Optax) and automatic differentiation
    with ``jax.grad`` and ``jax.jit``.

    Parameters
    ----------
    rollout : FlowBodyRollout
        The rollout object to simulate the soft body's advection and deformation.
        Must implement a ``rollout`` method compatible with JAX transformations.
    objective : callable
        Objective function with signature ``objective(rollout, design) -> scalar``.
        Computes the scalar value to optimize (minimize or maximize).
    optimizer : optax.GradientTransformation, optional
        Optax optimizer (e.g., ``optax.adam(1e-3)``). If None, defaults to Adam with learning rate 1e-3.

    Attributes
    ----------
    _grad_fn : callable
        JAX-jitted function that computes both the objective value and its gradient.

    Examples
    --------
    Minimize the Z displacement of a soft body:

    >>> def my_objective(rollout, design):
    ...     positions, _, _ = rollout.rollout(design, init_pos, init_ori, init_dofs)
    ...     return positions[-1, 2]   # final Z position

    >>> opt = FlowBodyOptimizer(rollout, my_objective, optax.adam(1e-3))
    >>> optimal_design = opt.run(init_design, n_steps=500, maximize=False)

    Notes
    -----
    - The optimizer uses JAX's ``jax.value_and_grad`` for efficient gradient computation.
    - Supports clipping of design parameters and gradients during optimization.
    - Compatible with ``jax.jit`` for just-in-time compilation of the optimization loop.
    """

    def __init__(self, rollout, objective, optimizer=None):
        self.rollout = rollout
        self.objective = objective
        self.optimizer = optimizer or optax.adam(1e-3)
        self._grad_fn = jax.jit(jax.value_and_grad(lambda d: self.objective(self.rollout, d)))

    def run(
        self,
        init_design,
        n_steps=500,
        print_every=100,
        clip_min=None,
        clip_max=None,
        grad_clip=None,
        maximize=True,
    ):
        """
        Run the optimization loop to find the optimal design parameters.

        Iteratively updates the design parameters using the specified optimizer and objective function.
        Supports clipping of design parameters and gradients, and can maximize or minimize the objective.

        Parameters
        ----------
        init_design : array-like
            Initial design parameters.
        n_steps : int, default=500
            Number of optimization steps.
        print_every : int, default=100
            Print progress every `print_every` steps.
        clip_min : float, optional
            Minimum allowed value for design parameters. If None, no lower bound.
        clip_max : float, optional
            Maximum allowed value for design parameters. If None, no upper bound.
        grad_clip : float, optional
            Maximum allowed norm for the gradient. If None, no clipping.
        maximize : bool, default=True
            If True, maximize the objective; if False, minimize.

        Returns
        -------
        jnp.ndarray
            Optimal design parameters after `n_steps` iterations.

        Notes
        -----
        - Uses JAX's ``jax.value_and_grad`` for efficient gradient computation.
        - If NaN is encountered in the loss or gradient, reverts to the best known design.
        - Supports clipping of design parameters and gradients to ensure stability.
        - Prints progress during optimization, including objective value and gradient norm.
        """
        sign = -1.0 if maximize else 1.0

        design = jnp.atleast_1d(jnp.array(init_design, dtype=float))
        opt_state = self.optimizer.init(design)
        best_design, best_value = design, -jnp.inf

        for step in range(n_steps):
            loss, grad = self._grad_fn(design)
            loss = sign * loss
            grad = sign * grad

            if jnp.isnan(loss) or jnp.any(jnp.isnan(grad)):
                print(f"NaN at step {step}, reverting to best known design")
                return best_design

            if grad_clip is not None:
                norm = jnp.linalg.norm(grad)
                grad = jnp.where(norm > grad_clip, grad * grad_clip / norm, grad)

            updates, opt_state = self.optimizer.update(grad, opt_state)
            design = optax.apply_updates(design, updates)

            if clip_min is not None or clip_max is not None:
                design = jnp.clip(design, clip_min, clip_max)

            value = float(-loss)
            if value > best_value:
                best_value, best_design = value, design

            if step % print_every == 0 or step == n_steps - 1:
                vals = "  ".join(f"design{i}={float(v):.2f}" for i, v in enumerate(jnp.atleast_1d(design)))
                print(
                    f"step {step:4d}  objective={value:.4f}  " f"|grad|={float(jnp.linalg.norm(grad)):.4f}  {vals}"
                )

        return best_design


# =============================================================================
# FlowBodyRL  (actor-critic skeleton)
# =============================================================================


class FlowBodyRL:
    """
    Experimental actor-critic-style optimizer for rollout rewards.

    The actor is the design vector, the environment is a ``FlowBodyRollout``,
    and the reward is supplied by the user. This class is currently a research
    skeleton rather than the recommended optimization interface; most users
    should start with ``FlowBodyOptimizer``.

    Parameters
    ----------
    rollout : FlowBodyRollout
        Rollout object defining the body dynamics.
    reward_fn : callable
        Callable with signature ``reward_fn(position, orientation, dofs)``.
    value_fn : callable, optional
        Optional baseline or critic with the same state signature as
        ``reward_fn``.

    Notes
    -----
    ``FlowBodyRL`` expects rollout stepping attributes that are still evolving.
    Treat it as an experimental helper for method development.

    Examples
    --------
    >>> def reward_fn(pos, ori, dofs):
    ...     return pos[2]
    >>> rl = FlowBodyRL(rollout, reward_fn, value_fn=None)
    >>> design = rl.run(init_design, init_position, init_orientation, init_dofs)
    """

    def __init__(self, rollout, reward_fn, value_fn=None):
        self.rollout = rollout
        self.reward_fn = reward_fn
        self.value_fn = value_fn  # None → pure policy gradient (no baseline)

    def _episode_return(self, design, init_position, init_orientation, init_dofs, gamma=1.0):
        """Compute discounted return for one episode — differentiable w.r.t. design."""
        carry = (init_position, init_orientation, init_dofs)
        gammas = gamma ** jnp.arange(self.rollout.n_steps)

        def step_with_reward(carry, inputs):
            t, discount = inputs
            (pos, ori, dofs), (new_pos, new_ori, new_dofs) = self.rollout.step(carry, t, design=design)
            reward = self.reward_fn(new_pos, new_ori, new_dofs)
            baseline = self.value_fn(pos, ori, dofs) if self.value_fn else 0.0
            return (new_pos, new_ori, new_dofs), discount * (reward - baseline)

        _, returns = jax.lax.scan(step_with_reward, carry, (jnp.arange(self.rollout.n_steps), gammas))
        return jnp.sum(returns)

    def run(
        self,
        init_design,
        init_position,
        init_orientation,
        init_dofs,
        n_steps=500,
        optimizer=None,
        print_every=100,
        gamma=1.0,
    ):
        """
        Optimize design variables using discounted rollout rewards.

        Parameters
        ----------
        init_design : array-like
            Initial design vector.
        init_position : array-like, shape (3,)
            Initial body-reference position.
        init_orientation : array-like, shape (3,)
            Initial Rodrigues orientation vector.
        init_dofs : array-like
            Initial degrees of freedom.
        n_steps : int, default=500
            Number of optimizer updates.
        optimizer : optax.GradientTransformation, optional
            Optax optimizer. Defaults to Adam with learning rate ``1e-3``.
        print_every : int, default=100
            Print progress every ``print_every`` updates.
        gamma : float, default=1.0
            Discount factor for rewards.

        Returns
        -------
        jnp.ndarray
            Optimized design vector.
        """

        optimizer = optimizer or optax.adam(1e-3)
        design = jnp.atleast_1d(jnp.array(init_design, dtype=float))
        opt_state = optimizer.init(design)

        grad_fn = jax.jit(
            jax.value_and_grad(
                lambda d: -self._episode_return(d, init_position, init_orientation, init_dofs, gamma)
            )
        )

        for step in range(n_steps):
            loss, grad = grad_fn(design)
            updates, opt_state = optimizer.update(grad, opt_state)
            design = optax.apply_updates(design, updates)

            if step % print_every == 0:
                print(f"step {step:4d}  return={float(-loss):.5f}  " f"|grad|={float(jnp.linalg.norm(grad)):.5f}")

        return design


# Useful functions for rotation with rotation vector ########################


@jax.jit
def rescale_orientation(rvec):
    """
    Keep a Rodrigues orientation vector within the principal rotation range.

    Parameters
    ----------
    rvec : array-like, shape (3,)
        Rodrigues rotation vector.

    Returns
    -------
    jnp.ndarray
        Rescaled rotation vector. Vectors with norm below ``pi`` are returned
        unchanged.
    """
    rvec = jnp.asarray(rvec, dtype=float)
    r_sq = jnp.dot(rvec, rvec)
    safe_r = jnp.sqrt(jnp.maximum(r_sq, 1e-12))  # gradient is 1/(2*sqrt(max(r²,ε))), always finite
    rescaled = rvec - 2 * jnp.pi * rvec / safe_r
    return jnp.where(r_sq >= jnp.pi**2, rescaled, rvec)


@jax.jit
def compute_bortz_operator(rvec):
    """
    Compute the Bortz operator for a Rodrigues vector.

    The operator maps angular velocity to the time derivative of the Rodrigues
    vector in the integration scheme.

    Parameters
    ----------
    rvec : array-like, shape (3,)
        Rodrigues rotation vector.

    Returns
    -------
    jnp.ndarray
        Matrix of shape ``(3, 3)``.
    """
    rvec = jnp.asarray(rvec, dtype=float)
    r_sq = jnp.dot(rvec, rvec)
    safe_r = jnp.sqrt(jnp.maximum(r_sq, 1e-12))  # ← same fix

    runit = rvec / safe_r
    kx, ky, kz = runit
    runitcross = jnp.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])

    half_theta = safe_r / 2.0
    cot_coeff = jnp.where(r_sq < 1e-12, 1.0 - r_sq / 12.0, half_theta / jnp.tan(half_theta))

    term1 = -half_theta * runitcross
    term2 = cot_coeff * jnp.eye(3)
    term3 = (1.0 - cot_coeff) * jnp.outer(runit, runit)

    return jnp.where(r_sq < 1e-12, jnp.eye(3), term1 + term2 + term3)


def _rotation_matrix_from_Rodrigues_impl(rvec, num_dofs):
    rvec = jnp.asarray(rvec, dtype=float)
    r_sq = jnp.dot(rvec, rvec)
    safe_r = jnp.sqrt(jnp.maximum(r_sq, 1e-12))  # ← key fix
    runit = rvec / safe_r

    kx, ky, kz = runit
    runitcross = jnp.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])

    R = jnp.eye(3) + jnp.sin(safe_r) * runitcross + (1 - jnp.cos(safe_r)) * runitcross @ runitcross

    sixc_R = jnp.block(
        [
            [R, jnp.zeros((3, 3)), jnp.zeros((3, num_dofs))],
            [jnp.zeros((3, 3)), R, jnp.zeros((3, num_dofs))],
            [jnp.zeros((num_dofs, 6)), jnp.eye(num_dofs)],
        ]
    )

    # Return identity at zero — value correct, gradient finite
    R_out = jnp.where(r_sq < 1e-12, jnp.eye(3), R)
    sixc_out = jnp.where(r_sq < 1e-12, jnp.eye(6 + num_dofs), sixc_R)
    return R_out, sixc_out


rotation_matrix_from_Rodrigues = jax.jit(
    lambda rvec, Ndof: _rotation_matrix_from_Rodrigues_impl(rvec, Ndof), static_argnums=(1,)
)


@jax.jit
def rotation_matrix(rvec):
    """
    Convert a Rodrigues vector to a rotation matrix.

    Parameters
    ----------
    rvec : array-like, shape (3,)
        Rodrigues rotation vector.

    Returns
    -------
    jnp.ndarray
        Rotation matrix of shape ``(3, 3)``.

    Examples
    --------
    The function accepts NumPy arrays or Python lists, so notebooks that do
    not import ``jax.numpy`` can still call it. Wrap with ``np.asarray`` if a
    strictly NumPy result is needed:

    >>> import numpy as np
    >>> import softmobility as sm
    >>> R = np.asarray(sm.rotation_matrix(np.array([0.0, 0.0, np.pi / 2])))
    >>> R.round(6)
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.]])
    """
    rvec = jnp.asarray(rvec, dtype=float)
    r_sq = jnp.dot(rvec, rvec)
    safe_r = jnp.sqrt(jnp.maximum(r_sq, 1e-12))
    runit = rvec / safe_r

    kx, ky, kz = runit
    runitcross = jnp.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])

    R = jnp.eye(3) + jnp.sin(safe_r) * runitcross + (1 - jnp.cos(safe_r)) * runitcross @ runitcross

    # Return identity at zero — value correct, gradient finite
    R_out = jnp.where(r_sq < 1e-12, jnp.eye(3), R)
    return R_out

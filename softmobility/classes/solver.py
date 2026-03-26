# softmobility/jax_solver.py
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
from jax import lax
import optax

from softmobility import SoftBody, Flow, Field, Scalar

# =============================================================================
# FlowBodyRollout
# =============================================================================


class FlowBodyRollout:
    """
    Pure-functional rollout to simulate the advection and deformation of a soft body in a flow.
    Compatible with jax.jit, jax.grad, jax.vmap.

    Usage
    -----
    rollout = FlowBodyRollout(soft_body, flow, input_map, dt, n_steps)
    pos, ori, dofs = rollout.rollout(design, init_pos, init_ori, init_dofs)

    # For optimization:
    grad_fn = jax.jit(jax.grad(lambda d: rollout.rollout(d, ...)[0][2]))
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

    def velocity(self, design, position, orientation, dofs, time):
        """Returns (v_lab, omega_lab, dot_dofs), the soft body's linear velocity,
        angular velocity and derivative of degrees-of-freedom, all in lab frame."""
        rot, sixc_rot = rotation_matrix_from_Rodrigues(orientation, Ndof=self.soft_body.Ndof)
        inputs = self._build_inputs(design, position, time, rot)
        u_lab = self.flow.velocity(position, time)
        omega_lab, E_lab = self.flow.omega_rate_of_strain(position, time)
        E_body = rot.T @ E_lab @ rot
        E_inf = jnp.array([E_body[0, 0], E_body[0, 1], E_body[0, 2], E_body[1, 1], E_body[1, 2]])
        tensors = self.soft_body.compute_mobility_problem(dofs, design)

        # Soft mobility equation in the body frame
        p_body = tensors.M_H @ inputs + tensors.M_K @ dofs + tensors.C_E @ E_inf

        # Rotate body-frame result back to lab frame
        p_lab = sixc_rot @ p_body

        v_lab = p_lab[:3] + u_lab
        omega = p_lab[3:6] + omega_lab
        dot_dofs = p_lab[6:]

        return v_lab, omega, dot_dofs

    def step(self, carry, t, design, dt):
        """Time stepping the soft mobility equation, lax.scan compatible"""
        position, orientation, dofs = carry
        time = t * dt
        bortz = compute_Bortz_operator(orientation)

        def vel(pos, ori, dof, time):
            return self.velocity(design, pos, ori, dof, time)

        v1, w1, d1 = vel(position, orientation, dofs, time)

        v2, w2, d2 = vel(
            position + dt * v1 / 2,
            orientation + dt * bortz @ w1 / 2,
            dofs + dt * d1 / 2,
            time + dt / 2,
        )

        pos_new = position + dt * (v1 + v2) / 2
        ori_new = orientation + dt * bortz @ (w1 + w2) / 2
        ori_new = rescale_orientation(ori_new)
        dof_new = dofs + dt * (d1 + d2) / 2

        return (pos_new, ori_new, dof_new), (pos_new, ori_new, dof_new)

    def rollout(
        self, dt, n_steps, init_position=jnp.zeros(3), init_orientation=jnp.zeros(3), init_dofs=None, design=None
    ):
        """
        Simulate n_steps from initial state.
        Returns: positions (N,3), orientations (N,3), dofs (N,Ndof)

        This is the core function to pass to jit/grad/vmap.
        """
        init_dofs = jnp.array(self.soft_body.dof_defaults) if init_dofs is None else jnp.asarray(init_dofs)
        design = jnp.asarray(self.soft_body.design_defaults) if design is None else jnp.asarray(design)

        dt = jnp.asarray(dt, dtype=float)
        carry = (init_position, init_orientation, init_dofs)
        _, (positions, orientations, dofs) = jax.lax.scan(
            partial(self.step, design=design, dt=dt), carry, jnp.arange(n_steps)
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
    Gradient-based optimizer for design parameters.

    The objective function has signature:
        objective(rollout, design) -> scalar

    Examples
    --------
    def my_objective(rollout, design):
        positions, _, _ = rollout.rollout(design, init_pos, init_ori, init_dofs)
        return positions[-1, 2] / final_time   # mean Z velocity

    opt = FlowBodyOptimizer(rollout, my_objective, optax.adam(1e-3))
    optimal_design = opt.run(init_design, n_steps=500)
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
    Online actor-critic RL where:
      - Actor  : design parameters (optimized by policy gradient or grad ascent)
      - Critic : user-supplied value network V(position, orientation, dofs) -> scalar
      - Env    : FlowBodyRollout.step (the Markov transition)

    The architecture is compatible with the optimizer above because the actor
    is just `design` and the environment step is already a pure function.

    reward_fn signature:  reward_fn(position, orientation, dofs) -> scalar
    value_fn signature:   value_fn(position, orientation, dofs) -> scalar  (critic)

    Examples
    --------
    def reward_fn(pos, ori, dofs):
        return pos[2]   # instantaneous Z position gain

    rl = FlowBodyRL(rollout, reward_fn, value_fn=None)
    design = rl.run(init_design, init_position, init_orientation, init_dofs)
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
    Rescale the orientation vector to avoid singularities.
    """
    rvec = jnp.array(rvec)
    r = jnp.linalg.norm(rvec)

    def rescale(_):
        return rvec - 2 * jnp.pi * rvec / r

    return lax.cond(r >= jnp.pi, rescale, lambda _: rvec, None)


@jax.jit
def compute_Bortz_operator(rvec):
    """
    Compute the time derivative of the orientation vector using the Bortz formula.
    """
    rvec = jnp.array(rvec)
    theta = jnp.linalg.norm(rvec)

    def small_r_case(_):
        """Return omega directly if r is small."""
        return jnp.eye(3)

    def normal_case(_):
        """Compute Bortz derivative normally."""
        runit = rvec / theta
        kx, ky, kz = runit
        runitcross = jnp.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        term1 = -(theta / 2) * runitcross
        term2 = (theta / 2) / jnp.tan(theta / 2) * jnp.eye(3)
        term3 = (1 - (theta / 2) / jnp.tan(theta / 2)) * jnp.outer(runit, runit)
        return term1 + term2 + term3

    return lax.cond(theta < 1e-6, small_r_case, normal_case, None)


rotation_matrix_from_Rodrigues = jax.jit(
    lambda rvec, Ndof: _rotation_matrix_from_Rodrigues_impl(rvec, Ndof), static_argnums=(1,)
)


def _rotation_matrix_from_Rodrigues_impl(rvec, Ndof):
    """
    Rotation matrix from rotation vector r using Rodrigues' rotation formula.
    """
    rvec = jnp.array(rvec)
    theta = jnp.linalg.norm(rvec)

    def no_rotation(_):
        """Return identity matrix when theta is very small."""
        R = jnp.eye(3)
        sixc_R = jnp.eye(Ndof + 6)
        return R, sixc_R

    def compute_rotation(_):
        """Compute Rodrigues' rotation matrix."""
        runit = rvec / theta
        kx, ky, kz = runit
        runitcross = jnp.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        R = jnp.eye(3) + jnp.sin(theta) * runitcross + (1 - jnp.cos(theta)) * jnp.dot(runitcross, runitcross)

        sixc_R = jnp.block(
            [
                [R, jnp.zeros((3, 3)), jnp.zeros((3, Ndof))],  # pos
                [jnp.zeros((3, 3)), R, jnp.zeros((3, Ndof))],  # ori
                [jnp.zeros((Ndof, 6)), jnp.eye(Ndof)],  # dof
            ]
        )
        return R, sixc_R

    return lax.cond(theta < 1e-6, no_rotation, compute_rotation, None)

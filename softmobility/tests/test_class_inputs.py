import jax
import jax.numpy as jnp
import pytest

from softmobility import (
    Field,
    Flow,
    Scalar,
    constant_scalar,
    extensional_flow,
    gravity_field,
    no_flow,
    oscillating_magnetic_field,
    oscillating_scalar,
    rotating_flow,
    rotating_magnetic_field,
    shear_flow,
    taylor_green_flow,
)


def test_named_scalar_constructors_evaluate_and_update():
    constant = constant_scalar(2.0)
    assert jnp.allclose(constant.value(), 2.0)

    constant.update_params(value=3.0)
    assert jnp.allclose(constant.value(), 3.0)

    signal = oscillating_scalar(amplitude=2.0, omega=3.0, phase=0.5)
    assert jnp.allclose(signal.value(time=0.25), 2.0 * jnp.sin(3.0 * 0.25 + 0.5))


def test_named_field_constructors_evaluate_and_update():
    gravity = gravity_field(g=9.81)
    assert jnp.allclose(gravity.vector(), jnp.array([0.0, 0.0, -9.81]))

    gravity.update_params(g=1.0)
    assert jnp.allclose(gravity.vector(), jnp.array([0.0, 0.0, -1.0]))

    rotating = rotating_magnetic_field(amp_x=1.0, amp_y=2.0, omega=3.0)
    assert jnp.allclose(
        rotating.vector(time=0.25),
        jnp.array([1.0, 2.0 * jnp.cos(0.75), 2.0 * jnp.sin(0.75)]),
    )

    oscillating = oscillating_magnetic_field(amp_x=1.0, amp_y=2.0, omega=3.0)
    assert jnp.allclose(
        oscillating.vector(time=0.25),
        jnp.array([1.0, 2.0 * jnp.sin(0.75), 0.0]),
    )


def test_named_flow_constructors_velocity_and_gradient():
    pos = jnp.array([1.0, 2.0, 3.0])

    assert jnp.allclose(no_flow().velocity(pos), jnp.zeros(3))
    assert jnp.allclose(shear_flow(2.0).velocity(pos), jnp.array([4.0, 0.0, 0.0]))
    assert jnp.allclose(rotating_flow(2.0).velocity(pos), jnp.array([-4.0, 2.0, 0.0]))
    assert jnp.allclose(extensional_flow(2.0).velocity(pos), jnp.array([2.0, -2.0, -3.0]))

    taylor = taylor_green_flow(2.0)
    assert jnp.allclose(
        taylor.velocity(pos),
        jnp.array([0.0, jnp.sin(2.0) * jnp.cos(3.0), -jnp.cos(2.0) * jnp.sin(3.0)]),
    )

    assert shear_flow(2.0).gradient(pos).shape == (3, 3)
    omega, strain = shear_flow(2.0).omega_rate_of_strain(pos)
    assert omega.shape == (3,)
    assert strain.shape == (3, 3)


def test_field_and_flow_validate_return_shape():
    bad_field = Field(lambda pos, time: jnp.array([1.0, 2.0]))
    bad_flow = Flow(lambda pos, time: jnp.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="Field must return a"):
        bad_field.vector()

    with pytest.raises(ValueError, match="Flow must return a"):
        bad_flow.velocity()


def test_parametric_inputs_are_jittable_and_differentiable():
    scalar = Scalar(lambda pos, time, p: p[0] * pos[0] + time, params=2.0, param_names="gain")
    field = Field(lambda pos, time, p: p[0] * pos, params=3.0, param_names="gain")
    flow = Flow(lambda pos, time, p: jnp.array([p[0] * pos[1], 0.0, 0.0]), params=4.0, param_names="gain")
    pos = jnp.array([1.0, 2.0, 3.0])

    assert jnp.allclose(jax.jit(scalar.value)(pos, 0.5), 2.5)
    assert jnp.allclose(jax.jit(field.vector)(pos, 0.0), 3.0 * pos)
    assert jnp.allclose(jax.jit(flow.velocity)(pos, 0.0), jnp.array([8.0, 0.0, 0.0]))

    grad = jax.grad(lambda x: flow.velocity(jnp.array([1.0, x, 3.0]), 0.0)[0])(2.0)
    assert jnp.allclose(grad, 4.0)

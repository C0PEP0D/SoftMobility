import jax.numpy as jnp

from softmobility import Sphere


def test_radius_entries() -> None:
    try:
        s = Sphere(radius="a")
    except TypeError as e:
        assert str(e) == "radius must be a callable or a scalar."

    s = Sphere(radius=1)

    radius_func = lambda _, __: 2.0
    s = Sphere(radius=radius_func)

    try:
        radius_func = lambda _: 1
        s = Sphere(radius=radius_func)
    except ValueError as e:
        assert str(e) == "radius must accept exactly two arguments: 'dofs', 'design'."


def test_position_entries():
    try:
        s = Sphere(position="a")
    except TypeError as e:
        assert str(e) == "position must be a callable, an array, or a list."

    s = Sphere(position=[1, 0, 0])

    position_func = lambda _, __, ___: [2.0, 0.0, 0.0]
    s = Sphere(position=position_func)

    try:
        position_func = lambda _: 1
        s = Sphere(position=position_func)
    except ValueError as e:
        assert str(e) == "position must accept exactly three arguments: 'dofs', 'design', 'time'."

    try:
        s = Sphere(position=[1, 2])
    except ValueError as e:
        assert str(e) == "position must have shape (3,), but got (2,)."


def test_orientation_entries():
    try:
        s = Sphere(orientation="a")
    except TypeError as e:
        assert str(e) == "orientation must be a callable, an array, or a list."

    s = Sphere(orientation=[1, 0, 0])

    orientation_func = lambda _, __, ___: [2.0, 0.0, 0.0]
    s = Sphere(orientation=orientation_func)

    try:
        orientation_func = lambda _: [2.0, 0.0, 0.0]
        s = Sphere(orientation=orientation_func)
    except ValueError as e:
        assert str(e) == "orientation must accept exactly three arguments: 'dofs', 'design', 'time'."

    try:
        s = Sphere(orientation=[1, 2, 3, 4])
    except ValueError as e:
        assert str(e) == "orientation must have shape (3,), but got (4,)."


def test_force_and_torque_entries():
    s = Sphere(
        force=[lambda dofs, design, inputs: inputs[0], 2.0, 0.0],
        torque=lambda dofs, design, inputs: jnp.array([0.0, inputs[1], 3.0]),
    )
    force = s.force(jnp.zeros(0), jnp.zeros(0), jnp.array([4.0, 5.0]))
    torque = s.torque(jnp.zeros(0), jnp.zeros(0), jnp.array([4.0, 5.0]))
    six_component_force = s.six_component_force(jnp.zeros(0), jnp.zeros(0), jnp.array([4.0, 5.0]))

    assert jnp.allclose(force, jnp.array([4.0, 2.0, 0.0]))
    assert jnp.allclose(torque, jnp.array([0.0, 5.0, 3.0]))
    assert jnp.allclose(six_component_force, jnp.array([4.0, 2.0, 0.0, 0.0, 5.0, 3.0]))

    try:
        Sphere(force=lambda dofs, design: jnp.zeros(3))
    except ValueError as e:
        assert str(e) == "force must accept exactly three arguments: 'dofs', 'design', 'inputs'."

    try:
        Sphere(force=[1.0, 2.0])
    except ValueError as e:
        assert str(e) == "force must have shape (3,), but got (2,)."

    try:
        Sphere(force=[0.0, 0.0, 0.0], C_H=jnp.zeros((6, 0)))
    except ValueError as e:
        assert str(e) == "Specify either force/torque or C_H/C_K, not both."

    s = Sphere(1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], C_H=jnp.ones((6, 0)), C_K=2.0 * jnp.ones((6, 0))) 
    assert jnp.allclose(s.C_H(jnp.zeros(0), jnp.zeros(0)), jnp.ones((6, 0)))
    assert jnp.allclose(s.C_K(jnp.zeros(0), jnp.zeros(0)), 2.0 * jnp.ones((6, 0)))

import jax.numpy as jnp

from softmobility import Sphere


def test_radius_entries() -> None:
    try:
        Sphere(radius="a")
    except TypeError as e:
        assert str(e) == "radius must be a callable or a scalar."

    Sphere(radius=1)

    def radius_func(_, __):
        return 2.0
    Sphere(radius=radius_func)

    try:
        def radius_func(_):
            return 1
        Sphere(radius=radius_func)
    except ValueError as e:
        assert str(e) == "radius must accept exactly two arguments: 'dofs', 'design'."


def test_position_entries():
    try:
        Sphere(position="a")
    except TypeError as e:
        assert str(e) == "position must be a callable, an array, or a list."

    Sphere(position=[1, 0, 0])

    def position_func(_, __, ___):
        return [2.0, 0.0, 0.0]
    Sphere(position=position_func)

    try:
        def position_func(_):
            return 1
        Sphere(position=position_func)
    except ValueError as e:
        assert str(e) == "position must accept exactly three arguments: 'dofs', 'design', 'time'."

    try:
        Sphere(position=[1, 2])
    except ValueError as e:
        assert str(e) == "position must have shape (3,), but got (2,)."


def test_orientation_entries():
    try:
        Sphere(orientation="a")
    except TypeError as e:
        assert str(e) == "orientation must be a callable, an array, or a list."

    Sphere(orientation=[1, 0, 0])

    def orientation_func(_, __, ___):
        return [2.0, 0.0, 0.0]
    Sphere(orientation=orientation_func)

    try:
        def orientation_func(_):
            return [2.0, 0.0, 0.0]
        Sphere(orientation=orientation_func)
    except ValueError as e:
        assert str(e) == "orientation must accept exactly three arguments: 'dofs', 'design', 'time'."

    try:
        Sphere(orientation=[1, 2, 3, 4])
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

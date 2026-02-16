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
        assert str(e) == "radius must accept exactly two arguments: 'dofs' and 'params'."


def test_position_entries():
    try:
        s = Sphere(position="a")
    except TypeError as e:
        assert str(e) == "position must be a callable, an array, or a list."

    s = Sphere(position=[1, 0, 0])

    position_func = lambda _, __: [2.0, 0.0, 0.0]
    s = Sphere(position=position_func)

    try:
        position_func = lambda _: 1
        s = Sphere(position=position_func)
    except ValueError as e:
        assert str(e) == "position must accept exactly two arguments: 'dofs' and 'params'."

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

    orientation_func = lambda _, __: [2.0, 0.0, 0.0]
    s = Sphere(orientation=orientation_func)

    try:
        orientation_func = lambda _: [2.0, 0.0, 0.0]
        s = Sphere(orientation=orientation_func)
    except ValueError as e:
        assert str(e) == "orientation must accept exactly two arguments: 'dofs' and 'params'."

    try:
        s = Sphere(orientation=[1, 2, 3, 4])
    except ValueError as e:
        assert str(e) == "orientation must have shape (3,), but got (4,)."


def test_force_entries():
    try:
        s = Sphere(force="string")
    except TypeError as e:
        assert str(e) == "force must be a callable, an array, or a list."

    s = Sphere(force=[1, 0, 0])

    force_func = lambda _, __: [2.0, 0.0, 0.0]
    s = Sphere(force=force_func)

    try:
        force_func = lambda _: 1
        s = Sphere(force=force_func)
    except ValueError as e:
        assert str(e) == "force must accept exactly two arguments: 'dofs' and 'params'."

    try:
        s = Sphere(force=[1, 2])
    except ValueError as e:
        assert str(e) == "force must have shape (3,), but got (2,)."


def test_torque_entries():
    try:
        s = Sphere(torque="a")
    except TypeError as e:
        assert str(e) == "torque must be a callable, an array, or a list."

    s = Sphere(torque=[1, 0, 0])

    torque_func = lambda _, __: [2.0, 0.0, 0.0]
    s = Sphere(torque=torque_func)

    try:
        torque_func = lambda _: 1
        s = Sphere(torque=torque_func)
    except ValueError as e:
        assert str(e) == "torque must accept exactly two arguments: 'dofs' and 'params'."

    try:
        s = Sphere(torque=[1, 2])
    except ValueError as e:
        assert str(e) == "torque must have shape (3,), but got (2,)."

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

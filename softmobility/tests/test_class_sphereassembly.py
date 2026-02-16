import jax.numpy as jnp
from softmobility import Sphere, SphereAssembly


def test_init():
    sa = SphereAssembly()
    assert True
    assert sa.Ndof == 0
    assert sa.Nparam == 0
    assert sa.Nspheres == 0
    assert isinstance(sa.spheres, list)
    assert len(sa.spheres) == sa.Nspheres


def test_add_sphere():
    sa = SphereAssembly()
    sa.add_sphere(sphere=Sphere(radius=1))
    assert len(sa.spheres) == 1
    assert sa.Nspheres == 1


def test_init_from_file():
    sa = SphereAssembly("./softmobility/tests/parameters.yaml")
    assert str(sa) == "Assembly with 2 spheres, 2 degrees of freedom, and 8 fixed parameters"
    assert sa.dof_variables == ["x0", "x1"]
    assert sa.param_variables == [
        "distance",
        "gravity0",
        "gravity1",
        "gravity2",
        "k",
        "mass0",
        "mass1",
        "myradius",
    ]

    assert jnp.allclose(sa.dof_defaults, jnp.array([1, 0])).item()
    assert jnp.allclose(sa.param_defaults, jnp.array([1, 0, 0, 0, 1, 0, 0, 0.25])).item()


def test_set_dof_defaults():
    sa = SphereAssembly("./softmobility/tests/parameters.yaml")
    sa.set_dof_defaults(new_dict={"x0": 3})
    assert jnp.allclose(sa.dof_defaults, jnp.array([3, 0])).item()
    sa.set_dof_defaults(new_dofs=[-1, 1])
    assert jnp.allclose(sa.dof_defaults, jnp.array([-1, 1])).item()
    try:
        sa.set_dof_defaults(new_dict={"x2": 3})
    except ValueError as e:
        assert str(e) == "Invalid variable name: x2"
    try:
        sa.set_dof_defaults(new_dofs=[1])
    except ValueError as e:
        assert str(e) == "new_dofs array must have shape (2,)"


def test_set_param_defaults():
    sa = SphereAssembly("./softmobility/tests/parameters.yaml")
    sa.set_param_defaults(new_dict={"distance": 3})
    assert jnp.allclose(sa.param_defaults, jnp.array([3, 0, 0, 0, 1, 0, 0, 0.25])).item()
    sa.set_param_defaults(new_params=[1, 1, 0.5, 0, 0, 0, 1, -1.2])
    assert jnp.allclose(sa.param_defaults, jnp.array([1, 1, 0.5, 0, 0, 0, 1, -1.2])).item()
    try:
        sa.set_param_defaults(new_dict={"x2": 3})
    except ValueError as e:
        assert str(e) == "Invalid variable name: x2"
    try:
        sa.set_param_defaults(new_params=[1, 0])
    except ValueError as e:
        assert str(e) == "new_params array must have shape (8,)"


def test_kinematic_tensors():
    sa = SphereAssembly("./softmobility/tests/parameters.yaml")

    # compute_velocity_matrix()
    M1 = sa.compute_Jass()
    M1test = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.84147096],
            [0.0, 0.45969772],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-0.33333334, 0.0],
            [0.0, -0.32719472],
            [0.0, 0.05504304],
        ]
    )
    assert jnp.allclose(M1, M1test).item()

    # compute_composition_of_velocity()
    M2 = sa.compute_C_U()
    M2test = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, -0.0],
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, -0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert jnp.allclose(M2, M2test)

    M3 = sa.compute_Jacobian_matrix()
    print(M3)
    M3test = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.84147096][0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.45969772][
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
            ],
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.33333334, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.32719472],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.05504304],
        ]
    )
    assert jnp.allclose(M3, M3test)

    M4 = sa.compute_stiffness_matrix()
    M4test = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-1.0, -0.0],
            [-0.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    assert jnp.allclose(M4, M4test)

    M5 = sa.compute_C_U()
    M5test = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, -0.0],
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, -0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert jnp.allclose(M5, M5test)

    M6 = sa.compute_composition_of_forces()
    M6test = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )


test_kinematic_tensors()

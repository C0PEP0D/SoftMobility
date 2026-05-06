import jax.numpy as jnp
import numpy as np
import pytest

from softmobility import Sphere, SphereAssembly

CALLABLE_COUPLING_YAML = """
dof_names: [x]
design_names: [k]
input_names: [gravity, active_force]

defaults:
  x0: 0.5
  k: 2.0

spheres:
  - radius: 1
    position: [-1, 0, 0]
    force: [gravity0, 0, 0]
    torque: [-k * x0 + active_force, 0, 0]

  - radius: 1
    position: [1, 0, 0]
    orientation: [0, 0, x0]
    force: [-gravity0, 0, 0]
    torque: [k * x0 - active_force, 0, 0]
"""


def _callable_coupling_assembly():
    sa = SphereAssembly()
    sa.add_dof("x0", default=0.5)
    sa.add_design("k", default=2.0)
    sa.add_input("gravity", kind="field")
    sa.add_input("active_force")

    sa.add_sphere(
        Sphere(
            radius=1,
            position=[-1, 0, 0],
            force=[lambda dofs, design, inputs: inputs[0], 0.0, 0.0],
            torque=[lambda dofs, design, inputs: -design[0] * dofs[0] + inputs[3], 0.0, 0.0],
        )
    )
    sa.add_sphere(
        Sphere(
            radius=1,
            position=[1, 0, 0],
            orientation=lambda dofs, design, time: jnp.array([0.0, 0.0, dofs[0]]),
            force=[lambda dofs, design, inputs: -inputs[0], 0.0, 0.0],
            torque=[lambda dofs, design, inputs: design[0] * dofs[0] - inputs[3], 0.0, 0.0],
        )
    )

    return sa


def test_init():
    sa = SphereAssembly()
    assert True
    assert sa.Ndof == 0
    assert sa.Ndesign == 0
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
    assert sa.design_variables == [
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
    assert jnp.allclose(sa.design_defaults, jnp.array([1, 0, 0, 0, 1, 0, 0, 0.25])).item()


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
    sa.set_design_defaults(new_dict={"distance": 3})
    assert jnp.allclose(sa.design_defaults, jnp.array([3, 0, 0, 0, 1, 0, 0, 0.25])).item()
    sa.set_design_defaults(new_design=[1, 1, 0.5, 0, 0, 0, 1, -1.2])
    assert jnp.allclose(sa.design_defaults, jnp.array([1, 1, 0.5, 0, 0, 0, 1, -1.2])).item()
    try:
        sa.set_design_defaults(new_dict={"x2": 3})
    except ValueError as e:
        assert str(e) == "Invalid variable name: x2"
    try:
        sa.set_design_defaults(new_design=[1, 0])
    except ValueError as e:
        assert str(e) == "new_design array must have shape (8,)"


def test_kinematic_tensors():
    sa = SphereAssembly("./softmobility/tests/parameters.yaml")

    # compute_velocity_matrix()
    M1, _ = sa.compute_Jassembly()
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

    M3, _ = sa.compute_Jacobian_matrix()
    print(M3)
    M3test = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.84147096],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.45969772],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.33333334, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.32719472],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.05504304],
        ]
    )
    assert jnp.allclose(M3, M3test)

    M4 = sa.grand_C_K()
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


def test_callable_force_torque_couplings_match_yaml():
    yaml_assembly = SphereAssembly(CALLABLE_COUPLING_YAML, verbose=False)
    callable_assembly = _callable_coupling_assembly()

    assert callable_assembly.dof_variables == yaml_assembly.dof_variables
    assert callable_assembly.design_variables == yaml_assembly.design_variables
    assert callable_assembly.input_variables == yaml_assembly.input_variables
    assert jnp.allclose(callable_assembly.dof_defaults, yaml_assembly.dof_defaults)
    assert jnp.allclose(callable_assembly.design_defaults, yaml_assembly.design_defaults)
    assert jnp.allclose(callable_assembly.grand_C_H(), yaml_assembly.grand_C_H())
    assert jnp.allclose(callable_assembly.grand_C_K(), yaml_assembly.grand_C_K())


def test_callable_force_torque_input_nonlinearity_raises():
    sa = SphereAssembly()
    sa.add_input("active_force")

    with pytest.raises(ValueError, match="force/torque must be linear in inputs"):
        sa.add_sphere(Sphere(force=[lambda dofs, design, inputs: inputs[0] ** 2, 0.0, 0.0]))


def test_explicit_couplings_still_work():
    sa = SphereAssembly()
    sa.add_dof("x0")
    sa.add_input("active_force")

    C_H = jnp.ones((6, 1))
    C_K = 2.0 * jnp.ones((6, 1))
    sa.add_sphere(Sphere(C_H=C_H, C_K=C_K))

    assert jnp.allclose(sa.grand_C_H(), C_H)
    assert jnp.allclose(sa.grand_C_K(), C_K)


def _numpy_callable_assembly():
    """Same dumbbell as CALLABLE_COUPLING_YAML but using np.array and list returns."""
    sa = SphereAssembly()
    sa.add_dof("x0", default=0.5)
    sa.add_design("k", default=2.0)
    sa.add_input("gravity", kind="field")
    sa.add_input("active_force")

    sa.add_sphere(
        Sphere(
            radius=1,
            position=np.array([-1, 0, 0]),
            force=lambda dofs, design, inputs: np.array([inputs[0], 0.0, 0.0]),
            torque=lambda dofs, design, inputs: np.array([-design[0] * dofs[0] + inputs[3], 0.0, 0.0]),
        )
    )
    sa.add_sphere(
        Sphere(
            radius=1,
            position=np.array([1, 0, 0]),
            orientation=lambda dofs, design, time: np.array([0.0, 0.0, dofs[0]]),
            force=lambda dofs, design, inputs: np.array([-inputs[0], 0.0, 0.0]),
            torque=lambda dofs, design, inputs: np.array([design[0] * dofs[0] - inputs[3], 0.0, 0.0]),
        )
    )
    return sa


def _list_callable_assembly():
    """Same dumbbell using Python list returns instead of np.array."""
    sa = SphereAssembly()
    sa.add_dof("x0", default=0.5)
    sa.add_design("k", default=2.0)
    sa.add_input("gravity", kind="field")
    sa.add_input("active_force")

    sa.add_sphere(
        Sphere(
            radius=1,
            position=[-1, 0, 0],
            force=lambda dofs, design, inputs: [inputs[0], 0.0, 0.0],
            torque=lambda dofs, design, inputs: [-design[0] * dofs[0] + inputs[3], 0.0, 0.0],
        )
    )
    sa.add_sphere(
        Sphere(
            radius=1,
            position=[1, 0, 0],
            orientation=lambda dofs, design, time: [0.0, 0.0, dofs[0]],
            force=lambda dofs, design, inputs: [-inputs[0], 0.0, 0.0],
            torque=lambda dofs, design, inputs: [design[0] * dofs[0] - inputs[3], 0.0, 0.0],
        )
    )
    return sa


def test_numpy_array_callables_match_yaml():
    """np.array-returning callables must produce the same C_H and C_K as YAML."""
    yaml_assembly = SphereAssembly(CALLABLE_COUPLING_YAML, verbose=False)
    np_assembly = _numpy_callable_assembly()

    assert jnp.allclose(np_assembly.grand_C_H(), yaml_assembly.grand_C_H(), atol=1e-6)
    assert jnp.allclose(np_assembly.grand_C_K(), yaml_assembly.grand_C_K(), atol=1e-6)


def test_list_callables_match_yaml():
    """List-returning callables must produce the same C_H and C_K as YAML."""
    yaml_assembly = SphereAssembly(CALLABLE_COUPLING_YAML, verbose=False)
    list_assembly = _list_callable_assembly()

    assert jnp.allclose(list_assembly.grand_C_H(), yaml_assembly.grand_C_H(), atol=1e-6)
    assert jnp.allclose(list_assembly.grand_C_K(), yaml_assembly.grand_C_K(), atol=1e-6)


def test_numpy_nonlinearity_raises():
    """np.array-returning force that is nonlinear in inputs must still raise."""
    sa = SphereAssembly()
    sa.add_input("active_force")

    with pytest.raises(ValueError, match="force/torque must be linear in inputs"):
        sa.add_sphere(Sphere(force=lambda dofs, design, inputs: np.array([inputs[0] ** 2, 0.0, 0.0])))

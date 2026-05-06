import jax.numpy as jnp
import numpy as np
import pytest
from softmobility import SoftBody, Sphere


def test_mobility_matrices():
    sp = SoftBody("./softmobility/tests/parameters.yaml")
    M = sp.compute_mobility_tensor()
    Mexpected = jnp.array(
        [
            [0.21220659, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04807806, 0.0, 0.0, 0.0, -0.03978874, 0.0],
            [0.0, 0.21220659, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04807806, 0.0, 0.03978874, 0.0, 0.0],
            [0.0, 0.0, 0.21220659, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06299883, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.546479, 0.0, 0.0, 0.0, -0.03978873, 0.0, -0.01989437, 0.0, -0.0],
            [0.0, 0.0, 0.0, 0.0, 2.546479, 0.0, 0.03978873, -0.0, 0.0, 0.0, -0.01989437, -0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.546479, 0.0, -0.0, 0.0, -0.0, -0.0, 0.03978874],
            [0.04807806, 0.0, 0.0, 0.0, 0.03978873, 0.0, 0.07073554, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.04807806, 0.0, -0.03978873, -0.0, -0.0, 0.0, 0.07073554, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.06299883, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07073554, 0.0, 0.0, 0.0],
            [0.0, 0.03978874, 0.0, -0.01989437, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09431405, 0.0, 0.0],
            [-0.03978874, 0.0, 0.0, 0.0, -0.01989437, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09431405, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.03978874, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09431405],
        ]
    )
    assert jnp.allclose(M, Mexpected)
    assert jnp.allclose(M - M.transpose(), jnp.zeros((12, 12)))


def test_mobility_tensor_symmetric_for_chain():
    """The grand mobility tensor must be symmetric for any geometry."""
    sp = SoftBody()
    for k in range(5):
        sp.add_sphere(Sphere(radius=0.4, position=[float(k), 0.0, 0.0]))
    M = sp.compute_mobility_tensor()
    assert M.shape == (30, 30)
    assert jnp.allclose(M, M.T, atol=1e-7)
    assert jnp.all(jnp.isfinite(M))


def test_jit_mobility_tensor_matches_eager():
    """jax.jit on compute_mobility_tensor must match the non-jit result."""
    import jax

    sp = SoftBody()
    for k in range(4):
        sp.add_sphere(Sphere(radius=0.3, position=[float(k), 0.0, 0.0]))
    M_eager = sp.compute_mobility_tensor()
    M_jit = jax.jit(sp.compute_mobility_tensor)()
    assert jnp.allclose(M_eager, M_jit, atol=1e-7)


def test_validate_no_overlap_method():
    sp = SoftBody()
    sp.add_sphere(Sphere(radius=1.0, position=[0.0, 0.0, 0.0]))
    sp.add_sphere(Sphere(radius=1.0, position=[1.5, 0.0, 0.0]))
    with pytest.raises(ValueError, match="overlap"):
        sp.validate_no_overlap()

    sp2 = SoftBody()
    sp2.add_sphere(Sphere(radius=1.0, position=[0.0, 0.0, 0.0]))
    sp2.add_sphere(Sphere(radius=1.0, position=[3.0, 0.0, 0.0]))
    sp2.validate_no_overlap()  # no raise


# def test_fast_mobility_problem():
#     sp = SoftBody("./splank/tests/parameters.yaml")
#     sp.set_dof_defaults(new_dict={"x0": 0, "x1": 0})
#     Mk, Mmean, _, Gk, Mdof, _, Gdof, *_ = sp.compute_fast_mobility_problem()
#     Mkexpected = jnp.array(
#         [
#             [0.131, 0.0, 0.0, 0.0, -0.07, 0.0, 0.131, 0.0, 0.0, 0.0, -0.07, 0.0],
#             [0.0, 0.131, 0.0, 0.07, 0.0, 0.0, 0.0, 0.131, 0.0, 0.07, 0.0, 0.0],
#             [0.0, 0.0, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07, 0.0, 0.0, 0.0],
#             [0.0, 0.07, 0.0, 0.077, 0.0, 0.0, 0.0, 0.07, 0.0, 0.077, 0.0, 0.0],
#             [-0.07, 0.0, 0.0, 0.0, 0.077, 0.0, -0.07, 0.0, 0.0, 0.0, 0.077, 0.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.093, 0.0, 0.0, 0.0, 0.0, 0.0, 0.093],
#         ]
#     )

#     Mmeanexpected = 0.5 * jnp.array(
#         [
#             [0.262, 0.0, 0.0, 0.0, -0.14, 0.0],
#             [0.0, 0.262, 0.0, 0.14, 0.0, 0.0],
#             [0.0, 0.0, 0.141, 0.0, 0.0, 0.0],
#             [0.0, 0.14, 0.0, 0.154, 0.0, 0.0],
#             [-0.14, 0.0, 0.0, 0.0, 0.154, 0.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.186],
#         ]
#     )

#     Gkexpected = jnp.array(
#         [
#             [0.0, 0.0, 0.664, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.0, 0.664],
#             [-0.967, 0.0, 0.0, -0.967, 0.0],
#             [0.0, 0.0, 0.0, 0.0, -0.239],
#             [0.0, 0.0, 0.239, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0],
#         ]
#     )

#     Mdofexpected = jnp.array(
#         [
#             [0.0, 0.118, 0.0, 0.905, 0.0, 0.0, 0.0, -0.132, 0.0, -0.095, 0.0, 0.0],
#             [-0.118, 0.0, 0.0, 0.0, 0.905, 0.0, 0.132, 0.0, 0.0, 0.0, -0.095, 0.0],
#         ]
#     )

#     Gdofexpected = jnp.array([[0.0, 0.0, 0.0, 0.0, -1.034], [0.0, 0.0, 1.034, 0.0, 0.0]])

#     assert jnp.allclose(Mk, Mkexpected, atol=1e-3)
#     assert jnp.allclose(Mmean, Mmeanexpected, atol=1e-3)
#     assert jnp.allclose(Gk, Gkexpected, atol=1e-3)
#     assert jnp.allclose(Mdof, Mdofexpected, atol=1e-3)
#     assert jnp.allclose(Gdof, Gdofexpected, atol=1e-3)

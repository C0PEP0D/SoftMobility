import pytest
import jax
import jax.numpy as jnp
import numpy as np
from softmobility.classes.inputs import Scalar


def test_no_params():
    # Test function with no params
    func = lambda pos, time: 0.5 * jnp.sin(2 * time)
    scalar = Scalar(func)
    pos = jnp.array([1.0, 2.0, 3.0])
    time = 1.0
    result = scalar.value(pos, time)
    assert isinstance(result, (float, jnp.ndarray))
    assert result == 0.5 * jnp.sin(2 * time)


def test_params_float():
    # Test with float param
    func = lambda pos, time, p: p[0] * jnp.sin(2 * time)
    scalar = Scalar(func, params=1.0, param_names="magnitude")
    pos = jnp.array([1.0, 2.0, 3.0])
    time = 1.0
    result = scalar.value(pos, time)
    assert result == 1.0 * jnp.sin(2 * time)


def test_params_array():
    # Test with np.array param
    func = lambda pos, time, p: p[0] * jnp.sin(2 * time)
    scalar = Scalar(func, params=np.array([1.0]), param_names="magnitude")
    pos = jnp.array([1.0, 2.0, 3.0])
    time = 1.0
    result = scalar.value(pos, time)
    assert result == 1.0 * jnp.sin(2 * time)


def test_params_list_of_arrays():
    # Test with list of arrays
    func = lambda pos, time, p: p[0] * jnp.sin(2 * time) + p[1] * jnp.cos(2 * time)
    scalar = Scalar(
        func,
        params=[np.array([1.0]), np.array([0.5])],
        param_names=["magnitude", "phase"],
    )
    pos = jnp.array([1.0, 2.0, 3.0])
    time = 1.0
    result = scalar.value(pos, time)
    assert result == 1.0 * jnp.sin(2 * time) + 0.5 * jnp.cos(2 * time)


def test_update_params_by_name():
    # Test updating params by name
    func = lambda pos, time, p: p[0] * jnp.sin(2 * time)
    scalar = Scalar(func, params=1.0, param_names="magnitude")
    pos = jnp.array([1.0, 2.0, 3.0])
    time = 1.0
    scalar.update_params(magnitude=2.0)
    result = scalar.value(pos, time)
    assert result == 2.0 * jnp.sin(2 * time)


def test_update_params_full_list():
    # Test updating params with a full list
    func = lambda pos, time, p: p[0] * jnp.sin(2 * time) + p[1] * jnp.cos(2 * time)
    scalar = Scalar(
        func,
        params=[np.array([1.0]), np.array([0.5])],
        param_names=["magnitude", "phase"],
    )
    pos = jnp.array([1.0, 2.0, 3.0])
    time = 1.0
    scalar.update_params(magnitude=2.0, phase=1.0)
    result = scalar.value(pos, time)
    assert result == 2.0 * jnp.sin(2 * time) + 1.0 * jnp.cos(2 * time)


def test_get_param():
    # Test get_param method
    func = lambda pos, time, p: p * jnp.sin(2 * time)
    scalar = Scalar(func, params=1.0, param_names="magnitude")
    assert scalar.get_param("magnitude") == 1.0


def test_jittable():
    # Test jittability
    func = lambda pos, time: jnp.sin(2 * time)
    scalar = Scalar(func)
    jitted_scalar = jax.jit(scalar._func)
    pos = jnp.array([1.0, 2.0, 3.0])
    time = 1.0
    result = jitted_scalar(pos, time)
    assert isinstance(result, jnp.ndarray)


def test_differentiable():
    # Test differentiability
    func = lambda pos, time, p: p * jnp.sin(2 * time)
    scalar = Scalar(func, params=1.0, param_names="magnitude")
    pos = jnp.array([1.0, 2.0, 3.0])
    time = 1.0
    grad_fn = jax.grad(lambda p: scalar._func(pos, time, p))
    gradient = grad_fn(scalar.get_param("magnitude"))
    assert isinstance(gradient, jnp.ndarray)


if __name__ == "__main__":
    test_no_params()
    test_params_float()
    test_params_array()
    test_params_list_of_arrays()
    test_update_params_by_name()
    test_update_params_full_list()
    test_jittable()
    test_differentiable()

import numpy as np

import softmobility as sm


def test_zero_rotation():
    R = np.asarray(sm.rotation_matrix([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-7)


def test_quarter_turn_x():
    R = np.asarray(sm.rotation_matrix([np.pi / 2, 0.0, 0.0]))
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(R, expected, atol=1e-7)


def test_quarter_turn_y():
    R = np.asarray(sm.rotation_matrix([0.0, np.pi / 2, 0.0]))
    expected = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(R, expected, atol=1e-7)


def test_quarter_turn_z():
    R = np.asarray(sm.rotation_matrix([0.0, 0.0, np.pi / 2]))
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(R, expected, atol=1e-7)


def test_orthogonality():
    rng = np.random.default_rng(0)
    for _ in range(8):
        rvec = rng.uniform(-np.pi, np.pi, size=3)
        R = np.asarray(sm.rotation_matrix(rvec))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_accepts_numpy_input():
    rvec = np.array([0.3, -0.7, 1.1])
    R = np.asarray(sm.rotation_matrix(rvec))
    assert R.shape == (3, 3)
    assert R.dtype == np.float64 or R.dtype == np.float32


def test_rescale_orientation_within_pi():
    rvec = np.array([0.5, -0.5, 0.5])
    out = np.asarray(sm.rescale_orientation(rvec))
    np.testing.assert_allclose(out, rvec, atol=1e-7)

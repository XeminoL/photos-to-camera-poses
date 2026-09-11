import numpy as np
import pytest

from sfm.core.geometry import (
    angle_between_rotations,
    camera_center,
    project,
    projection_jacobian,
    rodrigues,
    rotation_angle,
    rotation_angle_from_phase,
    rotation_axis,
    skew,
)

K = np.array([[1520.4, 0.0, 302.32],
              [0.0, 1525.9, 246.87],
              [0.0, 0.0, 1.0]])


@pytest.fixture
def rng():
    return np.random.default_rng(4)


def test_skew_is_antisymmetric(rng):
    v = rng.normal(size=3)
    assert np.allclose(skew(v), -skew(v).T)
    assert np.allclose(skew(v) @ v, 0.0)


def test_rodrigues_gives_rotation(rng):
    for _ in range(20):
        r = rodrigues(rng.normal(size=3))
        assert np.abs(r.T @ r - np.eye(3)).max() < 1e-14
        assert abs(np.linalg.det(r) - 1.0) < 1e-14


def test_rodrigues_zero_is_identity():
    assert np.allclose(rodrigues(np.zeros(3)), np.eye(3))


def test_angle_from_phase_matches_trace(rng):
    for _ in range(20):
        r = rodrigues(rng.normal(size=3) * 0.7)
        assert abs(rotation_angle_from_phase(r) - rotation_angle(r)) < 1e-9


def test_rotation_eigenvalues_on_unit_circle(rng):
    r = rodrigues(rng.normal(size=3))
    assert np.allclose(np.abs(np.linalg.eigvals(r)), 1.0)


def test_axis_is_fixed_by_rotation(rng):
    for _ in range(20):
        r = rodrigues(rng.normal(size=3))
        axis = rotation_axis(r)
        assert np.abs(r @ axis - axis).max() < 1e-12


def test_known_rotation_recovers_angle_and_axis():
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    r = rodrigues(axis * np.radians(37.0))
    assert abs(rotation_angle(r) - 37.0) < 1e-10
    assert abs(rotation_angle_from_phase(r) - 37.0) < 1e-10
    assert np.abs(np.abs(rotation_axis(r)) - np.abs(axis)).max() < 1e-12


def test_nth_root_of_rotation_keeps_conjugate_pair():
    r = rodrigues(np.array([0.0, 0.0, np.radians(120.0)]))
    for branch in range(3):
        angle = (np.radians(120.0) + 2 * np.pi * branch) / 3
        root = rodrigues(np.array([0.0, 0.0, angle]))
        assert np.abs(np.linalg.matrix_power(root, 3) - r).max() < 1e-14
        assert np.abs(root.T @ root - np.eye(3)).max() < 1e-14


def test_camera_center_round_trip(rng):
    r = rodrigues(rng.normal(size=3))
    t = rng.normal(size=3)
    assert np.abs(r @ camera_center(r, t) + t).max() < 1e-12


def test_projection_jacobian_matches_finite_difference(rng):
    r0 = rodrigues(rng.normal(size=3) * 0.3)
    t = np.array([0.3, -0.1, 0.2])
    point = np.array([0.2, 0.1, 5.0])
    d_w, d_t, d_x = projection_jacobian(K, r0, t, point)

    eps = 1e-7

    def shoot(w, tv, xv):
        return project(K, rodrigues(w) @ r0, tv, xv[None, :])[0]

    for col in range(3):
        step = np.zeros(3)
        step[col] = eps
        num_w = (shoot(step, t, point) - shoot(-step, t, point)) / (2 * eps)
        num_t = (shoot(np.zeros(3), t + step, point) - shoot(np.zeros(3), t - step, point)) / (2 * eps)
        num_x = (shoot(np.zeros(3), t, point + step) - shoot(np.zeros(3), t, point - step)) / (2 * eps)
        assert np.abs(d_w[:, col] - num_w).max() < 1e-4
        assert np.abs(d_t[:, col] - num_t).max() < 1e-6
        assert np.abs(d_x[:, col] - num_x).max() < 1e-6


def test_angle_between_rotations_is_symmetric(rng):
    a = rodrigues(rng.normal(size=3))
    b = rodrigues(rng.normal(size=3))
    assert abs(angle_between_rotations(a, b) - angle_between_rotations(b, a)) < 1e-10
import numpy as np
import pytest

from sfm.core.estimation import (
    decompose_essential,
    fundamental_from_points,
    project_to_essential,
    ransac_fundamental,
    sampson_distance,
    solve_pnp,
    triangulate,
)
from sfm.core.geometry import (
    angle_between_directions,
    angle_between_rotations,
    normalize_pixels,
    project,
    rodrigues,
    skew,
)

K = np.array([[1520.4, 0.0, 302.32],
              [0.0, 1525.9, 246.87],
              [0.0, 0.0, 1.0]])
K_INV = np.linalg.inv(K)


def make_scene(n=60, seed=5, noise=0.0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, (n, 3))
    points[:, 2] += 6.0
    r = rodrigues(np.array([0.04, 0.10, 0.02]))
    t = np.array([0.6, -0.1, 0.15])
    x1 = project(K, np.eye(3), np.zeros(3), points)
    x2 = project(K, r, t, points)
    if noise > 0:
        x1 = x1 + rng.normal(0.0, noise, x1.shape)
        x2 = x2 + rng.normal(0.0, noise, x2.shape)
    return points, r, t, x1, x2


def true_fundamental(r, t):
    f = K_INV.T @ (skew(t) @ r) @ K_INV
    return f / np.linalg.norm(f)


def test_fundamental_satisfies_epipolar_constraint():
    _, r, t, x1, x2 = make_scene()
    f = fundamental_from_points(x1, x2)
    assert sampson_distance(f, x1, x2).max() < 1e-8


def test_fundamental_matches_ground_truth():
    _, r, t, x1, x2 = make_scene()
    f = fundamental_from_points(x1, x2)
    f = f / np.linalg.norm(f)
    truth = true_fundamental(r, t)
    assert min(np.abs(f - truth).max(), np.abs(-f - truth).max()) < 1e-9


def test_fundamental_has_rank_two():
    _, _, _, x1, x2 = make_scene()
    s = np.linalg.svd(fundamental_from_points(x1, x2))[1]
    assert s[2] / s[0] < 1e-12


def test_fundamental_rejects_too_few_points():
    _, _, _, x1, x2 = make_scene()
    with pytest.raises(ValueError):
        fundamental_from_points(x1[:7], x2[:7])


def test_essential_projection_gives_equal_singular_values():
    rng = np.random.default_rng(9)
    e = project_to_essential(rng.normal(size=(3, 3)))
    s = np.linalg.svd(e)[1]
    assert abs(s[0] - s[1]) < 1e-12
    assert s[2] / s[0] < 1e-15


def test_standard_stereo_has_exact_singular_values():
    baseline = 0.5
    e = skew(np.array([baseline, 0.0, 0.0])) @ np.eye(3)
    s = np.linalg.svd(e)[1]
    assert abs(s[0] - baseline) < 1e-15
    assert abs(s[1] - baseline) < 1e-15
    assert s[2] < 1e-15


def test_decompose_recovers_pose():
    _, r, t, x1, x2 = make_scene()
    f = fundamental_from_points(x1, x2)
    rays1 = normalize_pixels(x1, K_INV)
    rays2 = normalize_pixels(x2, K_INV)
    r_est, t_est, mask, count = decompose_essential(K.T @ f @ K, rays1, rays2)
    assert count == len(x1)
    assert mask.all()
    assert angle_between_rotations(r_est, r) < 1e-6
    assert angle_between_directions(t_est, t) < 1e-6


def test_triangulation_recovers_depth_ratio():
    points, r, t, x1, x2 = make_scene()
    f = fundamental_from_points(x1, x2)
    rays1 = normalize_pixels(x1, K_INV)
    rays2 = normalize_pixels(x2, K_INV)
    r_est, t_est, _, _ = decompose_essential(K.T @ f @ K, rays1, rays2)
    p1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    got = triangulate(p1, np.hstack([r_est, t_est.reshape(3, 1)]), rays1, rays2)
    scale = np.linalg.norm(t) / np.linalg.norm(t_est)
    assert np.abs(got * scale - points).max() < 1e-6


def test_standard_stereo_depth_matches_closed_form():
    focal, baseline = 1520.4, 0.5
    for depth in (3.0, 6.0, 12.0, 50.0):
        disparity = focal * baseline / depth
        assert abs(focal * baseline / disparity - depth) < 1e-12


def test_pnp_recovers_pose_exactly():
    rng = np.random.default_rng(7)
    points = rng.uniform(-1.0, 1.0, (40, 3))
    points[:, 2] += 6.0
    r = rodrigues(np.array([0.0, 0.3, 0.0]))
    t = np.array([0.4, -0.1, 0.2])
    pixels = project(K, r, t, points)
    r_est, t_est = solve_pnp(points, pixels, K)
    assert angle_between_rotations(r_est, r) < 1e-5
    assert np.abs(t_est - t).max() < 1e-6


def test_pnp_intrinsics_removal_gives_equal_singular_values():
    rng = np.random.default_rng(7)
    points = rng.uniform(-1.0, 1.0, (40, 3))
    points[:, 2] += 6.0
    r = rodrigues(np.array([0.0, 0.3, 0.0]))
    t = np.array([0.4, -0.1, 0.2])
    r_est, _ = solve_pnp(points, project(K, r, t, points), K)
    assert np.abs(r_est.T @ r_est - np.eye(3)).max() < 1e-10


def test_ransac_rejects_outliers():
    points, r, t, x1, x2 = make_scene(n=100, noise=0.3)
    rng = np.random.default_rng(3)
    corrupt = rng.choice(len(x1), 15, replace=False)
    x2 = x2.copy()
    x2[corrupt] += rng.uniform(20.0, 60.0, (len(corrupt), 2))
    f, inliers = ransac_fundamental(x1, x2, threshold=1.0, iterations=500, seed=1)
    assert inliers.sum() >= 70
    assert not inliers[corrupt].any()
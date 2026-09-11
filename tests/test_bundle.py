import numpy as np

from sfm.core.bundle import SCALE_FREEDOM, Observation, Reconstruction, optimize
from sfm.core.geometry import angle_between_rotations, project, rodrigues

K = np.array([[1520.4, 0.0, 302.32],
              [0.0, 1525.9, 246.87],
              [0.0, 0.0, 1.0]])


def build(n_points=40, n_cameras=3, noise=0.0, seed=5):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, (n_points, 3))
    points[:, 2] += 6.0
    poses = {0: (np.eye(3), np.zeros(3))}
    for c in range(1, n_cameras):
        poses[c] = (rodrigues(np.array([0.03 * c, 0.09 * c, 0.02 * c])),
                    np.array([0.5 * c, -0.1 * c, 0.15 * c]))
    observations = []
    for c, (r, t) in poses.items():
        pixels = project(K, r, t, points)
        if noise > 0:
            pixels = pixels + rng.normal(0.0, noise, pixels.shape)
        for i, uv in enumerate(pixels):
            observations.append(Observation(i, c, uv))
    return Reconstruction(poses, points, observations, anchor=0)


def test_perfect_data_has_zero_residual():
    recon, report = optimize(build(), K, iterations=5)
    assert report.rms < 1e-8


def test_only_scale_is_free_when_anchor_is_fixed():
    _, report = optimize(build(noise=0.2), K, iterations=10)
    assert report.null_space_dimension == SCALE_FREEDOM
    assert not report.is_degenerate


def test_optimization_reduces_residual():
    truth = build(noise=0.4, seed=11)
    perturbed = Reconstruction(
        {c: (rodrigues(np.array([0.01, -0.015, 0.008])) @ r, t + np.array([0.02, -0.01, 0.015]))
         if c != truth.anchor else (r, t)
         for c, (r, t) in truth.poses.items()},
        truth.points + 0.02,
        truth.observations,
        truth.anchor,
    )
    before = optimize(perturbed, K, iterations=0)[1].rms
    after = optimize(perturbed, K, iterations=25)[1].rms
    assert after < before / 2


def test_recovers_pose_from_perturbed_start():
    truth = build(seed=3)
    start = Reconstruction(
        {c: (rodrigues(np.array([0.02, -0.02, 0.01])) @ r, t + np.array([0.03, 0.02, -0.02]))
         if c != truth.anchor else (r, t)
         for c, (r, t) in truth.poses.items()},
        truth.points * 1.02,
        truth.observations,
        truth.anchor,
    )
    recon, report = optimize(start, K, iterations=40)
    assert report.rms < 1e-4
    for c in truth.poses:
        assert angle_between_rotations(recon.poses[c][0], truth.poses[c][0]) < 1e-3


def test_camera_with_minimum_observations_is_not_degenerate():
    recon = build(n_points=40, n_cameras=2, noise=0.2, seed=8)
    extra = 99
    recon.poses[extra] = (rodrigues(np.array([0.05, 0.05, 0.0])), np.array([0.3, 0.0, 0.1]))
    r, t = recon.poses[extra]
    for i in range(3):
        recon.observations.append(Observation(i, extra, project(K, r, t, recon.points[i:i + 1])[0]))
    _, report = optimize(recon, K, iterations=5)
    assert report.null_space_dimension == SCALE_FREEDOM


def test_camera_seeing_too_few_points_is_degenerate():
    recon = build(n_points=40, n_cameras=2, noise=0.2, seed=8)
    starved = 99
    recon.poses[starved] = (rodrigues(np.array([0.05, 0.05, 0.0])), np.array([0.3, 0.0, 0.1]))
    r, t = recon.poses[starved]
    for i in range(2):
        recon.observations.append(Observation(i, starved, project(K, r, t, recon.points[i:i + 1])[0]))
    _, report = optimize(recon, K, iterations=5)
    assert report.is_degenerate
    assert report.excess_null_dimensions == 2


def test_collinear_points_leave_pose_underdetermined():
    recon = build(n_points=40, n_cameras=2, noise=0.0, seed=8)
    line = np.stack([np.linspace(-0.5, 0.5, 8), np.zeros(8), np.full(8, 6.0)], axis=1)
    base = len(recon.points)
    recon.points = np.vstack([recon.points, line])
    thin = 99
    recon.poses[thin] = (np.eye(3), np.array([0.2, 0.0, 0.0]))
    r, t = recon.poses[thin]
    for i in range(len(line)):
        recon.observations.append(
            Observation(base + i, thin, project(K, r, t, line[i:i + 1])[0]))
    _, report = optimize(recon, K, iterations=5)
    assert report.is_degenerate


def test_uncertainty_grows_when_observations_shrink():
    rich = optimize(build(n_points=60, noise=0.3, seed=2), K, iterations=10)[1]
    poor = optimize(build(n_points=12, noise=0.3, seed=2), K, iterations=10)[1]
    assert poor.angle_sigma[1] > rich.angle_sigma[1]


def test_anchor_camera_is_not_optimized():
    recon, _ = optimize(build(noise=0.3), K, iterations=10)
    assert np.allclose(recon.poses[recon.anchor][0], np.eye(3))
    assert np.allclose(recon.poses[recon.anchor][1], np.zeros(3))
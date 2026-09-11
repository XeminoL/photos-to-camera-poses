import json
from pathlib import Path

import numpy as np

from sfm.core.estimation import (
    decompose_essential,
    fundamental_from_points,
    ransac_fundamental,
    solve_pnp,
    triangulate,
)
from sfm.core.geometry import normalize_pixels, project, rodrigues

OUTPUT = Path("web/tests/reference.json")
K = [[1520.4, 0.0, 302.32], [0.0, 1525.9, 246.87], [0.0, 0.0, 1.0]]
SCENE_SEED = 5
OUTLIER_SEED = 3
POINT_COUNT = 60
NOISE = 0.3
OUTLIER_COUNT = 15
RANSAC_THRESHOLD = 1.0
RANSAC_ITERATIONS = 500
RANSAC_SEED = 1


def build_scene(count, seed, noise):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, (count, 3))
    points[:, 2] += 6.0
    r = rodrigues(np.array([0.04, 0.10, 0.02]))
    t = np.array([0.6, -0.1, 0.15])
    k = np.array(K)
    left = project(k, np.eye(3), np.zeros(3), points)
    right = project(k, r, t, points)
    if noise:
        left = left + rng.normal(0.0, noise, left.shape)
        right = right + rng.normal(0.0, noise, right.shape)
    return points, r, t, left, right


def main():
    k = np.array(K)
    k_inv = np.linalg.inv(k)
    points, r_true, t_true, left, right = build_scene(POINT_COUNT, SCENE_SEED, 0.0)

    f = fundamental_from_points(left, right)
    rays1 = normalize_pixels(left, k_inv)
    rays2 = normalize_pixels(right, k_inv)
    r_est, t_est, _, in_front = decompose_essential(k.T @ f @ k, rays1, rays2)
    p1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    cloud = triangulate(p1, np.hstack([r_est, t_est.reshape(3, 1)]), rays1, rays2)

    pnp_points = points[:40]
    pnp_r = rodrigues(np.array([0.0, 0.3, 0.0]))
    pnp_t = np.array([0.4, -0.1, 0.2])
    pnp_pixels = project(k, pnp_r, pnp_t, pnp_points)
    pnp_r_est, pnp_t_est = solve_pnp(pnp_points, pnp_pixels, k)

    noisy_points, _, _, noisy_left, noisy_right = build_scene(100, SCENE_SEED, NOISE)
    rng = np.random.default_rng(OUTLIER_SEED)
    corrupted = sorted(rng.choice(len(noisy_left), OUTLIER_COUNT, replace=False).tolist())
    dirty = noisy_right.copy()
    dirty[corrupted] += rng.uniform(20.0, 60.0, (OUTLIER_COUNT, 2))
    _, inliers = ransac_fundamental(noisy_left, dirty, RANSAC_THRESHOLD,
                                    RANSAC_ITERATIONS, RANSAC_SEED)

    payload = {
        "intrinsics": K,
        "scene": {
            "points": points.tolist(),
            "left": left.tolist(),
            "right": right.tolist(),
            "rotation": r_true.tolist(),
            "translation": t_true.tolist(),
        },
        "fundamental": (f / np.linalg.norm(f)).tolist(),
        "pose": {
            "rotation": r_est.tolist(),
            "translation": t_est.tolist(),
            "inFront": int(in_front),
        },
        "cloud": cloud.tolist(),
        "pnp": {
            "points": pnp_points.tolist(),
            "pixels": pnp_pixels.tolist(),
            "rotation": pnp_r_est.tolist(),
            "translation": pnp_t_est.tolist(),
        },
        "ransac": {
            "left": noisy_left.tolist(),
            "right": dirty.tolist(),
            "corrupted": corrupted,
            "inlierCount": int(inliers.sum()),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    print(f"  scene {len(points)} points, {in_front} in front")
    print(f"  ransac kept {inliers.sum()} of {len(noisy_left)}")


if __name__ == "__main__":
    main()
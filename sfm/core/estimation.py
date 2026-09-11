import numpy as np

from sfm.core.geometry import skew, to_homogeneous

SAMPSON_FLOOR = 1e-30
MIN_POINTS_FUNDAMENTAL = 8
MIN_POINTS_PNP = 6
PNP_SAMPLE_SIZE = 10
CHEIRALITY_MARGIN = 0.0


def _isotropic_normalizer(x, target_scale):
    center = x.mean(axis=0)
    spread = np.sqrt(((x - center) ** 2).sum(axis=1)).mean()
    s = target_scale / spread
    dim = x.shape[1]
    t = np.eye(dim + 1)
    t[:dim, :dim] *= s
    t[:dim, dim] = -s * center
    return t


def fundamental_from_points(x1, x2):
    if len(x1) < MIN_POINTS_FUNDAMENTAL:
        raise ValueError(f"need {MIN_POINTS_FUNDAMENTAL} points, got {len(x1)}")
    t1 = _isotropic_normalizer(x1, np.sqrt(2))
    t2 = _isotropic_normalizer(x2, np.sqrt(2))
    n1 = (t1 @ to_homogeneous(x1).T).T
    n2 = (t2 @ to_homogeneous(x2).T).T
    a = np.stack([n2[:, 0] * n1[:, 0], n2[:, 0] * n1[:, 1], n2[:, 0],
                  n2[:, 1] * n1[:, 0], n2[:, 1] * n1[:, 1], n2[:, 1],
                  n1[:, 0], n1[:, 1], np.ones(len(n1))], axis=1)
    f = np.linalg.svd(a)[2][-1].reshape(3, 3)
    u, s, vt = np.linalg.svd(f)
    return t2.T @ (u @ np.diag([s[0], s[1], 0.0]) @ vt) @ t1


def sampson_distance(f, x1, x2):
    a1 = to_homogeneous(x1)
    a2 = to_homogeneous(x2)
    fx1 = (f @ a1.T).T
    ftx2 = (f.T @ a2.T).T
    num = np.einsum("ij,ij->i", a2, fx1) ** 2
    den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, SAMPSON_FLOOR))


def ransac_fundamental(x1, x2, threshold, iterations, seed):
    rng = np.random.default_rng(seed)
    best_count, best_f = -1, None
    for _ in range(iterations):
        idx = rng.choice(len(x1), MIN_POINTS_FUNDAMENTAL, replace=False)
        try:
            f = fundamental_from_points(x1[idx], x2[idx])
        except np.linalg.LinAlgError:
            continue
        count = int((sampson_distance(f, x1, x2) < threshold).sum())
        if count > best_count:
            best_count, best_f = count, f
    if best_f is None:
        raise RuntimeError("ransac found no candidate")
    inliers = sampson_distance(best_f, x1, x2) < threshold
    if inliers.sum() < MIN_POINTS_FUNDAMENTAL:
        return best_f, inliers
    refined = fundamental_from_points(x1[inliers], x2[inliers])
    return refined, sampson_distance(refined, x1, x2) < threshold


def project_to_essential(e):
    u, s, vt = np.linalg.svd(e)
    mean = (s[0] + s[1]) / 2.0
    return u @ np.diag([mean, mean, 0.0]) @ vt


def triangulate_point(p1, p2, ray1, ray2):
    m = np.stack([ray1[0] * p1[2] - p1[0], ray1[1] * p1[2] - p1[1],
                  ray2[0] * p2[2] - p2[0], ray2[1] * p2[2] - p2[1]])
    x = np.linalg.svd(m)[2][-1]
    return x[:3] / x[3]


def triangulate(p1, p2, rays1, rays2):
    return np.array([triangulate_point(p1, p2, a, b) for a, b in zip(rays1, rays2)])


def _in_front(r, t, points):
    return (points[:, 2] > CHEIRALITY_MARGIN) & ((r @ points.T + t[:, None])[2] > CHEIRALITY_MARGIN)


def decompose_essential(e, rays1, rays2):
    u, _, vt = np.linalg.svd(project_to_essential(e))
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1
    if np.linalg.det(vt) < 0:
        vt[-1, :] *= -1
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    p1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    best = (-1, None, None, None)
    for r in (u @ w @ vt, u @ w.T @ vt):
        for t in (u[:, 2], -u[:, 2]):
            points = triangulate(p1, np.hstack([r, t.reshape(3, 1)]), rays1, rays2)
            mask = _in_front(r, t, points)
            if mask.sum() > best[0]:
                best = (int(mask.sum()), r, t, mask)
    return best[1], best[2], best[3], best[0]


def solve_pnp(points, pixels, k):
    if len(points) < MIN_POINTS_PNP:
        raise ValueError(f"need {MIN_POINTS_PNP} points, got {len(points)}")
    u = _isotropic_normalizer(points, np.sqrt(3))
    t = _isotropic_normalizer(pixels, np.sqrt(2))
    xh = (u @ to_homogeneous(points).T).T
    uh = (t @ to_homogeneous(pixels).T).T
    rows = []
    for xi, ui in zip(xh, uh):
        zero = np.zeros(4)
        rows.append(np.concatenate([xi, zero, -ui[0] * xi]))
        rows.append(np.concatenate([zero, xi, -ui[1] * xi]))
    p = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 4)
    p = np.linalg.inv(t) @ p @ u

    k_inv = np.linalg.inv(k)
    m = k_inv @ p[:, :3]
    tv = k_inv @ p[:, 3]
    if np.linalg.det(m) < 0:
        m, tv = -m, -tv
    um, sm, vtm = np.linalg.svd(m)
    r = um @ vtm
    if np.linalg.det(r) < 0:
        r, tv = -r, -tv
    return r, tv / sm.mean()


def ransac_pnp(points, pixels, k, threshold, iterations, seed):
    from sfm.core.geometry import reprojection_error

    rng = np.random.default_rng(seed)
    sample = min(max(PNP_SAMPLE_SIZE, MIN_POINTS_PNP), len(points))
    best_count, best = -1, None
    for _ in range(iterations):
        idx = rng.choice(len(points), sample, replace=False)
        try:
            r, t = solve_pnp(points[idx], pixels[idx], k)
        except (np.linalg.LinAlgError, ValueError):
            continue
        count = int((reprojection_error(k, r, t, points, pixels) < threshold).sum())
        if count > best_count:
            best_count, best = count, (r, t)
    if best is None:
        raise RuntimeError("pnp ransac found no candidate")

    inliers = reprojection_error(k, best[0], best[1], points, pixels) < threshold
    if inliers.sum() < MIN_POINTS_PNP:
        return best[0], best[1], inliers
    try:
        r, t = solve_pnp(points[inliers], pixels[inliers], k)
    except (np.linalg.LinAlgError, ValueError):
        return best[0], best[1], inliers
    refined = reprojection_error(k, r, t, points, pixels) < threshold
    if refined.sum() < inliers.sum():
        return best[0], best[1], inliers
    return r, t, refined
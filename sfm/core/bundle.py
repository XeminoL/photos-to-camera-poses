from dataclasses import dataclass, field

import numpy as np

from sfm.core.geometry import projection_jacobian, rodrigues, skew

SCALE_FREEDOM = 1
RANK_TOLERANCE = 1e-8
LAMBDA_INITIAL = 1e-3
LAMBDA_GROWTH = 10.0
LAMBDA_DECAY = 3.0
LAMBDA_CEILING = 1e12
LAMBDA_FLOOR = 1e-12
BACKTRACK_LIMIT = 12
DIAGONAL_FLOOR = 1e-12


@dataclass
class Observation:
    point: int
    camera: int
    pixel: np.ndarray


@dataclass
class Reconstruction:
    poses: dict
    points: np.ndarray
    observations: list
    anchor: int

    def free_cameras(self):
        return [c for c in sorted(self.poses) if c != self.anchor]


@dataclass
class BundleReport:
    rms: float
    null_space_dimension: int
    excess_null_dimensions: int
    angle_sigma: dict = field(default_factory=dict)
    center_sigma: dict = field(default_factory=dict)

    @property
    def is_degenerate(self):
        return self.excess_null_dimensions > 0


def _residual(recon, k):
    out = np.empty(2 * len(recon.observations))
    for n, obs in enumerate(recon.observations):
        r, t = recon.poses[obs.camera]
        y = r @ recon.points[obs.point] + t
        projected = (k @ y)[:2] / y[2]
        out[2 * n:2 * n + 2] = projected - obs.pixel
    return out


def _jacobian(recon, k, index):
    free = recon.free_cameras()
    n_free = len(free)
    columns = n_free * 6 + 3 * len(recon.points)
    j = np.zeros((2 * len(recon.observations), columns))
    for n, obs in enumerate(recon.observations):
        r, t = recon.poses[obs.camera]
        d_w, d_t, d_x = projection_jacobian(k, r, t, recon.points[obs.point])
        row = slice(2 * n, 2 * n + 2)
        if obs.camera in index:
            base = index[obs.camera] * 6
            j[row, base:base + 3] = d_w
            j[row, base + 3:base + 6] = d_t
        base = n_free * 6 + 3 * obs.point
        j[row, base:base + 3] = d_x
    return j


def _apply(recon, k, index, delta):
    n_free = len(index)
    poses = {}
    for camera, (r, t) in recon.poses.items():
        if camera in index:
            base = index[camera] * 6
            poses[camera] = (rodrigues(delta[base:base + 3]) @ r, t + delta[base + 3:base + 6])
        else:
            poses[camera] = (r, t)
    points = recon.points + delta[n_free * 6:].reshape(-1, 3)
    return Reconstruction(poses, points, recon.observations, recon.anchor)


def _null_space_dimension(j):
    scale = np.maximum(np.linalg.norm(j, axis=0), DIAGONAL_FLOOR)
    values = np.linalg.eigvalsh((j / scale).T @ (j / scale))
    return max(int((values < RANK_TOLERANCE * values[-1]).sum()), SCALE_FREEDOM)


def _covariance(h, residual, null_dim, n_residuals):
    values, vectors = np.linalg.eigh(h)
    dof = max(n_residuals - (h.shape[0] - null_dim), 1)
    sigma_squared = (residual @ residual) / dof
    inverse = values.copy()
    inverse[:null_dim] = 0.0
    inverse[null_dim:] = 1.0 / np.maximum(inverse[null_dim:], DIAGONAL_FLOOR)
    return (vectors * inverse) @ vectors.T * sigma_squared


def optimize(recon, k, iterations):
    index = {c: i for i, c in enumerate(recon.free_cameras())}
    residual = _residual(recon, k)
    lam = LAMBDA_INITIAL
    j = None

    for _ in range(iterations):
        j = _jacobian(recon, k, index)
        h = j.T @ j
        gradient = j.T @ residual
        improved = False
        for _ in range(BACKTRACK_LIMIT):
            damped = h + lam * np.diag(np.diag(h) + DIAGONAL_FLOOR)
            try:
                delta = np.linalg.solve(damped, -gradient)
            except np.linalg.LinAlgError:
                lam *= LAMBDA_GROWTH
                continue
            candidate = _apply(recon, k, index, delta)
            trial = _residual(candidate, k)
            if trial @ trial < residual @ residual:
                recon, residual = candidate, trial
                lam = max(lam / LAMBDA_DECAY, LAMBDA_FLOOR)
                improved = True
                break
            lam *= LAMBDA_GROWTH
        if not improved or lam > LAMBDA_CEILING:
            break

    j = _jacobian(recon, k, index)
    h = j.T @ j
    null_dim = _null_space_dimension(j)
    covariance = _covariance(h, residual, null_dim, len(residual))

    angle_sigma, center_sigma = {}, {}
    diagonal = np.diag(covariance)
    for camera, position in index.items():
        base = position * 6
        angle_sigma[camera] = np.degrees(np.sqrt(max(diagonal[base:base + 3].sum(), 0.0)))
        center_sigma[camera] = np.sqrt(max(diagonal[base + 3:base + 6].sum(), 0.0))

    report = BundleReport(
        rms=float(np.sqrt((residual @ residual) / len(recon.observations))),
        null_space_dimension=null_dim,
        excess_null_dimensions=null_dim - SCALE_FREEDOM,
        angle_sigma=angle_sigma,
        center_sigma=center_sigma,
    )
    return recon, report
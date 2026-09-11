import numpy as np

ROTATION_EPS = 1e-12


def skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def rodrigues(w):
    theta = np.linalg.norm(w)
    if theta < ROTATION_EPS:
        return np.eye(3)
    k = skew(w / theta)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * k @ k


def rotation_angle(r):
    return np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))


def rotation_angle_from_phase(r):
    w = np.linalg.eigvals(r)
    return np.degrees(abs(np.angle(w[np.argmax(np.abs(w.imag))])))


def rotation_axis(r):
    w, v = np.linalg.eig(r)
    axis = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    return axis / np.linalg.norm(axis)


def angle_between_rotations(a, b):
    return rotation_angle(a.T @ b)


def angle_between_directions(a, b):
    cos = abs(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def camera_center(r, t):
    return -r.T @ t


def relative_pose(r_from, t_from, r_to, t_to):
    r = r_to @ r_from.T
    return r, t_to - r @ t_from


def to_homogeneous(x):
    return np.hstack([x, np.ones((len(x), 1))])


def normalize_pixels(x, k_inv):
    return (k_inv @ to_homogeneous(x).T).T


def project(k, r, t, points):
    p = (k @ (r @ points.T + t[:, None])).T
    return p[:, :2] / p[:, 2:3]


def reprojection_error(k, r, t, points, observed):
    return np.linalg.norm(project(k, r, t, points) - observed, axis=1)


def projection_jacobian(k, r, t, point):
    y = r @ point + t
    z = y[2]
    fx, fy, s = k[0, 0], k[1, 1], k[0, 1]
    d_pi = np.array([[fx / z, s / z, -(fx * y[0] + s * y[1]) / z ** 2],
                     [0.0, fy / z, -fy * y[1] / z ** 2]])
    return d_pi @ (-skew(r @ point)), d_pi, d_pi @ r
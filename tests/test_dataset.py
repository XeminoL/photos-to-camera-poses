from pathlib import Path

import numpy as np
import pytest

from sfm.core.geometry import camera_center, rotation_angle, rotation_angle_from_phase
from sfm.dataset import duplicate_views, load_middlebury

PAR_FILE = Path("data/templeSparseRing/templeSR_par.txt")
EXPECTED_VIEWS = 16
DUPLICATE_PAIR = (0, 10)
LATITUDE_GROUPS = 2


@pytest.fixture(scope="module")
def cameras():
    if not PAR_FILE.exists():
        pytest.skip(f"{PAR_FILE} not downloaded")
    return load_middlebury(PAR_FILE)


def test_loads_all_views(cameras):
    assert len(cameras) == EXPECTED_VIEWS


def test_every_view_shares_one_intrinsic_matrix(cameras):
    for camera in cameras[1:]:
        assert np.allclose(camera.k, cameras[0].k)


def test_every_rotation_is_valid(cameras):
    for camera in cameras:
        assert np.abs(camera.r.T @ camera.r - np.eye(3)).max() < 1e-14
        assert abs(np.linalg.det(camera.r) - 1.0) < 1e-9


def test_phase_and_trace_agree_on_real_data(cameras):
    for camera in cameras:
        assert abs(rotation_angle_from_phase(camera.r) - rotation_angle(camera.r)) < 1e-12


def test_duplicate_view_is_detected(cameras):
    assert duplicate_views(cameras) == [DUPLICATE_PAIR]


def test_views_lie_on_a_thin_ring(cameras):
    centers = np.array([camera_center(c.r, c.t) for c in cameras])
    centered = centers - centers.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    assert s[2] / s[0] < 1e-2
    assert abs(s[0] - s[1]) / s[0] < 0.1


def test_ring_radius_is_consistent(cameras):
    centers = np.array([camera_center(c.r, c.t) for c in cameras])
    radius = np.linalg.norm(centers - centers.mean(axis=0), axis=1)
    assert radius.std() / radius.mean() < 0.06
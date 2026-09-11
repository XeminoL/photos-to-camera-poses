from dataclasses import dataclass, field

import numpy as np

from sfm.core.bundle import Observation, Reconstruction, optimize
from sfm.core.estimation import (
    MIN_POINTS_PNP,
    decompose_essential,
    ransac_fundamental,
    ransac_pnp,
    triangulate,
)
from sfm.core.geometry import (
    angle_between_directions,
    camera_center,
    normalize_pixels,
    reprojection_error,
)

FUNDAMENTAL_THRESHOLD = 0.3
FUNDAMENTAL_ITERATIONS = 1000
PNP_THRESHOLD = 20.0
PNP_ITERATIONS = 800
MIN_PNP_MATCHES = 8
BUNDLE_ITERATIONS = 12
TRIANGULATION_THRESHOLD = 4.0
MIN_TRIANGULATION_ANGLE = 3.0
GROW_POINTS = False
DEFAULT_SEED = 7


@dataclass
class Rejection:
    camera: int
    reason: str
    matches: int = 0
    inliers: int = 0
    excess_null_dimensions: int = 0


@dataclass
class PipelineResult:
    reconstruction: Reconstruction
    report: object
    rejected: list = field(default_factory=list)
    history: list = field(default_factory=list)


class Session:
    def __init__(self, features, matches, k, seed=DEFAULT_SEED, grow_points=GROW_POINTS):
        self.features = features
        self.matches = matches
        self.k = k
        self.k_inv = np.linalg.inv(k)
        self.seed = seed
        self.grow_points = grow_points

    def pixels(self, camera, indices):
        return self.features[camera].keypoints[indices]

    def pair(self, left, right):
        key = (left, right) if left < right else (right, left)
        pairs = self.matches[key]
        return pairs if left < key[1] or left == key[0] else pairs[:, ::-1]

    def initialize(self, left, right):
        pairs = self.matches[(left, right)]
        x1 = self.pixels(left, pairs[:, 0])
        x2 = self.pixels(right, pairs[:, 1])
        f, inliers = ransac_fundamental(x1, x2, FUNDAMENTAL_THRESHOLD,
                                        FUNDAMENTAL_ITERATIONS, self.seed)
        pairs = pairs[inliers]
        rays1 = normalize_pixels(x1[inliers], self.k_inv)
        rays2 = normalize_pixels(x2[inliers], self.k_inv)
        r, t, mask, _ = decompose_essential(self.k.T @ f @ self.k, rays1, rays2)

        p1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        points = triangulate(p1, np.hstack([r, t.reshape(3, 1)]), rays1, rays2)
        poses = {left: (np.eye(3), np.zeros(3)), right: (r, t)}
        observations, kept, tracks = [], [], {}
        for row, (pair, good) in enumerate(zip(pairs, mask)):
            if not good:
                continue
            index = len(kept)
            kept.append(points[row])
            tracks[(left, pair[0])] = index
            tracks[(right, pair[1])] = index
            observations.append(Observation(index, left, self.pixels(left, pair[0])))
            observations.append(Observation(index, right, self.pixels(right, pair[1])))
        recon = Reconstruction(poses, np.array(kept), observations, anchor=left)
        self.tracks = tracks
        return recon

    def _correspondences(self, recon, camera):
        found = {}
        for (seen_camera, feature), point in self.tracks.items():
            if seen_camera not in recon.poses:
                continue
            key = (seen_camera, camera) if seen_camera < camera else (camera, seen_camera)
            pairs = self.matches.get(key)
            if pairs is None:
                continue
            if seen_camera < camera:
                hits = pairs[pairs[:, 0] == feature][:, 1]
            else:
                hits = pairs[pairs[:, 1] == feature][:, 0]
            for hit in hits:
                found[point] = int(hit)
        return found

    def _try_register(self, recon, camera):
        found = self._correspondences(recon, camera)
        if len(found) < MIN_PNP_MATCHES:
            return None, Rejection(camera, "too few correspondences", len(found))

        indices = list(found)
        points = recon.points[indices]
        pixels = self.pixels(camera, [found[i] for i in indices])
        try:
            r, t, inliers = ransac_pnp(points, pixels, self.k, PNP_THRESHOLD,
                                       PNP_ITERATIONS, self.seed)
        except (RuntimeError, ValueError):
            return None, Rejection(camera, "pnp failed", len(found))
        if inliers.sum() < MIN_POINTS_PNP:
            return None, Rejection(camera, "too few pnp inliers", len(found), int(inliers.sum()))

        poses = dict(recon.poses)
        poses[camera] = (r, t)
        observations = list(recon.observations)
        errors = reprojection_error(self.k, r, t, points, pixels)
        for index, error in zip(indices, errors):
            if error < PNP_THRESHOLD:
                observations.append(Observation(index, camera, self.pixels(camera, found[index])))
                self.tracks[(camera, found[index])] = index
        candidate = Reconstruction(poses, recon.points.copy(), observations, recon.anchor)
        refined, report = optimize(candidate, self.k, BUNDLE_ITERATIONS)
        if report.is_degenerate:
            for index in indices:
                self.tracks.pop((camera, found[index]), None)
            return None, Rejection(camera, "would make pose underdetermined", len(found),
                                   int(inliers.sum()), report.excess_null_dimensions)

        if self.grow_points:
            points, observations = self._grow(refined, camera)
            grown = Reconstruction(refined.poses, points, observations, refined.anchor)
            refined, report = optimize(grown, self.k, BUNDLE_ITERATIONS)
        return (refined, report), Rejection(camera, "", len(found), int(inliers.sum()))

    def _grow(self, recon, camera):
        points = list(recon.points)
        observations = list(recon.observations)
        r_new, t_new = recon.poses[camera]
        p_new = np.hstack([r_new, t_new.reshape(3, 1)])
        for other in recon.poses:
            if other == camera:
                continue
            key = (other, camera) if other < camera else (camera, other)
            pairs = self.matches.get(key)
            if pairs is None:
                continue
            r_old, t_old = recon.poses[other]
            p_old = np.hstack([r_old, t_old.reshape(3, 1)])
            for row in pairs:
                a, b = (row[0], row[1]) if other < camera else (row[1], row[0])
                if (other, a) in self.tracks or (camera, b) in self.tracks:
                    continue
                pixel_old = self.pixels(other, a)
                pixel_new = self.pixels(camera, b)
                ray_old = normalize_pixels(pixel_old[None, :], self.k_inv)[0]
                ray_new = normalize_pixels(pixel_new[None, :], self.k_inv)[0]
                point = triangulate(p_old, p_new, ray_old[None, :], ray_new[None, :])[0]
                if (r_old @ point + t_old)[2] <= 0 or (r_new @ point + t_new)[2] <= 0:
                    continue
                to_old = camera_center(r_old, t_old) - point
                to_new = camera_center(r_new, t_new) - point
                if angle_between_directions(to_old, to_new) < MIN_TRIANGULATION_ANGLE:
                    continue
                errors = [reprojection_error(self.k, r_old, t_old, point[None, :], pixel_old[None, :])[0],
                          reprojection_error(self.k, r_new, t_new, point[None, :], pixel_new[None, :])[0]]
                if max(errors) > TRIANGULATION_THRESHOLD:
                    continue
                index = len(points)
                points.append(point)
                self.tracks[(other, a)] = index
                self.tracks[(camera, b)] = index
                observations.append(Observation(index, other, pixel_old))
                observations.append(Observation(index, camera, pixel_new))
        return np.array(points), observations

    def run(self, seed_pair):
        recon = self.initialize(*seed_pair)
        recon, report = optimize(recon, self.k, BUNDLE_ITERATIONS)
        result = PipelineResult(recon, report)
        result.history.append((seed_pair[1], report.rms, len(recon.poses), len(recon.points)))

        blocked = set()
        while True:
            remaining = [c for c in self.features
                         if c not in result.reconstruction.poses and c not in blocked]
            if not remaining:
                break
            scored = sorted(remaining,
                            key=lambda c: -len(self._correspondences(result.reconstruction, c)))
            progressed = False
            for camera in scored:
                outcome, rejection = self._try_register(result.reconstruction, camera)
                if outcome is None:
                    result.rejected.append(rejection)
                    blocked.add(camera)
                    continue
                result.reconstruction, result.report = outcome
                result.history.append((camera, result.report.rms,
                                       len(result.reconstruction.poses),
                                       len(result.reconstruction.points)))
                progressed = True
                break
            if not progressed:
                break
        return result
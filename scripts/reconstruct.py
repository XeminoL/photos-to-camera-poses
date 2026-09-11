import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

from sfm.core.features import detect, match
from sfm.core.geometry import (
    angle_between_rotations,
    camera_center,
    relative_pose,
)
from sfm.core.pipeline import Session
from sfm.dataset import duplicate_views, load_middlebury

DATA_DIR = Path("data/templeSparseRing")
PAR_FILE = DATA_DIR / "templeSR_par.txt"
MIN_PAIR_MATCHES = 30


def build_session(cameras, seed):
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher()
    features = {}
    for i, camera in enumerate(cameras):
        image = cv2.imread(str(DATA_DIR / camera.name), cv2.IMREAD_GRAYSCALE)
        features[i] = detect(image, detector)
    matches = {}
    for i, j in itertools.combinations(range(len(cameras)), 2):
        pairs = match(features[i], features[j], matcher)
        if len(pairs) >= MIN_PAIR_MATCHES:
            matches[(i, j)] = pairs
    return Session(features, matches, cameras[0].k, seed), features, matches


def report(result, cameras, anchor):
    recon = result.reconstruction
    scale = None
    print(f"\n{len(recon.poses)} cameras, {len(recon.points)} points, "
          f"rms {result.report.rms:.4f} px, null space {result.report.null_space_dimension}")
    print("camera | angle error | predicted | center error | predicted")
    for c in sorted(recon.poses):
        r, t = recon.poses[c]
        r_true, t_true = relative_pose(cameras[anchor].r, cameras[anchor].t,
                                       cameras[c].r, cameras[c].t)
        if scale is None and np.linalg.norm(t_true) > 1e-9:
            scale = np.linalg.norm(t_true) / max(np.linalg.norm(t), 1e-12)
        angle = angle_between_rotations(r, r_true)
        center = np.linalg.norm(camera_center(r, t) * (scale or 1.0)
                                - camera_center(r_true, t_true))
        print(f"{c + 1:6d} | {angle:11.4f} | {result.report.angle_sigma.get(c, 0.0):9.4f} | "
              f"{center:12.6f} | {result.report.center_sigma.get(c, 0.0) * (scale or 1.0):9.6f}")
    for item in result.rejected:
        print(f"  rejected {item.camera + 1}: {item.reason} "
              f"(matches {item.matches}, inliers {item.inliers}"
              + (f", excess null {item.excess_null_dimensions}"
                 if item.excess_null_dimensions else "") + ")")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-pair", type=int, nargs=2, default=[12, 13])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cameras = load_middlebury(PAR_FILE)
    for i, j in duplicate_views(cameras):
        print(f"warning: views {i + 1} and {j + 1} share the same centre")

    session, _, _ = build_session(cameras, args.seed)
    left, right = args.seed_pair[0] - 1, args.seed_pair[1] - 1
    result = session.run((left, right))
    report(result, cameras, left)


if __name__ == "__main__":
    main()
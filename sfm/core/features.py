from dataclasses import dataclass

import numpy as np

LOWE_RATIO = 0.75


@dataclass
class ImageFeatures:
    keypoints: np.ndarray
    descriptors: np.ndarray

    def __len__(self):
        return len(self.keypoints)


def detect(image, detector):
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return ImageFeatures(np.array([kp.pt for kp in keypoints]), descriptors)


def match(left, right, matcher, ratio=LOWE_RATIO):
    pairs = matcher.knnMatch(left.descriptors, right.descriptors, k=2)
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    return np.array([[m.queryIdx, m.trainIdx] for m in good], dtype=int).reshape(-1, 2)
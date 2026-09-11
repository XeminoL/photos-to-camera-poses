from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GroundTruthCamera:
    name: str
    k: np.ndarray
    r: np.ndarray
    t: np.ndarray


def load_middlebury(par_file):
    par_file = Path(par_file)
    lines = par_file.read_text().strip().splitlines()
    cameras = []
    for line in lines[1:1 + int(lines[0])]:
        parts = line.split()
        values = np.array([float(v) for v in parts[1:]])
        cameras.append(GroundTruthCamera(parts[0],
                                         values[:9].reshape(3, 3),
                                         values[9:18].reshape(3, 3),
                                         values[18:]))
    return cameras


def duplicate_views(cameras, tolerance=1e-9):
    duplicates = []
    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            if np.abs(cameras[i].t - cameras[j].t).max() < tolerance:
                duplicates.append((i, j))
    return duplicates
import json
from pathlib import Path

import cv2
import numpy as np

OUTPUT = Path("web/tests/temple.json")
DATA = Path("data/templeSparseRing")
SIZE = (640, 480)


def main():
    lines = (DATA / "templeSR_par.txt").read_text().strip().splitlines()
    count = int(lines[0])
    cameras = []
    images = []
    for line in lines[1:1 + count]:
        parts = line.split()
        values = np.array([float(v) for v in parts[1:]])
        full = cv2.imread(str(DATA / parts[0]), cv2.IMREAD_GRAYSCALE)
        scale = SIZE[0] / full.shape[1]
        small = cv2.resize(full, SIZE)
        k = values[:9].reshape(3, 3).copy()
        k[:2] *= scale
        cameras.append({
            "name": parts[0],
            "k": k.tolist(),
            "r": values[9:18].reshape(3, 3).tolist(),
            "t": values[18:].tolist(),
        })
        images.append((small.astype(np.float32) / 255.0).ravel().tolist())

    payload = {"width": SIZE[0], "height": SIZE[1], "cameras": cameras, "images": images}
    OUTPUT.write_text(json.dumps(payload))
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"wrote {OUTPUT} ({size_mb:.1f} MB)")
    print(f"  {count} views at {SIZE[0]}x{SIZE[1]}, intrinsics rescaled by {SIZE[0] / 640:.3f}")


if __name__ == "__main__":
    main()
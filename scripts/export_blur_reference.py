import json
from pathlib import Path

import cv2
import numpy as np

OUTPUT = Path("web/tests/blur-reference.json")
SIZE = (64, 48)
SIGMAS = [0.8, 1.6, 3.2]


def main():
    image = cv2.imread("data/templeSparseRing/templeSR0012.png", cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(image, SIZE).astype(np.float32) / 255.0

    blurred = {}
    for sigma in SIGMAS:
        radius = max(1, int(np.ceil(4 * sigma)))
        size = 2 * radius + 1
        out = cv2.GaussianBlur(small, (size, size), sigma, borderType=cv2.BORDER_REFLECT101)
        blurred[str(sigma)] = out.ravel().tolist()

    payload = {
        "width": SIZE[0],
        "height": SIZE[1],
        "image": small.ravel().tolist(),
        "blurred": blurred,
        "kernels": {
            str(s): cv2.getGaussianKernel(2 * max(1, int(np.ceil(4 * s))) + 1, s).ravel().tolist()
            for s in SIGMAS
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    for sigma in SIGMAS:
        arr = np.array(blurred[str(sigma)])
        print(f"  sigma {sigma}: mean {arr.mean():.6f} min {arr.min():.6f} max {arr.max():.6f}")


if __name__ == "__main__":
    main()
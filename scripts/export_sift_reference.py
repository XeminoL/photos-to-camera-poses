import json
from pathlib import Path

import cv2
import numpy as np

OUTPUT = Path("web/tests/sift-reference.json")
SIZE = (160, 120)
IMAGE = "data/templeSparseRing/templeSR0012.png"


def main():
    full = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(full, SIZE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(small, None)

    order = np.argsort([-kp.response for kp in keypoints])
    listed = []
    for index in order:
        kp = keypoints[index]
        listed.append({
            "x": float(kp.pt[0]),
            "y": float(kp.pt[1]),
            "size": float(kp.size),
            "angle": float(kp.angle),
            "response": float(kp.response),
            "octave": int(kp.octave & 255),
        })

    payload = {
        "width": SIZE[0],
        "height": SIZE[1],
        "image": (small.astype(np.float32) / 255.0).ravel().tolist(),
        "keypoints": listed,
        "descriptorSample": descriptors[order[:20]].tolist(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    print(f"  {len(listed)} keypoints on {SIZE[0]}x{SIZE[1]}")
    sizes = np.array([k["size"] for k in listed])
    print(f"  size range {sizes.min():.3f} to {sizes.max():.3f}, median {np.median(sizes):.3f}")
    responses = np.array([k["response"] for k in listed])
    print(f"  response range {responses.min():.6f} to {responses.max():.6f}")
    octaves = {}
    for k in listed:
        octaves[k["octave"]] = octaves.get(k["octave"], 0) + 1
    print(f"  per octave: {dict(sorted(octaves.items()))}")


if __name__ == "__main__":
    main()
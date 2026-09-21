> [!NOTE]
> **Status:** This project is inactive and no longer updated.

Photos taken around an object, in. Camera positions and a sparse point cloud, out.

![The interface after a run](docs/screenshot.png)

Python 3.11+, Node 18+. 

The hard part is knowing when *not* to place a camera. Two photos from the same
spot give near-zero residuals and no geometry. Reprojection error misses this;
the rank of `JᵀJ` catches it. One null direction is scale, which photos can
never fix. Wider than that, the camera is refused.

Tuning came from measurement: RANSAC is tight because 2 outliers in 263 points
cost 9× the direction error, and track growth is off because 2× the points cost
8× the error. Hessian is 95.7% zeros, solved by Schur complement.

SIFT written from scratch, not imported. Against OpenCV: every keypoint within
1 px both ways, descriptor cosine 1.0000.

16-image Middlebury temple: 4 cameras, 247 points, 0.1511 px reprojection, true
angle error 0.12° to 0.34° against gantry-measured positions. Data not bundled:
[vision.middlebury.edu/mview/data](https://vision.middlebury.edu/mview/data/).

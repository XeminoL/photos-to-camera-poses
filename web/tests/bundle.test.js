import assert from 'node:assert/strict';
import test from 'node:test';

import { SCALE_FREEDOM, optimize } from '../src/bundle.js';
import { angleBetweenRotations, identity, matmul, project, rodrigues } from '../src/geometry.js';

const K = [[1520.4, 0, 302.32], [0, 1525.9, 246.87], [0, 0, 1]];

function seeded(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function build({ pointCount = 40, cameraCount = 3, noise = 0, seed = 5 } = {}) {
  const random = seeded(seed);
  const points = Array.from({ length: pointCount }, () =>
    [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1 + 6]);
  const poses = new Map([[0, { r: identity(), t: [0, 0, 0] }]]);
  for (let c = 1; c < cameraCount; c += 1) {
    poses.set(c, {
      r: rodrigues([0.03 * c, 0.09 * c, 0.02 * c]),
      t: [0.5 * c, -0.1 * c, 0.15 * c],
    });
  }
  const observations = [];
  for (const [camera, pose] of poses) {
    points.forEach((point, index) => {
      const pixel = project(K, pose.r, pose.t, point);
      observations.push({
        point: index,
        camera,
        pixel: noise ? pixel.map((v) => v + (random() * 2 - 1) * noise) : pixel,
      });
    });
  }
  return { poses, points, observations, anchor: 0 };
}

test('perfect data leaves no residual', () => {
  const { report } = optimize(build(), K, 5);
  assert.ok(report.rms < 1e-8, `rms was ${report.rms}`);
});

test('only scale is free when the anchor is fixed', () => {
  const { report } = optimize(build({ noise: 0.2 }), K, 10);
  assert.equal(report.nullSpaceDimension, SCALE_FREEDOM);
  assert.ok(!report.isDegenerate);
});

test('optimisation reduces the residual', () => {
  const truth = build({ noise: 0.4, seed: 11 });
  const poses = new Map();
  for (const [camera, pose] of truth.poses) {
    poses.set(camera, camera === truth.anchor ? pose : {
      r: matmul(rodrigues([0.01, -0.015, 0.008]), pose.r),
      t: pose.t.map((v, i) => v + [0.02, -0.01, 0.015][i]),
    });
  }
  const start = { ...truth, poses, points: truth.points.map((p) => p.map((v) => v + 0.02)) };
  const before = optimize(start, K, 0).report.rms;
  const after = optimize(start, K, 25).report.rms;
  assert.ok(after < before / 2, `${after} not much better than ${before}`);
});

test('recovers pose from a perturbed start', () => {
  const truth = build({ seed: 3 });
  const poses = new Map();
  for (const [camera, pose] of truth.poses) {
    poses.set(camera, camera === truth.anchor ? pose : {
      r: matmul(rodrigues([0.02, -0.02, 0.01]), pose.r),
      t: pose.t.map((v, i) => v + [0.03, 0.02, -0.02][i]),
    });
  }
  const start = { ...truth, poses, points: truth.points.map((p) => p.map((v) => v * 1.02)) };
  const { reconstruction, report } = optimize(start, K, 40);
  assert.ok(report.rms < 1e-4, `rms was ${report.rms}`);
  for (const [camera, pose] of truth.poses) {
    assert.ok(angleBetweenRotations(reconstruction.poses.get(camera).r, pose.r) < 1e-3);
  }
});

test('a camera seeing too few points is degenerate', () => {
  const recon = build({ pointCount: 40, cameraCount: 2, noise: 0.2, seed: 8 });
  const starved = 99;
  const pose = { r: rodrigues([0.05, 0.05, 0]), t: [0.3, 0, 0.1] };
  recon.poses.set(starved, pose);
  for (let i = 0; i < 2; i += 1) {
    recon.observations.push({
      point: i,
      camera: starved,
      pixel: project(K, pose.r, pose.t, recon.points[i]),
    });
  }
  const { report } = optimize(recon, K, 5);
  assert.ok(report.isDegenerate);
  assert.equal(report.excessNullDimensions, 2);
});

test('uncertainty grows when observations shrink', () => {
  const rich = optimize(build({ pointCount: 60, noise: 0.3, seed: 2 }), K, 10).report;
  const poor = optimize(build({ pointCount: 12, noise: 0.3, seed: 2 }), K, 10).report;
  assert.ok(poor.angleSigma.get(1) > rich.angleSigma.get(1));
});

test('anchor camera stays fixed', () => {
  const { reconstruction } = optimize(build({ noise: 0.3 }), K, 10);
  const anchor = reconstruction.poses.get(reconstruction.anchor);
  assert.ok(Math.max(...anchor.r.flat().map((v, i) => Math.abs(v - identity().flat()[i]))) < 1e-15);
  assert.ok(Math.max(...anchor.t.map(Math.abs)) < 1e-15);
});
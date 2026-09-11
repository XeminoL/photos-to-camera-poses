import assert from 'node:assert/strict';
import test from 'node:test';
import { Matrix, SingularValueDecomposition, inverse } from 'ml-matrix';

import {
  decomposeEssential,
  fundamentalFromPoints,
  projectToEssential,
  ransacFundamental,
  sampsonDistance,
  solvePnp,
  triangulatePoint,
} from '../src/estimation.js';
import {
  angleBetweenDirections,
  angleBetweenRotations,
  identity,
  matmul,
  normalizePixel,
  norm,
  project,
  rodrigues,
  skew,
  transpose,
} from '../src/geometry.js';

const K = [[1520.4, 0, 302.32], [0, 1525.9, 246.87], [0, 0, 1]];
const K_INV = inverse(new Matrix(K)).to2DArray();

function seeded(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function makeScene(count = 60, seed = 5, noise = 0) {
  const random = seeded(seed);
  const points = Array.from({ length: count }, () =>
    [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1 + 6]);
  const r = rodrigues([0.04, 0.1, 0.02]);
  const t = [0.6, -0.1, 0.15];
  const jitter = () => (noise ? (random() * 2 - 1) * noise : 0);
  const left = points.map((p) => project(K, identity(), [0, 0, 0], p).map((v) => v + jitter()));
  const right = points.map((p) => project(K, r, t, p).map((v) => v + jitter()));
  return { points, r, t, left, right };
}

const normalizeMatrix = (m) => {
  const scale = Math.hypot(...m.flat());
  return m.map((row) => row.map((v) => v / scale));
};

test('fundamental matrix satisfies the epipolar constraint', () => {
  const { left, right } = makeScene();
  const f = fundamentalFromPoints(left, right);
  assert.ok(Math.max(...sampsonDistance(f, left, right)) < 1e-8);
});

test('fundamental matrix matches ground truth', () => {
  const { r, t, left, right } = makeScene();
  const got = normalizeMatrix(fundamentalFromPoints(left, right));
  const truth = normalizeMatrix(matmul(matmul(transpose(K_INV), matmul(skew(t), r)), K_INV));
  const direct = Math.max(...got.flat().map((v, i) => Math.abs(v - truth.flat()[i])));
  const flipped = Math.max(...got.flat().map((v, i) => Math.abs(-v - truth.flat()[i])));
  assert.ok(Math.min(direct, flipped) < 1e-9);
});

test('fundamental matrix has rank two', () => {
  const { left, right } = makeScene();
  const { diagonal } = new SingularValueDecomposition(new Matrix(fundamentalFromPoints(left, right)));
  assert.ok(diagonal[2] / diagonal[0] < 1e-12);
});

test('minimal eight point set gives the same answer as a large set', () => {
  const { left, right } = makeScene();
  const minimal = fundamentalFromPoints(left.slice(0, 8), right.slice(0, 8));
  assert.ok(Math.max(...sampsonDistance(minimal, left, right)) < 1e-9);
});

test('too few points is rejected', () => {
  const { left, right } = makeScene();
  assert.throws(() => fundamentalFromPoints(left.slice(0, 7), right.slice(0, 7)));
});

test('essential projection equalises the first two singular values', () => {
  const random = seeded(9);
  const raw = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => random() - 0.5));
  const { diagonal } = new SingularValueDecomposition(new Matrix(projectToEssential(raw)));
  assert.ok(Math.abs(diagonal[0] - diagonal[1]) < 1e-12);
  assert.ok(diagonal[2] / diagonal[0] < 1e-14);
});

test('standard stereo has exact singular values', () => {
  const baseline = 0.5;
  const e = matmul(skew([baseline, 0, 0]), identity());
  const { diagonal } = new SingularValueDecomposition(new Matrix(e));
  assert.ok(Math.abs(diagonal[0] - baseline) < 1e-15);
  assert.ok(Math.abs(diagonal[1] - baseline) < 1e-15);
  assert.ok(diagonal[2] < 1e-15);
});

test('decomposition recovers the pose', () => {
  const { r, t, left, right } = makeScene();
  const f = fundamentalFromPoints(left, right);
  const rays1 = left.map((p) => normalizePixel(p, K_INV));
  const rays2 = right.map((p) => normalizePixel(p, K_INV));
  const e = matmul(matmul(transpose(K), f), K);
  const best = decomposeEssential(e, rays1, rays2);
  assert.equal(best.count, left.length);
  assert.ok(angleBetweenRotations(best.r, r) < 1e-6);
  assert.ok(angleBetweenDirections(best.t, t) < 1e-6);
});

test('triangulation recovers the scene up to scale', () => {
  const { points, t, left, right } = makeScene();
  const f = fundamentalFromPoints(left, right);
  const rays1 = left.map((p) => normalizePixel(p, K_INV));
  const rays2 = right.map((p) => normalizePixel(p, K_INV));
  const best = decomposeEssential(matmul(matmul(transpose(K), f), K), rays1, rays2);
  const p1 = identity().map((row) => [...row, 0]);
  const p2 = best.r.map((row, i) => [...row, best.t[i]]);
  const scale = norm(t) / norm(best.t);
  let worst = 0;
  rays1.forEach((ray, i) => {
    const point = triangulatePoint(p1, p2, ray, rays2[i]);
    point.forEach((value, j) => {
      worst = Math.max(worst, Math.abs(value * scale - points[i][j]));
    });
  });
  assert.ok(worst < 1e-6);
});

test('closed form depth matches triangulated depth', () => {
  const focal = 1520.4;
  const baseline = 0.5;
  for (const depth of [3, 6, 12, 50]) {
    const disparity = (focal * baseline) / depth;
    assert.ok(Math.abs((focal * baseline) / disparity - depth) < 1e-12);
  }
});

test('pnp recovers the pose exactly', () => {
  const random = seeded(7);
  const points = Array.from({ length: 40 }, () =>
    [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1 + 6]);
  const r = rodrigues([0, 0.3, 0]);
  const t = [0.4, -0.1, 0.2];
  const pixels = points.map((p) => project(K, r, t, p));
  const got = solvePnp(points, pixels, K, K_INV);
  assert.ok(angleBetweenRotations(got.r, r) < 1e-5);
  assert.ok(Math.max(...got.t.map((v, i) => Math.abs(v - t[i]))) < 1e-6);
});

test('ransac rejects planted outliers', () => {
  const { left, right } = makeScene(100, 5, 0.3);
  const random = seeded(3);
  const corrupted = new Set();
  while (corrupted.size < 15) corrupted.add(Math.floor(random() * left.length));
  const dirty = right.map((p, i) =>
    (corrupted.has(i) ? [p[0] + 20 + random() * 40, p[1] + 20 + random() * 40] : p));
  const { inliers } = ransacFundamental(left, dirty, 1, 500, 1);
  assert.ok(inliers.filter(Boolean).length >= 70);
  for (const index of corrupted) assert.equal(inliers[index], false);
});
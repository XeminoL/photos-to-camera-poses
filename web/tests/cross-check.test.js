import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { Matrix, inverse } from 'ml-matrix';

import {
  decomposeEssential,
  fundamentalFromPoints,
  ransacFundamental,
  solvePnp,
  triangulatePoint,
} from '../src/estimation.js';
import {
  angleBetweenDirections,
  angleBetweenRotations,
  identity,
  matmul,
  normalizePixel,
  transpose,
} from '../src/geometry.js';

const reference = JSON.parse(
  readFileSync(fileURLToPath(new URL('./reference.json', import.meta.url)), 'utf8'),
);
const K = reference.intrinsics;
const K_INV = inverse(new Matrix(K)).to2DArray();

const worstDiff = (a, b) => Math.max(...a.flat().map((v, i) => Math.abs(v - b.flat()[i])));

const signAgnosticDiff = (a, b) => Math.min(
  worstDiff(a, b),
  worstDiff(a.map((row) => row.map((v) => -v)), b),
);

test('javascript fundamental matrix matches python', () => {
  const { left, right } = reference.scene;
  const got = fundamentalFromPoints(left, right);
  const scale = Math.hypot(...got.flat());
  const normalized = got.map((row) => row.map((v) => v / scale));
  assert.ok(signAgnosticDiff(normalized, reference.fundamental) < 1e-9);
});

test('javascript pose matches python', () => {
  const { left, right } = reference.scene;
  const f = fundamentalFromPoints(left, right);
  const rays1 = left.map((p) => normalizePixel(p, K_INV));
  const rays2 = right.map((p) => normalizePixel(p, K_INV));
  const best = decomposeEssential(matmul(matmul(transpose(K), f), K), rays1, rays2);

  assert.equal(best.count, reference.pose.inFront);
  assert.ok(angleBetweenRotations(best.r, reference.pose.rotation) < 1e-5);
  assert.ok(angleBetweenDirections(best.t, reference.pose.translation) < 1e-5);
});

test('javascript point cloud matches python', () => {
  const { left, right } = reference.scene;
  const f = fundamentalFromPoints(left, right);
  const rays1 = left.map((p) => normalizePixel(p, K_INV));
  const rays2 = right.map((p) => normalizePixel(p, K_INV));
  const best = decomposeEssential(matmul(matmul(transpose(K), f), K), rays1, rays2);
  const p1 = identity().map((row) => [...row, 0]);
  const p2 = best.r.map((row, i) => [...row, best.t[i]]);

  let worst = 0;
  rays1.forEach((ray, i) => {
    const point = triangulatePoint(p1, p2, ray, rays2[i]);
    point.forEach((value, j) => {
      worst = Math.max(worst, Math.abs(value - reference.cloud[i][j]));
    });
  });
  assert.ok(worst < 1e-8, `cloud differs by ${worst}`);
});

test('javascript pnp matches python', () => {
  const { points, pixels, rotation, translation } = reference.pnp;
  const got = solvePnp(points, pixels, K, K_INV);
  assert.ok(angleBetweenRotations(got.r, rotation) < 1e-5);
  assert.ok(Math.max(...got.t.map((v, i) => Math.abs(v - translation[i]))) < 1e-8);
});

test('javascript ransac keeps the same inliers as python', () => {
  const { left, right, corrupted, inlierCount } = reference.ransac;
  const { inliers } = ransacFundamental(left, right, 1, 500, 1);
  const kept = inliers.filter(Boolean).length;

  for (const index of corrupted) {
    assert.equal(inliers[index], false, `outlier ${index} was kept`);
  }
  assert.ok(Math.abs(kept - inlierCount) <= 3,
    `javascript kept ${kept}, python kept ${inlierCount}`);
});
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  angleBetweenRotations,
  apply,
  cameraCenter,
  identity,
  matmul,
  norm,
  project,
  projectionJacobian,
  rodrigues,
  rotationAngle,
  rotationAngleFromPhase,
  rotationAxis,
  skew,
  subtract,
  transpose,
} from '../src/geometry.js';

const K = [[1520.4, 0, 302.32], [0, 1525.9, 246.87], [0, 0, 1]];

function seeded(seed) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648 - 0.5;
  };
}

const maxAbsDiff = (a, b) =>
  Math.max(...a.flat().map((value, i) => Math.abs(value - b.flat()[i])));

test('skew is antisymmetric and kills its own vector', () => {
  const v = [0.3, -1.2, 0.7];
  assert.ok(maxAbsDiff(skew(v), transpose(skew(v)).map((r) => r.map((x) => -x))) < 1e-15);
  assert.ok(norm(apply(skew(v), v)) < 1e-15);
});

test('rodrigues produces a rotation matrix', () => {
  const random = seeded(7);
  for (let i = 0; i < 20; i += 1) {
    const r = rodrigues([random(), random(), random()]);
    assert.ok(maxAbsDiff(matmul(transpose(r), r), identity()) < 1e-14);
  }
});

test('rodrigues of zero is the identity', () => {
  assert.ok(maxAbsDiff(rodrigues([0, 0, 0]), identity()) < 1e-15);
});

test('phase angle matches trace angle', () => {
  const random = seeded(11);
  for (let i = 0; i < 20; i += 1) {
    const r = rodrigues([random(), random(), random()]);
    assert.ok(Math.abs(rotationAngleFromPhase(r) - rotationAngle(r)) < 1e-9);
  }
});

test('known rotation gives back its angle and axis', () => {
  const axis = [1, 2, 3];
  const length = norm(axis);
  const unit = axis.map((v) => v / length);
  const r = rodrigues(unit.map((v) => v * (37 * Math.PI) / 180));
  assert.ok(Math.abs(rotationAngle(r) - 37) < 1e-10);
  assert.ok(Math.abs(rotationAngleFromPhase(r) - 37) < 1e-10);
  const recovered = rotationAxis(r);
  assert.ok(Math.max(...unit.map((v, i) => Math.abs(Math.abs(v) - Math.abs(recovered[i])))) < 1e-10);
});

test('nth root of a rotation stays a rotation', () => {
  const full = (120 * Math.PI) / 180;
  const r = rodrigues([0, 0, full]);
  for (let branch = 0; branch < 3; branch += 1) {
    const root = rodrigues([0, 0, (full + 2 * Math.PI * branch) / 3]);
    const cubed = matmul(matmul(root, root), root);
    assert.ok(maxAbsDiff(cubed, r) < 1e-13);
    assert.ok(maxAbsDiff(matmul(transpose(root), root), identity()) < 1e-14);
  }
});

test('camera centre round trips', () => {
  const r = rodrigues([0.2, -0.4, 0.1]);
  const t = [0.5, -0.2, 0.3];
  const back = apply(r, cameraCenter(r, t)).map((value, i) => value + t[i]);
  assert.ok(norm(back) < 1e-12);
});

test('projection jacobian matches finite differences at zero', () => {
  const base = rodrigues([0.3, -0.1, 0.2]);
  const t = [0.3, -0.1, 0.2];
  const point = [0.2, 0.1, 5];
  const { dAngle, dTranslation, dPoint } = projectionJacobian(K, base, t, point);
  const eps = 1e-7;

  for (let col = 0; col < 3; col += 1) {
    const step = [0, 0, 0];
    step[col] = eps;

    const angleUp = project(K, matmul(rodrigues(step), base), t, point);
    const angleDown = project(K, matmul(rodrigues(step.map((v) => -v)), base), t, point);
    const translationUp = project(K, base, t.map((v, i) => v + step[i]), point);
    const translationDown = project(K, base, t.map((v, i) => v - step[i]), point);
    const pointUp = project(K, base, t, point.map((v, i) => v + step[i]));
    const pointDown = project(K, base, t, point.map((v, i) => v - step[i]));

    for (let row = 0; row < 2; row += 1) {
      assert.ok(Math.abs(dAngle[row][col] - (angleUp[row] - angleDown[row]) / (2 * eps)) < 1e-3);
      assert.ok(Math.abs(dTranslation[row][col]
        - (translationUp[row] - translationDown[row]) / (2 * eps)) < 1e-5);
      assert.ok(Math.abs(dPoint[row][col] - (pointUp[row] - pointDown[row]) / (2 * eps)) < 1e-5);
    }
  }
});

test('angle between rotations is symmetric', () => {
  const a = rodrigues([0.1, 0.2, -0.3]);
  const b = rodrigues([-0.2, 0.05, 0.4]);
  assert.ok(Math.abs(angleBetweenRotations(a, b) - angleBetweenRotations(b, a)) < 1e-10);
});
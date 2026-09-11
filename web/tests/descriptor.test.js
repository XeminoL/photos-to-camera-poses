import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
  DESCRIPTOR_LENGTH,
  DESCRIPTOR_MAX,
  DESCRIPTOR_OUTPUT_SCALE,
  computeDescriptor,
  computeOrientations,
  describeKeypoints,
} from '../src/descriptor.js';

const unit = (descriptor) => {
  const norm = Math.hypot(...descriptor);
  return [...descriptor].map((value) => value / norm);
};

const cosine = (a, b) => {
  const x = unit(a);
  const y = unit(b);
  return x.reduce((sum, value, i) => sum + value * y[i], 0);
};
import { findKeypoints } from '../src/keypoints.js';
import { buildDifferenceOfGaussian, buildGaussianPyramid, octaveCount } from '../src/pyramid.js';

const reference = JSON.parse(
  readFileSync(fileURLToPath(new URL('./sift-reference.json', import.meta.url)), 'utf8'),
);

const source = {
  width: reference.width,
  height: reference.height,
  data: Float32Array.from(reference.image),
};

function pipeline(image) {
  const pyramid = buildGaussianPyramid(image, octaveCount(image.width, image.height));
  const keypoints = findKeypoints(buildDifferenceOfGaussian(pyramid));
  return describeKeypoints(pyramid, keypoints);
}

function gradientImage(size, angle) {
  const data = new Float32Array(size * size);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const cx = x - size / 2;
      const cy = y - size / 2;
      data[y * size + x] = 0.5 + 0.4 * Math.tanh((cx * cos + cy * sin) / 3);
    }
  }
  return { width: size, height: size, data };
}

test('orientation follows the image gradient', () => {
  for (const radians of [0, Math.PI / 4, Math.PI / 2, -Math.PI / 3]) {
    const image = gradientImage(64, radians);
    const angles = computeOrientations(image, 32, 32, 2);
    assert.ok(angles.length >= 1, 'no orientation found');
    const expected = (((radians * 180) / Math.PI) % 360 + 360) % 360;
    const closest = angles.reduce((best, angle) => {
      let diff = Math.abs(angle - expected);
      if (diff > 180) diff = 360 - diff;
      return Math.min(best, diff);
    }, Infinity);
    assert.ok(closest < 12, `expected ${expected.toFixed(1)} deg, off by ${closest.toFixed(1)}`);
  }
});

test('a flat patch has no orientation', () => {
  const flat = { width: 40, height: 40, data: new Float32Array(1600).fill(0.4) };
  assert.equal(computeOrientations(flat, 20, 20, 2).length, 0);
});

test('descriptor uses the opencv integer scale', () => {
  const image = gradientImage(64, 0.3);
  const descriptor = computeDescriptor(image, 32, 32, 2, 17);
  assert.equal(descriptor.length, DESCRIPTOR_LENGTH);
  assert.equal(DESCRIPTOR_LENGTH, 128);
  for (const value of descriptor) {
    assert.ok(value >= 0 && value <= DESCRIPTOR_MAX, `value ${value} out of range`);
    assert.equal(value, Math.round(value), 'entries must be whole numbers');
  }
  assert.ok(Math.hypot(...descriptor) < DESCRIPTOR_OUTPUT_SCALE + 1);
});

test('no single entry dominates the descriptor', () => {
  const image = gradientImage(64, 0.8);
  const descriptor = computeDescriptor(image, 32, 32, 2, 46);
  const scaled = unit(descriptor);
  const largest = Math.max(...scaled);
  assert.ok(largest < 0.6, `one entry dominates at ${largest.toFixed(3)}`);
  const above = scaled.filter((value) => value > 0.2).length;
  assert.ok(above < scaled.length / 3, `${above} entries exceed the clip`);
});

test('descriptor is invariant to rotating the patch', () => {
  const base = gradientImage(96, 0);
  const turned = gradientImage(96, Math.PI / 3);
  const baseAngle = computeOrientations(base, 48, 48, 3)[0];
  const turnedAngle = computeOrientations(turned, 48, 48, 3)[0];
  let turn = Math.abs(turnedAngle - baseAngle);
  if (turn > 180) turn = 360 - turn;
  assert.ok(Math.abs(turn - 60) < 8, `patch turned by ${turn.toFixed(1)} deg, expected 60`);
  const first = computeDescriptor(base, 48, 48, 3, baseAngle);
  const second = computeDescriptor(turned, 48, 48, 3, turnedAngle);
  const similarity = cosine(first, second);
  assert.ok(similarity > 0.9, `rotated descriptor only matched at ${similarity.toFixed(3)}`);
});

test('different patterns give different descriptors', () => {
  const first = computeDescriptor(gradientImage(64, 0), 32, 32, 2, 0);
  const blob = { width: 64, height: 64, data: new Float32Array(4096).fill(0.2) };
  for (let y = 24; y < 40; y += 1) {
    for (let x = 24; x < 40; x += 1) blob.data[y * 64 + x] = 0.9;
  }
  const second = computeDescriptor(blob, 32, 32, 2, 0);
  const similarity = cosine(first, second);
  assert.ok(similarity < 0.8, `unrelated patches matched at ${similarity.toFixed(3)}`);
});

test('describing a real image yields opencv style descriptors', () => {
  const described = pipeline(source);
  assert.ok(described.length > 50, `only described ${described.length}`);
  for (const point of described) {
    assert.equal(point.descriptor.length, 128);
    assert.ok(Math.max(...point.descriptor) > 0, 'descriptor is empty');
    assert.ok(Math.max(...point.descriptor) <= DESCRIPTOR_MAX);
    assert.ok(point.angle >= 0 && point.angle < 360, `angle ${point.angle} is not in degrees`);
  }
});

test('orientation assignment can split one keypoint into several', () => {
  const described = pipeline(source);
  const seen = new Map();
  for (const point of described) {
    const key = `${point.x.toFixed(4)},${point.y.toFixed(4)}`;
    seen.set(key, (seen.get(key) ?? 0) + 1);
  }
  assert.ok(described.length > seen.size, 'no keypoint received a second orientation');
});

test('descriptor count lands near opencv', () => {
  const described = pipeline(source);
  const ratio = described.length / reference.keypoints.length;
  assert.ok(ratio > 0.7 && ratio < 1.4,
    `described ${described.length}, opencv had ${reference.keypoints.length}`);
});
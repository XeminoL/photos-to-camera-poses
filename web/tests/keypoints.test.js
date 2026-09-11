import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { CONTRAST_THRESHOLD, findKeypoints } from '../src/keypoints.js';
import { buildDifferenceOfGaussian, buildGaussianPyramid, octaveCount } from '../src/pyramid.js';

const reference = JSON.parse(
  readFileSync(fileURLToPath(new URL('./sift-reference.json', import.meta.url)), 'utf8'),
);

const source = {
  width: reference.width,
  height: reference.height,
  data: Float32Array.from(reference.image),
};

const detect = () => {
  const octaves = octaveCount(source.width, source.height);
  return findKeypoints(buildDifferenceOfGaussian(buildGaussianPyramid(source, octaves)));
};

const nearest = (point, list) => list.reduce((best, other) => {
  const distance = Math.hypot(point.x - other.x, point.y - other.y);
  return distance < best.distance ? { distance, other } : best;
}, { distance: Infinity, other: null });

test('finds a sensible number of keypoints', () => {
  const found = detect();
  assert.ok(found.length > 20, `only found ${found.length}`);
  assert.ok(found.length < 10 * reference.keypoints.length,
    `found ${found.length}, opencv found ${reference.keypoints.length}`);
});

test('keypoints stay inside the image', () => {
  for (const point of detect()) {
    assert.ok(point.x >= 0 && point.x <= source.width);
    assert.ok(point.y >= 0 && point.y <= source.height);
    assert.ok(point.sigma > 0);
    assert.ok(Number.isFinite(point.response));
  }
});

test('every keypoint clears the contrast threshold', () => {
  for (const point of detect()) {
    assert.ok(point.response * 3 >= CONTRAST_THRESHOLD * 0.9,
      `response ${point.response} is too weak`);
  }
});

test('most opencv keypoints have a nearby match', () => {
  const found = detect();
  const strong = reference.keypoints.slice(0, 40);
  const matched = strong.filter((point) => nearest(point, found).distance < 2).length;
  assert.ok(matched >= strong.length * 0.6,
    `only ${matched} of ${strong.length} strong opencv keypoints were found`);
});

test('sub-pixel refinement gives fractional coordinates', () => {
  const found = detect();
  const fractional = found.filter((point) =>
    Math.abs(point.x - Math.round(point.x)) > 1e-6
    || Math.abs(point.y - Math.round(point.y)) > 1e-6);
  assert.ok(fractional.length > found.length / 2,
    `only ${fractional.length} of ${found.length} were refined`);
});

test('a flat image yields no keypoints', () => {
  const flat = { width: 64, height: 64, data: new Float32Array(64 * 64).fill(0.5) };
  const pyramid = buildGaussianPyramid(flat, 2);
  assert.equal(findKeypoints(buildDifferenceOfGaussian(pyramid)).length, 0);
});

test('a single blob is found once', () => {
  const size = 64;
  const data = new Float32Array(size * size).fill(0.2);
  for (let y = 28; y < 36; y += 1) {
    for (let x = 28; x < 36; x += 1) data[y * size + x] = 0.9;
  }
  const pyramid = buildGaussianPyramid({ width: size, height: size, data }, 2);
  const found = findKeypoints(buildDifferenceOfGaussian(pyramid));
  assert.ok(found.length >= 1, 'blob was missed');
  const closest = nearest({ x: 32, y: 32 }, found);
  assert.ok(closest.distance < 3, `nearest keypoint was ${closest.distance} away`);
});

test('keypoints spread across more than one octave', () => {
  const octaves = new Set(detect().map((point) => point.octave));
  assert.ok(octaves.size > 1, `all keypoints landed in octave ${[...octaves]}`);
});